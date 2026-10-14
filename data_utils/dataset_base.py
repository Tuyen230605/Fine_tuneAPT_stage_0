import os
import cv2
import time
import h5py
import copy
import torch
import random
import numpy as np
from collections import OrderedDict
from torch import Tensor
from einops import rearrange
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional
from torchvision.transforms import v2
from torch.utils.data import (
    Dataset, IterableDataset, 
    ConcatDataset, ChainDataset, 
    get_worker_info, DataLoader
)

from . import h5io, align


# collections of wrist camera keys for all datasets
WRIST_CAMERA_NAMES = (
    "eye_in_hand", # libero
    "wrist_camera", # maniskill
    "gripperPOV", # metaworld
    "wrist_left", # droid
    "eih_cam", # pickplace
    "left_cam", # agibot
    "right_cam", # agibot
    "left_camera", # robotwin
    "right_camera", # robotwin
    "ego_view", # internvl-m1
    "hand_cam", # internvl-a1
    "lh_cam", # internvl-a1, aloha
    "rh_cam",  # internvl-a1, aloha
    "wrist_cam", # franka
    )

def infer_record_dt(t: np.ndarray, default: float = 1.0):
    if len(t) < 2:
        return default
    else:
        return (t[-1] - t[0]) / (len(t) - 1)


def find_closest_ind(train: np.ndarray, query: np.ndarray):
    bin_indices = np.digitize(query, train)
    sample_indices = []
    
    # query < train
    mask = (bin_indices == 0)
    if np.any(mask):
        sample_indices.append(np.array([0]*mask.sum(), dtype=bin_indices.dtype))
    
    # query in train
    mask = (bin_indices > 0) & (bin_indices < len(train))
    r_ind = bin_indices[mask]
    l_ind = r_ind - 1
    
    dist0 = np.abs(train[l_ind] - query[mask])
    dist1 = np.abs(train[r_ind] - query[mask])
    sample_indices.append(np.where(dist0 < dist1, l_ind, r_ind))

    # query > train
    mask = (bin_indices == len(train))
    if np.any(mask):
        sample_indices.append(np.array([len(train)-1]*mask.sum(), dtype=bin_indices.dtype))
    
    sample_indices = np.concatenate(sample_indices)
    return sample_indices


def axis_angle_to_matrix(rotvec: np.ndarray):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float32)

    axis = rotvec / theta
    x, y, z = axis.astype(np.float32)
    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float32)
    eye = np.eye(3, dtype=np.float32)
    return eye + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class DataSampler(object):
    @staticmethod
    def _random_photometric_and_noise_aug(
        rgb: np.ndarray,
        mask: np.ndarray,
        photometric_aug: bool,
        noise_aug: bool,
        background_aug: bool
    ):
        """
        Args:
            rgb: (T, 3, H, W), float32 in [0, 1]
            mask: (T, H, W), bool
        """
        T, _, H, W = rgb.shape
        out = rgb.copy()

        # Robustness to lighting/background variation.
        if photometric_aug:
            brightness = np.random.uniform(0.75, 1.25)
            contrast = np.random.uniform(0.75, 1.25)
            saturation = np.random.uniform(0.75, 1.25)
            gamma = np.random.uniform(0.85, 1.15)

            out = out * brightness
            mean = out.mean(axis=(2, 3), keepdims=True)
            out = (out - mean) * contrast + mean
            gray = out.mean(axis=1, keepdims=True)
            out = (out - gray) * saturation + gray
            out = np.clip(out, 0.0, 1.0)
            out = np.power(out, gamma)

        # Robustness to background and clutter: random soft patches and tint.
        if background_aug:
            for t in range(T):
                if np.random.rand() < 0.5:
                    # Random global tint
                    tint = np.random.uniform(-0.06, 0.06, size=(3, 1, 1)).astype(np.float32)
                    out[t] = out[t] + tint

                # Random cutout-like background clutter; avoid masked foreground when possible.
                for _ in range(np.random.randint(1, 4)):
                    if np.random.rand() > 0.4:
                        continue
                    h = np.random.randint(max(8, H // 16), max(12, H // 5))
                    w = np.random.randint(max(8, W // 16), max(12, W // 5))
                    y0 = np.random.randint(0, max(1, H - h))
                    x0 = np.random.randint(0, max(1, W - w))
                    patch = np.random.uniform(0.0, 1.0, size=(3, h, w)).astype(np.float32)

                    # If valid mask is provided, bias patching to background region.
                    # Here `mask=True` means valid pixels; when mask is all ones this is still safe.
                    region_mask = mask[t, y0:y0+h, x0:x0+w]
                    bg_ratio = 1.0 - region_mask.mean()
                    if bg_ratio > 0.1:
                        fg = region_mask[None, ...].astype(np.float32)
                        out[t, :, y0:y0+h, x0:x0+w] = (
                            out[t, :, y0:y0+h, x0:x0+w] * fg + patch * (1.0 - fg)
                        )
                    else:
                        alpha = np.random.uniform(0.15, 0.35)
                        out[t, :, y0:y0+h, x0:x0+w] = (
                            (1.0 - alpha) * out[t, :, y0:y0+h, x0:x0+w] + alpha * patch
                        )

        # Robustness to sensor noise/compression-like degradation.
        if noise_aug:
            # if np.random.rand() < 0.8:
            #     sigma = np.random.uniform(0.005, 0.03)
            #     out = out + np.random.normal(0.0, sigma, size=out.shape).astype(np.float32)

            if np.random.rand() < 0.142857:
                # Per-image blur with random odd kernel.
                for t in range(T):
                    k = int(np.random.choice([3, 5]))
                    img = np.transpose(out[t], (1, 2, 0))  # HWC
                    img = cv2.GaussianBlur(img, (k, k), sigmaX=np.random.uniform(0.2, 1.0))
                    out[t] = np.transpose(img, (2, 0, 1))

        out = np.clip(out, 0.0, 1.0).astype(np.float32)
        return out

    @classmethod
    def pad2ncam(self, x: np.ndarray, num_camera: int, dim: int, zero_init: bool):
        pad_ncam = num_camera - x.shape[dim]
        if pad_ncam < 0:
            raise ValueError("[ERR ] current ncam = {}, which is larger than desired ncam = {}"
                             .format(x.shape[dim], num_camera))
        if pad_ncam > 0:
            pad = x.take([-1]*pad_ncam, axis=dim)
            if zero_init:
                pad[:] = 0
            x = np.concatenate([x, pad], axis=dim)
        return x

    @classmethod
    def pad2nee(self, x: np.ndarray, num_ee: int, dim: int):
        pad_nee = num_ee - x.shape[dim]
        if pad_nee < 0:
            raise ValueError("[ERR ] current nee = {}, which is larger than desired nee = {}"
                             .format(x.shape[dim], num_ee))
        if pad_nee > 0:
            pad = x.take([-1]*pad_nee, axis=dim)
            x = np.concatenate([x, pad], axis=dim)
        return x

    @classmethod
    def preprocess_images(
        cls,
        cam_names: List[str],
        Ks: List[np.ndarray],
        rgbs: List[np.ndarray],
        masks: Optional[List[np.ndarray]],
        output_image_hw: Optional[Tuple[int, int]] = None,
        geom_aug: Optional[bool] = False,
        photometric_aug: Optional[bool] = False,
        noise_aug: Optional[bool] = False,
        background_aug: Optional[bool] = False
    ):
        processed_Ks = []
        processed_rgbs = []
        processed_masks = []
        
        # Determine geometric augmentation parameters
        if geom_aug and output_image_hw is not None:
            # Generate random crop offset and scale for perspective shift
            # Crop between 85% to 100% of original image
            crop_scale = np.random.uniform(0.95, 1.0)
            # Center offset up to available margin
            max_offset_x = 1.0 - crop_scale
            max_offset_y = 1.0 - crop_scale
            offset_x = np.random.uniform(-max_offset_x/2, max_offset_x/2)
            offset_y = np.random.uniform(-max_offset_y/2, max_offset_y/2)
        else:
            crop_scale = 1.0
            offset_x = 0.0
            offset_y = 0.0

        for cam_idx, rgb in enumerate(rgbs):
            # rgb: (T, 3, H, W)
            K = Ks[cam_idx]  # (3, 3)

            if rgb.dtype == np.uint8:
                rgb = rgb.astype(np.float32) / 255.
            
            if masks is None or masks[cam_idx] is None:
                T, _, Hin, Win = rgb.shape
                mask = np.ones((T, Hin, Win), dtype=bool)
            else:
                mask = masks[cam_idx]
            
            # !!! dont do it for wrist camera !!!
            if cam_names[cam_idx] not in WRIST_CAMERA_NAMES:
                if geom_aug and output_image_hw is not None:
                    # Apply random crop/shift by cropping original image before scaling
                    T, _, Hin, Win = rgb.shape
                    crop_H = int(Hin * crop_scale)
                    crop_W = int(Win * crop_scale)
                    start_y = int((Hin - crop_H) / 2 + offset_y * Hin)
                    start_x = int((Win - crop_W) / 2 + offset_x * Win)
                    start_y = np.clip(start_y, 0, Hin - crop_H)
                    start_x = np.clip(start_x, 0, Win - crop_W)
                    
                    rgb = rgb[..., start_y:start_y+crop_H, start_x:start_x+crop_W]
                    mask = mask[..., start_y:start_y+crop_H, start_x:start_x+crop_W]
                    
                    # Update K for crop
                    K = ImageProcessor.tform_K_for_center_view(K, -start_x, -start_y)
            
            if output_image_hw is not None:
                Hout, Wout = output_image_hw
                rgb, metadata = ImageProcessor.scale_to_fit(rgb, Hout, Wout)
                mask, metadata = ImageProcessor.scale_to_fit(mask, Hout, Wout)
                K = ImageProcessor.tform_K_for_scale_to_fit(K, **metadata)

                rgb, metadata = ImageProcessor.center_view(rgb, Hout, Wout)
                mask, metadata = ImageProcessor.center_view(mask, Hout, Wout)
                K = ImageProcessor.tform_K_for_center_view(K, **metadata)

            rgb = cls._random_photometric_and_noise_aug(
                rgb=rgb,
                mask=mask,
                photometric_aug=photometric_aug,
                noise_aug=noise_aug,
                background_aug=background_aug
            )
            
            processed_Ks.append(K)
            processed_rgbs.append(rgb)
            processed_masks.append(mask)
        
        processed_Ks = np.stack(processed_Ks, axis=0)  # (Ncam, 3, 3)
        processed_rgbs = np.stack(processed_rgbs, axis=1)  # (T, Ncam, 3, H, W)
        processed_masks = np.stack(processed_masks, axis=1)  # (T, Ncam, H, W)
        return processed_Ks, processed_rgbs, processed_masks

    @classmethod
    def sample_framedict(
        cls,
        obs_traj: List[Dict[str, np.ndarray]], 
        ee_indices: List[int], 
        camera_names: List[str], 
        ee_ref_cams: Optional[Dict[str, Tuple[str]]], 
        num_history_cameras: int, 
        num_history_states: int, 
        num_future_states: int, 
        latest: bool = False, 
        sample_state_gaps: int = 1,
        sample_camera_gaps: int = 1, 
        sample_dt: float = 1.0,
        record_dt: Optional[float] = None, 
        output_image_hw: Optional[Tuple[int, int]] = None, 
        enable_seg: bool = False,
        geom_aug: bool = False, 
        photometric_aug: bool = False,
        noise_aug: bool = False,
        background_aug: bool = False,
        pad2ncam: int = -1,
        pad2nee: int = -1,
    ):
        """
        obs_traj is a list of dict containing necessary keys listed as followings.
        - ee_pose: np.ndarray of shape (4, 4) or (nee, 4, 4), ^{world}_{ee} T
        - gripper: float or np.ndarray of shape (nee,), value from [0 (close), 1 (open)]
        - timestamp: float, current timestamp
        - CAMERA_NAME_0: 
            - model: pinhole
            - camera:
                - width: int
                - height: int
                - K: np.ndarray of shape (3, 3) or (9,)
            - data:
                - color: np.ndarray, shape=(H, W, C)
                - seg: None | np.ndarray of shape (H, W) | isaacsim seg output
                - wcT: np.ndarray of shape (4, 4), ^{world}_{cam} T
        
        - CAMERA_NAME_1: similar as CAMERA_NAME_0
            - model: pinhole
            - ...
        """
        if not latest:
            last_obs_index = np.random.choice(len(obs_traj), 1)[0]
        else:
            last_obs_index = len(obs_traj) - 1

        all_timestamps = np.array([tau["timestamp"] for tau in obs_traj]).astype(np.float64)
        if record_dt is not None:
            # all_timestamps_calibrated = all_timestamps[-1] - np.arange(len(all_timestamps))[::-1] * record_dt
            all_timestamps_calibrated = all_timestamps
        else:
            all_timestamps_calibrated = all_timestamps
            record_dt = infer_record_dt(all_timestamps, default=sample_dt)
        
        if sample_dt is None:
            sample_dt = record_dt
        
        # we don't interpolate the images, but find the image with closest timestamp
        current_time = all_timestamps_calibrated[last_obs_index]
        prev_obs_sample_time = current_time + np.arange(-num_history_cameras+1, 1) * sample_camera_gaps * sample_dt
        prev_obs_sample_ind = find_closest_ind(all_timestamps_calibrated, prev_obs_sample_time)

        obs_across_cams: Dict[str, list] = {}  # property_name -> list[property], list len = num_cameras
        for cam_name in camera_names:
            obs_cam = h5io.gather_frames(obs_traj, cam_name, prev_obs_sample_ind, compress=False)
            for k, v in obs_cam.items():
                if k not in obs_across_cams:
                    obs_across_cams[k] = []
                obs_across_cams[k].append(v)

        K, rgbs, masks = cls.preprocess_images(
            cam_names=camera_names,
            Ks=obs_across_cams["K"],
            rgbs=obs_across_cams["rgb"],
            masks=obs_across_cams.get("mask", None) if enable_seg else None,
            output_image_hw=output_image_hw,
            geom_aug=geom_aug,
            photometric_aug=photometric_aug,
            noise_aug=noise_aug,
            background_aug=background_aug
        )
        # K: (ncam, 3, 3); rgbs: (T, ncam, 3, H, W); masks: (T, ncam, H, W)
        
        cam_poses = np.stack(obs_across_cams["pose"], axis=1)   # (T, ncam, 4, 4)
        ee_poses = h5io.gather_ee_poses(obs_traj, prev_obs_sample_ind)  # (T, nee, 4, 4)
        
        # interpolate the robot states
        all_ee_poses = np.stack([tau["ee_pose"] for tau in obs_traj], axis=0)  # (L, nee, 4, 4)
        all_grippers = np.array([tau["gripper"] for tau in obs_traj])  # (L, nee)
        all_states = {"ee_pose": all_ee_poses, "gripper": all_grippers}
        interp_funcs = {"ee_pose": align.interp_SE3_sep, "gripper": align.interp_linear}
        
        history_time = current_time + np.arange(-num_history_states+1, 1) * sample_state_gaps * sample_dt
        future_time = current_time + np.arange(1, num_future_states+1) * sample_state_gaps * sample_dt

        history_queries = align.align_data(
            query_time=history_time,
            train_time=all_timestamps_calibrated,
            train_data=all_states,
            interp_funcs=interp_funcs
        )
        history_states = h5io.compose_ee_gripper(
            ee_poses=history_queries["ee_pose"], 
            grippers=history_queries["gripper"]
        )

        future_queries = align.align_data(
            query_time=future_time,
            train_time=all_timestamps_calibrated,
            train_data=all_states,
            interp_funcs=interp_funcs,
        )
        future_states = h5io.compose_ee_gripper(
            ee_poses=future_queries["ee_pose"],
            grippers=future_queries["gripper"]
        )

        # pad to n camera
        if pad2ncam > 0:
            rgbs = cls.pad2ncam(rgbs, pad2ncam, dim=1, zero_init=True)
            masks = cls.pad2ncam(masks, pad2ncam, dim=1, zero_init=True)
            cam_poses = cls.pad2ncam(cam_poses, pad2ncam, dim=1, zero_init=False)
            K = cls.pad2ncam(K, pad2ncam, dim=0, zero_init=False)
        
        # previous data has only one ee, therefore we preserve this for compatility
        if ee_poses.ndim == 3:
            ee_poses = ee_poses[:, None]  # (To, 4, 4) -> (To, Nee=1, 4, 4)
        if history_states.ndim == 2:
            history_states = history_states[:, None]  # (nhist, 17) -> (nhist, Nee=1, 17)
        if future_states.ndim == 2:
            future_states = future_states[:, None]  # (Ta, 17) -> (Ta, Nee=1, 17)
        
        # select ee by indices
        assert isinstance(ee_indices, (list, tuple))
        ee_poses = ee_poses.take(ee_indices, axis=1)
        history_states = history_states.take(ee_indices, axis=1)
        future_states = future_states.take(ee_indices, axis=1)
        
        # pad to n ee
        current_nee = len(ee_indices)
        if pad2nee > 0:
            ee_poses = cls.pad2nee(ee_poses, pad2nee, dim=1)
            history_states = cls.pad2nee(history_states, pad2nee, dim=1)
            future_states = cls.pad2nee(future_states, pad2nee, dim=1)
            valid_ee_mask = np.zeros(pad2nee, dtype=bool)
            valid_ee_mask[:current_nee] = True
        else:
            valid_ee_mask = np.ones(current_nee, dtype=bool)
        
        # read reference camera frame to express action
        padded_nee = valid_ee_mask.shape[0]
        action_ref_pose = cam_poses[-1, 0:1].copy().repeat(padded_nee, 0)  # (nee, 4, 4)
        if ee_indices is not None:
            for i, ee_ind in enumerate(ee_indices):
                candidate_cam_names = ee_ref_cams[ee_ind]
                if isinstance(candidate_cam_names, str):
                    sel_cam_name = candidate_cam_names
                else:
                    sel_cam_name = random.sample(candidate_cam_names, k=1)[0]
                sel_cam_idx = camera_names.index(sel_cam_name)
                action_ref_pose[i] = cam_poses[-1, sel_cam_idx]

        return (
            rgbs,                               # (To, ncam, 3, H, W)
            masks,                              # (To, ncam, H, W)
            cam_poses.astype(np.float32),       # (To, ncam, 4, 4)
            ee_poses.astype(np.float32),        # (To, nee, 4, 4)
            action_ref_pose.astype(np.float32), # (nee, 4, 4)
            history_states.astype(np.float32),  # (nhist, nee, 17)
            future_states.astype(np.float32),   # (Ta, nee, 17)
            current_time,                       # scalar,
            K.astype(np.float32),               # (ncam, 3, 3)
            valid_ee_mask,                      # (nee,)
        )

    @classmethod
    def sample_hdf5(
        cls,
        obs_traj: h5py.File, 
        default_ee_indices: List[int], 
        camera_names: List[str], 
        ee_ref_cams: Optional[Dict[str, Tuple[str]]], 
        num_history_cameras: int, 
        num_history_states: int, 
        num_future_states: int, 
        latest: bool = False, 
        sample_state_gaps: int = 1,
        sample_camera_gaps: int = 1,
        sample_dt: float = 1.0,
        record_dt: Optional[float] = None, 
        output_image_hw: Optional[Tuple[int, int]] = None, 
        enable_seg: bool = False, 
        geom_aug: bool = False,
        photometric_aug: bool = False,
        noise_aug: bool = False,
        background_aug: bool = False,
        pad2ncam: int = -1,
        pad2nee: int = -1, 
        video_root: Optional[str] = None,
        debug_sample_index: Optional[int] = None,
        require_full_future: bool = False,
    ):
        """
        obs_traj is a tree-like data structure
        - ee_pose: np.ndarray of shape (T, nee, 4, 4)
        - gripper: np.ndarray of shape (T, nee)
        - ee_pose_desired (optional): np.ndarray of shape (T, nee, 4, 4)
        - gripper_desired (optional): np.ndarray of shape (T, nee)
        - timestamp: np.ndarray of shape (T,)
        - CAMERA_NAME_0:
            - rgb: np.ndarray of shape (T, 3, H, W) or list of bytes (jpeg encoding)
            - pose: np.ndarray of shape (T, 4, 4)
            - K: np.ndarray of shape (3, 3), camera intrinsic
        - CAMERA_NAME_1:
            - rgb: np.ndarray of shape (T, 3, H, W) or list of vlen
            - ...
        """

        all_timestamps = obs_traj["timestamp"][:].astype(np.float64)  # (L,)
        if record_dt is not None:
            all_timestamps_calibrated = np.arange(len(all_timestamps)) * record_dt
        else:
            all_timestamps_calibrated = all_timestamps
            record_dt = infer_record_dt(all_timestamps, default=sample_dt)
        
        if sample_dt is None:
            sample_dt = record_dt

        if latest:
            last_obs_index = len(all_timestamps) - 1
        elif require_full_future:
            last_time = all_timestamps_calibrated[-1] - num_future_states * sample_state_gaps * sample_dt
            max_index = np.searchsorted(all_timestamps_calibrated, last_time, side="right") - 1
            if max_index < 0:
                raise ValueError("Episode is shorter than the required future action horizon")
            last_obs_index = np.random.randint(max_index + 1)
        else:
            last_obs_index = np.random.randint(len(all_timestamps))

        if debug_sample_index is not None:
            print("[INFO] Debug sample index set, overwrite")
            last_obs_index = debug_sample_index


        # we don't interpolate the images, but find the image with closest timestamp
        current_time = all_timestamps_calibrated[last_obs_index]
        prev_obs_sample_time = current_time + np.arange(-num_history_cameras+1, 1) * sample_camera_gaps * sample_dt
        prev_obs_sample_ind = find_closest_ind(all_timestamps_calibrated, prev_obs_sample_time)

        obs_across_cams: Dict[str, list] = {}  # 
        for cam_name in camera_names:
            # if cam_name in obs_traj.keys():
            obs_cam = h5io.slice_encoded_frames(
                obs_traj[cam_name], 
                prev_obs_sample_ind,
                timestamp=all_timestamps,  # use original timestamp to iter video file
                video_root=video_root
            )
            for k, v in obs_cam.items():
                if k not in obs_across_cams:
                    obs_across_cams[k] = []
                obs_across_cams[k].append(v)
        
        K, rgbs, masks = cls.preprocess_images(
            cam_names=camera_names,
            Ks=obs_across_cams["K"],
            rgbs=obs_across_cams["rgb"],
            masks=obs_across_cams.get("mask", None) if enable_seg else None,
            output_image_hw=output_image_hw,
            geom_aug=geom_aug,
            photometric_aug=photometric_aug,
            noise_aug=noise_aug,
            background_aug=background_aug
        )
        # K: (ncam, 3, 3); rgbs: (T, ncam, 3, H, W); masks: (T, ncam, H, W)

        cam_poses = np.stack(obs_across_cams["pose"], axis=1)   # (T, ncam, 4, 4)
        ee_poses = h5io.slice_dset(obs_traj["ee_pose"], prev_obs_sample_ind)  # (T, 4, 4)

        # interpolate the robot states
        all_states = {
            "ee_pose": obs_traj["ee_pose"][:],  # (L, 4, 4)
            "gripper": obs_traj["gripper"][:],  # (L,)
        }

        interp_funcs = {
            "ee_pose": align.interp_SE3_sep, 
            "gripper": align.interp_linear
        }

        history_time = current_time + np.arange(-num_history_states+1, 1) * sample_state_gaps * sample_dt
        future_time = current_time + np.arange(1, num_future_states+1) * sample_state_gaps * sample_dt
        future_desired_time = current_time + np.arange(num_future_states) * sample_state_gaps * sample_dt
        # since desired has been given, this is one step behind future_time

        history_queries = align.align_data(
            query_time=history_time,
            train_time=all_timestamps_calibrated,
            train_data=all_states,
            interp_funcs=interp_funcs
        )
        history_states = h5io.compose_ee_gripper(
            ee_poses=history_queries["ee_pose"], 
            grippers=history_queries["gripper"]
        )
                
        if "gripper_desired" in obs_traj.keys():
            future_grippers = align.align_data(
                query_time=future_desired_time,
                train_time=all_timestamps_calibrated,
                train_data={"gripper": obs_traj["gripper_desired"][:]},
                interp_funcs={"gripper": align.interp_linear}
            )["gripper"]  # (L, nee)
        else:
            future_grippers = align.align_data(
                query_time=future_time,
                train_time=all_timestamps_calibrated,
                train_data={"gripper": all_states["gripper"]},
                interp_funcs={"gripper": align.interp_linear}
            )["gripper"]  # (L, nee)
        
        if "ee_pose_desired" in obs_traj.keys():
            future_ee_poses = align.align_data(
                query_time=future_desired_time,
                train_time=all_timestamps_calibrated,
                train_data={"ee_pose": obs_traj["ee_pose_desired"][:]},
                interp_funcs={"ee_pose": align.interp_SE3_sep}
            )["ee_pose"]  # (L, nee, 4, 4)
        else:
            future_ee_poses = align.align_data(
                query_time=future_time,
                train_time=all_timestamps_calibrated,
                train_data={"ee_pose": all_states["ee_pose"]},
                interp_funcs={"ee_pose": align.interp_SE3_sep}
            )["ee_pose"]  # (L, nee, 4, 4)
        
        future_states = h5io.compose_ee_gripper(
            ee_poses=future_ee_poses,
            grippers=future_grippers
        )  # (L, nee, 17)

        # pad to n camera
        if pad2ncam > 0:
            rgbs = cls.pad2ncam(rgbs, pad2ncam, dim=1, zero_init=True)
            masks = cls.pad2ncam(masks, pad2ncam, dim=1, zero_init=True)
            cam_poses = cls.pad2ncam(cam_poses, pad2ncam, dim=1, zero_init=False)
            K = cls.pad2ncam(K, pad2ncam, dim=0, zero_init=False)
        
        # previous data has only one ee, therefore we preserve this for compatility
        if ee_poses.ndim == 3:
            ee_poses = ee_poses[:, None]  # (To, 4, 4) -> (To, Nee=1, 4, 4)
        if history_states.ndim == 2:
            history_states = history_states[:, None]  # (nhist, 17) -> (nhist, Nee=1, 17)
        if future_states.ndim == 2:
            future_states = future_states[:, None]  # (Ta, 17) -> (Ta, Nee=1, 17)
        
        # select ee by indices
        if "ee_indices" in obs_traj.attrs:
            # this overwrite the config's ee_indices by h5 data's ee_indices
            # allow sample specific enabling/disabling which ee pose to predict
            ee_indices = obs_traj.attrs["ee_indices"]
            assert len(ee_indices) <= len(default_ee_indices), (
                "sample use ee_indices = {}, while default_ee_indices = {}".format(
                    ee_indices, default_ee_indices
                )
            )
        else:
            ee_indices = default_ee_indices
        
        assert isinstance(ee_indices, (list, tuple, np.ndarray))
        ee_poses = ee_poses.take(ee_indices, axis=1)
        history_states = history_states.take(ee_indices, axis=1)
        future_states = future_states.take(ee_indices, axis=1)
        
        # pad to n ee
        current_nee = len(ee_indices)
        if pad2nee > 0:
            ee_poses = cls.pad2nee(ee_poses, pad2nee, dim=1)
            history_states = cls.pad2nee(history_states, pad2nee, dim=1)
            future_states = cls.pad2nee(future_states, pad2nee, dim=1)
            valid_ee_mask = np.zeros(pad2nee, dtype=bool)
            valid_ee_mask[:current_nee] = True
        else:
            valid_ee_mask = np.ones(current_nee, dtype=bool)
        if "valid_ee_mask" in obs_traj.attrs:
            episode_mask = np.asarray(obs_traj.attrs["valid_ee_mask"], dtype=bool)
            if episode_mask.shape != (current_nee,) or not episode_mask.any():
                raise ValueError("valid_ee_mask must identify at least one selected end effector")
            valid_ee_mask[:current_nee] &= episode_mask
        
        if obs_traj.attrs.get("is_bgr", False):
            # revert bgr to rgb
            rgbs = np.ascontiguousarray(np.flip(rgbs, axis=2))
        
        # read reference camera frame to express action
        padded_nee = valid_ee_mask.shape[0]
        action_ref_pose = cam_poses[-1, 0:1].copy().repeat(padded_nee, 0)  # (nee, 4, 4)
        if ee_indices is not None:
            for i, ee_ind in enumerate(ee_indices):
                candidate_cam_names = ee_ref_cams[ee_ind]
                if isinstance(candidate_cam_names, str):
                    sel_cam_name = candidate_cam_names
                else:
                    sel_cam_name = random.sample(candidate_cam_names, k=1)[0]
                sel_cam_idx = camera_names.index(sel_cam_name)
                action_ref_pose[i] = cam_poses[-1, sel_cam_idx]
        
        # read subtask
        if "subtask_description" in obs_traj.keys():
            bin_ind = np.digitize(last_obs_index, obs_traj["subtask_start_index"]) - 1
            bin_ind = np.clip(bin_ind, None, len(obs_traj["subtask_description"]) - 1)

            if bin_ind < 0:
                description = ""
            else:
                description = obs_traj["subtask_description"][bin_ind]
                if isinstance(description, bytes):
                    description = description.decode("utf-8")
        else:
            description = ""

        return (
            rgbs,                               # (To, ncam, 3, H, W)
            masks,                              # (To, ncam, H, W)
            cam_poses.astype(np.float32),       # (To, ncam, 4, 4)
            ee_poses.astype(np.float32),        # (To, nee, 4, 4)
            action_ref_pose.astype(np.float32), # (nee, 4, 4)
            history_states.astype(np.float32),  # (nhist, nee, 17)
            future_states.astype(np.float32),   # (Ta, nee, 17)
            current_time,                       # scalar,
            K.astype(np.float32),               # (ncam, 3, 3)
            valid_ee_mask,                      # (nee,)
            description
        )


def gen_norm_xy_map(H: int, W: int, K: np.ndarray):
    """
    Args:
        H (int): image height
        W (int): image width
        K (np.ndarray): (Ncam, 3, 3)
    
    Returns:
        norm_xy (np.ndarray): (Ncam, 2, H, W)
    """
    fx = K[:, 0, 0]; fy = K[:, 1, 1]; cx = K[:, 0, 2]; cy = K[:, 1, 2]  # (ncam,)
    XX, YY = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    grid = np.stack([XX, YY], axis=0)  # (2, H, W)
    cxy = np.stack([cx, cy], axis=-1)  # (ncam, 2)
    fxy = np.stack([fx, fy], axis=-1)  # (ncam, 2)
    norm_xy = (grid - cxy[:, :, None, None]) / fxy[:, :, None, None]  # (ncam, 2, H, W)
    return norm_xy


@dataclass
class DataConfig(object):
    ### total traj time of gt future action is `sample_dt * num_future_states`
    sample_dt: float
    ### if None, inferenced from `timestamp` key from data, otherwise overwrite the data
    record_dt: Optional[float]
    ### image height and width, none means remain unchanged
    output_image_hw: Optional[Tuple[int, int]]

    ### used in training and real-world execution
    ee_indices: Tuple[int]
    ee_ref_cams: Optional[Dict[int, Tuple[str]]]  # which camera is chosen to express the action of ee
    camera_names: Tuple[str]
    enable_seg: bool = False  # segment image patches if mask is available
    
    geom_aug: bool = False  # random crop and shift for geometric augmentation
    photometric_aug: bool = True  # lighting/color augmentation
    noise_aug: bool = True  # sensor noise / blur augmentation
    background_aug: bool = False  # background clutter perturbation 
    # !!! should be true for bg?

    sample_state_gaps: int = 1
    sample_camera_gaps: int = 4

    num_history_cameras: int = 1
    num_history_states: int = 1
    num_future_states: int = 32  # future states as gt action
    require_full_future: bool = False

    video_root: Optional[str] = None  # for those rgb key as string, which means load video from video_root/rgb
    shuffle_cameras: bool = True
    extrinsic_jitter_prob: float = 0.0
    extrinsic_jitter_translation: float = 0.0  # meters, uniform xyz jitter range
    extrinsic_jitter_rotation_deg: float = 0.0
    extrinsic_jitter_camera_names: Tuple[str, ...] = ()


class ImageProcessor(object):
    @classmethod
    def scale_to_fit(cls, x: np.ndarray, H, W):
        old_H, old_W = x.shape[-2:]
        scale_H = H / old_H
        scale_W = W / old_W
        scale = min(scale_H, scale_W)

        new_H = int(scale * old_H)
        new_W = int(scale * old_W)

        metadata = dict(old_H=old_H, old_W=old_W, new_H=new_H, new_W=new_W)

        if new_H == old_H and new_W == old_W:
            return x, metadata

        # Use cv2 directly to avoid numpy→tensor→numpy round-trip.
        # INTER_AREA gives anti-aliased results for downsampling (matches torchvision BILINEAR+antialias).
        is_bool = x.dtype == bool
        work = x.astype(np.uint8) if is_bool else x
        interp = cv2.INTER_NEAREST if (is_bool or not np.issubdtype(x.dtype, np.floating)) \
                 else (cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

        if work.ndim == 4:  # (T, C, H, W)
            T, C = work.shape[:2]
            out = np.empty((T, C, new_H, new_W), dtype=work.dtype)
            for t in range(T):
                frame = np.transpose(work[t], (1, 2, 0))          # CHW→HWC
                resized = cv2.resize(frame, (new_W, new_H), interpolation=interp)
                if C == 1:
                    resized = resized[:, :, np.newaxis]
                out[t] = np.transpose(resized, (2, 0, 1))
        elif work.ndim == 3:  # (T, H, W)
            T = work.shape[0]
            out = np.empty((T, new_H, new_W), dtype=work.dtype)
            for t in range(T):
                out[t] = cv2.resize(work[t], (new_W, new_H), interpolation=interp)
        else:
            raise ValueError(f"scale_to_fit: unsupported ndim={work.ndim}")

        if is_bool:
            out = out.astype(bool)

        return out, metadata

    @classmethod
    def tform_K_for_scale_to_fit(
        cls, 
        K: np.ndarray, 
        old_H: int, old_W: int, new_H: int, new_W: int
    ):
        fx = K[..., 0, 0]
        fy = K[..., 1, 1]
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        
        scale_x = new_W / old_W
        scale_y = new_H / old_H
        
        new_fx = fx * scale_x
        new_fy = fy * scale_y
        new_cx = cx * scale_x
        new_cy = cy * scale_y
        
        K_new = K.copy()
        K_new[..., 0, 0] = new_fx
        K_new[..., 1, 1] = new_fy
        K_new[..., 0, 2] = new_cx
        K_new[..., 1, 2] = new_cy
        return K_new

    @classmethod
    def center_view(cls, x: np.ndarray, H, W):
        old_H, old_W = x.shape[-2:]
        dx = (W - old_W) // 2
        dy = (H - old_H) // 2
        metadata = dict(dcx=dx, dcy=dy)

        if old_H == H and old_W == W:
            return x, metadata

        # Pure numpy pad/crop — no tensor allocation needed.
        out = np.zeros(x.shape[:-2] + (H, W), dtype=x.dtype)
        # Compute the overlapping region between src and dst
        src_y0 = max(0, -dy);  src_y1 = min(old_H, H - dy)
        src_x0 = max(0, -dx);  src_x1 = min(old_W, W - dx)
        dst_y0 = max(0,  dy);  dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = max(0,  dx);  dst_x1 = dst_x0 + (src_x1 - src_x0)
        out[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
        return out, metadata
    
    @classmethod
    def tform_K_for_center_view(cls, K: np.ndarray, dcx: int, dcy: int):
        cx = K[..., 0, 2]
        cy = K[..., 1, 2]
        K_new = K.copy()
        K_new[..., 0, 2] = cx + dcx
        K_new[..., 1, 2] = cy + dcy
        return K_new


class H5DatasetMapBase(Dataset):

    config = DataConfig(
        sample_dt=1.0,
        record_dt=None,
        output_image_hw=None, 
        ee_indices=(),
        camera_names=(),
        ee_ref_cams=None,
    )

    def __init__(
        self,
        h5_filelist: List[str],
    ):
        self.h5_filelist = h5_filelist
        self.data_sampler = DataSampler()
        self._h5_cache_pid = None
        self._h5_cache: OrderedDict[str, h5py.File] = OrderedDict()
        # Keep more files open: each worker samples from several datasets, and
        # re-opening an H5 file costs ~5-10 ms (metadata read + kernel file-table entry).
        self._max_open_h5_files = 32
        self._access_count = 0
        # Periodic full reset disabled (set to 0): the LRU eviction above already
        # bounds memory, and resetting everything every 1000 accesses was wasteful.
        self._reset_interval = 0

        if isinstance(self.config.camera_names, str):
            # wrap to tuple
            self.config.camera_names = (self.config.camera_names,)

        if isinstance(self.config.ee_indices, int):
            # wrap to tuple
            self.config.ee_indices = (self.config.ee_indices,)

        self.cam_num = len(self.config.camera_names)
        self.ee_num = len(self.config.ee_indices)
        self.pad2ncam = self.cam_num
        self.pad2nee = self.ee_num

    def _reset_h5_cache(self):
        for h5 in self._h5_cache.values():
            try:
                h5.close()
            except Exception:
                pass
        self._h5_cache = OrderedDict()
        self._h5_cache_pid = os.getpid()

    def _get_h5_handle(self, h5_file: str) -> h5py.File:
        """
        MEMORY OPTIMIZATION: LRU cache with periodic full reset.
        Balances speed (keep files open) with memory (periodic cleanup).
        """
        import gc
        current_pid = os.getpid()
        if self._h5_cache_pid != current_pid:
            self._reset_h5_cache()

        self._access_count += 1

        # Periodic full reset — disabled when _reset_interval == 0 (rely on LRU eviction only)
        if self._reset_interval > 0 and self._access_count % self._reset_interval == 0:
            self._reset_h5_cache()
            gc.collect()

        # Check if file is already cached
        h5 = self._h5_cache.pop(h5_file, None)
        if h5 is None:
            # Open with moderate chunk cache (512KB per file)
            h5 = h5py.File(
                h5_file, "r",
                rdcc_nbytes=4 * 1024 * 1024,  # 4 MB chunk cache (helps ee_pose/timestamp)
                rdcc_nslots=4099,              # Prime number of cache slots
            )

        # Add to cache (moves to end for LRU)
        self._h5_cache[h5_file] = h5

        # Evict oldest files if cache is full
        while len(self._h5_cache) > self._max_open_h5_files:
            _, old_h5 = self._h5_cache.popitem(last=False)
            try:
                old_h5.close()
            except Exception:
                pass

        return h5
    
    @classmethod
    def inst(cls, train_stage: int = 0) -> "H5DatasetMapBase":
        raise NotImplementedError

    def __len__(self):
        return len(self.h5_filelist)

    def _sample_extrinsic_jitter_transform(self):
        delta = np.eye(4, dtype=np.float32)

        if self.config.extrinsic_jitter_rotation_deg > 0:
            axis = np.random.randn(3).astype(np.float32)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-12:
                axis = axis / axis_norm
                angle = np.deg2rad(
                    np.random.uniform(
                        -self.config.extrinsic_jitter_rotation_deg,
                        self.config.extrinsic_jitter_rotation_deg
                    )
                ).astype(np.float32)
                delta[:3, :3] = axis_angle_to_matrix(axis * angle)

        if self.config.extrinsic_jitter_translation > 0:
            delta[:3, 3] = np.random.uniform(
                -self.config.extrinsic_jitter_translation,
                self.config.extrinsic_jitter_translation,
                size=3,
            ).astype(np.float32)

        return delta

    def _apply_extrinsic_jitter(self, obs_cam_poses: np.ndarray, camera_names: List[str]):
        target_names = set(self.config.extrinsic_jitter_camera_names)
        if self.config.extrinsic_jitter_prob <= 0 or not target_names:
            return obs_cam_poses

        obs_cam_poses = obs_cam_poses.copy()
        for cam_idx, cam_name in enumerate(camera_names):
            if cam_name not in target_names:
                continue
            if np.random.rand() >= self.config.extrinsic_jitter_prob:
                continue

            delta = self._sample_extrinsic_jitter_transform()
            obs_cam_poses[:, cam_idx] = delta[None] @ obs_cam_poses[:, cam_idx]

        return obs_cam_poses
    
    def sample_from_hdf5(
        self, 
        h5: h5py.File, 
        latest: bool = False, 
        debug_sample_index: Optional[int] = None,
        video_root: Optional[str] = None,
    ):
        prompt_candidates = []
        for k, v in h5.attrs.items():
            if "prompt_text" in k:
                v = v.strip()
                if len(v):
                    prompt_candidates.append(v)
        
        prompt_text = random.sample(prompt_candidates, 1)[0] if len(prompt_candidates) else ""
        # if len(prompt_text) == 0:
        #     prompt_text = "Do any possible actions"

        if self.config.shuffle_cameras:
            camera_names = list(self.config.camera_names).copy()
            random.shuffle(camera_names)
        else:
            camera_names = self.config.camera_names

        (
            obs_rgbs, obs_masks, obs_cam_poses, obs_ee_poses, action_ref_pose, 
            history_states, future_states, timestamps, K, valid_ee_mask, sub_task_description
        ) = self.data_sampler.sample_hdf5(
            obs_traj=h5, 
            default_ee_indices=self.config.ee_indices,
            camera_names=camera_names, 
            ee_ref_cams=self.config.ee_ref_cams, 
            num_history_cameras=self.config.num_history_cameras, 
            num_history_states=self.config.num_history_states, 
            num_future_states=self.config.num_future_states,
            latest=latest, 
            sample_state_gaps=self.config.sample_state_gaps, 
            sample_camera_gaps=self.config.sample_camera_gaps, 
            sample_dt=self.config.sample_dt,
            record_dt=self.config.record_dt, 
            output_image_hw=self.config.output_image_hw,
            enable_seg=self.config.enable_seg,
            geom_aug=self.config.geom_aug,
            photometric_aug=self.config.photometric_aug,
            noise_aug=self.config.noise_aug,
            background_aug=self.config.background_aug,
            pad2ncam=self.pad2ncam,
            pad2nee=self.pad2nee,
            video_root=video_root if video_root is not None else self.config.video_root,
            debug_sample_index=debug_sample_index,
            require_full_future=self.config.require_full_future,
        )
        # obs_cam_poses = self._apply_extrinsic_jitter(obs_cam_poses, camera_names)

        T, ncam, C, H, W = obs_rgbs.shape
        norm_xys = gen_norm_xy_map(H, W, K).astype(np.float32)
        norm_xys = norm_xys[None].repeat(T, axis=0)  # (T, ncam, 2, H, W)
        
        # join prompt_text and sub_task_description
        if len(prompt_text) and not prompt_text.endswith("."):
            join_str = ". "
        else:
            join_str = " "
        if len(sub_task_description):
            prompt_text = prompt_text + join_str + sub_task_description
        
        out = {
            "K": K,                                                     # (ncam, 3, 3)
            "obs_rgbs": obs_rgbs,                                       # (To, ncam, 3, H, W)
            "obs_masks": obs_masks,                                     # (To, ncam, H, W)
            "prompt_text": prompt_text,                                 # str
            "obs_norm_xys": norm_xys,                                   # (To, ncam, 2, H, W)
            "obs_extrinsics": obs_cam_poses,                            # (To, ncam, 4, 4)
            "current_ee_pose": obs_ee_poses[-1],                        # (nee, 4, 4)
            "action_ref_pose": action_ref_pose,                         # (nee, 4, 4)
            "history_ee_states": history_states,                        # (nhist, nee, 17)
            "gt_future_ee_states": future_states,                       # (Ta, nee, 17)
            "timestamps": timestamps,                                   # (To,)
            "valid_ee_mask": valid_ee_mask,                             # (nee,)
            "gripper_mask": np.ones_like(valid_ee_mask, dtype=bool),    # (nee,) True for gripper EEs, False for dexhand EEs
        }
        return out

    def __getitem__(self, i):
        """
        Load sample from HDF5 file with LRU caching.
        Files are kept open for speed, with periodic full reset for memory.
        """
        h5_file = self.h5_filelist[i]
        try:
            h5 = self._get_h5_handle(h5_file)
            out = self.sample_from_hdf5(h5, latest=False, debug_sample_index=None)
        except Exception as e:
            print(f"Error loading {h5_file}: {e}")
            raise e
        # Note: h5 file stays in cache, will be closed by LRU eviction or periodic reset
        return out
    
    def visualize(self, model_proxy=None):
        return visualize_dataset(self, model_proxy=model_proxy)

    def save_visualization(self, model_proxy=None, save_dir: str = "visualization", video_fps: int = 10):
        return save_visualization(self, model_proxy=model_proxy, save_dir=save_dir, video_fps=video_fps)


class H5DatasetIterBase(H5DatasetMapBase, IterableDataset):
    
    def __init__(self, h5_filelist: List[str]):
        super().__init__(h5_filelist)
        self._shuffle_h5_list = False
    
    def __iter__(self):
        indices = np.arange(len(self.h5_filelist))
        if self._shuffle_h5_list:
            np.random.shuffle(indices)
            # print("[INFO] {} shuffles dataset".format(os.getpid()))
        
        # worker_info = get_worker_info()
        # if (worker_info is not None) and (worker_info.num_workers > 1):
        #     # split workload
        #     splits = np.linspace(0, len(indices), num=worker_info.num_workers+1, endpoint=True)
        #     splits = splits.astype(np.int64).tolist()
        #     indices = indices[splits[worker_info.id]:splits[worker_info.id+1]].copy()
        
        for i in indices:
            # print(os.getpid())
            yield self[int(i)]


def concat_datasets(
    datasets: List[Union[H5DatasetMapBase, H5DatasetIterBase]],
    shuffle: bool = None
):
    num_cams = [d.cam_num for d in datasets]
    num_ees = [d.ee_num for d in datasets]
    pad2ncam = max(num_cams)
    pad2nee = max(num_ees)
    for d in datasets:
        d.pad2ncam = pad2ncam
        d.pad2nee = pad2nee
        print("[INFO] dataset {} uses {} cameras, {} end-effectors".format(d, d.cam_num, d.ee_num))
    print("[INFO] Final padded camera num: {}, end-effector num: {}".format(pad2ncam, pad2nee))
    
    if isinstance(datasets[0], H5DatasetIterBase):
        if shuffle:
            for d in datasets:
                d._shuffle_h5_list = True
        return ChainDataset(datasets)
    else:
        return ConcatDataset(datasets)


def make_multiplex_inplace(dataset, multiplex: int):
    if isinstance(dataset, (list, tuple)):
        for d in dataset:
            make_multiplex_inplace(d, multiplex)
    elif isinstance(dataset, H5DatasetMapBase):
        dataset.h5_filelist = dataset.h5_filelist * multiplex
        print("[INFO] Dataset {} multiplexed by {}, new length {}".format(
            dataset, multiplex, len(dataset.h5_filelist)))
    elif isinstance(dataset, (ChainDataset, ConcatDataset)):
        for d in dataset.datasets:
            make_multiplex_inplace(d, multiplex)
    else:
        raise TypeError("Unsupported dataset type {}".format(type(dataset)))


def get_dataloader(
    datasets: List[Union[H5DatasetMapBase, H5DatasetIterBase]],
    batch_size: int,
    num_workers: int = 0,
    shuffle: Optional[bool] = None, 
    persistent_workers: bool = False,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    sample_weights: Optional[list] = None,
    sample_multiplex: int = 1
):
    if sample_weights is not None:
        sample_weights = np.array(sample_weights)
        sample_weights = sample_weights / sample_weights.sum()
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights.tolist(), 
            num_samples=len(sample_weights) * sample_multiplex, 
            replacement=True
        )
    else:
        sampler = None
    
    if isinstance(datasets, (list, tuple)):
        datasets = concat_datasets(datasets, shuffle)
    elif isinstance(datasets, H5DatasetIterBase):
        if shuffle == True:
            datasets._shuffle_h5_list = True
    
    if isinstance(datasets, (H5DatasetIterBase, ChainDataset)):
        shuffle = None  # overwrite shuffle args
    
    dataloader_kwargs = dict(
        dataset=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = persistent_workers
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(**dataloader_kwargs)
    return dataloader


def generate_sample_weights(
    datasets: List[H5DatasetMapBase],
    dataset_weights: List[float]
):
    sample_weights = []
    for i, dataset in enumerate(datasets):
        sample_weights.append(
            np.array([dataset_weights[i] / len(dataset)] * len(dataset))
        )
    sample_weights = np.concatenate(sample_weights)
    sample_weights = sample_weights / sample_weights.sum()
    return sample_weights.tolist()


def rbd(d: Dict[str, Tensor]):
    """remove batch dimension"""
    return {k:v[0] if v is not None else v for k, v in d.items()}


def proj(K: np.ndarray, cwT: np.ndarray, pos: np.ndarray):
    pos_in_cam = pos @ cwT[:3, :3].T + cwT[:3, 3]
    xy = pos_in_cam[:2] / pos_in_cam[-1:]

    fx = K[0, 0]; fy = K[1, 1]; cx = K[0, 2]; cy = K[1, 2]
    fxy = np.array([fx, fy]); cxy = np.array([cx, cy])
    uv = xy * fxy + cxy
    return uv


def draw_ee_proj(bgr: np.ndarray, K: np.ndarray, cwT: np.ndarray, pose: np.ndarray):
    alen = 0.05  # 5cm
    pos = pose[:3, 3]
    x_end = tuple(proj(K, cwT, pos + pose[:3, 0] * alen).astype(int).tolist())
    y_end = tuple(proj(K, cwT, pos + pose[:3, 1] * alen).astype(int).tolist())
    z_end = tuple(proj(K, cwT, pos + pose[:3, 2] * alen).astype(int).tolist())
    origin = tuple(proj(K, cwT, pos).astype(int).tolist())
    
    cv2.line(bgr, origin, x_end, (0, 0, 255), thickness=2)
    cv2.line(bgr, origin, y_end, (0, 255, 0), thickness=2)
    cv2.line(bgr, origin, z_end, (255, 0, 0), thickness=2)
    return bgr


def visualize_traj(
    data: Dict[str, Tensor], 
    future_ee_states: Optional[List[Tensor]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None
):
    """draw current ee pose and 

    Args:
        data (Dict[str, Tensor]): remove the batch dimension
        future_ee_states (List[Tensor]): you can plot multiple trajs, each of shape (Ta, nee, 4*4+1)
        colors (List[Tuple[int, int, int]]): each traj can has different color

    Returns:
        bgrs: image to visualize
    """
    
    if future_ee_states is None:
        future_ee_states = [data["gt_future_ee_states"]]
    if colors is None:
        colors = [(0, 255, 0)]
    
    # data["obs_rgbs"]: (To, ncam, C, H, W)
    To, ncam, _, H, W = data["obs_rgbs"].shape
    if data["obs_rgbs"].dtype == torch.float32:
        data["obs_rgbs"] = (data["obs_rgbs"] * 255.0).to(torch.uint8)
    rgb = rearrange(data["obs_rgbs"][-1], "n c h w -> n h w c")  # latest time
    
    # data["K"]: (ncam, 3, 3)
    K = data["K"]  # (ncam, 3, 3)
    K_np = K.cpu().numpy()
    
    # data["obs_extrinsics"]: (To, ncam, 4, 4)
    wcT = data["obs_extrinsics"][-1]  # (ncam, 4, 4)
    wcT_np = wcT.cpu().numpy()
    
    valid_ee_mask = data["valid_ee_mask"]  # (nee)
    valid_ee_mask_np = valid_ee_mask.cpu().numpy()
    
    nhist, nee, _ = data["history_ee_states"].shape
    history_weTs = data["history_ee_states"][:, :, :16].view(nhist, nee, 4, 4)
    history_weTs_np = history_weTs.cpu().numpy()
    
    bgrs = np.ascontiguousarray(rgb.flip(-1).cpu().numpy())  # (ncam, H, W, C)
    
    for typei, ee_states in enumerate(future_ee_states):
        # future_ee_states: (Ta, nee, 4*4+1)
        Ta, nee, _ = ee_states.shape
        weTs = ee_states[:, :, :16].view(Ta, nee, 4, 4)  # (Ta, nee, 4, 4)

        ceTs = (
            rearrange(torch.inverse(wcT), "ncam r c -> () () ncam r c") @ 
            rearrange(weTs, "Ta nee r c -> Ta nee () r c")
        )  # (Ta, nee, ncam, 4, 4)
        cets = ceTs[..., :3, 3]  # (Ta, nee, ncam, 3)

        proj_norm = cets[..., :2] / cets[..., 2:3]  # (Ta, nee, ncam, 2)
        fxy = K[..., [0, 1], [0, 1]]; cxy = K[..., [0, 1], [2, 2]]  # (ncam, 2)
        proj_pix = proj_norm * fxy + cxy  # (Ta, nee, ncam, 2)
        proj_pix_np: np.ndarray = proj_pix.cpu().numpy()  # (Ta, nee, ncam, 2)
    
        for eei in range(nee):
            if not valid_ee_mask_np[eei]:
                continue
            
            for cami in range(ncam):
                for x, y in proj_pix_np[:, eei, cami]:
                    cv2.circle(bgrs[cami], (int(x), int(y)), 
                               radius=2, color=colors[typei], thickness=-1)

                if typei == 0:
                    # draw current ee poses only once
                    draw_ee_proj(
                        bgrs[cami], 
                        K=K_np[cami], 
                        cwT=np.linalg.inv(wcT_np[cami]), 
                        pose=history_weTs_np[-1, eei]
                    )

    bgrs = rearrange(bgrs, "n h w c -> h (n w) c")
    bgrs = np.ascontiguousarray(bgrs)
    return bgrs


def visualize_dataset(
    dataset: H5DatasetMapBase,
    model_proxy = None
):
    import copy
    
    episode_idx = 0
    frame_idx = 0

    original_config = copy.deepcopy(dataset.config)
    if dataset.config.shuffle_cameras:
        dataset.config.shuffle_cameras = False
        print("[INFO] In dataset visualization, temporarily set shuffle_cameras to False!!!")
    for ee_ind, cam_names in dataset.config.ee_ref_cams.items():
        if isinstance(cam_names, (list, tuple)) and len(cam_names) > 1:
            dataset.config.ee_ref_cams[ee_ind] = (cam_names[0],)  # only choose the first one
            print("[INFO] In dataset visualization, temporarily set ref cam of ee_ind {} to {}"
                  .format(ee_ind, cam_names[0]))

    h5 = h5py.File(dataset.h5_filelist[episode_idx], mode="r")
    max_frames = len(h5["ee_pose"][:])

    step = 1
    model_infer_enabled = True
    
    while True:
        print("-"*61)
        print("[INFO] Usage:")
        print("q: quit")
        print("p: previous episode")
        print("n: next episode")
        print("<-: previous frame")
        print("->: next frame")
        print("a: toggle step size between 1 and 10")
        print("i: toggle model inference mode")
        print()

        print("[INFO] Step size: {}".format(step))
        print("[INFO] Model inference mode: {}".format(model_infer_enabled))
        print("[INFO] Episode: {}/{}, frame: {}/{}".format(
            episode_idx+1, len(dataset), frame_idx, max_frames))
        print("[INFO] H5 file: {}".format(dataset.h5_filelist[episode_idx]))
        
        out = dataset.sample_from_hdf5(
            h5=h5,
            latest=False,
            debug_sample_index=frame_idx
        )
        
        print("[INFO] Prompt text: {}".format(out["prompt_text"]))
        
        for k, v in out.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v)
        
        future_ee_states = [out["gt_future_ee_states"]]
        traj_colors = [(0, 255, 0)]
        
        if model_infer_enabled and (model_proxy is not None):
            data_inputs = out.copy()  # shallow copy
            # make batch size = 1
            for k, v in data_inputs.items():
                if isinstance(v, torch.Tensor):
                    data_inputs[k] = v.unsqueeze(0)  # add batch dim
            data_inputs["prompt_text"] = [data_inputs["prompt_text"]]
            # run prediction
            actions_pred = model_proxy.run_inference(data_inputs).cpu()  # (B, Ta, Nee, 17+?)
            
            future_ee_states.append(actions_pred[0, ..., :17])  # remove extra info if has
            traj_colors.append((255, 0, 0))
        
        bgrs = visualize_traj(
            data=out,
            future_ee_states=future_ee_states,
            colors=traj_colors
        )
        
        print("[INFO] future grippers = \n{}".format(
            out["gt_future_ee_states"][:, out["valid_ee_mask"], -1].transpose(0, 1)))
        
        cv2.imshow("gt traj", bgrs)
        key = cv2.waitKey(1)
        
        if key == ord("n"):
            h5.close()
            episode_idx = (episode_idx + 1) % len(dataset.h5_filelist)
            frame_idx = 0
            h5 = h5py.File(dataset.h5_filelist[episode_idx], mode="r")
            max_frames = len(h5["ee_pose"][:])
        elif key == ord('p'):
            h5.close()
            episode_idx = (episode_idx - 1) % len(dataset.h5_filelist)
            frame_idx = 0
            h5 = h5py.File(dataset.h5_filelist[episode_idx], mode="r")
            max_frames = len(h5["ee_pose"][:])
        elif key == 83:
            frame_idx = (frame_idx + step) % max_frames
        elif key == 81:
            frame_idx = (frame_idx - step) % max_frames
        elif key == ord('q'):
            h5.close()
            break
        elif key == ord('a'):
            step = {1: 10, 10: 1}[step]
        elif key == ord('i'):
            model_infer_enabled = (not model_infer_enabled) and (model_proxy is not None)
    
    dataset.config = original_config
    return


def save_visualization(
    dataset: H5DatasetMapBase,
    model_proxy = None,
    model_infer_enabled = True,
    save_dir: str = "visualization",
    video_fps: int = 10
):
    """Save visualization videos for each episode in the dataset.
    Args:
        model_proxy: model proxy for model inference
        model_infer_enabled: whether to enable model inference
        save_dir: save directory
        video_fps: video frame rate
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    original_config = copy.deepcopy(dataset.config)
    if dataset.config.shuffle_cameras:
        dataset.config.shuffle_cameras = False
        print("[INFO] In dataset visualization, temporarily set shuffle_cameras to False!!!")
    for ee_ind, cam_names in dataset.config.ee_ref_cams.items():
        if isinstance(cam_names, (list, tuple)) and len(cam_names) > 1:
            dataset.config.ee_ref_cams[ee_ind] = (cam_names[0],)  # only choose the first one
            print("[INFO] In dataset visualization, temporarily set ref cam of ee_ind {} to {}"
                    .format(ee_ind, cam_names[0]))

    for episode_idx, h5_path in enumerate(dataset.h5_filelist):
        video_name = os.path.splitext(os.path.basename(h5_path))[0] + ".mp4"
        video_path = os.path.join(save_dir, video_name)
        print(f"[INFO] Saving episode {episode_idx+1}/{len(dataset)}: {h5_path} -> {video_path}")

        h5 = h5py.File(h5_path, mode="r")
        max_frames = len(h5["ee_pose"][:])
        print(f"[INFO] max frame: {max_frames}")
        video_writer = None

        for frame_idx in range(max_frames):
            out = dataset.sample_from_hdf5(
                h5=h5,
                latest=False,
                debug_sample_index=frame_idx
            )

            print("[INFO] Prompt text: {}".format(out["prompt_text"]))

            for k, v in out.items():
                if isinstance(v, np.ndarray):
                    out[k] = torch.from_numpy(v)
            future_ee_states = [out["gt_future_ee_states"]]
            traj_colors = [(0, 255, 0)]

            if (model_proxy is not None):
                data_inputs = out.copy()  # shallow copy
                # make batch size = 1
                for k, v in data_inputs.items():
                    if isinstance(v, torch.Tensor):
                        data_inputs[k] = v.unsqueeze(0)  # add batch dim
                data_inputs["prompt_text"] = [data_inputs["prompt_text"]]
                # run prediction
                actions_pred = model_proxy.run_inference(data_inputs).cpu()  # (B, Ta, Nee, 17+?)
                
                future_ee_states.append(actions_pred[0, ..., :17])  # remove extra info if has
                traj_colors.append((255, 0, 0))
                
            # Use existing visualize_traj function, assumed imported
            bgrs = visualize_traj(
                data=out,
                future_ee_states=future_ee_states,
                colors=traj_colors
            )
            if isinstance(bgrs, list):
                # In case visualize_traj returns a list of images
                bgr = bgrs[0]
            else:
                bgr = bgrs

            # Initialize video writer if not yet done
            if video_writer is None:
                H, W = bgr.shape[:2]
                video_writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    video_fps,
                    (W, H)
                )
            # Ensure BGR is uint8
            if bgr.dtype != np.uint8:
                bgr = (np.clip(bgr, 0, 1) * 255).astype(np.uint8)
            video_writer.write(bgr)

        if video_writer is not None:
            video_writer.release()
        h5.close()
    # Restore config
    dataset.config = original_config
    print("[INFO] Finished saving visualization videos to {}".format(save_dir))
