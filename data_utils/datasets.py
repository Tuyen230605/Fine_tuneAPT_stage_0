import os
import sys
import json
import glob
import copy
import h5py
import torch
import random
import inspect
import traceback
import numpy as np
from typing import Dict, Union
from torchvision.transforms import v2
from .data_loc import DSET, LOC
from .dataset_base import DataConfig, H5DatasetMapBase, gen_norm_xy_map
from .vid_dec import decode_video_frames_torchcodec

# __getitem__(self, i) of H5DatasetMapBase return a dict of Tensors (except for `prompt_text`),
# We list the name and shape as follows:

# K:                    (ncam, 3, 3)
# obs_rgbs:             (To, ncam, 3, H, W)
# obs_masks:            (To, ncam, H, W)
# prompt_text:          str
# obs_norm_xys:         (To, ncam, 2, H, W), coordinates in normalized camera plane, 2 = (x, y)
# obs_extrinsics:       (To, ncam, 4, 4)
# current_ee_pose:      (nee, 4, 4)
# history_ee_states:    (nhist, nee, 17), 17 = (16 for flattened 4x4 pose matrix (row major), 1 for gripper)
# gt_future_ee_states:  (Ta, nee, 17)
# timestamps:           (To,)
# valid_ee_mask:        (nee,)

# NOTE:
# To -> number of image observations
# ncam -> number of cameras
# nee -> number of end-effectors
# nhist -> number of historical actions (including current, therefore nhist always >= 1)
# Ta -> number of future actions

def quat_correct(out: Dict[str, np.ndarray]):
    from scipy.spatial.transform import Rotation
    for key in ["current_ee_pose"]:
        quat_raw = Rotation.from_matrix(out[key][..., :3, :3]).as_quat()
        quat_corrected = quat_raw[..., [1, 2, 3, 0]]
        out[key][..., :3, :3] = Rotation.from_quat(quat_corrected).as_matrix()
    for key in ["history_ee_states", "gt_future_ee_states"]:
        ee = out[key]  # (T, Nee, 17)
        Ta, Nee, _ = ee.shape
        pose = ee[..., :16].reshape(-1, 4, 4)
        quat_raw = Rotation.from_matrix(pose[..., :3, :3]).as_quat()
        quat_corrected = quat_raw[..., [1, 2, 3, 0]]
        pose[..., :3, :3] = Rotation.from_quat(quat_corrected).as_matrix()
        pose = pose.reshape(-1, 16)
        out[key][..., :16] = pose.reshape(Ta, Nee, 16)   
    return out


def fwd_ee_origin(
    out: Dict[str, np.ndarray], 
    fwd_axis: int, 
    distance: float
):
    for key in ["current_ee_pose"]:
        out[key][..., :3, 3] += out[key][..., :3, fwd_axis] * distance
    for key in ["history_ee_states", "gt_future_ee_states"]:
        ee = out[key]  # (B, T, Nee, 17)
        Ta, Nee, _ = ee.shape
        pose = ee[..., :16].reshape(Ta, Nee, 4, 4)
        pose[..., :3, 3] += pose[..., :3, fwd_axis] * distance
        out[key][..., :16] = pose.reshape(Ta, Nee, 16)
    return out


def get_loc(cls: Union[str, H5DatasetMapBase]):
    # get file location for dataset cls
    if isinstance(cls, str):
        cls_name = cls
    else:
        cls_name = cls.__name__

    path = LOC[getattr(DSET, cls_name)]
    assert path is not None, "No path defined for the dataset `{}` in this machine".format(cls_name)
    return path


class Libero(H5DatasetMapBase):
    # camera_drop_prob: float = 0.0

    config = DataConfig(
        record_dt=None,
        sample_dt=1.0,
        output_image_hw=(256, 256),
        ee_indices=(0,),
        camera_names=("agentview", "eye_in_hand"),
        ee_ref_cams={0: ("agentview", "eye_in_hand")},
    )

    @classmethod
    def inst(cls, task_suites = [], train_stage: int = 1):
        if isinstance(task_suites, str):
            task_suites = [task_suites]
        
        h5_files = []
        for suite in task_suites:
            if suite == "spatial":
                h5_files.extend(glob.glob(LOC[DSET.LiberoSpatial], recursive=True))
            elif suite == "object":
                h5_files.extend(glob.glob(LOC[DSET.LiberoObject], recursive=True))
            elif suite == "goal":
                h5_files.extend(glob.glob(LOC[DSET.LiberoGoal], recursive=True))
            elif suite == "10":
                h5_files.extend(glob.glob(LOC[DSET.Libero10], recursive=True))
            else:
                raise TypeError("Unknown task suite: {}".format(suite))
        
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files)
    
    def modify_prompt(self, lang: str):
        # remove something like "LIVING ROOM SCENE6" in libero10 and libero90
        index = lang.find("SCENE")
        if index >= 0:
            lang = lang[index:]
            lang = " ".join(lang.split(" ")[1:])
        return lang
    
    def __getitem__(self, i):
        out = super().__getitem__(i)
        out["prompt_text"] = self.modify_prompt(out["prompt_text"])
        # out["camera_drop_prob"] = np.float32(self.camera_drop_prob)
        return out


class LiberoSpatial(Libero):
    # camera_drop_prob = 0.15

    @classmethod
    def inst(cls, train_stage: int = 1):
        return super().inst(["spatial"], train_stage)


class LiberoObject(Libero):
    # camera_drop_prob = 0.15

    @classmethod
    def inst(cls, train_stage: int = 1):
        return super().inst(["object"], train_stage)


class LiberoGoal(Libero):
    # camera_drop_prob = 0.0

    @classmethod
    def inst(cls, train_stage: int = 1):
        return super().inst(["goal"], train_stage)


class Libero10(Libero):
    # camera_drop_prob = 0.0

    @classmethod
    def inst(cls, train_stage: int = 1):
        return super().inst(["10"], train_stage)


class Droid(H5DatasetMapBase):
    data_root = LOC.get(DSET.Droid)

    config = DataConfig(
        record_dt=1.0/15,
        sample_dt=1.0/15,
        output_image_hw=(256, 256),
        ee_indices=(0,),
        camera_names=("exterior_2_left", "exterior_1_left", "wrist_left"),
        ee_ref_cams={0: ("exterior_2_left", "exterior_1_left", "wrist_left")}, 
        sample_state_gaps=2,
        video_root=data_root,
    )

    @classmethod
    def filter_files(cls, filelist):
        filtered = []
        for f in filelist:
            # remove .h5 and episode_
            episode_index = int(os.path.split(f)[-1][:-3].replace("episode_", ""))
            if episode_index in [11907, 14419, 24440, 64837, 64871]:
                print("[INFO] in Droid dataset, remove file: {}".format(f))
            else:
                filtered.append(f)
        return filtered

    @classmethod
    def inst(cls, train_stage: int = 0):
        if cls.data_root is None:
            raise ValueError("No path defined for Droid on this machine")
        h5_files = glob.glob(os.path.join(cls.data_root, "data/*/*.h5"), recursive=True)
        h5_files = cls.filter_files(h5_files)
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        if train_stage == 0:
            h5_files = h5_files[:int(len(h5_files) * 0.5)]
        # else:
        #     h5_files = h5_files[int(len(h5_files) * 0.5):]
        return cls(h5_files)
    
    def sample_from_hdf5(self, h5, latest = False, debug_sample_index = None):
        out = super().sample_from_hdf5(h5, latest, debug_sample_index)
        out = fwd_ee_origin(out, fwd_axis=2, distance=0.15)
        return out
    
    def __getitem__(self, i):
        try:
            out = super().__getitem__(i)
        except Exception as e:
            # This occasionally fails when reading video files, I don't know why
            traceback.print_exc()
            with open("error_filelist.txt", "a") as fp:
                fp.write("Error in file reading: ")
                fp.write(self.h5_filelist[i] + "\n")
            print("[INFO] Retry another index")
            out = super().__getitem__((i+1)%len(self))
        
        return out


class PickPlaceCan(H5DatasetMapBase):
    config = DataConfig(
        record_dt=None,
        sample_dt=1.0,
        output_image_hw=(256, 256),
        ee_indices=(0,),
        camera_names=("e2h_cam", "eih_cam"),
        ee_ref_cams={0: ("e2h_cam", "eih_cam")},
    )

    @classmethod
    def inst(cls, train_stage: int = 0):
        h5_files = glob.glob(get_loc(cls))
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        if train_stage == 0:
            h5_files = h5_files[:int(len(h5_files) * 0.5)]
        # else:
        #     h5_files = h5_files[int(len(h5_files) * 0.5):]
        return cls(h5_files)


class OpenArmFixedPlace(H5DatasetMapBase):
    """Converted bimanual OpenArm episodes for Stage-0 fixed-place training."""

    config = DataConfig(
        record_dt=None,
        sample_dt=0.1,
        output_image_hw=(256, 256),
        ee_indices=(0, 1),
        camera_names=("chest", "front"),
        ee_ref_cams={0: ("front",), 1: ("front",)},
        num_future_states=32,
        require_full_future=True,
        shuffle_cameras=False,
    )

    @classmethod
    def inst(cls, train_stage: int = 0):
        pattern = os.environ.get("APT_FIXED_PLACE_DATA_GLOB")
        if not pattern:
            raise ValueError("Set APT_FIXED_PLACE_DATA_GLOB to converted train/**/*.h5")
        h5_files = sorted(glob.glob(pattern, recursive=True))
        if not h5_files:
            raise ValueError(f"No converted OpenArm episodes match {pattern!r}")
        print(f"[INFO] {cls.__name__}: {len(h5_files)} episodes (stage {train_stage})")
        return cls(h5_files)


class AgibotWorldAlpha(H5DatasetMapBase):
    
    data_root = LOC.get(DSET.AgibotWorldAlpha)
    
    config = DataConfig(
        record_dt=1.0/30,
        sample_dt=1.0/30,
        output_image_hw=(256, 256),
        ee_indices=(1, 2),  # 0: head, 1: left, 2: right
        camera_names=("head_cam", "left_cam", "right_cam"),
        ee_ref_cams={1: ("head_cam", "left_cam"), 2: ("head_cam", "right_cam")}, 
        sample_state_gaps=3,
        video_root=data_root
    )

    @classmethod
    def inst(cls, train_stage: int = 0):
        if cls.data_root is None:
            raise ValueError("No path defined for AgibotWorldAlpha on this machine")
        with open(f"{cls.data_root}/head_color_filelist.txt", "r") as fp:
            head_video_fname = [line.strip() for line in fp.readlines()]
        head_video_fname = [line for line in head_video_fname if len(line)]

        h5_root = f"{cls.data_root}/hdf5"
        h5_files = []
        for head_fname in head_video_fname:
            parts = head_fname.split("/")
            task_id = parts[1]  # 352
            episode_id = parts[3]  # 674501
            h5_files.append(os.path.join(h5_root, task_id, episode_id + ".h5"))
        # test file existence
        assert os.path.exists(h5_files[0]), f"{h5_files[0]} does not exist!"

        # remove file with short video length
        import numpy as np
        video_file_sizes = np.load(f"{cls.data_root}/video_sizes.npy")  # (num_f, 3), for 3 views
        video_file_sizes_MB = video_file_sizes / (1024.0 * 1024.0)
        valid_file_mask = np.all(video_file_sizes_MB > 0.01, axis=-1)  # (num_f,)

        assert len(valid_file_mask) == len(h5_files), f"{len(valid_file_mask)} vs {len(h5_files)}"
        h5_files = [f for i, f in enumerate(h5_files) if valid_file_mask[i]]

        assert len(h5_files) > 0
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        if train_stage == 0:
            h5_files = h5_files[:int(len(h5_files) * 0.5)]
        # else:
        #     h5_files = h5_files[int(len(h5_files) * 0.5):]
        return cls(h5_files)

    def sample_from_hdf5(self, h5, latest = False, debug_sample_index = None):
        out = super().sample_from_hdf5(h5, latest, debug_sample_index)
        out = fwd_ee_origin(out, fwd_axis=2, distance=0.08)
        return out
    
    def __getitem__(self, i):
        try:
            out = super().__getitem__(i)
            obs_extrinsics = out["obs_extrinsics"]  # (To, Ncam, 4, 4)
            latest_extr = obs_extrinsics[-1]  # (Ncam, 4, 4)
            np.linalg.inv(latest_extr)
        except Exception as e:
            traceback.print_exc()
            with open("error_filelist.txt", "a") as fp:
                fp.write("Error in extr inversion: ")
                fp.write(self.h5_filelist[i] + "\n")
            print("[INFO] Retry another index")
            out = super().__getitem__((i+1)%len(self))
        return out


class InternM1Franka(H5DatasetMapBase):
    data_root = LOC.get(DSET.InternM1Franka)

    config = DataConfig(
        record_dt=None,
        sample_dt=None,
        output_image_hw=(256, 256),
        ee_indices=(0,),
        camera_names=("base_view", "base_view_2", "ego_view"),
        ee_ref_cams={0: ("base_view", "base_view_2", "ego_view")}, 
        sample_state_gaps=4,
        video_root=data_root
    )

    @classmethod
    def inst(cls, train_stage: int = 0):
        if cls.data_root is None:
            raise ValueError("No path defined for InternM1Franka on this machine")
        h5_files = glob.glob(os.path.join(cls.data_root, "**/*.h5"), recursive=True)
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        if train_stage == 0:
            h5_files = h5_files[:int(len(h5_files) * 0.5)]
        # else:
        #     h5_files = h5_files[int(len(h5_files) * 0.5):]
        return cls(h5_files)

    def __getitem__(self, i):
        try:
            out = super().__getitem__(i)
        except Exception as e:
            traceback.print_exc()
            with open("error_filelist.txt", "a") as fp:
                fp.write("Error in video decoding: ")
                fp.write(self.h5_filelist[i] + "\n")
            print("[INFO] Retry another index")
            out = super().__getitem__((i+1)%len(self))
        return out


class _InternA1Mixin(object):
    # The four InternA1 datasets all share the same on-disk root and are
    # discriminated by ``robot_name``; the directory layout pairs an
    # ``InternData-A1_h5`` tree (the .h5 files this loader reads) with a
    # sibling ``InternData-A1/sim_updated`` tree (the raw videos referenced
    # by ``video_root``). Both roots are sourced from the YAML loader:
    #   * ``h5_root`` ← ``LOC[DSET.InternA1Franka]``  (any of the 4 keys works)
    #   * ``raw_root`` ← ``LOC[DSET.InternA1Franka] + "_raw"`` by default,
    #     overridable via ``APT_INTERNA1_RAW_ROOT`` env var.
    h5_root = LOC.get(DSET.InternA1Franka)
    raw_root = os.environ.get(
        "APT_INTERNA1_RAW_ROOT",
        (h5_root + "_raw") if h5_root else None,
    )
    robot_name = None

    @classmethod
    def inst(cls, train_stage: int = 1):
        assert cls.robot_name is not None
        h5_files = glob.glob(
            os.path.join(cls.h5_root, "**", cls.robot_name, "**", "*.h5"),
            recursive=True
        )
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files)

    @classmethod
    def infer_video_root(cls, h5_file: str):
        rel_dir = os.path.relpath(os.path.dirname(h5_file), cls.h5_root)
        return os.path.join(cls.raw_root, rel_dir)

    def _getitem_with_video_root(self, i):
        # Pass video_root directly to sample_from_hdf5 to avoid mutating the
        # class-level config object, which is shared across all instances of the
        # same class in this process.
        h5_file = self.h5_filelist[i]
        video_root = self.infer_video_root(h5_file)
        h5 = self._get_h5_handle(h5_file)
        out = self.sample_from_hdf5(h5, latest=False, debug_sample_index=None,
                                    video_root=video_root)
        return out

    def __getitem__(self, i):
        try:
            return self._getitem_with_video_root(i)
        except Exception:
            traceback.print_exc()
            with open("error_filelist.txt", "a") as fp:
                fp.write("Error in video decoding: ")
                fp.write(self.h5_filelist[i] + "\n")
            print("[INFO] Retry another index")
            return self._getitem_with_video_root((i+1) % len(self))


class InternA1Franka(_InternA1Mixin, H5DatasetMapBase):
    robot_name = "franka"

    config = DataConfig(
        record_dt=1.0/10,
        sample_dt=1.0/10,
        output_image_hw=(256, 256),
        ee_indices=(0,),
        camera_names=("head_cam", "hand_cam"),
        ee_ref_cams={0: ("head_cam", "hand_cam")},
        sample_state_gaps=2,
        video_root=_InternA1Mixin.raw_root
    )


class InternA1Lift2(_InternA1Mixin, H5DatasetMapBase):
    robot_name = "lift2"

    config = DataConfig(
        record_dt=1.0/10,
        sample_dt=1.0/10,
        output_image_hw=(256, 256),
        ee_indices=(0, 1),
        camera_names=("head_cam", "lh_cam", "rh_cam"),
        ee_ref_cams={0: ("head_cam", "lh_cam"), 1: ("head_cam", "rh_cam")},
        sample_state_gaps=2,
        video_root=_InternA1Mixin.raw_root
    )


class InternA1Genie1(_InternA1Mixin, H5DatasetMapBase):
    robot_name = "genie1"

    config = DataConfig(
        record_dt=1.0/10,
        sample_dt=1.0/10,
        output_image_hw=(256, 256),
        ee_indices=(0, 1),
        camera_names=("head_cam", "lh_cam", "rh_cam"),
        ee_ref_cams={0: ("head_cam", "lh_cam"), 1: ("head_cam", "rh_cam")},
        sample_state_gaps=2,
        video_root=_InternA1Mixin.raw_root
    )


class InternA1SplitAloha(_InternA1Mixin, H5DatasetMapBase):
    robot_name = "split_aloha"

    config = DataConfig(
        record_dt=1.0/10,
        sample_dt=1.0/10,
        output_image_hw=(256, 256),
        ee_indices=(0, 1),
        camera_names=("head_cam", "lh_cam", "rh_cam"),
        ee_ref_cams={0: ("head_cam", "lh_cam"), 1: ("head_cam", "rh_cam")},
        sample_state_gaps=2,
        video_root=_InternA1Mixin.raw_root
    )


class AlohaTableStorage(H5DatasetMapBase):
    config = DataConfig(
        record_dt=1.0/10,
        sample_dt=1.0/10,
        output_image_hw=(256, 256),
        ee_indices=(0,),  # right arm
        camera_names=("head_cam", "rh_cam"),
        ee_ref_cams={0: ("head_cam", "rh_cam")}, 
        sample_state_gaps=2
    )
    
    @classmethod
    def inst(cls, train_stage: int = 1):
        h5_files = glob.glob(get_loc(cls))
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files)


class AlohaPickPlace(H5DatasetMapBase):
    config = DataConfig(
        record_dt=1.0/10,
        sample_dt=1.0/10,
        output_image_hw=(256, 256),
        ee_indices=(0,),  # right arm
        camera_names=("head_cam", "rh_cam"),
        ee_ref_cams={0: ("head_cam", "rh_cam")}, 
        sample_state_gaps=2
    )
    
    @classmethod
    def inst(cls, train_stage: int = 1):
        h5_files = glob.glob(get_loc(cls))
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files)


class AlohaPickPlaceClutter(H5DatasetMapBase):
    config = DataConfig(
        record_dt=1.0/10,
        sample_dt=1.0/10,
        output_image_hw=(256, 256),
        ee_indices=(0,),  # right arm
        camera_names=("head_cam", "rh_cam"),
        ee_ref_cams={0: ("head_cam", "rh_cam")}, 
        sample_state_gaps=2
    )
    
    @classmethod
    def inst(cls, train_stage: int = 1):
        h5_files = glob.glob(get_loc(cls))
        print("[INFO] num samples of {}: {}".format(cls.__name__, len(h5_files)))
        assert len(h5_files) > 0
        h5_files.sort()
        return cls(h5_files)


def get_subclasses(base_class):
    current_module = sys.modules[__name__]
    subclasses = []
    for name, obj in inspect.getmembers(current_module, inspect.isclass):
        if issubclass(obj, base_class) and obj is not base_class:
            subclasses.append(obj)
    return subclasses


DATA_CONFIGS: Dict[str, DataConfig] = {
    c.__name__: c.config for c in get_subclasses(H5DatasetMapBase)
}


if __name__ == "__main__":

    import torch
    from torch.utils.data import DataLoader
    dataset = LiberoSpatial.inst()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    for data in dataloader:
        print("-"*61)
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                print("  - {}: {}".format(k, v.shape))
            else:
                print("  - {}: {}".format(k, v))
        
        input("[INFO] Press Enter to continue: ")

