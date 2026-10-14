import copy
import glob
import os
import threading
from typing import Union

import cv2
import numpy as np
import torch
from torch import Tensor

from data_utils.dataset_base import DataConfig, DataSampler, gen_norm_xy_map, rbd
from data_utils.datasets import DATA_CONFIGS
from infer_utils.draw_traj import visualize_traj
from infer_utils.ensemble import TrajEnsembler

from .. import vla
from ..configs import TrainConfig


def parse_config(ckpt_dir: str):
    config_files = glob.glob(os.path.join(ckpt_dir, "*.json"))
    config_files.sort()
    
    assert len(config_files), "No config files found in {}".format(ckpt_dir)
    config_file = config_files[-1]
    print("[INFO] Use config file {}".format(config_file))
    
    cfg = TrainConfig.load(config_file)
    data_config = cfg.dataset_classes[0].config
    model_name = cfg.model
    vlm_finetune_mode = getattr(cfg, "vlm_finetune_mode", "frozen")

    data_config.shuffle_cameras = False  # overwrite
    print("[INFO] model = {}".format(model_name))
    print("[INFO] data config = {}".format(data_config))
    print("[INFO] vlm_finetune_mode = {}".format(vlm_finetune_mode))

    return model_name, data_config, vlm_finetune_mode


def load_model(path, device, use_ema: bool = False):
    model_name, data_config, vlm_finetune_mode = parse_config(os.path.dirname(path))

    # Load to CPU first so optimizer/scheduler states (can be >15 GB after
    # all-rank ZeRO gathering) never touch GPU memory.  Only model weights
    # get moved to the device later via load_state_dict.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for _k in ("optimizer", "scheduler", "ema"):
        ckpt.pop(_k, None)
    # For inference, always create with vlm_finetune_mode="frozen":
    #   - "frozen" and "full" share identical architecture; "frozen" skips
    #     requires_grad and uses torch.no_grad() in forward, saving ~16 GB
    #     of activation memory.
    #   - "lora" needs its own LoRA layers to load adapter weights; after
    #     loading we merge and the result is effectively frozen.
    infer_vlm_mode = "lora" if vlm_finetune_mode == "lora" else "frozen"
    model: vla.VLA = getattr(vla, "vla_{}".format(model_name))(
        train_stage=1, vlm_finetune_mode=infer_vlm_mode).to(device)
    model.actor.load_state_dict(ckpt["weights"], strict=True)

    if "vlm_weights" in ckpt:
        # named_parameters() deduplicates tied weights (e.g. lm_head.weight == embed_tokens.weight),
        # so the saved dict may be missing some keys. strict=False tolerates this;
        # we still assert no unexpected keys to catch real mismatches.
        incompat = model.vlm.load_state_dict(ckpt["vlm_weights"], strict=False)
        assert not incompat.unexpected_keys, \
            f"Unexpected VLM keys in checkpoint: {incompat.unexpected_keys}"
        print("[INFO] Loaded VLM weights from checkpoint")
        # Merge LoRA deltas into base weights for inference efficiency
        if vlm_finetune_mode == "lora" and hasattr(model.vlm, "merge_and_unload"):
            model.vlm = model.vlm.merge_and_unload()
            print("[INFO] LoRA weights merged into base model")
    # incompatiable = model.load_state_dict(ckpt["module"], strict=False)
    # assert not incompatiable.unexpected_keys
    print("[INFO] Load weights from iter: {}".format(ckpt["current_iters"]))
    model.eval()
    save_gate_img(model)
    return model, data_config


def save_gate_img(model: vla.VLA):
    with torch.no_grad():
        gate = model.actor.dp_head.traj_context_attn.gate.sigmoid().cpu().numpy()  # (n_layers, embed_dim)
        print("[INFO] layerwise mean gate value:")
        print(np.mean(gate, axis=-1))
        gate = (gate * 255).astype(np.uint8)
    
    here = os.path.dirname(__file__)
    cv2.imwrite(os.path.join(here, "..", "gate.png"), gate)


class TrajPlanner(object):
    def __init__(
        self, 
        ckpt_path: str, 
        device: str = "cuda:0", 
        ensemble: int = -1,
        use_ema: bool = False
    ):
        self.model, self.config = load_model(ckpt_path, device, use_ema)

        self.ensemble = int(ensemble)
        self.ensembler_lock = threading.Lock()
        self.pos_ensembler = TrajEnsembler(int(ensemble))
        self.rot_ensembler = TrajEnsembler(int(ensemble))
        self.gripper_ensembler = TrajEnsembler(int(ensemble))

        self.obs_frames = []
        self.obs_lock = threading.Lock()
        
        self.device = device
        self.last_obs_data = None
    
    def reset(self):
        with self.ensembler_lock:
            self.pos_ensembler.reset()
            self.rot_ensembler.reset()
            self.gripper_ensembler.reset()
        with self.obs_lock:
            self.obs_frames.clear()
        return self
    
    def set_config(self, config: Union[str, dict, DataConfig]):
        if isinstance(config, str):
            config = DATA_CONFIGS[config]
        elif isinstance(config, dict):
            config = DataConfig(**config)
        elif isinstance(config, DataConfig):
            pass
        else:
            raise TypeError("Unsupported type of config: {}".format(type(config)))
        
        config: DataConfig = copy.deepcopy(config)
        config.shuffle_cameras = False  # do not shuffle cameras when inference
        self.config = config
    
    def set_prompt(self, prompt_text: str):
        """
        Args:
            prompt_text (str):
        """
        self.prompt_text = prompt_text
        return self
    
    def add_obs_frame(self, obs_frame: dict):
        """
        Args:
            obs_frame (dict) should contains necessary keys listed as followings.

            - CAM_NAME_0: 
                - model: pinhole
                - camera:
                    - width: int
                    - height: int
                    - K: np.ndarray of shape 9 (3x3), flattened
                - data:
                    - color: np.ndarray, shape=(H, W, C)
                    - seg: None | np.ndarray of shape (H, W) | isaacsim seg output
                    - wcT: np.ndarray of shape (4, 4), ^{world}_{cam} T
                    - timestep: float, current timestamp used for sync
            
            - CAM_NAME_1: similar as CAM_NAME_0
            - ee_pose: np.ndarray of shape (4, 4), ^{world}_{ee} T
            - gripper: float, value from [0 (close), 1 (open)]
            - timestamp: float
        """
        max_frames = max(
            self.config.num_history_cameras * self.config.sample_camera_gaps,
            self.config.num_history_states * self.config.sample_state_gaps
        )
        
        def max_time(a: float, b: float):
            if a is None: return b
            elif b is None: return a
            else: return max(a, b)
        
        if (self.config.record_dt is None) and (self.config.sample_dt is None):
            with self.obs_lock:
                self.obs_frames.append(obs_frame)
                while len(self.obs_frames) > max_frames:
                    self.obs_frames.pop(0)
        else:
            time_interval = max_frames * max_time(self.config.record_dt, self.config.sample_dt)
            latest_time = obs_frame["timestamp"]
            earliest_time_thersh = latest_time - time_interval
            with self.obs_lock:
                self.obs_frames.append(obs_frame)
                pop_counts = 0
                for frame in self.obs_frames[1:]:
                    if frame["timestamp"] < earliest_time_thersh:
                        pop_counts += 1
                    else:
                        break                
                if pop_counts > 0:
                    self.obs_frames = self.obs_frames[pop_counts:]
            
        return self
    
    def _make_data_for_infer(self, obs_frames: list, sample_num: int=1):
        """
        Args:
            obs_frames (list[dict]): list of obs_frame, 
                see annotations above
        """
        (
            obs_rgbs, obs_masks, obs_cam_poses, obs_ee_poses, action_ref_pose, 
            history_actions, future_actions, current_time, K, valid_ee_mask
        ) = DataSampler.sample_framedict(
            obs_traj=obs_frames,
            ee_indices=self.config.ee_indices,
            camera_names=self.config.camera_names,
            ee_ref_cams=self.config.ee_ref_cams, 
            num_history_cameras=self.config.num_history_cameras,
            num_history_states=self.config.num_history_states,
            num_future_states=self.config.num_future_states,
            latest=True,
            sample_camera_gaps=self.config.sample_camera_gaps,
            sample_state_gaps=self.config.sample_state_gaps,
            sample_dt=self.config.sample_dt,
            record_dt=self.config.record_dt,
            output_image_hw=self.config.output_image_hw,
            enable_seg=self.config.enable_seg,
        )

        T, ncam, C, H, W = obs_rgbs.shape
        norm_xys = gen_norm_xy_map(H, W, K).astype(np.float32)
        norm_xys = norm_xys[None].repeat(T, axis=0)  # (T, ncam, 2, H, W)

        obs_data = {
            "K": K,                                 # (ncam, 3, 3)
            "obs_rgbs": obs_rgbs,                   # (T, ncam, 3, H, W)
            "obs_masks": obs_masks,                 # (T, ncam, H, W)
            "prompt_text": [self.prompt_text],      # [str]
            "obs_norm_xys": norm_xys,               # (To, ncam, 2, H, W)
            "obs_extrinsics": obs_cam_poses,        # (To, ncam, 4, 4)
            "current_ee_pose": obs_ee_poses[-1],    # (nee, 4, 4)
            "action_ref_pose": action_ref_pose,     # (nee, 4, 4)
            "history_ee_states": history_actions,   # (nhist, nee, 17)
            "gt_future_ee_states": future_actions,  # (Ta, nee, 17)
            "timestamps": np.array(current_time),   # scalar
            "valid_ee_mask": valid_ee_mask,         # (nee,)
        }
        
        for k in obs_data:
            if isinstance(obs_data[k], np.ndarray):
                obs_data[k] = (torch.from_numpy(obs_data[k])
                                    .to(self.device)
                                    .unsqueeze(0)
                                    .repeat(sample_num, *([1] * obs_data[k].ndim)))
            if isinstance(obs_data[k], list):
                obs_data[k] = [obs_data[k][0]] * sample_num
        return obs_data
    
    def _run_inference(self, obs_data):
        for k in obs_data:
            if isinstance(obs_data[k], Tensor):
                obs_data[k] = obs_data[k].to(self.device, non_blocking=True)

        with torch.inference_mode():
            actions: Tensor = self.model(
                obs_rgbs=obs_data["obs_rgbs"], 
                obs_masks=obs_data.get("obs_masks", None),
                obs_norm_xys=obs_data["obs_norm_xys"],
                obs_extrinsics=obs_data["obs_extrinsics"],
                prompt_text=obs_data["prompt_text"],

                current_ee_pose=obs_data["current_ee_pose"],
                action_ref_pose=obs_data["action_ref_pose"],
                history_ee_states=obs_data["history_ee_states"],
                gt_future_ee_states=obs_data["gt_future_ee_states"], 
                valid_ee_mask=obs_data["valid_ee_mask"],
                inference=True,
                fp16=True,
            )  # (B, Ta, nee, 17)
        return actions
    
    def _make_empty_action(self, B, Ta, Nee):
        actions = np.zeros((B, Ta, Nee, 16+1))
        actions[..., :16] = np.eye(4).ravel()
        return actions
    
    def _scatter_to_original_order(
        self, 
        nee_total: int,
        ee_indices: tuple, 
        action_selected: np.ndarray
    ):
        B, Ta, nee_selected, _ = action_selected.shape
        action_full = self._make_empty_action(B, Ta, nee_total)
        
        for i, ee_ind in enumerate(ee_indices):
            action_full[:, :, ee_ind] = action_selected[:, :, i]
        return action_full

    def get_action(
        self, 
        sample_num: int = 1,
        draw_traj: bool = False,
        compress_traj_img: bool = False
    ):
        """
        Returns
        -------
            future_ee_poses (np.ndarray): shape (Ta, Nee, 4, 4), ^{world} _{ee} T
            future_grippers (np.ndarray): shape (Ta, Nee), range [0 (close), 1 (open)]
            future_time (np.ndarray): shape (Ta,)
            traj_img (np.ndarray | None): shape (H, Ncam*W, C) if not compressed else (nbytes,)
        """
        with self.obs_lock:
            obs_frames = self.obs_frames.copy()  # shallow copy
        
        if len(obs_frames) == 0:
            return None
        
        obs_data = self._make_data_for_infer(obs_frames, sample_num)
        actions = self._run_inference(obs_data) # (B, Ta, Nee, 17)
        
        if draw_traj:
            traj_img = visualize_traj(
                data=rbd(obs_data),
                future_ee_states=[actions[0]],
                colors=[(0, 0, 255)]
            )
            if traj_img.dtype == np.float32:
                traj_img = (traj_img * 255.).clip(0, 255).astype(np.uint8)
            if compress_traj_img:
                traj_img = cv2.imencode(".jpg", traj_img)[1]
        else:
            traj_img = None
        
        self.last_obs_data = obs_data
        actions = actions.detach().cpu().numpy()  # (B, Ta, nee_sel, 17)
        actions = self._scatter_to_original_order(
            nee_total=obs_frames[-1]["ee_pose"].shape[0],
            ee_indices=self.config.ee_indices,
            action_selected=actions
        )
        B, Ta, nee, _ = actions.shape

        ee_poses = np.reshape(actions[:, :, :, :16], (B, Ta, nee, 4, 4))
        grippers = actions[:, :, :, -1]  # (B, Ta, nee)
        
        # obs_data["timestamp"]: (B,)
        latest_time = obs_data["timestamps"][0].item()
        action_dt = self.config.sample_dt * self.config.sample_state_gaps
        future_time = (1 + np.arange(Ta)) * action_dt + latest_time
        future_ee_poses = ee_poses  # (B, Ta, nee, 4, 4)
        future_grippers = grippers  # (B, Ta, nee)

        # for i in range(B):
        #     for j in range(nee):
        #         future_ee_poses[i, :, j], future_grippers[i, :, j] = self.ensemble_traj(future_ee_poses[i, :, j], future_grippers[i, :, j], future_time)
        return future_ee_poses, future_grippers, future_time, traj_img
    
    def set_ensemble_nums(self, n: int):
        with self.ensembler_lock:
            self.ensemble = n
            self.pos_ensembler.reset()
            self.rot_ensembler.reset()
            self.gripper_ensembler.reset()

    def ensemble_traj(
        self, 
        future_ee_poses: np.ndarray,
        future_grippers: np.ndarray,
        future_time: np.ndarray
    ):
        if self.ensemble != 0:
            with self.ensembler_lock:
                future_ee_poses[..., :3, 3] = self.pos_ensembler.update(
                    future_ee_poses[..., :3, 3], future_time, on_SO3=False
                )
                # future_ee_poses[..., :3, :3] = self.rot_ensembler.update(
                #     future_ee_poses[..., :3, :3], future_time, on_SO3=True
                # )
                future_grippers = self.gripper_ensembler.update(
                    future_grippers, future_time, on_SO3=False
                )
        
        return future_ee_poses, future_grippers

