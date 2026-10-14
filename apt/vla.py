"""Top-level VLA wrapper: pairs a Qwen3-VL backbone with the action expert.

The two-stage APT recipe is realised by passing different ``train_stage``
values to the action expert:

* ``train_stage=0`` — Stage 1 of the paper (VA prior). Half the attention
  layers are active, language tokens are masked.
* ``train_stage=1`` — Stage 2 of the paper (VLA likelihood). Layers are
  doubled (interleaved with newly inserted language-injection layers) and
  language tokens participate in attention.
"""
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .action_expert import ActionExpert
from .vlm import VLM


class VLA(nn.Module):
    """Vision-Language-Action policy = frozen-or-tuned VLM + diffusion action expert."""

    def __init__(
        self,
        hdim: int,
        num_heads: int,
        diffusion_timesteps: int = 100,
        inference_timesteps: int = 20,
        train_stage: int = 0,
        camera_view_dropout: float = 0.0,
        vlm_finetune_mode: str = "frozen",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vlm = VLM(
            vlm_finetune_mode=vlm_finetune_mode,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.actor = ActionExpert(
            idim=self.vlm.output_dim, 
            hdim=hdim,
            num_heads=num_heads,
            num_diffusion_layers=self.vlm.num_layers, 
            diffusion_timesteps=diffusion_timesteps,
            inference_timesteps=inference_timesteps,
            train_stage=train_stage,
            camera_view_dropout=camera_view_dropout,
        )

    def parameter_groups(self):
        """Split trainable params into (decay, no_decay).

        VLM (Qwen3-VL) and the action expert use very different naming
        conventions, so the no-decay rules are defined separately."""
        vlm_decay, vlm_no_decay = self._vlm_parameter_groups()
        actor_decay, actor_no_decay = self._actor_parameter_groups()
        return vlm_decay + actor_decay, vlm_no_decay + actor_no_decay

    def _vlm_parameter_groups(self):
        # Qwen3-VL no-decay set:
        #   - all *Norm weights (RMSNorm / LayerNorm) — names contain "norm"
        #   - token / position embeddings (`embed_tokens`, `pos_embed`)
        #   - all biases
        # `patch_embed.proj` is a Conv and keeps weight decay, matching the
        # common VLM finetune recipe.
        decay, no_decay = [], []
        for name, param in self.vlm.named_parameters():
            if not param.requires_grad:
                continue
            lname = name.lower()
            is_no_decay = (
                name.endswith(".bias")
                or "norm" in lname
                or "embed_tokens" in lname
                or "pos_embed" in lname
            )
            (no_decay if is_no_decay else decay).append(param)
        return decay, no_decay

    def _actor_parameter_groups(self):
        # Action expert no-decay set:
        #   - all *norm* submodule weights
        #   - all biases
        #   - `gate` (zero-init residual gates inside HybridAttentionLayers)
        # NOTE: `denoising_time_embed` is an nn.Sequential of Linear layers —
        # its weights stay in the decay group.
        decay, no_decay = [], []
        for name, param in self.actor.named_parameters():
            if not param.requires_grad:
                continue
            lname = name.lower()
            is_no_decay = (
                name.endswith(".bias")
                or "norm" in lname
                or "embedding" in lname
            )
            (no_decay if is_no_decay else decay).append(param)
        return decay, no_decay
                
    def load_from_pretrain(
        self,
        state_dict: Dict[str, Tensor],
        load_from_va: bool = False,
    ):
        """Load actor weights, optionally expanding a Stage-0 VA prior into Stage-1.

        Parameters
        ----------
        state_dict : Dict[str, Tensor]
            The actor sub-state-dict from a saved checkpoint (i.e. ``ckpt["weights"]``).
        load_from_va : bool, default False
            When True, the source is treated as a Stage-0 (VA) checkpoint with
            ``N/2`` traj-context attention layers and the current model is the
            Stage-1 actor with ``N`` interleaved layers. Source layer ``i`` is
            copied into the odd-index target layer ``2*i + 1``; the even-index
            target layers (newly inserted language-injection layers) keep
            their fresh initialisation.
        """
        if load_from_va: # loading from an original va pretrained model
            actor_state = self.actor.state_dict()
            # load traj context attn layers
            s0_layer_idx = 0
            for s1_layer_idx in range(len(self.actor.dp_head.traj_context_attn.layers)):
                if s1_layer_idx % 2 == 1:
                    # 对应 Stage0 的 attention
                    for name, _ in self.actor.dp_head.traj_context_attn.layers[s1_layer_idx].named_parameters():
                        k1 = f"dp_head.traj_context_attn.layers.{s1_layer_idx}.{name}"
                        k0 = f"dp_head.traj_context_attn.layers.{s0_layer_idx}.{name}"
                        actor_state[k1] = state_dict[k0]
                    s0_layer_idx += 1
            # load other layers
            # !!! IMPORTANT !!!
            for name, param in self.actor.named_parameters():
                if "dp_head.traj_context_attn.layers" not in name:
                    actor_state[name] = param
            self.actor.load_state_dict(actor_state)
        else:
            self.actor.load_state_dict(state_dict)
        if "vlm_weights" in state_dict:
            incompat = self.vlm.load_state_dict(state_dict["vlm_weights"], strict=False)
            assert not incompat.unexpected_keys, \
                f"Unexpected VLM keys in checkpoint: {incompat.unexpected_keys}"

    def forward(
        self, 
        obs_rgbs: Tensor,
        obs_masks: Optional[Tensor], 
        obs_norm_xys: Tensor, 
        obs_extrinsics: Tensor, 
        prompt_text: Optional[Tensor], 

        current_ee_pose: Tensor, 
        action_ref_pose: Tensor,
        history_ee_states: Tensor, 
        gt_future_ee_states: Tensor, 
        valid_ee_mask: Tensor, 
        inference: bool, 
        fp16: bool,
        robot_pose_aug: bool = False,
        camera_drop_prob: Optional[Tensor] = None,
    ):
        """
        Args:
            obs_rgbs: (B, To, ncam, 3, H, W)
            obs_masks: (B, To, ncam, H, W)
            obs_norm_xys: (B, To, ncam, 2, H, W), coordinates in normalized camera plane
            obs_extrinsics: (B, To, ncam, 4, 4), ^{world}_{camera} T
            prompt_text: (B, Lang, E) or None, language instruction

            current_ee_pose: (B, Nee, 4, 4), ^{world}_{ee} T
            action_ref_pose: (B, Nee, 4, 4), ^{world}_{ref} T
            history_ee_states: (B, nhist, Nee, 4*4+1), in world frame,
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
            gt_future_ee_states: (B, Ta, Nee, 4*4+1), ground truth future actions, in world frame
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
                * Note: if `inference` is True, we only derive prediction actions shape from gt_future_ee_states
            valid_ee_mask: (B, Nee), only compute loss on these end-effectors
            inference: if True, returns the predicted trajectory, otherwise returns loss and metrics for logging
            fp16: if True, use bfloat16
            robot_pose_aug: if True, add noise to the robot states
        
        Returns
        -------
        (if inference is True)
            pred_future_ee_states (Tensor): (B, Ta, Nee, 4*4+1)
                * 4x4 is the flattened transformation matrix, 
                * 1 is gripper openness, range [0 (close), 1 (open)]
        (else)
            loss (Tensor): scalar tensor
            metrics (Dict[str, Tensor]): metrics for logging
        """
        vl_inputs, vl_outputs = self.vlm(
            obs_rgbs=obs_rgbs,
            obs_masks=obs_masks,
            obs_norm_xys=obs_norm_xys,
            obs_extrinsics=obs_extrinsics,
            prompt_text=prompt_text,
            fp16=fp16
        )

        outputs = self.actor(
            vl_inputs=vl_inputs,
            vl_outputs=vl_outputs,

            current_ee_pose=current_ee_pose,
            action_ref_pose=action_ref_pose, 
            history_ee_states=history_ee_states,
            gt_future_ee_states=gt_future_ee_states,
            valid_ee_mask=valid_ee_mask, 
            inference=inference,
            fp16=fp16,
            robot_pose_aug=robot_pose_aug,
            camera_drop_prob=camera_drop_prob,
        )
        return outputs


def vla_tiny(
    diffusion_timesteps: int = 100,
    inference_timesteps: int = 20,
    train_stage: int = 0,
    camera_view_dropout: float = 0.0,
    vlm_finetune_mode: str = "frozen",
    use_gradient_checkpointing: bool = False,
):
    return VLA(
        hdim=192,
        num_heads=3,
        diffusion_timesteps=diffusion_timesteps,
        inference_timesteps=inference_timesteps,
        train_stage=train_stage,
        camera_view_dropout=camera_view_dropout,
        vlm_finetune_mode=vlm_finetune_mode,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


def vla_small(
    diffusion_timesteps: int = 100,
    inference_timesteps: int = 20,
    train_stage: int = 0,
    camera_view_dropout: float = 0.0,
    vlm_finetune_mode: str = "frozen",
    use_gradient_checkpointing: bool = False,
):
    return VLA(
        hdim=384,
        num_heads=6,
        diffusion_timesteps=diffusion_timesteps,
        inference_timesteps=inference_timesteps,
        train_stage=train_stage,
        camera_view_dropout=camera_view_dropout,
        vlm_finetune_mode=vlm_finetune_mode,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


def vla_base(
    diffusion_timesteps: int = 100, 
    inference_timesteps: int = 20,
    train_stage: int = 0,
    camera_view_dropout: float = 0.0,
    vlm_finetune_mode: str = "frozen",
    use_gradient_checkpointing: bool = False,
):
    return VLA(
        hdim=768,
        num_heads=12,
        diffusion_timesteps=diffusion_timesteps,
        inference_timesteps=inference_timesteps,
        train_stage=train_stage,
        camera_view_dropout=camera_view_dropout,
        vlm_finetune_mode=vlm_finetune_mode,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


