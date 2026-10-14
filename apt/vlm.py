import torch
import contextlib
from typing import List
from torch import nn, Tensor
from .encoders.qwen3_vl import QwenVL3Encoder, VALID_VLM_FINETUNE_MODES


class VLM(nn.Module):
    def __init__(
        self,
        vlm_finetune_mode: str = "frozen",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        assert vlm_finetune_mode in VALID_VLM_FINETUNE_MODES, \
            f"vlm_finetune_mode must be one of {VALID_VLM_FINETUNE_MODES}, got {vlm_finetune_mode!r}"
        self.vlm_finetune_mode = vlm_finetune_mode
        self.vlm = QwenVL3Encoder(
            add_action_id=False,
            vlm_finetune_mode=vlm_finetune_mode,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.indices = list(range(0, self.vlm.num_llm_layers, 3))
    
    @property
    def num_layers(self):
        return len(self.indices)
    
    @property
    def output_dim(self):
        return self.vlm.hidden_size

    def forward(
        self, 
        obs_rgbs: Tensor,
        obs_masks: Tensor, 
        obs_norm_xys: Tensor, 
        obs_extrinsics: Tensor, 
        prompt_text: List[str], 
        fp16: bool,
    ):
        """
        Args:
            rgb (Tensor): (B, T, N, 3, H, W)
            mask (Tensor): (B, T, N, H, W)
            norm_xy (Tensor): (B, T, N, 2, H, W)
            extrinsics (Tensor): (B, T, N, 4, 4)
            text (List[Tensor]): (B,)

        Returns:
            inputs (Dict[str, Tensor]): inputs
            outputs (Dict[str, Tensor | List[Tensor]]):
                - camera_mask (Tensor): (B, Ncam)
                - norm_xy_ds (Tensor): (B, Ncam, Lv, 2)
                - valid_vision_mask (Tensor): (B, Ncam, Lv)
                - deepstack_visual_embeds (List[Tensor]): len = n_stack_layers = 3, shape = (S, C), S = sum(camera_mask) * Hds * Wds
                - inputs_embeds (Tensor): shape = (B, L, C), L = Lpad + Ncam*Lv + Llang
                - hidden_states (List[Tensor]): len = n_layers, shape = (B, L, C), L = Lpad + Ncam*Lv + Llang
                - attention_mask (Tensor): (B, L)
                - is_vision_mask (Tensor): (B, L)
                - pos_id_1d (Tensor): (B, L)
                - modality_type: (B, L), including PAD, VISION, LANGUAGE
                # - visual_embeds (List[Tensor]): len = 3 (n_deep_stack) + 1 (last_layer), shape = (B, Ncam, Lv, C)
                # - latent_planning_token (Tensor): (B, 2*C)
        """
        ctx = torch.no_grad() if self.vlm_finetune_mode == "frozen" else contextlib.nullcontext()
        with ctx:
            inputs, outputs = self.vlm(
                rgb=obs_rgbs, 
                mask=obs_masks, 
                norm_xy=obs_norm_xys, 
                text=prompt_text,
                select_layer_indices=self.indices,
                fp16=fp16
            )

        inputs["extrinsics"] = obs_extrinsics
        outputs["extrinsics"] = obs_extrinsics[:, -1]  # (B, Ncam, 4, 4)
        # NOTE: xxx[:, -1] selects the latest image observation. We don't use history images

        return inputs, outputs

