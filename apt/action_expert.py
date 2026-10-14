import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn, Tensor
from diffusers import DDIMScheduler
from typing import Optional, Tuple, Dict, List, Union

from .layers.utils import simple_mlp
from .layers.pe import SinusoidalPosEmb, RoPE
from .layers.norms import RMSNorm, AdaRMSNorm
from .layers.attn import MySimpleMHA, FFN, ProjOpt
from .layers.rot_transforms import rotation_6d_to_matrix, quaternion_to_matrix
from .action_transform import states2action, action2states
from .encoders.modality import ModalityType


def prepare_attention_mask(
    vl_mask: Tensor, 
    vl_modality: Tensor, 
    a_mask: Tensor
):
    """
    Args:
        vl_mask (Tensor): (B, len_vl)
        vl_modality (Tensor): (B, len_vl)
        a_mask (Tensor): (B, len_a)
        train_stage (int): 0 for va training (no language influence on action), 1 for vla training
    
    Returns:
    -------
        causal_mask (Tensor): (B, len_vl + len_a, len_vl + len_a)
        causal_mask_dilated (Tensor): (B, len_vl + len_a, len_vl + len_a)
    """
    B, len_a = a_mask.shape
    B, len_vl = vl_mask.shape
    len_vla = len_a + len_vl

    causal_mask = a_mask.new_zeros((B, len_vla, len_vla))
    causal_mask[:, :, :len_vl] = vl_mask[:, None, :]
    causal_mask[:, len_vl:, len_vl:] = a_mask[:, None, :]

    a_modality = vl_modality.new_full(a_mask.shape, ModalityType.PAD)
    a_modality[a_mask] = ModalityType.ACTION
    vla_modality = torch.cat([vl_modality, a_modality], dim=1)  # (B, len_vl + len_a)

    causal_mask_dilated = causal_mask.clone()
    # drop query = v | a, key = l
    row_mask = (vla_modality == ModalityType.VISION) | (vla_modality == ModalityType.ACTION)
    col_mask = (vla_modality == ModalityType.LANGUAGE)
    causal_mask_dilated[row_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)] = False

    # drop query = l, key = v
    row_mask = (vla_modality == ModalityType.LANGUAGE)
    col_mask = (vla_modality == ModalityType.VISION)
    causal_mask_dilated[row_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)] = False

    # drop query = l, key = a
    # To avoid language tokens attending to action tokens (i.e., prevent language from "seeing" action tokens as keys),
    # we need to set the mask at (query=language, key=action) to False.
    row_mask = (vla_modality == ModalityType.LANGUAGE)
    col_mask = (vla_modality == ModalityType.ACTION)
    causal_mask_dilated[row_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)] = False

    # For train_stage == 0: completely block all language-action interactions
    # This ensures language cannot influence action in any way.
    # Note: The blocking is already done above (lines 48-62), but we ensure it's applied
    # when train_stage == 0. The mask will be used for both vla_mask and vla_mask_dilated
    # in train_stage == 0 to completely isolate actions from language.

    return causal_mask, causal_mask_dilated


class SplitNorm(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm_vl = RMSNorm(embed_dim)
        self.norm_a = AdaRMSNorm(embed_dim)
    
    def forward(self, x: Tensor, t: Tensor, split: Tuple[int, int]):
        vl, a = x.split(split, dim=1)  # (B, L_vl, C), (B, La, C)
        vl = self.norm_vl(vl)
        a = self.norm_a(a, t)  # modulated by diffusion timestep
        return torch.cat([vl, a], dim=1)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, pe_type):
        super().__init__()
        self.norm1 = SplitNorm(embed_dim)
        self.attn = MySimpleMHA(
            embed_dim, num_heads, proj_opt=ProjOpt.QKV,
            bias=True, qk_norm=True, pe_type=pe_type
        )

        self.norm2 = SplitNorm(embed_dim)
        self.ffn = FFN(embed_dim, embed_dim * 2)
    
    def forward(
        self, 
        x: Tensor, 
        pe: Tensor, 
        mask: Tensor, 
        film: Tensor,
        split: Tuple[int, int], 
    ):
        # pre-norm SA
        residual = x
        x = self.norm1(x, film, split)
        x, attn_weight, mask = self.attn(
            x=x,
            c=x,
            x_pe=pe,
            c_pe=pe,
            attn_mask=mask,
        )
        x = x + residual

        # FFN
        x = x + self.ffn(self.norm2(x, film, split))
        return x, mask


class HybridAttentionLayers(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, train_stage: int):
        super().__init__()
        self.train_stage = train_stage
        blocks = []
        if num_layers % 2 == 0:
            va_num_layers = num_layers // 2
        else:
            va_num_layers = num_layers // 2 + 1
        if train_stage == 0:
            for i in range(va_num_layers):
                blocks.append(AttentionBlock(embed_dim, num_heads, pe_type=("prope", None))) # !!! if use prope, mask_dilated is needed !!!
        else:
            for i in range(num_layers):
                if i % 2 == 0:
                    blocks.append(AttentionBlock(embed_dim, num_heads, pe_type=(None, "rope")))
                else:
                    blocks.append(AttentionBlock(embed_dim, num_heads, pe_type=("prope", None)))
        self.layers = nn.ModuleList(blocks)
        self.gate = nn.Parameter(torch.zeros(num_layers, embed_dim))
    
    def forward(
        self, 
        x: Tensor, 
        pe: Tensor, 
        pe_gta: Tensor, 
        mask: Tensor, 
        mask_dilated: Tensor,
        film_cond: Tensor, 
        vla_split_size: Tuple[int, int], 
        vl_highways: List[Tensor],
    ):
        gate = self.gate.sigmoid()
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                x, mask = layer(x, pe, mask, film_cond, vla_split_size)
            else:
                x, mask_dilated = layer(x, pe_gta, mask_dilated, film_cond, vla_split_size)
            
            if i < len(vl_highways):
                vl, a = x.split(vla_split_size, dim=1)
                if self.train_stage == 0:
                    j = 2 * i + 1
                else:
                    j = i
                gi = gate[j]  # (hdim,)
                vl = vl * gi + vl_highways[j] * (1 - gi)
                x = torch.cat([vl, a], dim=1)
        return x


class ContextEncoder(nn.Module):
    def __init__(self, idim, hdim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.idim = idim
        self.hdim = hdim
        self.proj_input = nn.Linear(idim, hdim)
        self.proj_vl = nn.ModuleList([nn.Linear(idim, hdim) for _ in range(num_layers)])
        self.proj_pe = nn.Linear(2, hdim)  # for normalized coordinates
        self.rope = RoPE(hdim//2, 1, num_heads)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.proj_pe.weight)
        nn.init.zeros_(self.proj_pe.bias)

    def forward(
        self, 
        vl_outputs: Dict[str, Tensor], 
        action_mask: Tensor, 
        action_ref_pose: Tensor, 
        fp16: bool
    ):
        """
        Args:
            vl_outputs (Dict[str, Tensor]):
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

            action_mask: (B, Nee, nhist+Ta)
            action_ref_pose: (B, Nee, 4, 4)
            fp16: if True, use bfloat16        
        Returns
        -------
            vision_embeds: (B, Ncam*Lv, hdim)
            mask_v: (B, Ncam*Lv)
            extrinsic_v: (B, Ncam*Lv, 4, 4),
            c: (B, hdim)
        """
        ################## process VLM ##################
        B, Ncam, Lv, _ = vl_outputs["norm_xy_ds"].shape
        extrinsics = vl_outputs["extrinsics"]  # (B, Ncam, 4, 4)
        norm_xy_ds = vl_outputs["norm_xy_ds"]  # (B, Ncam, Lv, 2)
        camera_mask = vl_outputs["camera_mask"]  # (B, Ncam)

        # initial vl pose as eye(4)
        len_vl = vl_outputs["hidden_states"][0].shape[1]
        vl_extr = torch.eye(4).to(extrinsics)[None, None].repeat(B, len_vl, 1, 1)  # (B, len_vl, 4, 4)

        # camera pose as positional encoding
        v_extr = extrinsics[:, :, None, :, :].expand(B, Ncam, Lv, 4, 4)
        vl_extr[vl_outputs["is_vision_mask"]] = v_extr[camera_mask].view(-1, 4, 4)

        with torch.autocast(
            vl_outputs["inputs_embeds"].device.type, 
            torch.bfloat16 if fp16 else torch.float32
        ):
            # project initial and hidden vl features
            vl0: Tensor = self.proj_input(vl_outputs["inputs_embeds"])
            vl_highways: List[Tensor] = [proj(vl_outputs["hidden_states"][i]) 
                                        for i, proj in enumerate(self.proj_vl)]

            # add norm xy embed
            vl0 = vl0.clone()
            norm_xy_embed: Tensor = self.proj_pe(norm_xy_ds)  # (B, Ncam, Lv, hdim)
            vl0[vl_outputs["is_vision_mask"]] += norm_xy_embed[camera_mask].view(-1, self.hdim)
        
        # get attention_mask
        vl_mask = vl_outputs["attention_mask"]  # (B, L) alreay fine grined by valid_vision_mask
        

        ################## process Action Expert ##################
        B, Nee, La = action_mask.shape
        action_extr = action_ref_pose[:, :, None].expand(B, Nee, La, 4, 4).reshape(B, Nee*La, 4, 4)
        pos1d_offset = vl_outputs["pos_id_1d"][:, -1] + 1  # (B,)
        action_pos1d = torch.arange(La).to(pos1d_offset)[None, None, :] + pos1d_offset[:, None, None]  # (B, 1, La)
        action_pos1d = action_pos1d.expand(B, Nee, La).reshape(B, Nee*La)  # (B, Nee*La)


        ################## Merge together ##################
        vla_mask, vla_mask_dilated = prepare_attention_mask(
            vl_mask=vl_mask,
            vl_modality=vl_outputs["modality_type"],
            a_mask=action_mask.view(B, Nee*La)
        )

        vla_extr = torch.cat([vl_extr, action_extr], dim=1)  # (B, L+La, 4, 4)
        vla_pos1d = torch.cat([vl_outputs["pos_id_1d"], action_pos1d], dim=1)  # (B, L+La)
        vla_rope = self.rope(vla_pos1d.unsqueeze(-1))  # (B, L+La, D, 2)

        vla_pe = (None, vla_rope)  # rope only, half dimension
        vla_pe_gta = (vla_extr, None)  # GTA only, half dimension

        return vl0, vla_pe, vla_pe_gta, vla_mask, vla_mask_dilated, vl_highways


class DiffusionHead(nn.Module):
    """
    - Input: image observations and noisy action at `t`
    - Output: noisy action at `t-1`
    """

    def __init__(self, hdim: int, num_heads: int, act_dim: int, num_layers: int, train_stage: int):
        super().__init__()
        self.act_dim = act_dim
        self.num_layers = num_layers
        self.hist_enc = simple_mlp([act_dim-1, hdim, hdim], ln=True)
        self.traj_enc = simple_mlp([act_dim, hdim, hdim], ln=True)
        self.abs_pos_enc = simple_mlp([3, hdim, hdim], ln=True)

        self.denoising_time_embed = nn.Sequential(
            SinusoidalPosEmb(hdim, temperature=1000),  # train diffusion timestep = 100
            simple_mlp([hdim, hdim, hdim], ln=True)
        )

        ### traj-context attn
        self.max_nee = 4  # maximum number of end-effectors supported
        self.arm_id_embed = nn.Parameter(torch.randn(1, self.max_nee, 1, hdim))
        self.traj_context_attn = HybridAttentionLayers(
            hdim, num_heads, num_layers, train_stage
        )

        ### final mlp
        self.final_norm = RMSNorm(hdim)
        self.act_head = nn.Linear(hdim, act_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.trunc_normal_(self.arm_id_embed, std=0.02)
        nn.init.zeros_(self.abs_pos_enc[-1].weight)
        nn.init.zeros_(self.act_head.weight)
    
    def pos_rel2abs(self, cur_wcT: Tensor, cur_weT: Tensor, t3r6: Tensor):
        """
        Args:
            cur_wcT (Tensor): (..., 4, 4), ^{world} T _{cam}
            cur_weT (Tensor): (..., 4, 4), ^{world} T _{ee}
            t3r6 (Tensor): (..., T, 9)
        
        Returns:
            traj_cet (Tensor), traj ee pos in camera frame, shape (..., T, 3)
        """
        ecT = torch.inverse(cur_weT) @ cur_wcT  # (..., 4, 4)
        ecR = ecT[..., :3, :3]  # (..., 3, 3)
        ecR = ecR.unsqueeze(-3)  # (..., 1, 3, 3)
        
        e1e2R = ecR @ rotation_6d_to_matrix(t3r6[..., 3:]) @ ecR.transpose(-1, -2)  # (..., T, 3, 3)
        e1e2t = (ecR @ t3r6[..., :3].unsqueeze(-1)).squeeze(-1)  # (..., T, 3)
        
        e1e2T = e1e2t.new_zeros(*e1e2t.shape[:-1], 4, 4)  # (..., T, 4, 4)
        e1e2T[..., :3, :3] = e1e2R
        e1e2T[..., :3, 3] = e1e2t
        e1e2T[..., 3, 3] = 1

        traj_ceT = (torch.inverse(cur_wcT) @ cur_weT).unsqueeze(-3) @ e1e2T
        traj_cet = traj_ceT[..., :3, 3]  # (..., T, 3)
        return traj_cet

    def forward(
        self, 
        denoise_timestep: Tensor, 
        trajectory: Tensor, 
        history: Tensor, 
        cur_weT: Tensor, 
        cur_wrT: Tensor, 
        vl0: Tensor, 
        vla_pe: Tuple[Tensor], 
        vla_pe_gta, 
        vla_mask, 
        vla_mask_dilated, 
        vl_highways, 
        fp16: bool
    ):
        """
        Args:
            denoise_timestep: (B,), denoising time step
            trajectory: (B, Nee, Ta, act_dim)
            history: (B, Nee, nhist, act_dim)
            cur_weT: (B, Nee, 4, 4)
            cur_wrT: (B, Nee, 4, 4)
            cond: [(B, Lc, hdim)]
            cond_mask: B, Lc
            cond_pe: (B, Lc, 4, 4)
            action_mask: (B, Nee, nhist+Ta)
            action_ref_pose: (B, Nee, 4, 4)
            fp16 (bool): use bfloat16

        Returns:
            action_epsilon: (B, Nee, Ta, act_dim)
        """
        with torch.autocast(
            denoise_timestep.device.type,
            torch.bfloat16 if fp16 else torch.float32
        ):
            denoise_time_embed: Tensor = self.denoising_time_embed(denoise_timestep)  # (B, hdim)

            # noisy trajectory add temporal positional embeddings
            B, Nee, Ta, _ = trajectory.shape
            B, Nee, nhist, _ = history.shape
            hist_feats = self.hist_enc(history[..., :self.act_dim-1])  # (B, Nee, nhist, hdim)
            traj_feats = self.traj_enc(trajectory[..., :self.act_dim])  # (B, Nee, Ta, hdim)
            full_traj_feats = torch.cat([hist_feats, traj_feats], dim=-2)  # (B, Nee, nhist+Ta, hdim)

            # get absolute pos under camera, add additional positional encoding
            with torch.no_grad():
                full_traj_t3r6 = torch.cat([history[..., :9], trajectory[..., :9]], dim=-2)  # (B, Nee, nhist+Ta, 9)
                abs_pos = self.pos_rel2abs(cur_wrT, cur_weT, full_traj_t3r6)
            # let the model be aware of the absolute position of the end-effector
            full_traj_feats = full_traj_feats + self.abs_pos_enc(abs_pos)  # (B, Nee, nhist+Ta, hdim)
            full_traj_feats = full_traj_feats + self.arm_id_embed[:, :Nee, :, :]  # (B, Nee, nhist+Ta, hdim)
            full_traj_feats = rearrange(full_traj_feats, "b n l c -> b (n l) c")            # (B, Nee*(nhist+Ta), C)

            vla_split_size = [vl0.shape[1], full_traj_feats.shape[1]]
            x = torch.cat([vl0, full_traj_feats], dim=1)

            x = self.traj_context_attn(
                x=x,
                pe=vla_pe,
                pe_gta=vla_pe_gta,
                mask=vla_mask,
                mask_dilated=vla_mask_dilated,
                film_cond=denoise_time_embed,
                vla_split_size=vla_split_size,
                vl_highways=vl_highways,
            )

            full_traj_feats = x[:, vl0.shape[1]:]  # (B, Nee*(nhist+Ta), C)
            full_traj_feats = rearrange(full_traj_feats, "b (n l) c -> b n l c", n=Nee, l=nhist+Ta)
            traj_feats = full_traj_feats[:, :, nhist:nhist+Ta, :]  # (B, Nee, Ta, hdim)
            traj_feats = self.final_norm(traj_feats)
            action = self.act_head(traj_feats)
        return action


def generate_action_mask(nhist: int, Ta: int, valid_ee_mask: Tensor):
    """
    Args:
        nhist (int): number of history
        Ta (int): horizon of future actions
        valid_ee_mask (Tensor): shape (B, Nee)
    
    Returns:
        action_mask (Tensor): (B, Nee, nhist+Ta)
    """
    B, Nee = valid_ee_mask.shape
    mask = valid_ee_mask.new_ones(B, Nee, nhist + Ta)
    mask[~valid_ee_mask] = False
    return mask


def compute_mean_pose(vl_outputs: Dict[str, Tensor], inference: bool):
    cam_trans = vl_outputs["extrinsics"][:, :, :3, 3]  # (B, Ncam, 3)
    cam_mask = vl_outputs["camera_mask"].float()  # (B, Ncam)
    avg_trans = (cam_trans * cam_mask.unsqueeze(-1)).sum(dim=1) / cam_mask.unsqueeze(-1).sum(dim=1)  # (B, 3)

    B = avg_trans.shape[0]
    avg_pose = torch.eye(4).to(avg_trans)[None].repeat(B, 1, 1)
    avg_pose[:, :3, 3] = avg_trans

    if not inference:
        # generate random rotation for data augmentation
        rand_q = torch.randn(B, 4).to(avg_trans)
        rand_q = nn.functional.normalize(rand_q, dim=-1)
        avg_pose[:, :3, :3] = quaternion_to_matrix(rand_q)

    return avg_pose  # (B, 3, 3)


class ActionExpert(nn.Module):
    """Diffusion-based action expert with layer-wise VLM gated fusion.

    The expert consumes the per-layer VLM hidden states (one per attention
    layer, sampled at uniform depth) and a noisy action sequence. Each
    self-attention layer adds a sigmoid-gated VLM feature into its input
    stream (see paper Section 3.3 Eq. 3). ``train_stage`` switches the
    active depth and the language attention mask:

    * ``train_stage=0`` activates the first ``N/2`` traj-context layers and
      masks language tokens (pure VA prior).
    * ``train_stage=1`` activates all ``N`` interleaved layers and lets
      language tokens attend (VLA likelihood).
    """

    def __init__(
        self,
        idim: int,
        hdim: int,
        num_heads: int,
        num_diffusion_layers: int,
        diffusion_timesteps: int = 100,
        inference_timesteps: Optional[int] = None,
        train_stage: int = 0,
        camera_view_dropout: float = 0.0,
    ):
        super().__init__()
        self.context_encoder = ContextEncoder(idim, hdim, num_heads, num_diffusion_layers)
        self.dp_head = DiffusionHead(hdim, num_heads, self.act_dim, num_diffusion_layers, train_stage)

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False
        )

        self.diffusion_timesteps = diffusion_timesteps
        if inference_timesteps is None:
            inference_timesteps = max(diffusion_timesteps//5, 10)
        self.inference_timesteps = inference_timesteps
        self.inference_scheduler = self.noise_scheduler

        self.train_stage = train_stage
        self.camera_view_dropout = camera_view_dropout

    def _add_input_noise(
        self,
        history_action: Tensor,
        current_ee_pose: Tensor,
        inference: bool
    ):
        # Keep targets clean; only perturb model inputs during training.
        if inference:
            return history_action, current_ee_pose

        hist_noisy = history_action.clone()
        cur_noisy = current_ee_pose.clone()

        # action noise in camera-frame representation
        hist_noisy[..., :3] += torch.randn_like(hist_noisy[..., :3]) * 0.005
        hist_noisy[..., 3:9] += torch.randn_like(hist_noisy[..., 3:9]) * 0.01
        hist_noisy[..., 9:10] = (hist_noisy[..., 9:10] + torch.randn_like(hist_noisy[..., 9:10]) * 0.01).clamp(-1.0, 1.0)

        # small proprioceptive translation noise (no rotation perturbation for stability)
        cur_noisy[..., :3, 3] += torch.randn_like(cur_noisy[..., :3, 3]) * 0.003

        return hist_noisy, cur_noisy

    def _apply_camera_view_dropout(
        self, 
        vl_outputs: Dict[str, Tensor], 
        p_drop: Union[float, Tensor]
    ):
        """Randomly mask out all tokens from one camera view during training.

        For each sample in the batch, with probability p_drop, one active camera
        is randomly selected and its vision tokens are masked out in both the
        attention mask and modality type. This forces the model to handle missing
        camera views and develop more independent per-camera representations.

        Only applied when at least 2 cameras are active (never drops the last one).

        Args:
            p_drop: scalar float for uniform probability, or (B,) Tensor for
                    per-sample probabilities (e.g. different suites in a mixed batch).
        """
        camera_mask = vl_outputs["camera_mask"]  # (B, Ncam)
        is_vision_mask = vl_outputs["is_vision_mask"]  # (B, L)
        B, Ncam = camera_mask.shape
        Lv = vl_outputs["norm_xy_ds"].shape[2]

        if isinstance(p_drop, Tensor):
            drop_decisions = torch.rand(B, device=p_drop.device) < p_drop
        else:
            drop_decisions = torch.rand(B) < p_drop
        if not drop_decisions.any():
            return

        attention_mask = vl_outputs["attention_mask"].clone()
        modality_type = vl_outputs["modality_type"].clone()

        for b in range(B):
            if not drop_decisions[b]:
                continue
            active_cams = camera_mask[b].nonzero(as_tuple=True)[0]
            if len(active_cams) <= 1:
                continue

            drop_local_idx = torch.randint(len(active_cams), (1,)).item()
            vision_positions = is_vision_mask[b].nonzero(as_tuple=True)[0]
            start = drop_local_idx * Lv
            end = start + Lv
            drop_positions = vision_positions[start:end]

            attention_mask[b, drop_positions] = False
            modality_type[b, drop_positions] = ModalityType.PAD

        vl_outputs["attention_mask"] = attention_mask
        vl_outputs["modality_type"] = modality_type

    @property
    def act_dim(self):
        """dimension of action defined in camera frame"""
        return 10

    def iterative_denoise(
        self, 
        traj_shape: Tuple[int, int, int, int],
        fixed_inputs: Dict[str, Tensor],
        initial_noise: Optional[Tensor] = None
    ):
        """
        Args:
            trajectory_shape: (B, Nee, Ta, act_dim)
            fixed_inputs: inputs for diffusion head
            initial_noise: (B, Nee, Ta, act_dim) or None

        Returns:
            trajectory: (B, Nee, Ta, act_dim)
        """
        if initial_noise is None:
            B, Nee, Ta, _ = traj_shape
            device = next(iter(fixed_inputs.values())).device
            initial_noise = torch.randn(B, Nee, Ta, self.act_dim, device=device)
        
        self.inference_scheduler.set_timesteps(self.inference_timesteps)
        trajectory = initial_noise
        for t in self.inference_scheduler.timesteps:
            out = self.dp_head(
                t * torch.ones(trajectory.shape[0], device=trajectory.device), 
                trajectory,
                **fixed_inputs
            )
            trajectory = self.inference_scheduler.step(
                out[..., :self.act_dim], t, trajectory[..., :self.act_dim]
            ).prev_sample
        return trajectory

    def forward(
        self, 
        vl_inputs: Dict[str, Tensor],
        vl_outputs: Dict[str, Tensor], 
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
            vl_inputs: (Dict[str, Tensor]):
                - rgb: (B, To, ncam, 3, H, W)
                - mask: (B, To, ncam, H, W)
                - norm_xy: (B, To, ncam, 2, H, W), coordinates in normalized camera plane
                - text: List (length=B) of prompt
                - extrinsics: (B, To, ncam, 4, 4), ^{world}_{camera} T
            
            vl_outputs (Dict[str, Tensor]):
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
        # compute average pose
        avg_pose = compute_mean_pose(vl_outputs, inference=inference)  # (B, 4, 4)
        avg_pose_inv = torch.inverse(avg_pose)  # (B, 4, 4)

        # apply avg pose distraction to all pose-related inputs
        vl_outputs["extrinsics"] = avg_pose_inv.unsqueeze(1) @ vl_outputs["extrinsics"]
        current_ee_pose = avg_pose_inv.unsqueeze(1) @ current_ee_pose
        action_ref_pose = avg_pose_inv.unsqueeze(1) @ action_ref_pose
        
        history_ee_states = history_ee_states.clone()
        history_ee_states[..., :16] = (
            avg_pose_inv.unsqueeze(1).unsqueeze(1) @  # (B, 1, 1, 4, 4) 
            history_ee_states[..., :16].reshape(*history_ee_states.shape[:-1], 4, 4)  # (B, nhist, Nee, 4, 4)
        ).view(*history_ee_states.shape[:-1], 16)
        
        gt_future_ee_states = gt_future_ee_states.clone()
        gt_future_ee_states[..., :16] = (
            avg_pose_inv.unsqueeze(1).unsqueeze(1) @  # (B, 1, 1, 4, 4) 
            gt_future_ee_states[..., :16].reshape(*gt_future_ee_states.shape[:-1], 4, 4)  # (B, nhist, Nee, 4, 4)
        ).view(*gt_future_ee_states.shape[:-1], 16)

        # debug
        try:
            extrinsics = vl_outputs["extrinsics"]  # (B, Ncam, 4, 4)
            extr_inv = extrinsics.inverse()
        except Exception as e:
            print(extrinsics)
            print(e)

            B, Ncam = extrinsics.shape[:2]
            for b in range(B):
                for n in range(Ncam):
                    if extrinsics[b, n, -1, -1].abs() < 1e-6:
                        print(f"Invalid extrinsics at batch {b}, camera {n}:")
                        print(extrinsics[b, n])
                        print("prompt = {}".format(vl_inputs["text"][b]))
            from IPython import embed; embed()

        history_ee_states = history_ee_states.transpose(1, 2)  # (B, Nee, nhist, 4*4+1)
        gt_future_ee_states = gt_future_ee_states.transpose(1, 2)  # (B, Nee, Ta, 4*4+1)

        B, Nee, nhist, _ = history_ee_states.shape
        B, Nee, Ta, _ = gt_future_ee_states.shape

        if self.training:
            if camera_drop_prob is not None:
                self._apply_camera_view_dropout(vl_outputs, p_drop=camera_drop_prob)
            elif self.camera_view_dropout > 0:
                self._apply_camera_view_dropout(vl_outputs, p_drop=self.camera_view_dropout)

        action_mask = generate_action_mask(nhist=nhist, Ta=Ta, valid_ee_mask=valid_ee_mask)
        vl0, vla_pe, vla_pe_gta, vla_mask, vla_mask_dilated, vl_highways = self.context_encoder(
            vl_outputs=vl_outputs,
            action_mask=action_mask,
            action_ref_pose=action_ref_pose, 
            fp16=fp16
        )

        history_action = states2action(
            action_ref_pose, 
            current_ee_pose, 
            history_ee_states
        )  # (B, Nee, nhist, 10)

        if not inference:
            gt_future_action = states2action(
                action_ref_pose, 
                current_ee_pose, 
                gt_future_ee_states
            )  # (B, Nee, Ta, 10)

        if robot_pose_aug:
            history_action, current_ee_pose = self._add_input_noise(
                history_action=history_action,
                current_ee_pose=current_ee_pose,
                inference=inference
            )

        if self.train_stage == 0:
            fixed_inputs = dict(
                history=history_action,  # shape (B', nhist, act_dim)
                cur_weT=current_ee_pose,
                cur_wrT=action_ref_pose,
                vl0=vl0,
                vla_pe=vla_pe_gta,
                vla_pe_gta=vla_pe_gta,
                vla_mask=vla_mask_dilated, # mask l for v and a
                vla_mask_dilated=vla_mask_dilated,
                vl_highways=vl_highways,
                fp16=fp16
            )
        elif self.train_stage == 1:
            fixed_inputs = dict(
                history=history_action,  # shape (B', nhist, act_dim)
                cur_weT=current_ee_pose,
                cur_wrT=action_ref_pose,
                vl0=vl0,
                vla_pe=vla_pe,
                vla_pe_gta=vla_pe_gta,
                vla_mask=vla_mask,
                vla_mask_dilated=vla_mask_dilated,
                vl_highways=vl_highways,
                fp16=fp16
            )            

        ###################### Inference ######################
        if inference:
            pred_actions = self.iterative_denoise(
                traj_shape=(B, Nee, Ta, self.act_dim),
                fixed_inputs=fixed_inputs
            )  # (B, Nee, Ta, act_dim)
            pred_future_ee_states = action2states(
                action_ref_pose,    # (B, Nee, 4, 4)
                current_ee_pose,    # (B, Nee, 4, 4)
                pred_actions        # (B, Nee, Ta, act_dim)
            )  # (B, Nee, Ta, 4*4+1)

            # revert avg pose distraction
            pred_future_ee_states[..., :16] = (
                avg_pose.unsqueeze(1).unsqueeze(1) @  # (B, 1, 1, 4, 4) 
                pred_future_ee_states[..., :16].reshape(*pred_future_ee_states.shape[:-1], 4, 4)  # (B, Ta, Nee, 4, 4)
            ).view(*pred_future_ee_states.shape[:-1], 16)

            return pred_future_ee_states.transpose(1, 2).contiguous()  # (B, Ta, Nee, 17)

        ###################### Training ######################
        # sample noise
        noise = torch.randn(B, Nee, Ta, self.act_dim, 
                            device=gt_future_ee_states.device)

        # sample a random timestep
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            size=(B,), 
            device=noise.device
        )

        # add noise to the clean trajectory
        noisy_trajectory = self.noise_scheduler.add_noise(
            gt_future_action, noise,
            timesteps
        )

        # one step denoising
        pred = self.dp_head(timesteps, noisy_trajectory, **fixed_inputs)
        target = get_target(gt_future_action, noise, timesteps, self.noise_scheduler)

        # only calculate errors on valid_ee_indices
        pred = pred[valid_ee_mask]  # (B', Ta, act_dim)
        target = target[valid_ee_mask]  # (B', Ta, act_dim)
        
        # loss calculation
        pos_loss = F.l1_loss(pred[..., 0:3], target[..., 0:3], reduction="mean")
        rot_loss = F.l1_loss(pred[..., 3:9], target[..., 3:9], reduction="mean")
        openness_loss = F.l1_loss(pred[..., 9:10], target[..., 9:10], reduction="mean")

        total_loss = 30 * pos_loss + 10 * rot_loss + 10 * openness_loss
        metrics = {
            "pos_loss": pos_loss.item(),
            "rot_loss": rot_loss.item(),
            "openness_loss": openness_loss.item(),
            "total_loss": total_loss.item()
        }
        return total_loss, metrics


def get_target(traj: Tensor, noise: Tensor, timesteps: Tensor, scheduler: DDIMScheduler):
    """returns supervision depending on scheduler type"""
    pred_type = scheduler.config.prediction_type
    if pred_type == "epsilon":
        target = noise
    if pred_type == "sample":
        target = traj
    if pred_type == "v_prediction":
        target = scheduler.get_velocity(traj, noise, timesteps) 
    return target

