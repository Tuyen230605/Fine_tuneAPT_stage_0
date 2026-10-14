import torch
from torch import Tensor
from .layers.rot_transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


def space_ee2cam(cur_wcT: Tensor, cur_weT: Tensor, fut_weT: Tensor):
    """
    Args:
        cur_wcT (Tensor): (..., 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (..., 4, 4), ^{world} T _{ee}
        fut_weT (Tensor): (..., T, 4, 4), future ee pose in world frame
    
    Returns:
        t3r6 (Tensor): some repr of ^{cam} v _{ee} * dt, shape (..., T, 9)
    """
    e1e2T = torch.inverse(cur_weT.unsqueeze(-3)) @ fut_weT  # (..., T, 4, 4)
    e1e2R = e1e2T[..., :3, :3]  # (..., T, 3, 3)
    e1e2t = e1e2T[..., :3, 3]   # (..., T, 3)

    ceT = torch.inverse(cur_wcT) @ cur_weT  # (..., 4, 4)
    ceR = ceT[..., :3, :3]  # (..., 3, 3)
    ceR = ceR.unsqueeze(-3)  # (..., 1, 3, 3)
    
    r = matrix_to_rotation_6d(ceR @ e1e2R @ ceR.transpose(-1, -2))
    t = (ceR @ e1e2t.unsqueeze(-1)).squeeze(-1)
    t3r6 = torch.cat([t, r], dim=-1)
    return t3r6


def space_cam2ee(cur_wcT: Tensor, cur_weT: Tensor, t3r6: Tensor):
    """
    Args:
        cur_wcT (Tensor): (..., 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (..., 4, 4), ^{world} T _{ee}
        t3r6 (Tensor): (..., T, 9)
    
    Returns:
        fut_weT (Tensor), future ee pose in world frame, shape (..., T, 4, 4)
    """
    ecT = torch.inverse(cur_weT) @ cur_wcT  # (..., 4, 4)
    ecR = ecT[..., :3, :3]  # (..., 3, 3)
    ecR = ecR.unsqueeze(-3)  # (..., 1, 3, 3)
    
    e1e2R = ecR @ rotation_6d_to_matrix(t3r6[..., 3:]) @ ecR.transpose(-1, -2)  # (..., T, 3, 3)
    e1e2t = (ecR @ t3r6[..., :3].unsqueeze(-1)).squeeze(-1)  # (..., T, 3)
    
    e1e2T = e1e2t.new_zeros(*e1e2t.shape[:-1], 4, 4)
    e1e2T[..., :3, :3] = e1e2R
    e1e2T[..., :3, 3] = e1e2t
    e1e2T[..., 3, 3] = 1

    fut_weT = cur_weT.unsqueeze(-3) @ e1e2T
    return fut_weT


def states2action(cur_wcT: Tensor, cur_weT: Tensor, ee_states: Tensor):
    """
    Args:
        cur_wcT (Tensor): (..., 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (..., 4, 4), ^{world} T _{ee}
        ee_states (Tensor): (..., T, 16 or 17)
    
    Returns:
        action (Tensor): (..., T, 9 or 10)
    """
    Ta, C = ee_states.shape[-2:]
    batch_shape = ee_states.shape[:-2]
 
    weT = ee_states[..., :16].view(*batch_shape, Ta, 4, 4)
    t3r6 = space_ee2cam(cur_wcT, cur_weT, weT)
    
    if C == 16:
        return t3r6
    else:
        openness = (ee_states[..., -1:] - 0.5) * 2  # normalize gripper openness
        return torch.cat([t3r6, openness], dim=-1)


def action2states(cur_wcT: Tensor, cur_weT: Tensor, action: Tensor):
    """
    Args:
        cur_wcT (Tensor): (B, 4, 4), ^{world} T _{cam}
        cur_weT (Tensor): (B, 4, 4), ^{world} T _{ee}
        action (Tensor): (B, T, 9 or 10)
    
    Returns:
        ee_states (Tensor): (B, T, 16 or 17)
    """
    Ta, C = action.shape[-2:]
    batch_shape = action.shape[:-2]

    t3r6 = action[..., :9]
    weT = space_cam2ee(cur_wcT, cur_weT, t3r6).view(*batch_shape, Ta, 16)
    
    if C == 9:
        return weT
    else:
        openness = action[..., -1:] / 2 + 0.5  # denormalize gripper openness
        return torch.cat([weT, openness], dim=-1)

