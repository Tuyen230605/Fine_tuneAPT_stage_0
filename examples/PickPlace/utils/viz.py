"""Visualization helpers for episode rollouts.

Both functions are pure numpy + OpenCV; they have no dependency on Isaac Sim
and can be unit-tested in isolation.
"""

import cv2
import numpy as np


def plot_traj(bgr, cam_pose, ee_poses, K, color=(0, 0, 255)):
    """Overlay a sequence of EE positions onto a BGR image as projected dots.

    Args:
        bgr:       (H, W, 3) image, modified in place.
        cam_pose:  (4, 4) world-to-camera transform.
        ee_poses:  (Ta, 4, 4) end-effector poses to project.
        K:         (3, 3) pinhole intrinsics.
    """
    ceTs = np.linalg.inv(cam_pose)[None] @ ee_poses
    cets = ceTs[:, :3, 3]
    proj_norm = cets[:, :2] / cets[:, 2:3]
    fxy = K[[0, 1], [0, 1]]; cxy = K[[0, 1], [2, 2]]
    proj_pix = proj_norm * fxy + cxy
    for x, y in proj_pix:
        cv2.circle(bgr, (int(x), int(y)), radius=2, color=color, thickness=-1)
    return bgr


def draw_prompt(bgr, masks):
    """Dim the image and highlight `masks` regions with a white bounding box."""
    debug_bgr = (bgr * 0.3).astype(np.uint8)
    for mask in masks:
        debug_bgr[mask] = bgr[mask]
        rr, cc = np.nonzero(mask)
        debug_bgr = cv2.rectangle(
            img=debug_bgr,
            pt1=(cc.min() - 5, rr.min() - 5),
            pt2=(cc.max() + 5, rr.max() + 5),
            color=(255, 255, 255), thickness=2,
        )
    return debug_bgr
