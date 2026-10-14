"""Pose sampling helpers that do not depend on Isaac Sim runtime state.

Both helpers consume only numpy and the (Isaac-free) geometry utilities in
`isaac_env.non_isaac.perception` / `utils.sample2d`. The Isaac-bound pose
helpers (which need a live `World._backend_utils`) intentionally stay in
`eval/eval.py` for now.
"""

import numpy as np

from isaac_env.non_isaac import perception
from utils.sample2d import sample_xy


def sample_cam_extr():
    """Sample a third-person camera extrinsic on a hemisphere above the table.

    Hemisphere: radius in [2.3, 2.7] m around the workspace center
    (0.25, 0.25, 0); azimuth in [0, pi/2]; elevation in [pi/6, pi/3].
    Returns the (4, 4) ``world_T_cam`` looking at the workspace center.
    """
    origin = np.array([0.25, 0.25, 0])
    radius = np.random.uniform(2.3, 2.7)
    theta  = np.random.uniform(0, np.pi/2)            # azimuth from +X axis
    phi    = np.random.uniform(np.pi/6, np.pi/6*2)    # elevation above XY plane

    z = radius * np.sin(phi)
    x = radius * np.cos(phi) * np.cos(theta)
    y = radius * np.cos(phi) * np.sin(theta)

    return perception.look_at_view_transform(
        eye=origin + np.array([x, y, z]), to=origin, up=np.array([0, 0, 1])
    )


def sample_initial_pose():
    """Sample the gripper's initial 6-DoF pose for the rollout start state.

    XY drawn by `utils.sample2d.sample_xy`; Z uniform in [0.25, 0.5];
    orientation has Z pointing straight down (-Z world) and a randomized
    yaw in the XY plane.
    """
    xy  = sample_xy()
    xyz = np.concatenate([xy, [np.random.uniform(0.25, 0.5)]])

    z_axis = np.array([0, 0, -1.0])
    y_axis = np.concatenate([np.random.randn(2), [0.0]])
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-7)
    x_axis = np.cross(y_axis, z_axis)

    pose = np.eye(4)
    pose[:3, 3]    = xyz
    pose[:3, :3]   = np.stack([x_axis, y_axis, z_axis], axis=-1)
    return pose
