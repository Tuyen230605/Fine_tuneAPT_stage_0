"""Grasp-proposal base classes used by every object in pp_env.objs.

`GraspProposal` is the abstract parent: it holds an object's local bbox and
exposes pose-generation primitives, but leaves `best_grasp` unimplemented.

`ObjBase` is the reference concrete subclass tuned for cylindrical / box-like
items. Every concrete object class (cans, blocks, ycb, graspnet) inherits
from it and customises USD/OBJ paths, scale, and optionally overrides
`all_top_down_proposals` / `best_grasp` for non-cylindrical shapes.
"""

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation


class GraspProposal(object):
    """Generate candidate grasp poses around an object's bounding box.

    Subclasses override `best_grasp(obj_wcT, gripper_wcT, **kwargs)` to pick
    one pose per call given the object's current world transform.

    Local frame convention: object center at origin, XY = horizontal, +Z = up.
    `top_down_*` proposals point gripper -Z down; `bottom_up_*` point +Z up;
    `horizontal_*` point sideways at the object center.
    """

    def __init__(
        self,
        vmin: np.ndarray,
        vmax: np.ndarray,
        center_offset: np.ndarray = np.zeros(3),
        radius_scale: float = 1,
    ):
        """
        Args:
            vmin, vmax:     Object axis-aligned bounding-box extents in the
                            object's local frame (shape (3,)).
            center_offset:  XY shift of the grasp-center used for sampling.
            radius_scale:   Scales the polar radius for `*_border` / `horizontal`
                            proposals (1 = inscribed in the XY box diagonal).
        """
        self.vmin = vmin
        self.vmax = vmax

        self.center3d = (self.vmin + self.vmax) / 2.0
        self.center3d[:2] += center_offset[:2]
        self.center2d = np.array([self.center3d[0], self.center3d[1], 0])
        self.radius = np.linalg.norm((vmax - vmin)[:2]) / (2 * np.sqrt(2)) * radius_scale
    
    def top_down_border_proposal(self, theta):
        pos = np.array([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta),
            self.vmax[-1]
        ]) + self.center2d  # 3d center pos on the top plane

        axis_x = np.array([np.cos(theta), np.sin(theta), 0])
        axis_y = np.array([np.cos(theta-np.pi/2), np.sin(theta-np.pi/2), 0])
        axis_z = np.array([0, 0, -1])
        rot_mat = np.stack([axis_x, axis_y, axis_z], axis=0).T

        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = pos
        return pose
    
    def top_down_center_proposal(self, theta):
        pos = np.array([0, 0, self.vmax[-1]]) + self.center2d
        axis_x = np.array([np.cos(theta), np.sin(theta), 0])
        axis_y = np.array([np.cos(theta-np.pi/2), np.sin(theta-np.pi/2), 0])
        axis_z = np.array([0, 0, -1])
        rot_mat = np.stack([axis_x, axis_y, axis_z], axis=0).T

        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = pos
        return pose
    
    def bottom_up_center_proposal(self, theta):
        pos = np.array([0, 0, self.vmin[-1]]) + self.center2d
        axis_x = np.array([np.cos(theta), np.sin(theta), 0])
        axis_y = np.array([np.cos(theta-np.pi/2), np.sin(theta-np.pi/2), 0])
        axis_z = np.array([0, 0, 1])
        rot_mat = np.stack([axis_x, axis_y, axis_z], axis=0).T

        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = pos
        return pose

    def horizontal_proposal(self, theta, h_ratio=0.5):
        pos = np.array([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta),
            self.vmin[-1] + h_ratio * (self.vmax[-1] - self.vmin[-1])
        ]) + self.center2d
        pos2d = np.array([pos[0], pos[1], 0])

        axis_y = np.array([0, 0, -1])
        axis_z = self.center2d - pos2d
        axis_z = axis_z / np.linalg.norm(axis_z, axis=-1, keepdims=True)
        axis_x = np.cross(axis_y, axis_z)
        rot_mat = np.stack([axis_x, axis_y, axis_z], axis=0).T

        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = pos
        return pose
    
    def best_grasp(self, obj_wcT: np.ndarray, *arg_conds, **kwarg_conds):
        """Return the chosen 4x4 grasp pose in world frame.

        Args:
            obj_wcT:      (4, 4) ^{world}_{object} T at query time.
            *arg_conds:   Subclass-specific positional kwargs (usually `gripper_wcT`).
            **kwarg_conds: Subclass-specific keyword args (e.g. `flip=True`).
        """
        raise NotImplementedError


class ObjBase(GraspProposal):
    """Reference grasp implementation for cylindrical / box-like objects.

    Concrete subclasses must set:
        USD_PATH    Path to the Isaac Sim USD asset (relative to repo root).
        OBJ_PATH    Path to the OBJ file used at construction to derive the bbox.
        SCALE       (3,) numpy array scale applied to the loaded mesh.

    Subclasses may override `all_top_down_proposals` for non-cylindrical shapes
    (rectangular blocks, asymmetric YCB items, etc.) or `initial_pose` to spawn
    the object in a non-identity local frame.
    """

    # Concrete subclasses must set these.
    USD_PATH: str
    OBJ_PATH: str
    SCALE: np.ndarray

    def __init__(self):
        mesh: trimesh.Trimesh = trimesh.load(self.OBJ_PATH, force="mesh")
        mesh.apply_scale(self.SCALE)
        points = mesh.vertices
        vmin = points.min(axis=0)
        vmax = points.max(axis=0)
        super().__init__(vmin, vmax)

    def initial_pose(self):
        """Local-frame rotation applied at spawn (identity by default)."""
        return np.eye(3)

    def all_proposals(self):
        """Densely enumerate top-down / horizontal / bottom-up grasps for visualization."""
        grasp_poses = []
        for theta in np.linspace(0, np.pi*2, 36, endpoint=False):
            grasp_poses.append(self.top_down_center_proposal(theta))
            grasp_poses.append(self.horizontal_proposal(theta))
            grasp_poses.append(self.bottom_up_center_proposal(theta))
        return np.stack(grasp_poses, axis=0)

    def all_top_down_proposals(self, obj_wcT: np.ndarray):
        """Two grasp candidates picked based on the object's current Z-axis orientation."""
        zaxis = obj_wcT[:3, 2]
        grasp_poses = []
        for theta in np.linspace(0, np.pi*2, 2, endpoint=False):
            if zaxis[-1] > 0.7:          # upright
                grasp_poses.append(obj_wcT @ self.top_down_center_proposal(theta))
            elif zaxis[-1] > -0.7:       # on its side: grasp horizontally
                grasp_poses.append(obj_wcT @ self.horizontal_proposal(theta))
            else:                        # upside down: grasp from underneath
                grasp_poses.append(obj_wcT @ self.bottom_up_center_proposal(theta))
        return np.stack(grasp_poses, axis=0)

    def best_grasp(self, obj_wcT: np.ndarray, gripper_wcT: np.ndarray, flip=False):
        """Pick a single grasp that minimises pose change from `gripper_wcT`.

        Args:
            obj_wcT:     (4, 4) object world transform.
            gripper_wcT: (4, 4) current gripper world transform.
            flip:        If True and the object is on its side, mirror the grasp X/Y axes.
        """
        zaxis = obj_wcT[:3, 2]
        grasp_poses = []
        for theta in np.linspace(0, np.pi*2, 36, endpoint=False):
            if zaxis[-1] > 0.7:
                grasp_pose = obj_wcT @ self.top_down_center_proposal(theta)
            elif zaxis[-1] > -0.7:
                grasp_pose = obj_wcT @ self.horizontal_proposal(theta)
            else:
                grasp_pose = obj_wcT @ self.bottom_up_center_proposal(theta)
            grasp_poses.append(grasp_pose)

        grasp_poses = np.stack(grasp_poses, axis=0)
        if -0.7 < zaxis[-1] and zaxis[-1] <= 0.7:
            # On its side: pick the grasp pointing most directly down.
            zaxes = grasp_poses[:, :3, 2]
            index = np.argmin(zaxes[:, -1])
            grasp_pose = grasp_poses[index]
            if flip:
                grasp_pose[:3, :2] *= -1
        else:
            # Upright or upside down: pick the grasp closest to the current gripper pose
            # (also considering a 180° in-plane flip, which is usually still graspable).
            grasp_poses_flip = grasp_poses.copy()
            grasp_poses_flip[:, :3, :3] = np.einsum(
                "brc,cd->brd",
                grasp_poses_flip[:, :3, :3],
                Rotation.from_rotvec([0, 0, np.pi]).as_matrix(),
            )
            dT0 = np.einsum("rc,bcd->brd", np.linalg.inv(gripper_wcT), grasp_poses)
            dT1 = np.einsum("rc,bcd->brd", np.linalg.inv(gripper_wcT), grasp_poses_flip)
            du0 = np.linalg.norm(Rotation.from_matrix(dT0[:, :3, :3]).as_rotvec(), axis=-1)
            du1 = np.linalg.norm(Rotation.from_matrix(dT1[:, :3, :3]).as_rotvec(), axis=-1)
            du = np.concatenate([du0, du1], axis=0)

            dt0 = np.linalg.norm(dT0[:, :3, 3], axis=-1)
            dt1 = np.linalg.norm(dT1[:, :3, 3], axis=-1)
            dt = np.concatenate([dt0, dt1], axis=0)

            cost = 5 * du + dt
            index = np.argmin(cost)
            grasp_pose_index, need_flip = (index % len(dT0)), int(index > len(dT0))
            grasp_pose = (grasp_poses, grasp_poses_flip)[need_flip][grasp_pose_index]
        return grasp_pose
