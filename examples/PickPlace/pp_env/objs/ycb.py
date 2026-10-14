"""YCB-style single-object classes (USDs from assets/mesh/Axis_Aligned/)."""

import numpy as np
from scipy.spatial.transform import Rotation

from .base import ObjBase


class CanTomatoSoup(ObjBase):
    USD_PATH = "assets/mesh/Axis_Aligned/005_tomato_soup_can.usd"
    OBJ_PATH = "assets/mesh/Axis_Aligned/005_tomato_soup_can.obj"
    SCALE = np.ones(3) * 0.75
    
    def all_top_down_proposals(self, obj_wcT: np.ndarray):
        zaxis = obj_wcT[:3, 2]
        grasp_poses = []
        for theta in np.linspace(0, np.pi*2, 36, endpoint=False):
            if zaxis[-1] > 0.7 or zaxis[-1] < -0.7:  # can on its side: grasp horizontally
                grasp_poses.append(obj_wcT @ self.top_down_center_proposal(theta))
            # elif zaxis[-1] > -0.7:  # can upright
            #     grasp_poses.append(obj_wcT @ self.horizontal_proposal(theta))
            # else:  # can on its side, upside down
            #     grasp_poses.append(obj_wcT @ self.bottom_up_center_proposal(theta))
        grasp_poses = np.stack(grasp_poses, axis=0)

        # Filter grasp poses based on x-axis alignment with object y-axis
        obj_yaxis = obj_wcT[:3, 1]  # Get y-axis of object pose
        x_alignment = np.abs(np.einsum('bc,c->b', grasp_poses[:, :3, 0], obj_yaxis))  # Dot product with grasp x-axis
        # alignment_indices = np.argsort(np.abs(x_alignment))[-2:]
        alignment_indices = np.argsort(np.abs(x_alignment))[:2]
        grasp_poses = grasp_poses[alignment_indices]

        return grasp_poses


class WoodenBlock(ObjBase):
    USD_PATH = "assets/mesh/Axis_Aligned/036_wood_block.usd"
    OBJ_PATH = "assets/mesh/Axis_Aligned/036_wood_block.obj"
    SCALE = np.ones(3) * 0.6

    def all_top_down_proposals(self, obj_wcT: np.ndarray):
        zaxis = obj_wcT[:3, 2]
        grasp_poses = []
        for theta in np.linspace(0, np.pi*2, 36, endpoint=False):
            # if zaxis[-1] > 0.7 or zaxis[-1] < -0.7:  # can on its side: grasp horizontally
            grasp_poses.append(obj_wcT @ self.top_down_center_proposal(theta))
            # elif zaxis[-1] > -0.7:  # can upright
            #     grasp_poses.append(obj_wcT @ self.horizontal_proposal(theta))
            # else:  # can on its side, upside down
            #     grasp_poses.append(obj_wcT @ self.bottom_up_center_proposal(theta))
        grasp_poses = np.stack(grasp_poses, axis=0)

        # Filter grasp poses based on x-axis alignment with object y-axis
        obj_yaxis = obj_wcT[:3, 1]  # Get y-axis of object pose
        x_alignment = np.abs(np.einsum('bc,c->b', grasp_poses[:, :3, 0], obj_yaxis))  # Dot product with grasp x-axis
        # alignment_indices = np.argsort(np.abs(x_alignment))[-2:]
        alignment_indices = np.argsort(np.abs(x_alignment))[:2]
        grasp_poses = grasp_poses[alignment_indices]

        return grasp_poses

    def best_grasp(self, obj_wcT: np.ndarray, gripper_wcT: np.ndarray, flip=False):

        grasp_poses = self.all_top_down_proposals(obj_wcT)

        zaxis = obj_wcT[:3, 2]
        if zaxis[-1] > 0.7 or zaxis[-1] < -0.7:
            # bottle on its side: pick the grasp pointing most directly down
            zaxes = grasp_poses[:, :3, 2]  # (N, 3)
            index = np.argmin(zaxes[:, -1])  # most downwards
            grasp_pose = grasp_poses[index]
            if flip:
                grasp_pose[:3, :2] *= -1
        else:
            # bottle upright or upside down: pick the grasp requiring the smallest gripper pose change
            grasp_poses_flip = grasp_poses.copy()  # rotating the grasp by 180° is usually also valid
            grasp_poses_flip[:, :3, :3] = np.einsum("brc,cd->brd", 
                                                    grasp_poses_flip[:, :3, :3], 
                                                    Rotation.from_rotvec([0, 0, np.pi]).as_matrix())
            dT0 = np.einsum("rc,bcd->brd", np.linalg.inv(gripper_wcT), grasp_poses)
            dT1 = np.einsum("rc,bcd->brd", np.linalg.inv(gripper_wcT), grasp_poses_flip)
            du0 = np.linalg.norm(Rotation.from_matrix(dT0[:, :3, :3]).as_rotvec(), axis=-1)
            du1 = np.linalg.norm(Rotation.from_matrix(dT1[:, :3, :3]).as_rotvec(), axis=-1)
            du = np.concatenate([du0, du1], axis=0)  # (2*N,)
            
            dt0 = np.linalg.norm(dT0[:, :3, 3], axis=-1)
            dt1 = np.linalg.norm(dT1[:, :3, 3], axis=-1)
            dt = np.concatenate([dt0, dt1], axis=0)

            cost = 5 * du + dt
            index = np.argmin(cost)
            grasp_pose_index, need_flip = (index % len(dT0)), int(index > len(dT0))
            grasp_pose = (grasp_poses, grasp_poses_flip)[need_flip][grasp_pose_index]

        return grasp_pose


class Clamp(ObjBase):
    USD_PATH = "assets/mesh/Axis_Aligned/051_large_clamp.usd"
    OBJ_PATH = "assets/mesh/Axis_Aligned/051_large_clamp.obj"
    SCALE = np.ones(3) * 0.7

    def all_top_down_proposals(self, obj_wcT: np.ndarray):
        zaxis = obj_wcT[:3, 2]
        grasp_poses = []
        for theta in np.linspace(0, np.pi*2, 36, endpoint=False):
            # if zaxis[-1] > 0.7 or zaxis[-1] < -0.7:  # can on its side: grasp horizontally
            grasp_poses.append(obj_wcT @ self.top_down_center_proposal(theta))
            # elif zaxis[-1] > -0.7:  # can upright
            #     grasp_poses.append(obj_wcT @ self.horizontal_proposal(theta))
            # else:  # can on its side, upside down
            #     grasp_poses.append(obj_wcT @ self.bottom_up_center_proposal(theta))
        grasp_poses = np.stack(grasp_poses, axis=0)

        # Filter grasp poses based on x-axis alignment with object x-axis
        obj_xaxis = obj_wcT[:3, 0]  # Get x-axis of object pose
        x_alignment = np.abs(np.einsum('bc,c->b', grasp_poses[:, :3, 0], obj_xaxis))  # Dot product with grasp x-axis
        # alignment_indices = np.argsort(np.abs(x_alignment))[-2:]
        alignment_indices = np.argsort(np.abs(x_alignment))[:2]
        grasp_poses = grasp_poses[alignment_indices]

        return grasp_poses

    def best_grasp(self, obj_wcT: np.ndarray, gripper_wcT: np.ndarray, flip=False):

        grasp_poses = self.all_top_down_proposals(obj_wcT)

        zaxis = obj_wcT[:3, 2]
        if zaxis[-1] > 0.7 or zaxis[-1] < -0.7:
            # bottle on its side: pick the grasp pointing most directly down
            zaxes = grasp_poses[:, :3, 2]  # (N, 3)
            index = np.argmin(zaxes[:, -1])  # most downwards
            grasp_pose = grasp_poses[index]
            if flip:
                grasp_pose[:3, :2] *= -1
        else:
            # bottle upright or upside down: pick the grasp requiring the smallest gripper pose change
            grasp_poses_flip = grasp_poses.copy()  # rotating the grasp by 180° is usually also valid
            grasp_poses_flip[:, :3, :3] = np.einsum("brc,cd->brd", 
                                                    grasp_poses_flip[:, :3, :3], 
                                                    Rotation.from_rotvec([0, 0, np.pi]).as_matrix())
            dT0 = np.einsum("rc,bcd->brd", np.linalg.inv(gripper_wcT), grasp_poses)
            dT1 = np.einsum("rc,bcd->brd", np.linalg.inv(gripper_wcT), grasp_poses_flip)
            du0 = np.linalg.norm(Rotation.from_matrix(dT0[:, :3, :3]).as_rotvec(), axis=-1)
            du1 = np.linalg.norm(Rotation.from_matrix(dT1[:, :3, :3]).as_rotvec(), axis=-1)
            du = np.concatenate([du0, du1], axis=0)  # (2*N,)
            
            dt0 = np.linalg.norm(dT0[:, :3, 3], axis=-1)
            dt1 = np.linalg.norm(dT1[:, :3, 3], axis=-1)
            dt = np.concatenate([dt0, dt1], axis=0)

            cost = 5 * du + dt
            index = np.argmin(cost)
            grasp_pose_index, need_flip = (index % len(dT0)), int(index > len(dT0))
            grasp_pose = (grasp_poses, grasp_poses_flip)[need_flip][grasp_pose_index]

        return grasp_pose
