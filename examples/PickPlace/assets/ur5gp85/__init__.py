
from omni.isaac.core.objects import cuboid
from omni.isaac.core.prims import RigidPrim
import omni.isaac.core.utils.prims as prims_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.core.utils.string as strings_utils
from omni.isaac.core.utils.semantics import add_update_semantics

# motion, rmp
from omni.isaac.motion_generation.lula import RmpFlow
from omni.isaac.motion_generation import ArticulationMotionPolicy, MotionPolicyController

# ik solver
import lula
from omni.isaac.core.robots import Robot
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver
from omni.isaac.motion_generation.lula.kinematics import LulaKinematicsSolver
from omni.isaac.motion_generation.interface_config_loader import (
    load_supported_motion_policy_config,
    load_supported_lula_kinematics_solver_config
)
from pxr import Usd

import os
import time
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Optional, Tuple, List


_here = os.path.dirname(os.path.abspath(__file__))


def pbvs_straight(tcT: np.ndarray):
    """PBVS2: goes straight and shortest path"""
    u = Rotation.from_matrix(tcT[:3, :3]).as_rotvec()

    v = -tcT[:3, :3].T @ tcT[:3, 3]
    w = -u
    vel = np.concatenate([v, w])

    return vel


class KinematicsSolver(ArticulationKinematicsSolver):
    def __init__(self, robot_articulation: Articulation):
        kinematics_config = load_supported_lula_kinematics_solver_config("UR5")
        self._kinematics = LulaKinematicsSolver(**kinematics_config)

        end_effector_frame_name = "tool0"
        super().__init__(robot_articulation, self._kinematics, end_effector_frame_name)
        self.kinematics = lula.load_robot(kinematics_config["robot_description_path"], 
                                          kinematics_config["urdf_path"]).kinematics()
    
    def tr2jac(self, T: np.ndarray):
        Z = np.zeros((3, 3), dtype=T.dtype)
        R = T[:3, :3]
        return np.block([[R, Z], [Z, R]])
    
    def jacobian(self) -> np.ndarray:
        q = self._robot_articulation.get_joint_positions()[:6]
        T = self.kinematics.pose(q, self._ee_frame, "base_link").matrix()  # (4, 4)
        J = self.kinematics.jacobian(q, self._ee_frame)  # (6, n_dof)
        return self.tr2jac(T.T) @ J
    
    def compute_action(
        self, 
        target_position: np.ndarray, 
        target_orientation: np.ndarray
    ) -> Tuple[ArticulationAction, bool]:
        robot_base_translation, robot_base_orientation = self._robot_articulation.get_world_pose()
        self._kinematics.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        action, success = self._kinematics.compute_inverse_kinematics(
            self._ee_frame, target_position, target_orientation, 
            warm_start=self._robot_articulation.get_joint_positions()[:6]
        )
        if isinstance(action, np.ndarray):
            action = ArticulationAction(action, joint_indices=np.arange(6))
        return action, success


class RMPFlowController(MotionPolicyController):
    def __init__(self, name: str, robot_articulation: Articulation, physics_dt: float = 1.0 / 60):

        self.rmp_config = load_supported_motion_policy_config("UR5", "RMPflow")
        self.rmp_config.update({"end_effector_frame_name": "tool0"})
        self.rmpflow = RmpFlow(**self.rmp_config)
        self.articulation_rmp = ArticulationMotionPolicy(robot_articulation, self.rmpflow, physics_dt)
        super().__init__(name, self.articulation_rmp)

        self._robot_articulation = robot_articulation
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )

    def reset(self):
        MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
    
    def compute_action(self, target_position: np.ndarray, target_orientation: np.ndarray, updated_obstacles = None):
        self.rmpflow.set_end_effector_target(target_position, target_orientation)
        self.rmpflow.update_world(updated_obstacles)

        robot_base_translation, robot_base_orientation = self._robot_articulation.get_world_pose()
        self.rmpflow.set_robot_base_pose(robot_base_translation, robot_base_orientation)
        action = self.articulation_rmp.get_next_articulation_action()
        return action


class GripperController(object):
    def __init__(self, joint_prim_path: str, damping: Optional[float] = None):
        self.joint_prim = prims_utils.get_prim_at_path(joint_prim_path)
        if damping is not None:
            self.joint_prim.GetAttribute("drive:angular:physics:damping").Set(damping)
    
    def close(self, speed: float = 130):
        self.joint_prim.GetAttribute("drive:angular:physics:targetVelocity").Set(abs(speed))
    
    def open(self, speed: float = 130):
        self.joint_prim.GetAttribute("drive:angular:physics:targetVelocity").Set(-abs(speed))


class UR5GP85(object):
    USD_PATH = f"{_here}/ur5gp85v4p1.usd"
    # USD_PATH = "robot/ur5gp85v2.usd"
    # USD_PATH = "robot/ur5e_wi_gp.usd"

    def __init__(self, name: str, prim_path: str, physics_dt: float = 1.0 / 60, **robot_init_kwargs):
        self.prim_path = prim_path
        matches = prims_utils.find_matching_prim_paths(prim_path)
        if len(matches) == 0:
            stage_utils.add_reference_to_stage(
                usd_path=self.USD_PATH,
                prim_path=prim_path
            )
        
        add_update_semantics(
            prims_utils.get_prim_at_path(prim_path + "/ur5gp85/ur5"),
            semantic_label="ur5"
        )
        add_update_semantics(
            prims_utils.get_prim_at_path(prim_path + "/ur5gp85/Robotiq_2F_85_edit"),
            semantic_label="gripper"
        )

        self._robot = Robot(
            prim_path=prim_path + "/ur5gp85/ur5/base_link",
            name=name,
            **robot_init_kwargs
        )

        # overwrite default angle limit
        wrist23_joint_prim = prims_utils.get_prim_at_path(prim_path + "/ur5gp85/ur5/wrist_2_link/wrist_3_joint")
        wrist23_joint_prim.GetAttribute("physics:lowerLimit").Set(-3600.0)
        wrist23_joint_prim.GetAttribute("physics:upperLimit").Set(3600.0)
        # overwrite default gripper joint force
        prims_utils.get_prim_at_path(prim_path + "/ur5gp85/Robotiq_2F_85_edit/Robotiq_2F_85/finger_joint"
                                     ).GetAttribute("drive:angular:physics:maxForce").Set(0.4)

        self.ee_link = RigidPrim(
            prim_path=prim_path+"/ur5gp85/ur5/tool0",
            name="ee_link"
        )

        self.physics_dt = physics_dt
        self.ik_solver = KinematicsSolver(self._robot)
        self.rmp_controller = RMPFlowController("arm_rmp", self._robot, physics_dt)
        self.rmp_controller.reset()

        self.gripper = GripperController(prim_path + "/ur5gp85/Robotiq_2F_85_edit/Robotiq_2F_85/finger_joint")

        ee2tip = np.eye(4)
        # ee2tip[:3, :3] = Rotation.from_rotvec([0, 0, -np.pi/2]).as_matrix()
        ee2tip[:3, 3] = np.array([0, 0, 0.15])
        self.ee2tip = ee2tip
        self.tip2ee = np.linalg.inv(self.ee2tip)
    
    def ee_home_pose(self):
        base_p, base_q = self._robot.get_world_pose()
        base_pose = self.pq2mat(base_p, base_q)
        tcp_pose = self.pq2mat(np.array([0.5, 0, 0.5]), 
                               np.array([0, 0, 1.0, 0]))
        return base_pose @ tcp_pose
    
    def tip_home_pose(self):
        return self.ee_home_pose() @ self.ee2tip
    
    def set_to_default_state(self):
        half_pi = np.pi / 2
        pos = [0, -half_pi, half_pi, -half_pi, -half_pi,  half_pi, 0]
        self._robot.set_joint_positions(pos, np.arange(len(pos)))
        # self._robot.set_joint_velocities([-130.0/180 * np.pi], [6])
        prims_utils.get_prim_at_path(self.prim_path + "/ur5gp85/Robotiq_2F_85_edit/Robotiq_2F_85/finger_joint"
                                     ).GetAttribute("state:angular:physics:position").Set(0.0)
        prims_utils.get_prim_at_path(self.prim_path + "/ur5gp85/Robotiq_2F_85_edit/Robotiq_2F_85/right_outer_knuckle_joint"
                                     ).GetAttribute("state:angular:physics:position").Set(0.0)
    
    def mat2quat(self, m: np.ndarray):
        return self._robot._backend_utils.rot_matrices_to_quats(m)

    def quat2mat(self, q: np.ndarray):
        return self._robot._backend_utils.quats_to_rot_matrices(q)
    
    def pq2mat(self, pos: np.ndarray, quat: np.ndarray):
        T = np.eye(4)
        T[:3, :3] = self.quat2mat(quat)
        T[:3, 3] = pos
        return T

    def mat2pq(self, T: np.ndarray):
        pos = T[:3, 3]
        quat = self.mat2quat(T[:3, :3])
        return pos, quat

    def get_ee_pose(self, return_mat: bool = True):
        pos, quat = self.ee_link.get_world_pose()
        if return_mat:
            pose = self.pq2mat(pos, quat)
        else:
            pose = (pos, quat)
        return pose
    
    def get_tip_pose(self, return_mat: bool = True):
        ee_pose = self.get_ee_pose(return_mat=True)
        tip_pose = ee_pose @ self.ee2tip
        if not return_mat:
            tip_pose = self.mat2pq(tip_pose)
        return tip_pose
    
    def ee_vel_control(self, ee_vel_ee: np.ndarray):
        """
        - ee_vel_ee: [v, w], end-effector velocity in end-effector frame
        """
        q = self._robot.get_joint_positions()[:6]
        J = self.ik_solver.jacobian()
        joint_vel = np.linalg.pinv(J) @ ee_vel_ee

        return ArticulationAction(
            joint_positions=q + joint_vel * self.physics_dt,
            joint_velocities=joint_vel, 
            joint_indices=np.arange(6),
        )

    def _move_ee_action_rmp(self, pos, quat, updated_obstacles = None):
        action = self.rmp_controller.compute_action(pos, quat, updated_obstacles)
        return action
    
    def _move_ee_action_ik(self, pos, quat):
        action, success = self.ik_solver.compute_action(pos, quat)
        if not success:
            print("[WARN] Failed to solve with ik")
            print("[INFO] Fallback to RMP control")
            action = self._move_ee_action_rmp(pos, quat)
        return action
    
    def _move_ee_action_pbvs(self, pos, quat, vfixed = 0.3):
        action, success = self.ik_solver.compute_action(pos, quat)
        q5 = self._robot.get_joint_positions()[5]
        q5_desired = action.joint_positions[5]
        candidates = np.array([q5_desired, 
                               q5_desired - 2*np.pi, q5_desired + 2*np.pi,
                               q5_desired - 4*np.pi, q5_desired + 4*np.pi])
        min_index = np.argmin(np.abs(candidates - q5))
        action.joint_positions = action.joint_positions.copy()
        action.joint_positions[5] = candidates[min_index]

        if success:
            wdTe = self.pq2mat(pos, quat)
            wcTe = self.get_ee_pose(return_mat=True)
            dcT = np.linalg.inv(wdTe) @ wcTe
            ee_vel_ee = pbvs_straight(dcT)

            # When the PBVS linear velocity exceeds fixed_vnorm_thresh, rescale ee_vel_ee so that
            # the linear component of ee_vel_ee equals vfixed.
            fixed_vnorm_thresh = 0.02
            vnorm = np.linalg.norm(ee_vel_ee[:3])
            if vnorm >= fixed_vnorm_thresh:
                scale = vfixed / vnorm
                ee_vel_ee *= scale

            vel_action = self.ee_vel_control(ee_vel_ee)
            if vnorm < fixed_vnorm_thresh:
                vel_action.joint_positions = action.joint_positions
                # vel_action.joint_velocities = np.zeros(6)
            # if vnorm < fixed_vnorm_thresh:
            #     vel_action.joint_positions = action.joint_positions
            #     vel_action.joint_velocities = np.zeros(6)
            # if vnorm < fixed_vnorm_thresh:
            #     vel_action = self._move_ee_action_rmp(pos, quat)
            return vel_action

        else:
            print("[WARN] Failed to solve with pbvs")
            print("[INFO] Fallback to RMP control")
            return self._move_ee_action_rmp(pos, quat)
    
    def move_ee_action(self, pose, updated_obstacles: list = None, method: str = "pbvs"):
        if isinstance(pose, (list, tuple)):
            p, q = pose
        else:
            p, q = self.mat2pq(pose)
        
        method = method.strip().lower()
        if "rmp" in method:
            t0 = time.perf_counter()
            action = self._move_ee_action_rmp(p, q, updated_obstacles)
            t1 = time.perf_counter()
            print("[INFO] Rmp flow planner dt: {:.3f}ms".format((t1 - t0) * 1000))
        elif "ik" in method:
            action = self._move_ee_action_ik(p, q)
        else:
            action = self._move_ee_action_pbvs(p, q)
        return action
    
    def move_tip_action(self, pose, updated_obstacles: list = None, method: str = "pbvs"):
        if isinstance(pose, (list, tuple)):
            pose = self.pq2mat(*pose)
        
        pose = pose @ self.tip2ee
        return self.move_ee_action(pose, updated_obstacles, method)
    
    def open_gripper_action(self, speed: float = 130.0/180*np.pi):
        return ArticulationAction(
            joint_positions=[45.0 / 180 * np.pi],
            joint_velocities=[-abs(speed)],
            joint_indices=[6]
        )
    
    def close_gripper_action(self, speed: float = 130.0/180*np.pi):
        return ArticulationAction(
            joint_positions=[0.],
            joint_velocities=[abs(speed)],
            joint_indices=[6]
        )
    
    def get_gripper_norm_width(self):
        qpos = self._robot.get_joint_positions()[6]
        width = 1 - qpos / (45.0/180*np.pi)
        return width
    
    def apply_actions(self, actions: List[ArticulationAction]):
        if isinstance(actions, ArticulationAction):
            return self._robot.apply_action(actions)
        for a in actions:
            self._robot.apply_action(a)
    
    def default_eih_cam_extr(self):
        return dict(
            prim_path=self.prim_path + "/ur5gp85/Robotiq_2F_85_edit/Robotiq_2F_85/base_link/rgbd_cam",
            translation=np.array([0.036, 0, 0.05]),
            # orientation=np.array([0.5, 0.5, -0.5, -0.5])
        )

