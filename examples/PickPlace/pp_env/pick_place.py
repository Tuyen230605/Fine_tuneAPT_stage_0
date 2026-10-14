import numpy as np
from enum import Flag, auto
from scipy.spatial.transform import Rotation

from omni.isaac.core.prims import RigidPrim

from assets.ur5gp85 import UR5GP85
from .objs.base import GraspProposal


class PickPlace(object):
    """Analytic pick-and-place state machine driven by the scene/grasp proposals.

    Used to bring the UR5 to a pregrasp pose before handing control to the
    learned policy; also exposes `pose_close_to` for the policy-driven phase
    to decide when a commanded waypoint has been reached.
    """

    class State(Flag):
        init          = auto()  # move to `initial_pose` (or robot home if None)
        to_pregrasp   = auto()  # approach a pose offset above the grasp
        to_grasp      = auto()  # descend to the final grasp pose
        close_gripper = auto()  # wait for the gripper to close
        to_preplace   = auto()  # lift to a pose offset above the place pose
        to_place      = auto()  # descend to the final place pose
        open_gripper  = auto()  # wait for the gripper to open
        done          = auto()  # hold the last commanded pose

    def __init__(
        self,
        ur5: UR5GP85,
        obj: RigidPrim,
        grasp_proposal: GraspProposal,
        initial_pose: np.ndarray = None,
        place_pose: np.ndarray = None,
        grasp_condition: dict = None,
    ):
        """
        Args:
            ur5:             UR5 + Robotiq wrapper that this controller commands.
            obj:             Rigid prim being grasped; its world pose feeds `grasp_proposal`.
            grasp_proposal:  Per-object grasp generator (subclass of `GraspProposal`).
            initial_pose:    Optional 4x4 starting EE pose; defaults to the robot's tip home pose.
            place_pose:      Optional 4x4 target place pose; if None, skip the place phase.
            grasp_condition: Extra kwargs forwarded to `grasp_proposal.best_grasp(...)`
                             (e.g. {"flip": True} to optionally flip the gripper).
        """
        self.ur5 = ur5
        self.obj = obj
        self.grasp_proposal = grasp_proposal
        self.initial_pose = initial_pose
        self.place_pose = place_pose
        self.grasp_conditon = grasp_condition

        self.state = self.State.init
        self._grasp_pose = None
        self._last_pose = None
        self._waited_gripper_steps = 0

        # Convergence thresholds used by `pose_close_to`.
        self.du_thresh = 5 / 180.0 * np.pi   # ~5 degrees
        self.dt_thresh = 0.01                # 1 cm
    
    def reset(self):
        self.state = self.State.init
        self._grasp_pose = None
        self._last_pose = None
        self._waited_gripper_steps = 0
    
    def pose_close_to(self, T0: np.ndarray, T1: np.ndarray):
        """True iff `T1` is within `du_thresh`/`dt_thresh` of `T0` in SE(3)."""
        dT = np.linalg.inv(T0) @ T1
        du = np.linalg.norm(Rotation.from_matrix(dT[:3, :3]).as_rotvec())
        dt = np.linalg.norm(dT[:3, 3])
        print("[INFO] du = {:.2f}°, dt = {:.2f}mm"
              .format(du / np.pi * 180, dt * 1000))
        print("[INFO] du = {:.3f}, dt = {:.3f}"
              .format(du, dt))
        print("[INFO] du_thresh = {:.3f}, dt_thresh = {:.3f}"
              .format(self.du_thresh, self.dt_thresh))
        return (du < self.du_thresh) and (dt < self.dt_thresh)
    
    def get_grasp_pose(self, pregrasp_dist: float):
        # ee_pose = self.ur5.get_tip_pose()
        obj_pos, obj_orn = self.obj.get_world_pose()
        obj_pose = np.eye(4)
        obj_pose[:3, 3] = obj_pos
        obj_pose[:3, :3] = self.obj._backend_utils.quats_to_rot_matrices(obj_orn)
        ee_pose = self.ur5.get_tip_pose()
        grasp_pose = self.grasp_proposal.best_grasp(obj_pose, ee_pose, **self.grasp_conditon)
        grasp_pose[:3, 3] -= grasp_pose[:3, 2] * pregrasp_dist

        return grasp_pose
    
    def get_preplace_pose(self, preplace_dist: float):
        place_pose = self.place_pose.copy()
        place_pose[:3, 3] -= place_pose[:3, 2] * preplace_dist
        return place_pose

    def grasp(self, grasp_pose: np.array, gripper_steps=50):
        if self.state == self.State.to_pregrasp:
            # get pregrasp poses
            pregrasp_dist = 0.25
            pregrasp_pose = grasp_pose.copy()
            pregrasp_pose[:3, 3] -= pregrasp_pose[:3, 2] * pregrasp_dist
            robot_action = self.ur5.move_tip_action(pregrasp_pose)
            gripper_action = self.ur5.open_gripper_action()
            action = [robot_action, gripper_action]

            if self.pose_close_to(pregrasp_pose, self.ur5.get_tip_pose()):
                self.state = self.State.to_grasp
            return action, self.state    

        if self.state == self.State.to_grasp:
            tip_pos = grasp_pose[:3, 3] + grasp_pose[:3, 2] * 0.02
            if tip_pos[-1] < 0:
                grasp_pose[2, 3] += -tip_pos[-1]
            
            robot_action = self.ur5.move_tip_action(grasp_pose)
            gripper_action = self.ur5.open_gripper_action()
            action = [robot_action, gripper_action]

            if self.pose_close_to(grasp_pose, self.ur5.get_tip_pose()):
                self.state = self.State.close_gripper
            return action, self.state    

        if self.state == self.State.close_gripper:
            robot_action = self.ur5.move_tip_action(grasp_pose)
            gripper_action = self.ur5.close_gripper_action()
            action = [robot_action, gripper_action]

            self._waited_gripper_steps += 1

            if self._waited_gripper_steps > gripper_steps:
                self._waited_gripper_steps = 0
                self.state = self.State.to_preplace
            return action, self.state        


    def get_action(self, gripper_steps=50):
        if self.state == self.State.init:
            if self.initial_pose is None:
                desired_pose = self.ur5.ee_home_pose()
            else:
                desired_pose = self.initial_pose
            robot_action = self.ur5.move_tip_action(desired_pose, method="rmp")
            gripper_action = self.ur5.open_gripper_action()
            action = [robot_action, gripper_action]

            if self.pose_close_to(desired_pose, self.ur5.get_tip_pose()):
                self.state = self.State.to_pregrasp
            return action, self.state

        if self.state == self.State.to_pregrasp:
            if self._grasp_pose is None:
                self._grasp_pose = self.get_grasp_pose(pregrasp_dist=0.25)
            desired_pose = self._grasp_pose
            robot_action = self.ur5.move_tip_action(desired_pose,
                                                    # method="rmp"
                                                   )
            gripper_action = self.ur5.open_gripper_action()
            action = [robot_action, gripper_action]

            if self.pose_close_to(desired_pose, self.ur5.get_tip_pose()):
                self.state = self.State.to_grasp
                self._grasp_pose = None
            return action, self.state

        if self.state == self.State.to_grasp:
            if self._grasp_pose is None:
                self._grasp_pose = self.get_grasp_pose(pregrasp_dist=-0.05)
 
            tip_pos = self._grasp_pose[:3, 3] + self._grasp_pose[:3, 2] * 0.02
            if tip_pos[-1] < 0:
                self._grasp_pose[2, 3] += -tip_pos[-1]
        
            desired_pose = self._grasp_pose
            robot_action = self.ur5.move_tip_action(desired_pose,
                                                    # method="rmp"
                                                   )
            gripper_action = self.ur5.open_gripper_action()
            action = [robot_action, gripper_action]

            if self.pose_close_to(desired_pose, self.ur5.get_tip_pose()):
                self.state = self.State.close_gripper
            return action, self.state

        if self.state == self.State.close_gripper:
            desired_pose = self._grasp_pose
            robot_action = self.ur5.move_tip_action(desired_pose,
                                                    # method="rmp"
                                                   )
            gripper_action = self.ur5.close_gripper_action()
            action = [robot_action, gripper_action]
            self._waited_gripper_steps += 1

            if self._waited_gripper_steps > gripper_steps:
                self._waited_gripper_steps = 0
                if self.place_pose is not None:
                    self.state = self.State.to_preplace
                else:
                    self.state = self.State.open_gripper
            return action, self.state
        
        if self.state == self.State.to_preplace:
            desired_pose = self.get_preplace_pose(preplace_dist=0.2)
            robot_action = self.ur5.move_tip_action(desired_pose,
                                                    # method="rmp"
                                                    )
            gripper_action = self.ur5.close_gripper_action()
            action = [robot_action, gripper_action]

            if self.pose_close_to(desired_pose, self.ur5.get_tip_pose()):
                self.state = self.State.to_place
            return action, self.state

        if self.state == self.State.to_place:
            desired_pose = self.place_pose
            robot_action = self.ur5.move_tip_action(desired_pose,
                                                    # method="rmp",
                                                   )
            gripper_action = self.ur5.close_gripper_action()
            action = [robot_action, gripper_action]

            if self.pose_close_to(desired_pose, self.ur5.get_tip_pose()):
                self.state = self.State.open_gripper
            return action, self.state

        if self.state == self.State.open_gripper:
            desired_pose = self.place_pose
            robot_action = self.ur5.move_tip_action(desired_pose,
                                                    # method="rmp"
                                                   )
            gripper_action = self.ur5.open_gripper_action()
            action = [robot_action, gripper_action]
            self._waited_gripper_steps += 1

            if self._waited_gripper_steps > gripper_steps:
                self._waited_gripper_steps = 0
                self._last_pose = desired_pose.copy()
                self.state = self.State.done
            return action, self.state

        if self.state == self.State.done:
            # hold at current pose
            if self._last_pose is None:
                self._last_pose = self.ur5.get_tip_pose()
            robot_action = self.ur5.move_tip_action(self._last_pose,
                                                    # method="rmp"
                                                   )
            action = [robot_action]
            return action, self.state

