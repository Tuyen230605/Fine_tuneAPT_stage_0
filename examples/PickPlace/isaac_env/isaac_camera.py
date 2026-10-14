import numpy as np
from typing import Tuple
from copy import deepcopy

import omni.usd
import omni.timeline
import omni.replicator.core as rep
from pxr import Sdf
from omni.isaac.kit import carb
from omni.isaac.core import SimulationContext
from omni.isaac.sensor import Camera as _IsaacCamera
from .non_isaac.perception import PinholeCamera, OpenglCamera, Frame


class _IsaacCameraWrapper(_IsaacCamera):
    usd2opencv = np.array([[1.0,  0.0,  0.0],
                           [0.0, -1.0,  0.0],
                           [0.0,  0.0, -1.0]])
    opencv2usd = np.linalg.inv(usd2opencv)

    def __init__(
        self, 
        prim_path: str, 
        name: str = "camera", 
        frequency: int | None = None, 
        dt: str | None = None, 
        resolution: Tuple[int, int] | None = None, 
        position: np.ndarray | None = None, 
        orientation: np.ndarray | None = None, 
        translation: np.ndarray | None = None, 
        render_product_path: str = None
    ):
        super().__init__(prim_path, name, frequency, dt, resolution, 
                         position, orientation, translation, render_product_path)
        self.timed_wcT = None

    def get_timed_wcT(self):
        timeline = omni.timeline.get_timeline_interface()
        timecode = timeline.get_current_time() * timeline.get_time_codes_per_seconds()
        wcT = omni.usd.get_world_transform_matrix(self._camera_prim, timecode)
        wcT = np.asarray(wcT).transpose().copy()
        wcT[:3, :3] = wcT[:3, :3] @ self.usd2opencv
        return wcT, timecode
    
    def _data_acquisition_callback(self, event: carb.events.IEvent):
        ret = super()._data_acquisition_callback(event)
        wcT, timecode = self.get_timed_wcT()
        self.timed_wcT = (wcT, timecode)
        return ret
    
    def add_cross_corr_to_frame(self, other_cam_name: str):
        self._camera_prim.CreateAttribute("crossCameraReferenceName", Sdf.ValueTypeNames.String)
        self._camera_prim.GetAttribute("crossCameraReferenceName").Set(other_cam_name)
        anno = rep.annotators.get("cross_correspondence")
        anno.attach([self._render_product_path])
        self._custom_annotators["cross_correspondence"] = anno


class IsaacCamera(object):
    def __init__(
        self, 
        prim_path: str, 
        name: str, 
        opengl_cam: OpenglCamera,
        enable_depth: bool = False,
        enable_segmentation: bool = False,
        enable_rep_pcd: bool = False, 
        **init_isaac_cam_kwargs
    ):
        self.rotmat2quat = SimulationContext._instance._backend_utils.rot_matrices_to_quats
        self.quat2rotmat = SimulationContext._instance._backend_utils.quats_to_rot_matrices
        self.usd2opencv = np.array([[1.0,  0.0,  0.0],
                                    [0.0, -1.0,  0.0],
                                    [0.0,  0.0, -1.0]])
        self.opencv2usd = np.linalg.inv(self.usd2opencv)
        self.isaac_cam = _IsaacCameraWrapper(prim_path=prim_path, name=name, 
                                             **init_isaac_cam_kwargs)
        self.opengl_cam = opengl_cam
        self.enable_depth = enable_depth
        self.enable_segmentation = enable_segmentation
        self.enable_rep_pcd = enable_rep_pcd
    
    def initialize(self):
        """Please ensure this method is called after SimulationContext (world)'s reset 
        and before SimulationApp's update
        """
        self.isaac_cam.initialize()
        if self.enable_depth:
            self.isaac_cam.add_distance_to_image_plane_to_frame()
        if self.enable_segmentation:
            self.isaac_cam.add_semantic_segmentation_to_frame()
        if self.enable_rep_pcd:
            self.isaac_cam.add_pointcloud_to_frame(include_unlabelled=True)

        # setup isaac camera
        self.setup_intrinsic(self.opengl_cam)

    def setup_intrinsic(self, opengl_cam: OpenglCamera):
        # setup isaac camera
        intrinsic = opengl_cam.intrinsic
        if intrinsic.fx != intrinsic.fy:
            print("[WARN] Unequal fx and fy, modified here since IsaacSim only supports "
                  "pinhole model with equal focal length")
            intrinsic: PinholeCamera = deepcopy(intrinsic)
            intrinsic.fx = intrinsic.fy = (intrinsic.fx + intrinsic.fy) / 2.0
        
        self.opengl_cam = OpenglCamera(intrinsic, opengl_cam.near, opengl_cam.far)
        camera_settings = self.opengl_cam.to_isaac()
        self.isaac_cam.set_resolution(camera_settings["resolution"])
        self.isaac_cam.set_focal_length(
            (camera_settings["focal_length_x"] + camera_settings["focal_length_y"]) / 2 / 10.0)
        self.isaac_cam.set_horizontal_aperture(camera_settings["horizontal_aperture"] / 10.0)
        self.isaac_cam.set_vertical_aperture(camera_settings["vertical_aperture"] / 10.0)
        self.isaac_cam.set_clipping_range(*camera_settings["clipping_range"])

    def set_wcT(self, wcT: np.ndarray):
        self.isaac_cam.set_world_pose(
            position=wcT[:3, 3],
            orientation=self.rotmat2quat(wcT[:3, :3] @ self.opencv2usd),
            camera_axes="usd"
        )
    
    def get_timed_wcT(self):
        return self.isaac_cam.get_timed_wcT()

    def get_wcT(self):
        return self.isaac_cam.get_timed_wcT()[0]
    
    def render(self, clone: bool = False, fill_inf_depth: bool = False):
        raw_frame = self.isaac_cam.get_current_frame(clone=clone)
        rgba: np.ndarray = raw_frame["rgba"]
        if rgba.size == 0:
            return None

        pcd: np.ndarray = raw_frame.get("pointcloud", None)
        if pcd is not None:
            pcd = pcd["data"]
            desired_H = self.opengl_cam.intrinsic.height
            desired_W = self.opengl_cam.intrinsic.width
            if pcd.shape[0] < desired_H * desired_W:
                print("[INFO] too less point, pcd shape: {}, desired: {}"
                      .format(pcd.shape, (desired_H*desired_W, 3)))
                pcd = None
            else:
                pcd = np.reshape(pcd, (desired_H, desired_W, 3))
        
        depth: np.ndarray = raw_frame.get("distance_to_image_plane", None)
        seg: dict = raw_frame.get("semantic_segmentation", None)

        if fill_inf_depth and depth is not None:
            mask = np.isnan(depth) | np.isinf(depth)
            if mask.any():
                depth[mask] = self.opengl_cam.far
            depth = np.clip(depth, self.opengl_cam.near, self.opengl_cam.far)

        if self.isaac_cam.timed_wcT is None:
            wcT, timecode = self.get_timed_wcT()
        else:
            wcT, timecode = self.isaac_cam.timed_wcT

        return Frame(
            camera=self.opengl_cam.intrinsic, 
            color=rgba,
            depth=depth,
            seg=seg,
            wcT=wcT.copy(),
            pc_world=pcd,
            timestep=timecode
        )


def mask_from_semantics(data: dict, valid_label_subnames = []):
    seg_data: np.ndarray = data["data"]
    seg_info: dict = data["info"]
    mask = np.zeros(seg_data.shape, dtype=bool)

    for label, name in seg_info["idToLabels"].items():
        for query_name in valid_label_subnames:
            if query_name in name["class"]:
                mask[seg_data == int(label)] = True
                break
    return mask
