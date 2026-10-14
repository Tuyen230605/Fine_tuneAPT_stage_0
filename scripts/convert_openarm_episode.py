"""Convert OpenArm joint/RGB episodes to APT's Stage-0 HDF5 format.

The source has joint positions, not end-effector poses. This converter uses
the robot URDF and either calibrated camera poses or explicitly approximate
fixed-camera geometry. Run from the repository root with
``configs/openarm_fixed_place.template.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


def _numbers(text: str | None, count: int, default: list[float]) -> np.ndarray:
    values = default if text is None else [float(x) for x in text.split()]
    if len(values) != count or not np.isfinite(values).all():
        raise ValueError(f"Expected {count} finite numbers, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * cross


def _pose(value: object, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-3):
        raise ValueError(f"{name} rotation determinant must be 1")
    return pose


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class URDFFK:
    def __init__(self, path: Path):
        root = ET.parse(path).getroot()
        self.by_child: dict[str, Joint] = {}
        for item in root.findall("joint"):
            name = item.attrib["name"]
            kind = item.attrib["type"]
            parent = item.find("parent").attrib["link"]
            child = item.find("child").attrib["link"]
            origin_tag = item.find("origin")
            xyz = _numbers(origin_tag.get("xyz") if origin_tag is not None else None, 3, [0, 0, 0])
            rpy = _numbers(origin_tag.get("rpy") if origin_tag is not None else None, 3, [0, 0, 0])
            origin = np.eye(4)
            origin[:3, :3] = _rpy_matrix(rpy)
            origin[:3, 3] = xyz
            axis_tag = item.find("axis")
            axis = _numbers(axis_tag.get("xyz") if axis_tag is not None else None, 3, [1, 0, 0])
            if kind in ("revolute", "continuous", "prismatic"):
                norm = np.linalg.norm(axis)
                if norm < 1e-9:
                    raise ValueError(f"URDF joint {name} has a zero axis")
                axis = axis / norm
            if child in self.by_child:
                raise ValueError(f"URDF has more than one parent for link {child}")
            self.by_child[child] = Joint(name, kind, parent, child, origin, axis)

    def chain(self, base_link: str, ee_link: str) -> list[Joint]:
        chain: list[Joint] = []
        current = ee_link
        while current != base_link:
            if current not in self.by_child:
                raise ValueError(f"No URDF chain from {base_link} to {ee_link}; stopped at {current}")
            joint = self.by_child[current]
            chain.append(joint)
            current = joint.parent
            if len(chain) > len(self.by_child):
                raise ValueError("URDF joint graph contains a cycle")
        return list(reversed(chain))

    @staticmethod
    def forward(chain: list[Joint], values: dict[str, float]) -> np.ndarray:
        result = np.eye(4)
        for joint in chain:
            result = result @ joint.origin
            if joint.kind == "fixed":
                continue
            if joint.name not in values:
                raise ValueError(f"Missing joint value for URDF joint {joint.name}")
            value = values[joint.name]
            motion = np.eye(4)
            if joint.kind in ("revolute", "continuous"):
                motion[:3, :3] = _rotation(joint.axis, value)
            elif joint.kind == "prismatic":
                motion[:3, 3] = joint.axis * value
            else:
                raise ValueError(f"Unsupported URDF joint type {joint.kind}: {joint.name}")
            result = result @ motion
        return result


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    for name in ("urdf", "base_link", "arms", "cameras"):
        if not config.get(name):
            raise ValueError(f"Config requires {name}")
    urdf_path = Path(config["urdf"]).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = (path.parent / urdf_path).resolve()
    if not urdf_path.is_file():
        raise ValueError(f"URDF not found: {urdf_path}")
    config["urdf"] = str(urdf_path)
    if len(config["arms"]) != 2:
        raise ValueError("OpenArmFixedPlace requires left and right arms")
    for arm in config["arms"]:
        for name in ("ee_link", "joint_map", "gripper_raw_name", "gripper_closed_m", "gripper_open_m"):
            if name not in arm or arm[name] in (None, ""):
                raise ValueError(f"Each arm requires {name}")
        if not arm["joint_map"]:
            raise ValueError("Each arm needs raw-name to URDF-name joint_map")
        if not isinstance(arm["joint_map"], dict) or len(set(arm["joint_map"].values())) != len(arm["joint_map"]):
            raise ValueError("Each arm needs a unique raw-name to URDF-name joint_map")
        if "link_T_tcp" in arm:
            _pose(arm["link_T_tcp"], f"arms.{arm['ee_link']}.link_T_tcp")
        for name in ("gripper_closed_m", "gripper_open_m"):
            if not isinstance(arm[name], (int, float)) or not math.isfinite(arm[name]):
                raise ValueError(f"arms.{arm['ee_link']}.{name} must be a finite number")
        if arm["gripper_open_m"] <= arm["gripper_closed_m"]:
            raise ValueError("gripper_open_m must exceed gripper_closed_m")
    if set(config["cameras"]) != {"chest", "front"}:
        raise ValueError("This dataset config requires chest and front cameras")
    geometry_mode = config.setdefault("camera_geometry_mode", "calibrated")
    if geometry_mode not in ("calibrated", "approximate_static"):
        raise ValueError("camera_geometry_mode must be calibrated or approximate_static")
    assumed_focal_px = config.get("assumed_focal_px", 640.0)
    if not isinstance(assumed_focal_px, (int, float)) or not math.isfinite(assumed_focal_px) or assumed_focal_px <= 0:
        raise ValueError("assumed_focal_px must be a positive finite number")
    for name, camera in config["cameras"].items():
        if camera.get("raw_color_order") not in ("rgb", "bgr"):
            raise ValueError(f"cameras.{name}.raw_color_order must be rgb or bgr")
        if geometry_mode == "approximate_static":
            camera.setdefault("D", [])
            camera["K"] = None
            camera["base_T_camera"] = None
            continue
        K = np.asarray(camera.get("K"), dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f"cameras.{name}.K must be a calibrated 3x3 intrinsic matrix")
        D = np.asarray(camera.get("D", []), dtype=np.float64)
        if D.ndim != 1 or len(D) not in (0, 4, 5, 8, 12, 14) or not np.isfinite(D).all():
            raise ValueError(f"cameras.{name}.D must be [] or OpenCV distortion coefficients")
        _pose(camera.get("base_T_camera"), f"cameras.{name}.base_T_camera")
    config.setdefault("prompt_text", "dat_vat")
    if config["prompt_text"] != "dat_vat":
        raise ValueError("This Stage-0 task uses the fixed prompt dat_vat")
    config.setdefault("frame_stride", 5)
    config.setdefault("jpeg_quality", 90)
    if not isinstance(config["frame_stride"], int) or config["frame_stride"] < 1:
        raise ValueError("frame_stride must be a positive integer")
    if not isinstance(config["jpeg_quality"], int) or not 1 <= config["jpeg_quality"] <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    return config


def detect_active_arm(poses: np.ndarray, override: str | None = None) -> tuple[str, list[float], list[float]]:
    """Assign a single active arm; reject weak or simultaneous motion."""
    travel = np.linalg.norm(np.diff(poses[..., :3, 3], axis=0), axis=-1).sum(axis=0)
    displacement = np.linalg.norm(poses[..., :3, 3] - poses[0:1, :, :3, 3], axis=-1).max(axis=0)
    suggested = int(np.argmax(travel))
    other = 1 - suggested
    if override is None:
        if (travel[suggested] < 0.05 or displacement[suggested] < 0.03 or
                travel[suggested] < 3.0 * max(travel[other], 1e-6)):
            raise ValueError(
                "Ambiguous active arm: ee_travel_m=" + str(travel.round(4).tolist()) +
                ", max_displacement_m=" + str(displacement.round(4).tolist()) +
                "; review the episode and provide --arm-overrides"
            )
        active = ("left", "right")[suggested]
    else:
        if override not in ("left", "right"):
            raise ValueError(f"Invalid active-arm override: {override!r}")
        active = override
    return active, travel.tolist(), displacement.tolist()


def convert_episode(source: Path, destination: Path, config: dict, fk: URDFFK,
                    active_arm_override: str | None = None) -> dict:
    import cv2  # imported only when conversion is requested

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".partial.h5")
    try:
        with h5py.File(source, "r") as src, h5py.File(tmp, "w") as dst:
            raw_names = json.loads(src.attrs["joint_names_json"])
            if len(raw_names) != 16 or len(set(raw_names)) != 16:
                raise ValueError(f"{source}: expected 16 unique OpenArm joint names")
            name_to_index = {name: index for index, name in enumerate(raw_names)}
            if src.attrs.get("robot_type") != "OpenArm_Bimanual_16DOF":
                raise ValueError(f"{source}: unexpected robot_type")
            qpos = src["observations/qpos"]
            images = src["observations/images"]
            times_ns = src["timestamp_ns"][:]
            n_source = len(times_ns)
            if n_source < 2 or not np.issubdtype(times_ns.dtype, np.integer):
                raise ValueError(f"{source}: timestamp_ns must contain at least two integer values")
            if qpos.shape != (n_source, 16):
                raise ValueError(f"{source}: qpos shape does not match timestamps")
            if any(
                images[f"{name}_rgb"].ndim != 4
                or images[f"{name}_rgb"].shape[0] != n_source
                or images[f"{name}_rgb"].shape[-1] != 3
                or images[f"{name}_rgb"].dtype != np.uint8
                for name in config["cameras"]
            ):
                raise ValueError(f"{source}: expected synchronized uint8 HWC RGB frames")
            if np.any(np.diff(times_ns) <= 0):
                raise ValueError(f"{source}: timestamps are not strictly increasing")
            indices = np.arange(0, n_source, config["frame_stride"], dtype=np.int64)
            if len(indices) < 33:
                raise ValueError(f"{source}: fewer than 33 downsampled frames")
            q = qpos[indices]
            if not np.isfinite(q).all():
                raise ValueError(f"{source}: qpos contains NaN or Inf")
            timestamp = (times_ns[indices] - times_ns[0]).astype(np.float64) * 1e-9
            if timestamp[-1] < 3.2:
                raise ValueError(f"{source}: less than 3.2 seconds after downsampling")
            poses = np.empty((len(indices), len(config["arms"]), 4, 4), dtype=np.float32)
            grippers = np.empty((len(indices), len(config["arms"])), dtype=np.float32)
            for arm_index, arm in enumerate(config["arms"]):
                chain = fk.chain(config["base_link"], arm["ee_link"])
                link_T_tcp = _pose(arm.get("link_T_tcp", np.eye(4)), f"arms.{arm['ee_link']}.link_T_tcp")
                raw_to_urdf = arm["joint_map"]
                for joint in chain:
                    if joint.kind != "fixed" and joint.name not in raw_to_urdf.values():
                        raise ValueError(f"{source}: no raw mapping for URDF joint {joint.name}")
                for raw_name in raw_to_urdf:
                    if raw_name not in name_to_index:
                        raise ValueError(f"{source}: raw joint {raw_name} not found")
                gripper_name = arm["gripper_raw_name"]
                if gripper_name not in name_to_index:
                    raise ValueError(f"{source}: gripper joint {gripper_name} not found")
                for out_index, row in enumerate(q):
                    values = {urdf_name: float(row[name_to_index[raw_name]])
                              for raw_name, urdf_name in raw_to_urdf.items()}
                    poses[out_index, arm_index] = fk.forward(chain, values) @ link_T_tcp
                stroke = q[:, name_to_index[gripper_name]]
                if (stroke.min() < arm["gripper_closed_m"] - 1e-3 or
                        stroke.max() > arm["gripper_open_m"] + 1e-3):
                    raise ValueError(f"{source}: {gripper_name} stroke is outside configured limits")
                grippers[:, arm_index] = np.clip(
                    (stroke - arm["gripper_closed_m"])
                    / (arm["gripper_open_m"] - arm["gripper_closed_m"]), 0, 1
                )
            active_arm, travel, displacement = detect_active_arm(poses, active_arm_override)
            active_index = 0 if active_arm == "left" else 1
            dst.create_dataset("timestamp", data=timestamp)
            dst.create_dataset("ee_pose", data=poses)
            dst.create_dataset("gripper", data=grippers)
            dst.attrs["prompt_text"] = config["prompt_text"]
            dst.attrs["is_bgr"] = True  # cv2.imdecode returns BGR; APT flips it to RGB
            dst.attrs["ee_indices"] = np.arange(len(config["arms"]), dtype=np.int64)
            dst.attrs["valid_ee_mask"] = np.asarray([active_index == 0, active_index == 1], dtype=bool)
            dst.attrs["active_arm"] = active_arm
            dst.attrs["source_episode"] = str(source)
            dst.attrs["frame_stride"] = config["frame_stride"]
            dst.attrs["camera_geometry_mode"] = config["camera_geometry_mode"]
            if config["camera_geometry_mode"] == "approximate_static":
                dst.attrs["assumed_focal_px"] = float(config["assumed_focal_px"])
            for name, camera in config["cameras"].items():
                group = dst.create_group(name)
                rgb_in = images[f"{name}_rgb"]
                height, width = rgb_in.shape[1:3]
                if config["camera_geometry_mode"] == "approximate_static":
                    f = float(config["assumed_focal_px"])
                    camera_K = np.asarray([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float32)
                    base_T_camera = np.eye(4, dtype=np.float32)
                else:
                    camera_K = np.asarray(camera["K"], dtype=np.float32)
                    base_T_camera = _pose(camera["base_T_camera"], f"{name}.base_T_camera").astype(np.float32)
                group.create_dataset("K", data=camera_K)
                group.create_dataset(
                    "pose", data=np.repeat(
                        base_T_camera[None],
                        len(indices), axis=0
                    ).astype(np.float32)
                )
                rgb_out = group.create_dataset(
                    "rgb", shape=(len(indices),), dtype=h5py.vlen_dtype(np.uint8)
                )
                D = np.asarray(camera.get("D", []), dtype=np.float64)
                K = camera_K.astype(np.float64)
                for out_index, source_index in enumerate(indices):
                    frame = rgb_in[int(source_index)]
                    if camera["raw_color_order"] == "rgb":
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if len(D):
                        frame = cv2.undistort(frame, K, D)
                    okay, jpeg = cv2.imencode(
                        ".jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, config["jpeg_quality"]],
                    )
                    if not okay:
                        raise ValueError(f"{source}: JPEG encode failed for {name} frame {source_index}")
                    rgb_out[out_index] = jpeg.reshape(-1)
        os.replace(tmp, destination)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return {"source": str(source), "output": str(destination),
            "source_frames": n_source, "output_frames": len(indices),
            "duration_seconds": float(timestamp[-1]),
            "active_arm": active_arm,
            "ee_travel_m": [round(value, 4) for value in travel],
            "max_displacement_m": [round(value, 4) for value in displacement],
            "gripper_range": [[round(float(grippers[:, i].min()), 4), round(float(grippers[:, i].max()), 4)]
                              for i in range(len(config["arms"]))]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Convert only the first N episodes")
    parser.add_argument("--val-fraction", type=float, default=0.0,
                        help="Stable episode-level validation fraction (default: 0)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--arm-overrides", type=Path,
                        help="JSON mapping raw paths relative to input root to left/right")
    args = parser.parse_args()
    if not 0 <= args.val_fraction < 1:
        parser.error("--val-fraction must be in [0, 1)")
    config = load_config(args.config)
    if config["camera_geometry_mode"] == "approximate_static":
        print("[WARN] Using approximate fixed-camera geometry: K is estimated from image size and both base_T_camera poses are identity.")
    overrides = json.loads(args.arm_overrides.read_text(encoding="utf-8")) if args.arm_overrides else {}
    if not isinstance(overrides, dict) or any(value not in ("left", "right") for value in overrides.values()):
        parser.error("--arm-overrides must be a JSON object mapping paths to left/right")
    fk = URDFFK(Path(config["urdf"]))
    sources = sorted(set(args.input_root.rglob("*.h5")) | set(args.input_root.rglob("*.hdf5")))
    if args.limit > 0:
        sources = sources[:args.limit]
    if not sources:
        parser.error(f"No HDF5 episodes under {args.input_root}")
    if args.output_root.resolve().is_relative_to(args.input_root.resolve()):
        parser.error("--output-root must be outside --input-root")
    assigned = []
    outputs = set()
    split_counts = {"train": 0, "val": 0}
    for source in sources:
        relative = source.relative_to(args.input_root).with_suffix(".h5")
        digest = hashlib.sha256(relative.as_posix().encode("utf-8")).digest()
        holdout = int.from_bytes(digest[:8], "big") / 2**64 < args.val_fraction
        split = "val" if holdout else "train"
        split_counts[split] += 1
        destination = args.output_root / split / relative
        if destination in outputs:
            parser.error(f"Output path collision: {destination}")
        outputs.add(destination)
        assigned.append((source, destination))
    if split_counts["train"] == 0:
        parser.error("Episode split contains no training files; reduce --val-fraction")
    print(json.dumps({"split": split_counts}))
    converted_counts = {"left": 0, "right": 0}
    for source, destination in assigned:
        if destination.exists() and not args.overwrite:
            print(json.dumps({"source": str(source), "output": str(destination), "skipped": "exists"}))
            continue
        relative_name = source.relative_to(args.input_root).as_posix()
        result = convert_episode(source, destination, config, fk, overrides.get(relative_name))
        converted_counts[result["active_arm"]] += 1
        print(json.dumps(result))
    print(json.dumps({"converted_by_arm": converted_counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
