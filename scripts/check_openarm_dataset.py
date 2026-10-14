"""Check converted OpenArm episodes and optionally sample the APT adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def check_file(path: Path) -> tuple[int, float, str, str]:
    with h5py.File(path, "r") as h5:
        timestamp = h5["timestamp"][:]
        n = len(timestamp)
        if n < 33 or not np.isfinite(timestamp).all() or np.any(np.diff(timestamp) <= 0):
            raise ValueError(f"{path}: invalid timestamp series")
        if h5.attrs.get("prompt_text") != "dat_vat":
            raise ValueError(f"{path}: prompt_text must be dat_vat")
        geometry_mode = h5.attrs.get("camera_geometry_mode")
        if geometry_mode not in ("calibrated", "approximate_static"):
            raise ValueError(f"{path}: missing/invalid camera_geometry_mode")
        active_arm = h5.attrs.get("active_arm")
        expected_mask = {"left": [True, False], "right": [False, True]}.get(active_arm)
        if expected_mask is None or not np.array_equal(h5.attrs.get("valid_ee_mask"), expected_mask):
            raise ValueError(f"{path}: invalid active_arm or valid_ee_mask")
        if not h5.attrs.get("is_bgr", False):
            raise ValueError(f"{path}: JPEGs must be declared BGR for APT decoding")
        poses = h5["ee_pose"][:]
        grippers = h5["gripper"][:]
        if poses.shape != (n, 2, 4, 4) or grippers.shape != (n, 2):
            raise ValueError(f"{path}: expected two end effectors")
        if not np.isfinite(poses).all() or not np.isfinite(grippers).all():
            raise ValueError(f"{path}: nonfinite pose or gripper")
        if not np.allclose(poses[..., 3, :], [0, 0, 0, 1], atol=1e-4):
            raise ValueError(f"{path}: invalid EE homogeneous pose")
        if not np.allclose(np.swapaxes(poses[..., :3, :3], -1, -2) @ poses[..., :3, :3],
                           np.eye(3), atol=2e-3):
            raise ValueError(f"{path}: EE rotations are not orthonormal")
        if np.any((grippers < 0) | (grippers > 1)):
            raise ValueError(f"{path}: gripper must be normalized to [0, 1]")
        for name in ("chest", "front"):
            camera = h5[name]
            K = camera["K"][:]
            cam_poses = camera["pose"][:]
            if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
                raise ValueError(f"{path}: invalid {name}/K")
            if cam_poses.shape != (n, 4, 4) or not np.isfinite(cam_poses).all():
                raise ValueError(f"{path}: invalid {name}/pose")
            if not np.allclose(cam_poses[..., 3, :], [0, 0, 0, 1], atol=1e-4):
                raise ValueError(f"{path}: invalid {name} homogeneous pose")
            if not np.allclose(np.swapaxes(cam_poses[..., :3, :3], -1, -2) @
                               cam_poses[..., :3, :3], np.eye(3), atol=2e-3):
                raise ValueError(f"{path}: invalid {name} rotation")
            rgb = camera["rgb"]
            if rgb.shape != (n,):
                raise ValueError(f"{path}: invalid {name}/rgb length")
            for index in (0, n - 1):
                frame = cv2.imdecode(rgb[index], cv2.IMREAD_COLOR)
                if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError(f"{path}: cannot decode {name} JPEG frame {index}")
        return n, float(timestamp[-1] - timestamp[0]), active_arm, geometry_mode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="One converted .h5 file or output root")
    parser.add_argument("--apt-sample", action="store_true",
                        help="Also load one item through the APT dataset adapter (requires train dependencies)")
    args = parser.parse_args()
    paths = [args.path] if args.path.is_file() else sorted(args.path.rglob("*.h5"))
    if not paths:
        parser.error(f"No converted .h5 files at {args.path}")
    total = 0
    counts = {"left": 0, "right": 0}
    for path in paths:
        n, duration, active_arm, geometry_mode = check_file(path)
        print(f"[OK] {path}: {n} frames, {duration:.2f} s, active={active_arm}, camera={geometry_mode}")
        total += n
        counts[active_arm] += 1
    print(f"[OK] {len(paths)} episodes, {total} frames, by arm: {counts}")
    if args.apt_sample:
        from data_utils.datasets import OpenArmFixedPlace
        sample = OpenArmFixedPlace([str(paths[0])])[0]
        if sample["prompt_text"] != "dat_vat":
            raise ValueError("APT adapter changed the prompt")
        if sample["gt_future_ee_states"].shape != (32, 2, 17):
            raise ValueError("APT adapter returned an unexpected action shape")
        if sample["obs_rgbs"].shape[1:] != (2, 3, 256, 256):
            raise ValueError("APT adapter returned an unexpected camera shape")
        with h5py.File(paths[0], "r") as h5:
            expected_mask = np.asarray(h5.attrs["valid_ee_mask"], dtype=bool)
        if not np.array_equal(sample["valid_ee_mask"], expected_mask):
            raise ValueError("APT adapter changed the active-arm mask")
        print("[OK] APT sample:", {key: getattr(value, "shape", value)
                                  for key, value in sample.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
