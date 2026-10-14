"""LIBERO evaluation driver for the APT remote inference service.

Connects to a running APT policy server (via ``shm_transport``) and rolls
out each task in a LIBERO task suite, optionally recording success/fail
to CSV and a per-episode MP4.

Supports three benchmark families with identical entry-point logic — the
only difference is the value passed to ``--libero_task_suite`` and the
typical ``--num_trials_per_task``:

  * **LIBERO**       — 4 suites: ``libero_{object,spatial,goal,10}``        (typ. 50 trials/task)
  * **LIBERO-PRO**   — 8 suites: above × {``_swap``, ``_task``}             (typ. 50 trials/task)
  * **LIBERO-PLUS**  — 4 suites: ``libero_{object,spatial,goal,10}``        (typ.  1 trial/task)

Launch the script as a module so its relative imports resolve:

    python -m examples.libero.eval_libero_pred \\
        --libero_task_suite libero_object \\
        --model_name <model> --controller_name <ctrl> --controller_port 9091 \\
        --num_trials_per_task 50 \\
        --save_root  results/<benchmark>/<run> \\
        --video_root videos/<benchmark>/<run> \\
        --save --video --all --conti_save
"""
import argparse
import os

import cv2
import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from einops import rearrange
from libero.libero import benchmark
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)
from scipy.spatial.transform import Rotation as R

# ``libero_utils`` lives in OpenVLA-OFT (https://github.com/openvla-oft/openvla-oft).
# Clone it next to this checkout and put it on the PYTHONPATH, e.g.
#     PYTHONPATH=/path/to/openvla-oft:$PYTHONPATH
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)

from shm_transport import get_shm_proxy, setup_log_level

from . import align
from .video_writer import VideoWriter


IMAGE_RESOLUTION = 256
MAX_EVAL_EP_LEN = 500   # hard cap on per-episode env steps
SAMPLE_STEPS_PER_QUERY = 3  # how many sub-steps to execute per inference query


# ─────────────────────────────────────────────────────────────────────────────
# Projection / drawing helpers (used for the per-frame visualisation overlay)
# ─────────────────────────────────────────────────────────────────────────────

def proj(K: np.ndarray, cwT: np.ndarray, pos: np.ndarray):
    pos_in_cam = pos @ cwT[:3, :3].T + cwT[:3, 3]
    xy = pos_in_cam[..., :2] / pos_in_cam[..., -1:]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return xy * np.array([fx, fy]) + np.array([cx, cy])


def draw_ee_proj(bgr: np.ndarray, K: np.ndarray, cwT: np.ndarray, pose: np.ndarray):
    alen = 0.05  # 5 cm axis length
    pos = pose[:3, 3]
    origin = tuple(proj(K, cwT, pos).astype(int).tolist())
    x_end = tuple(proj(K, cwT, pos + pose[:3, 0] * alen).astype(int).tolist())
    y_end = tuple(proj(K, cwT, pos + pose[:3, 1] * alen).astype(int).tolist())
    z_end = tuple(proj(K, cwT, pos + pose[:3, 2] * alen).astype(int).tolist())
    cv2.line(bgr, origin, x_end, (0, 0, 255), thickness=2)
    cv2.line(bgr, origin, y_end, (0, 255, 0), thickness=2)
    cv2.line(bgr, origin, z_end, (255, 0, 0), thickness=2)
    return bgr


def vis_regen_obs(obs: dict, eef_pos: np.ndarray, eef_rotmat: np.ndarray, sim,
                  pred_poses=None):
    """Render a side-by-side BGR frame of the third-person and wrist views with
    the executed EE pose drawn as RGB axes, plus optional predicted future
    positions as red dots."""
    cam_K_e2h = get_camera_intrinsic_matrix(sim, "agentview", IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    cam_K_eih = get_camera_intrinsic_matrix(sim, "robot0_eye_in_hand", IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    cam_wcT_e2h = get_camera_extrinsic_matrix(sim, "agentview")
    cam_wcT_eih = get_camera_extrinsic_matrix(sim, "robot0_eye_in_hand")

    # MuJoCo returns the image upside-down; flip the H dimension and swap to BGR.
    bgr_e2h = np.ascontiguousarray(obs["agentview_image"][::-1, :, [2, 1, 0]])
    bgr_eih = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, :, [2, 1, 0]])

    act_img_e2h = proj(cam_K_e2h, np.linalg.inv(cam_wcT_e2h), eef_pos)
    act_img_eih = proj(cam_K_eih, np.linalg.inv(cam_wcT_eih), eef_pos)
    cv2.circle(bgr_e2h, tuple(act_img_e2h.astype(int).tolist()), 2, (0, 0, 255), -1)
    cv2.circle(bgr_eih, tuple(act_img_eih.astype(int).tolist()), 2, (0, 0, 255), -1)

    eef_pose = np.eye(4).astype(eef_pos.dtype)
    eef_pose[:3, :3] = eef_rotmat
    eef_pose[:3, 3] = eef_pos
    draw_ee_proj(bgr_e2h, cam_K_e2h, np.linalg.inv(cam_wcT_e2h), eef_pose)
    draw_ee_proj(bgr_eih, cam_K_eih, np.linalg.inv(cam_wcT_eih), eef_pose)

    if pred_poses is not None:
        future_act_e2h = proj(cam_K_e2h, np.linalg.inv(cam_wcT_e2h),
                              pred_poses[..., :3, 3])
        for pts in future_act_e2h:
            cv2.circle(bgr_e2h, tuple(pts.astype(int).tolist()), 2, (0, 0, 255), -1)

    return np.concatenate([bgr_e2h, bgr_eih], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Action / observation conversions between LIBERO and APT
# ─────────────────────────────────────────────────────────────────────────────

def action_from_apt_to_libero(
    future_ee_poses: np.ndarray,     # (T, 4, 4)
    future_grippers: np.ndarray,     # (T,)
    current_eef_pos: np.ndarray,     # (3,)
    current_eef_rotmat: np.ndarray,  # (3, 3)
    robosuite_action_scale: np.ndarray,
):
    """Convert APT's absolute future EE poses into LIBERO's OSC-pose action format.

    LIBERO's robosuite OSC controller (v1.4.1) takes a scaled delta pose:
    https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1_libero/robosuite/controllers/osc.py#L237
    """
    delta_pos = future_ee_poses[..., :3, 3] - current_eef_pos
    # robosuite computes goal as ``delta_rot @ current`` (left-multiply convention).
    delta_rotmat = future_ee_poses[..., :3, :3] @ np.linalg.inv(current_eef_rotmat)
    delta_ori = R.from_matrix(delta_rotmat).as_rotvec()
    delta_pose_scaled = np.concatenate([delta_pos, delta_ori], axis=-1) / robosuite_action_scale

    # APT outputs gripper in [0, 1] (1 = open); LIBERO expects {-1, +1} (-1 = open).
    binary = (future_grippers > 0.8).astype(np.float32)
    delta_gripper = 1 - 2 * binary

    return np.concatenate([delta_pose_scaled, delta_gripper[..., None]], axis=-1)


def obs_libero2apt(obs: dict, time: int, env) -> dict:
    """Repackage a LIBERO observation dict into the APT planner's expected schema."""
    ee_pose = np.eye(4)
    ee_pose[:3, 3] = obs["robot0_eef_pos"]
    ee_pose[:3, :3] = T.quat2mat(obs["robot0_eef_quat"])

    gripper_state = obs["robot0_gripper_qpos"]      # (2,)
    open_gripper_qpos, close_gripper_qpos = 0.04, 0.0
    gripper = (gripper_state[0] - close_gripper_qpos) / (open_gripper_qpos - close_gripper_qpos)

    # Cameras: MuJoCo renders upside-down — flip H to bring image right-side up.
    e2h_rgb = np.ascontiguousarray(obs["agentview_image"][::-1])
    eih_rgb = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    e2h_pose = get_camera_extrinsic_matrix(env.sim, "agentview")
    eih_pose = get_camera_extrinsic_matrix(env.sim, "robot0_eye_in_hand")
    e2h_K = get_camera_intrinsic_matrix(env.sim, "agentview",
                                        IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    eih_K = get_camera_intrinsic_matrix(env.sim, "robot0_eye_in_hand",
                                        IMAGE_RESOLUTION, IMAGE_RESOLUTION)
    return {
        "ee_pose":   ee_pose,
        "gripper":   gripper,
        "timestamp": time,
        "agentview": {
            "model":  "pinhole",
            "camera": {"height": IMAGE_RESOLUTION, "width": IMAGE_RESOLUTION,
                       "K": e2h_K.flatten().tolist()},
            "data":   {"color": e2h_rgb, "wcT": e2h_pose,
                       "seg": None, "depth": None},
        },
        "eye_in_hand": {
            "model":  "pinhole",
            "camera": {"height": IMAGE_RESOLUTION, "width": IMAGE_RESOLUTION,
                       "K": eih_K.flatten().tolist()},
            "data":   {"color": eih_rgb, "wcT": eih_pose,
                       "seg": None, "depth": None},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main rollout loop
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    controller = get_shm_proxy(args.controller_name, ns_port=args.controller_port)
    setup_log_level("WARNING")

    print("!" * 101)
    print(f"[INFO] model_name = {args.model_name}")
    print(f"[INFO] Task suite = {args.libero_task_suite}")
    print(f"[INFO] {args.num_trials_per_task} trials/task, {num_tasks_in_suite} tasks")

    suffix = args.suffix
    save_path = os.path.join(args.save_root,
                             f"{args.libero_task_suite}_{suffix}",
                             f"{args.model_name}.csv")
    video_root = os.path.join(args.video_root,
                              f"{args.libero_task_suite}_{suffix}",
                              args.model_name)

    fp = None
    if args.save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mode = "a" if args.conti_save else "w"
        fp = open(save_path, mode)
        if args.conti_save:
            fp.write("\n")
        else:
            fp.write("task_id, episode_id, success, prompt\n")
            fp.flush()

    if args.video:
        os.makedirs(video_root, exist_ok=True)

    num_replays = 0
    num_success = 0
    success_records = []

    for task_id in tqdm.tqdm(range(args.begin_task_id, num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, _env_args, task_description = get_libero_env(task, resolution=IMAGE_RESOLUTION)
        print(f"[INFO] task_description = {task_description}")

        for i in range(args.num_trials_per_task):
            save_video_path = os.path.join(
                video_root, f"task{task_id:0>3d}", f"ep{i:0>3d}.mp4")
            vid_writer = None
            if args.video:
                os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
                vid_writer = VideoWriter(save_video_path, 30)

            env.reset()
            env.set_init_state(initial_states[i])
            # Let the env settle for a few steps before querying the policy.
            for _ in range(10):
                obs, _r, _d, _info = env.step(get_libero_dummy_action())
                vis_regen_obs(obs, obs["robot0_eef_pos"],
                              T.quat2mat(obs["robot0_eef_quat"]), env.sim)

            prompt_text = task_description
            print(f"[INFO] prompt_text = {prompt_text}")

            controller.reset()
            controller.set_config("Libero")
            runtime_config = controller.get_config()
            sample_state_gaps = runtime_config["sample_state_gaps"]

            controller.set_prompt(prompt_text)
            a_idx = 0
            done = False

            while True:
                controller.add_obs_frame(obs_libero2apt(obs, time=a_idx, env=env))
                future_ee_poses, future_grippers, future_time, _ = \
                    controller.get_action(sample_num=1)
                future_ee_poses = future_ee_poses[0, :, 0]
                future_grippers = future_grippers[0, :, 0]

                if sample_state_gaps != 1:
                    query_time = a_idx + np.arange(
                        len(future_ee_poses) * sample_state_gaps)
                    train_data = {"ee_pose": future_ee_poses,
                                  "gripper": future_grippers}
                    interp_funcs = {"ee_pose": align.interp_SE3_sep,
                                    "gripper": align.interp_linear}
                    query_data = align.align_data(
                        query_time, future_time, train_data, interp_funcs)
                else:
                    query_data = {"ee_pose": future_ee_poses,
                                  "gripper": future_grippers}

                for s_id in range(SAMPLE_STEPS_PER_QUERY):
                    action = action_from_apt_to_libero(
                        future_ee_poses=query_data["ee_pose"][s_id],
                        future_grippers=query_data["gripper"][s_id],
                        current_eef_pos=obs["robot0_eef_pos"],
                        current_eef_rotmat=T.quat2mat(obs["robot0_eef_quat"]),
                        robosuite_action_scale=env.env.robots[0].controller.action_scale,
                    )
                    obs, _r, done, _info = env.step(action.tolist())
                    a_idx += 1

                    if vid_writer is not None:
                        debug_bgr = vis_regen_obs(
                            obs, obs["robot0_eef_pos"],
                            T.quat2mat(obs["robot0_eef_quat"]), env.sim,
                            pred_poses=query_data["ee_pose"])
                        vid_writer.write(debug_bgr)

                    if done or a_idx > MAX_EVAL_EP_LEN:
                        break

                if done or a_idx > MAX_EVAL_EP_LEN:
                    break

            if vid_writer is not None:
                vid_writer.finalize()

            if done:
                num_success += 1
            num_replays += 1
            print(
                f"Total # episodes replayed: {num_replays}, "
                f"Total # successes: {num_success} "
                f"({num_success / num_replays * 100:.1f} %)"
            )

            if fp is not None:
                fp.write(f"{task_id}, {i}, {int(done)}, {prompt_text}\n")
                fp.flush()
            success_records.append(int(done))

            if not args.all:
                break  # one trial per task in single-shot mode

    if fp is not None:
        fp.write(f" , , {np.mean(success_records) * 100:.2f}%, \n")
        fp.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Roll out a LIBERO task suite against an APT policy server.")
    p.add_argument("--libero_task_suite", type=str, required=True,
                   help="LIBERO suite name, e.g. libero_object, libero_object_swap (PRO).")
    p.add_argument("--model_name", type=str, default="debug",
                   help="Used as a label in the CSV / video output paths.")
    p.add_argument("--controller_name", type=str, default="control",
                   help="Pyro4 / shm_transport name of the running APT policy server.")
    p.add_argument("--controller_port", type=int, default=9091,
                   help="Naming-server port of the running APT policy server.")
    p.add_argument("--num_trials_per_task", type=int, default=50,
                   help="Number of initial-state trials per task in the suite.")
    p.add_argument("--begin_task_id", type=int, default=0,
                   help="Skip the first N tasks (useful for resuming).")
    p.add_argument("--save_root",  type=str, default="results",
                   help="Root directory under which to write the CSV success log.")
    p.add_argument("--video_root", type=str, default="videos",
                   help="Root directory under which to write rollout MP4s.")
    p.add_argument("--suffix", type=str, default="one",
                   help="Subdirectory suffix appended to the per-suite output dirs.")
    p.add_argument("--all", action="store_true", default=False,
                   help="Run all `num_trials_per_task` trials per task; without this only one runs.")
    p.add_argument("--save", action="store_true", default=False,
                   help="Write CSV success log.")
    p.add_argument("--conti_save", action="store_true", default=False,
                   help="Append to CSV instead of overwriting.")
    p.add_argument("--video", action="store_true", default=False,
                   help="Record per-episode MP4.")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
