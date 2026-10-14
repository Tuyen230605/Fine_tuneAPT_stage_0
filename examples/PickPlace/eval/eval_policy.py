"""Unified PickPlace evaluation entry point.

Dispatches on `--setting {so,uo,uc}` and reads `eval/configs/<setting>.yaml`.
The YAML schema is documented at the top of `eval/configs/so.yaml`.

Run as:
    python -m eval.eval --setting so -i 0 --uri control --port 9091 \\
        --no_gui --save --save_dir ./data/exp_results/run/so

The `uoue` split is produced by:
    python -m eval.eval --setting uo --novel_bg --random_env --seed N ...
"""

import os
import cv2
import json
import random
import argparse
import yaml
from pathlib import Path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--setting", required=True, choices=["so", "uo", "uc"],
                    help="Evaluation setting; selects eval/configs/<setting>.yaml")
parser.add_argument("-i", "--index", required=True, type=int,
                    help="GPU index; also sets CUDA_VISIBLE_DEVICES")
parser.add_argument("--no_gui",     action="store_true", default=False, help="Run Isaac Sim headless")
parser.add_argument("--save",       action="store_true", default=False, help="Save per-episode video + metric JSON")
parser.add_argument("--save_dir",   type=str, default=None, help="Output directory")
parser.add_argument("--no_cv_win",  action="store_true", default=False, help="Disable OpenCV preview windows")
parser.add_argument("--uri",        type=str, default="control", help="Pyro URI of the policy service")
parser.add_argument("--port",       type=int, default=9091,     help="Pyro nameserver port")
parser.add_argument("--seed",       type=int, default=0,        help="Random seed offset (uoue sweep)")

# Environment-randomization flags. Honored only when the loaded YAML sets
# `support_env_randomization: true` (currently only `uo`).
parser.add_argument("--random_plate", action="store_true", default=False)
parser.add_argument("--random_bg",    action="store_true", default=False)
parser.add_argument("--random_env",   action="store_true", default=False)
parser.add_argument("--novel_plate",  action="store_true", default=False)
parser.add_argument("--novel_bg",     action="store_true", default=False)
parser.add_argument("--novel_env",    action="store_true", default=False)
parser.add_argument("--plate",        type=str, default=None, help="Specific plate key")
parser.add_argument("--bg",           type=str, default=None, help="Specific background key")
parser.add_argument("--env",          type=str, default=None, help="Specific environment key")
parser.add_argument("--list_plates",  action="store_true", default=False, help="List plate keys and exit")
parser.add_argument("--list_bgs",     action="store_true", default=False, help="List background keys and exit")
parser.add_argument("--list_envs",    action="store_true", default=False, help="List environment keys and exit")
opt = parser.parse_args()


# -----------------------------------------------------------------------------
# Load setting config (early — no Isaac dependency)
# -----------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).parent / "configs"
with open(CONFIG_DIR / f"{opt.setting}.yaml") as fp:
    SETTING = yaml.safe_load(fp)

SUPPORT_ENV_RAND = SETTING.get("support_env_randomization", False)

# Handle list-and-exit flags before Isaac Sim init so they're cheap.
if opt.list_plates or opt.list_bgs or opt.list_envs:
    from pp_env.random_envs import plates as _plates
    from pp_env.random_envs import backgrounds as _backgrounds
    from pp_env.random_envs import environments as _environments
    if opt.list_plates:
        print("Available plates:")
        for k, d in _plates.list_available_plates():
            print(f"  - {k}: {d}")
    if opt.list_bgs:
        print("Available backgrounds:")
        for k, d in _backgrounds.list_available_backgrounds():
            print(f"  - {k}: {d}")
    if opt.list_envs:
        print("Available environments:")
        for k, d in _environments.list_available_environments():
            print(f"  - {k}: {d}")
    exit(0)


# -----------------------------------------------------------------------------
# Boot Isaac Sim
# -----------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.index)

import omni
from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless":   opt.no_gui,
    "multi_gpu":  False,
    "active_gpu": opt.index,
})


import numpy as np
from typing import List
from itertools import combinations
from scipy.spatial.transform import Rotation
from omni.isaac.core import World
from omni.isaac.core.utils.semantics import add_update_semantics
import omni.isaac.core.utils.prims as prims_utils
from pxr import Gf

from assets.ur5gp85 import UR5GP85
from isaac_env.isaac_camera import IsaacCamera
from isaac_env.non_isaac import perception
from isaac_env import utils
from isaac_env.add_default_ground import add_ground_plane
from pp_env.pick_place import PickPlace
from pp_env import objs
from pp_env.combo import enrich_pick_prompt, is_valid_combination
from pp_env.scene import (
    SceneConfig,
    setup_pickable_objects, remove_pickable_objects,
    setup_placeable_objects, remove_placeable_objects,
    sample_scene_config, apply_scene_config,
)
from pp_env.random_envs import plates, backgrounds, environments
from utils.sample2d import sample_xy, sample_obj_xys2
from utils import align
from utils.video_writer import VideoWriter
from utils.viz import plot_traj, draw_prompt
from utils.poses import sample_cam_extr, sample_initial_pose
from shm_transport import get_shm_proxy


# -----------------------------------------------------------------------------
# World / robot / cameras (identical across settings)
# -----------------------------------------------------------------------------
my_world: World = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
my_world.set_simulation_dt(1/240.0, 1/60.0)
my_world.set_block_on_render(True)
my_world.get_physics_context().set_physx_update_transformations_settings(update_to_usd=True)
timeline = omni.timeline.get_timeline_interface()

ur5 = UR5GP85(name="UR5", prim_path="/World/robot", physics_dt=my_world.get_physics_dt())


# -----------------------------------------------------------------------------
# Resolve environment-randomization choices (plate / bg / env)
# -----------------------------------------------------------------------------
def _resolve_env_keys():
    """Pick plate / background / ground-env USD keys from CLI flags or fall back to defaults.

    The flags are honored only when the YAML enables them. For `so` and `uc`
    we always use the canonical defaults (`plate` / `default` / `default`).
    """
    if SUPPORT_ENV_RAND:
        if opt.plate:        plate_key = opt.plate
        elif opt.random_plate: plate_key = plates.get_random_plate()
        elif opt.novel_plate:  plate_key = plates.get_novel_plate()
        else:                  plate_key = "plate"

        if opt.bg:           bg_key = opt.bg
        elif opt.random_bg:  bg_key = backgrounds.get_random_background()
        elif opt.novel_bg:   bg_key = backgrounds.get_novel_background()
        else:                bg_key = "default"

        if opt.env:          env_key = opt.env
        elif opt.random_env: env_key = environments.get_random_environment(seed=opt.seed + 10)
        elif opt.novel_env:  env_key = environments.get_novel_environment(seed=opt.seed + 10)
        else:                env_key = "default"
    else:
        plate_key, bg_key, env_key = "plate", "default", "default"

    return plate_key, bg_key, env_key


plate_key, bg_key, env_key = _resolve_env_keys()
plate_config = plates.get_plate_config(plate_key)
bg_config = backgrounds.get_background_config(bg_key)
env_config = environments.get_environment_config(env_key)


# -----------------------------------------------------------------------------
# Pickable object combinations
# -----------------------------------------------------------------------------
available_objs   = list(SETTING["pickable_objs"])
non_target_objs  = list(SETTING["non_target_objs"])     # lowercase prim names
conflicts        = SETTING.get("conflicts", {}) or {}
prompts          = SETTING["prompts"]


valid_combinations = [
    c for c in combinations(available_objs, 4)
    if is_valid_combination(c, conflicts)
]
print(f"valid combinations: {len(valid_combinations)}")


# -----------------------------------------------------------------------------
# Scene setup
# -----------------------------------------------------------------------------
# Placeable spec frozen at startup from YAML, so we don't re-parse on every episode.
PLACEABLE_SPECS = SETTING["placeable"]


# -----------------------------------------------------------------------------
# Cameras
# -----------------------------------------------------------------------------
e2h_wcT = perception.look_at_view_transform(
    eye=np.array([1.5, 1.5, 1.5]), to=np.array([0.25, 0.25, 0]), up=np.array([0, 0, 1])
)
camera_e2h = IsaacCamera(
    prim_path="/World/rgbd_e2h", name="eye-to-hand",
    opengl_cam=perception.OpenglCamera(
        intrinsic=perception.PinholeCamera(width=256, height=256, fx=800, fy=800, cx=128, cy=128),
        near=0.01, far=10.0,
    ),
    enable_segmentation=True,
)
camera_e2h.set_wcT(e2h_wcT)

camera_eih = IsaacCamera(
    name="eye-in-hand",
    opengl_cam=perception.OpenglCamera(
        intrinsic=perception.PinholeCamera(width=256, height=256, fx=256, fy=256, cx=128, cy=128),
        near=0.01, far=10.0,
    ),
    enable_depth=True, enable_segmentation=True,
    **ur5.default_eih_cam_extr(),
)


# -----------------------------------------------------------------------------
# Ground + lighting
# -----------------------------------------------------------------------------
add_ground_plane(my_world, usd_path=env_config.usd_path)
add_update_semantics(prims_utils.get_prim_at_path("/World/defaultGroundPlane"), "ground")
my_world.scene.add(ur5._robot)

hdr_file = bg_config.hdr_file or "./assets/hdr/symmetrical_garden_02_2k.hdr"
hdr_intensity = bg_config.intensity if bg_config.hdr_file else 1000
hdr_light = utils.add_hdr_light(prim_path="/World/my_hdr_light", hdr_file=hdr_file, intensity=hdr_intensity)

default_light = prims_utils.get_prim_at_path(prim_path="/World/defaultGroundPlane/SphereLight")
default_light.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, -10.0))
default_light.GetAttribute("intensity").Set(0)

my_world.reset()
camera_e2h.initialize()
camera_eih.initialize()


# -----------------------------------------------------------------------------
# Scene config (derived once from the loaded SETTING)
# -----------------------------------------------------------------------------
PLACE_RADIUS = SETTING["scene"]["place_radius"]
EXTRA_ROT_X  = [float(s.get("extra_rotation_x_deg", 0.0)) for s in PLACEABLE_SPECS]


def init_world_from_config(config: SceneConfig,
                           pickable_objects: List[utils.UsdObject],
                           placable_objects: List[utils.UsdObject],
                           proposals):
    initial_pose = sample_initial_pose()
    pp_controller = PickPlace(
        ur5=ur5,
        obj=pickable_objects[config.pick_obj_index],
        grasp_proposal=proposals[config.pick_obj_index],
        initial_pose=initial_pose,
        place_pose=config.place_gripper_pose,
        grasp_condition={"flip": config.flip_grasp},
    )

    init_steps = 40
    while simulation_app.is_running() and init_steps > 0:
        my_world.step()
        init_steps -= 1
        if my_world.is_playing():
            if my_world.current_time_step_index == 0:
                my_world.reset()
                ur5._robot.disable_gravity()
            else:
                ur5.set_to_default_state()

    apply_scene_config(config, pickable_objects, placable_objects, my_world)
    for _ in range(10):
        my_world.step()
    return pp_controller


# -----------------------------------------------------------------------------
# Policy connection + output dirs
# -----------------------------------------------------------------------------
policy = get_shm_proxy(uri_name=opt.uri, ns_port=opt.port)

if opt.save_dir is None:
    raise ValueError("--save_dir is required")
save_dir        = opt.save_dir
save_vid_dir    = os.path.join(save_dir, "videos")
save_metric_dir = os.path.join(save_dir, "metrics")
if opt.save:
    os.makedirs(save_vid_dir,    exist_ok=True)
    os.makedirs(save_metric_dir, exist_ok=True)
    print(f"[INFO] videos saved to {save_vid_dir}")
    print(f"[INFO] metrics saved to {save_metric_dir}")


# -----------------------------------------------------------------------------
# Episode rollout
# -----------------------------------------------------------------------------
GRIPPER_THRESH = 0.8
N_ACTION_STEPS = 8           # number of policy actions to apply per query
MAX_STEPS      = 1000        # per-episode hard cap on simulation steps
POLICY_SET_CONFIG = SETTING.get("policy_set_config")


def sample_one_episode(seed: int):
    """Run a single episode at combination index `seed`. Returns final XY distance."""
    np.random.seed(seed)
    random.seed(seed)

    comb = valid_combinations[seed]
    pickable_objects, proposals, pick_languages = setup_pickable_objects(comb)
    placable_objects, place_languages          = setup_placeable_objects(PLACEABLE_SPECS)

    for obj in (placable_objects + pickable_objects):
        utils.loaded_usd_texture_path_rel2abs(obj.prim)
        my_world.scene.add(obj)

    current_config = sample_scene_config(
        len(placable_objects), len(pickable_objects), pickable_objects, proposals, my_world,
        non_target_objs=non_target_objs,
        place_radius=PLACE_RADIUS,
        extra_rot_x=EXTRA_ROT_X,
    )
    pp_controller = init_world_from_config(
        current_config, pickable_objects, placable_objects, proposals)
    camera_e2h.set_wcT(sample_cam_extr())

    # Drive the analytic state machine until the gripper reaches its pregrasp pose,
    # then hand control off to the policy.
    while simulation_app.is_running():
        my_world.step()
        actions, state = pp_controller.get_action()
        if state == pp_controller.State.to_pregrasp:
            break
        ur5.apply_actions(actions)

    pause = False
    current_binary_gripper_state = "open"

    # Build a language prompt and a debug overlay highlighting target + container.
    init_frame = camera_e2h.render(clone=True)
    prompt_rgb = np.ascontiguousarray(init_frame.color[:, :, :3])
    prompt_pick_mask  = init_frame.semantic_mask([pickable_objects[current_config.pick_obj_index].name])
    prompt_place_mask = init_frame.semantic_mask([placable_objects[current_config.place_obj_index].name])
    prompt_bgr        = np.ascontiguousarray(prompt_rgb[:, :, ::-1])
    prompt_debug_img  = draw_prompt(prompt_bgr, [prompt_pick_mask, prompt_place_mask])
    if not opt.no_cv_win:
        cv2.imshow("prompt rgb & mask", prompt_debug_img)
        cv2.waitKey(100)

    template    = "move the {} to the {}"
    prompt_lang = template.format(
        enrich_pick_prompt(pick_languages[current_config.pick_obj_index], prompts),
        place_languages[current_config.place_obj_index],
    )
    print(f"prompt language: {prompt_lang}")

    policy.reset()
    policy.set_prompt(prompt_lang)
    if POLICY_SET_CONFIG:
        policy.set_config(POLICY_SET_CONFIG)

    def _get_obs_frame(show=True):
        """Render an observation tick: state + dual-cam RGB+seg as a dict."""
        obs_frame = {
            "ee_pose": ur5.get_tip_pose()[None, ...],
            "gripper": ur5.get_gripper_norm_width()[None, ...],
        }
        my_world.step(render=True)

        nonlocal current_binary_gripper_state
        current_binary_gripper_state = int(obs_frame["gripper"] > GRIPPER_THRESH)

        frame_e2h = camera_e2h.render(clone=True)
        frame_eih = camera_eih.render(clone=True)
        frame_e2h.timestep = mimic_real_record_steps
        frame_eih.timestep = mimic_real_record_steps

        bgr_e2h = np.ascontiguousarray(frame_e2h.color[:, :, [2, 1, 0]])
        bgr_eih = np.ascontiguousarray(frame_eih.color[:, :, [2, 1, 0]])
        bgr = np.concatenate([bgr_e2h, bgr_eih], axis=1)

        if show and not opt.no_cv_win:
            cv2.imshow("bgr", bgr)
            key = cv2.waitKey(1)
        else:
            key = -1

        obs_frame["e2h_cam"]   = frame_e2h.to_dict()
        obs_frame["eih_cam"]   = frame_eih.to_dict()
        obs_frame["timestamp"] = mimic_real_record_steps
        return obs_frame, bgr, key

    steps                   = 0
    mimic_real_record_steps = 0
    final_dist              = float("inf")
    vid_writer              = None
    if opt.save:
        vid_writer = VideoWriter(
            output_path=os.path.join(save_vid_dir, "{:0>4d}.mp4".format(seed)),
            frame_rate=30,
        )

    obs_frame, debug_bgr, key = _get_obs_frame()
    policy.add_obs_frame(obs_frame)

    while simulation_app.is_running():
        if pause:
            my_world.step()
            obs_frame, debug_bgr, key = _get_obs_frame()
            key = cv2.waitKey(1)
            if key == ord('q'): quit()
            elif key == ord('p'): pause = not pause
            elif key == ord('b'): break
            continue

        prev_binary_gripper_state = current_binary_gripper_state

        # Query policy. Output shapes: (B, Ta, 1) → grab batch 0 / sample 0.
        future_ee_poses, future_grippers, future_time, _ = policy.get_action(sample_num=1)
        all_future_ee_poses = future_ee_poses[:, :, 0]   # (B, Ta, 4, 4)
        all_future_grippers = future_grippers[:, :, 0]   # (B, Ta)
        future_ee_poses     = future_ee_poses[0, :, 0]
        future_grippers     = future_grippers[0, :, 0]

        # If the policy emits actions at a coarser tick than the sim, interpolate
        # in SE(3) for the EE pose and linearly for the gripper state.
        runtime_option   = policy.get_config()
        sample_state_gaps = runtime_option["sample_state_gaps"]

        if sample_state_gaps != 1:
            query_time = mimic_real_record_steps + np.arange(len(future_ee_poses) * sample_state_gaps)
            train_data = {"ee_pose": future_ee_poses, "gripper": future_grippers}
            interp_funcs = {"ee_pose": align.interp_SE3_sep, "gripper": align.interp_linear}
            query_data = align.align_data(query_time, future_time, train_data, interp_funcs)
            all_query_data = []
            for i in range(all_future_ee_poses.shape[0]):
                td = {"ee_pose": all_future_ee_poses[i], "gripper": all_future_grippers[i]}
                all_query_data.append(align.align_data(query_time, future_time, td, interp_funcs))
        else:
            query_data = {"ee_pose": future_ee_poses, "gripper": future_grippers}
            all_query_data = [
                {"ee_pose": all_future_ee_poses[i], "gripper": all_future_grippers[i]}
                for i in range(all_future_ee_poses.shape[0])
            ]

        all_query_ee_poses = np.stack(
            [q["ee_pose"] for q in all_query_data], axis=0).reshape(-1, 4, 4)

        # Snapshot + overlay all sampled trajectories
        obs_frame, debug_bgr, key = _get_obs_frame(show=False)
        debug_bgr = plot_traj(debug_bgr, camera_e2h.get_wcT(),
                              all_query_ee_poses, camera_e2h.opengl_cam.intrinsic.K,
                              color=(0, 0, 255))
        if opt.save:
            vid_writer.write(np.concatenate([prompt_debug_img, debug_bgr], axis=1))
        if opt.no_cv_win:
            key = -1
        else:
            cv2.imshow("bgr", debug_bgr)
            key = cv2.waitKey(1)
        if   key == ord('b'): break
        elif key == ord('q'): quit()

        # Apply the next N_ACTION_STEPS waypoints.
        for i, (ee_pose, finger) in enumerate(
                zip(query_data["ee_pose"][:N_ACTION_STEPS],
                    query_data["gripper"][:N_ACTION_STEPS])):
            mimic_real_record_steps += 1
            ee_pose[2, 3] = np.clip(ee_pose[2, 3], 0.02, None)

            if steps >= MAX_STEPS:
                break

            # Servo until the gripper reaches `ee_pose` or 50 sim steps elapse.
            for _ in range(50):
                action_body = ur5.move_tip_action(ee_pose)
                action_gripper = ur5.open_gripper_action() if finger > GRIPPER_THRESH \
                                 else ur5.close_gripper_action()
                ur5.apply_actions([action_body, action_gripper])
                obs_frame, debug_bgr, key = _get_obs_frame(show=False)
                steps += 1

                debug_bgr = plot_traj(debug_bgr, camera_e2h.get_wcT(),
                                      query_data["ee_pose"],
                                      camera_e2h.opengl_cam.intrinsic.K,
                                      color=(0, 0, 255))
                if opt.save:
                    vid_writer.write(np.concatenate([prompt_debug_img, debug_bgr], axis=1))
                if opt.no_cv_win:
                    key = -1
                else:
                    cv2.imshow("bgr", debug_bgr)
                    key = cv2.waitKey(1)
                if   key == ord('b'): break
                elif key == ord('q'): quit()
                pp_controller.dt_thresh = 0.005
                if pp_controller.pose_close_to(ee_pose, ur5.get_tip_pose()):
                    break

            policy.add_obs_frame(obs_frame)

        if steps >= MAX_STEPS:           break
        if key == ord('b'):              break
        # Episode ends once the gripper re-opens after a closed-grasp transition.
        if prev_binary_gripper_state == 0 and current_binary_gripper_state == 1:
            break

        grasp_obj = pickable_objects[current_config.pick_obj_index]
        grasp_obj_pos, _ = grasp_obj.get_world_pose()
        final_dist = np.linalg.norm(
            (current_config.place_gripper_pose[:3, 3] - grasp_obj_pos)[:2])
        obj_in_workspace = (
            grasp_obj_pos[0] > -0.1 and grasp_obj_pos[0] < 0.7 and
            grasp_obj_pos[1] > -0.1 and grasp_obj_pos[1] < 0.7)
        if not obj_in_workspace:
            break

    if opt.save:
        vid_writer.finalize()
        metric_path = os.path.join(save_metric_dir, "{:0>4d}.json".format(seed))
        with open(metric_path, "w") as fp:
            json.dump({"dist": float(final_dist)}, fp, indent=4, ensure_ascii=False)

    remove_pickable_objects(pickable_objects, my_world)
    remove_placeable_objects(placable_objects, my_world)
    return final_dist


# -----------------------------------------------------------------------------
# Episode driver
# -----------------------------------------------------------------------------
exclude_samples = list(SETTING["episodes"]["exclude"])
total_episodes  = int(SETTING["episodes"]["total"])

novel_bg_active = SUPPORT_ENV_RAND and opt.novel_bg
if novel_bg_active:
    # uoue: walk a 10-episode window offset by the seed, skipping both the
    # standard exclude list and the per-setting novel_bg exclude list.
    novel_bg_exclude = list((SETTING.get("novel_bg") or {}).get("exclude", []))
    skip = set(exclude_samples) | set(novel_bg_exclude)
    indices = range(opt.seed + 10, opt.seed + 20)
else:
    skip = set(exclude_samples)
    indices = range(total_episodes)

for i in indices:
    if i in skip:
        continue
    sample_one_episode(i)

simulation_app.close()
