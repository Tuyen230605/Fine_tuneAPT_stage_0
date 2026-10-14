"""Episode-scene construction and sampling helpers.

These functions are called from `eval/eval.py` once per episode to:
  - spawn pickable + placeable USD objects in the world,
  - sample target / placement poses,
  - apply the sampled poses to live prims and let physics settle.

Functions that touch the live world (`my_world.scene`, `my_world._backend_utils`,
`my_world.step`) take the World instance as an explicit argument rather than
reaching for a module global. This keeps `pp_env.scene` import-time
independent of the eval driver's runtime state.

This module imports `omni.isaac.*` at the top level, so it must only be
imported AFTER the eval driver has constructed its `SimulationApp(...)`.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Mapping, Sequence
from scipy.spatial.transform import Rotation

from omni.isaac.core.utils.semantics import add_update_semantics
import omni.isaac.core.utils.prims as prims_utils

from isaac_env import utils as isaac_utils
from pp_env import objs as objs_registry
from utils.sample2d import sample_obj_xys2


@dataclass
class SceneConfig(object):
    """Frozen per-episode pose plan produced by `sample_scene_config`."""
    place_obj_poses:    np.ndarray  # (N_place, 4, 4)
    place_obj_index:    int
    place_gripper_pose: np.ndarray  # (4, 4)
    pick_obj_poses:     list        # list of (pos, orn) pairs
    pick_obj_index:     int
    flip_grasp:         bool


# -----------------------------------------------------------------------------
# Object spawn / despawn
# -----------------------------------------------------------------------------
def setup_pickable_objects(comb: Sequence[str]):
    """Spawn the pickable objects in `comb` at a holding position above the table.

    Returns a tuple ``(pickable_objects, proposals, pick_languages)`` of equal length.
    """
    pickable_objects, proposals, pick_languages = [], [], []
    for obj_name in comb:
        obj_class = getattr(objs_registry, obj_name)
        usd_obj = isaac_utils.add_object(
            usd_path=obj_class.USD_PATH,
            prim_path=f"/World/interference_objs/{obj_name.lower()}",
            name=obj_name.lower(),
            fixed=False, collision=True, approx="none", mass=0.1,
            scale=obj_class.SCALE,
            position=np.array([0.5, 0, 0.5]),
            orientation=np.array([1, 0, 0, 0]),
        )
        add_update_semantics(usd_obj.prim, usd_obj.name)
        pickable_objects.append(usd_obj)
        proposals.append(obj_class())
        pick_languages.append(obj_name)
    return pickable_objects, proposals, pick_languages


def remove_pickable_objects(pickable_objects, my_world):
    """Remove every prim in `pickable_objects` from the scene and the USD stage."""
    for obj in pickable_objects:
        my_world.scene.remove_object(obj.name)
        if obj.prim.IsValid():
            prims_utils.delete_prim(obj.prim_path)


def setup_placeable_objects(placeable_specs: Sequence[Mapping]):
    """Spawn the target containers declared in `placeable_specs` (from YAML).

    Each spec must provide ``usd``, ``name``, ``language``, ``scale`` keys; see
    `eval/configs/so.yaml` for the schema.
    """
    placable_objects, place_languages = [], []
    for spec in placeable_specs:
        obj = isaac_utils.add_object(
            usd_path=spec["usd"],
            prim_path=f"/World/{spec['name']}",
            name=spec["name"],
            fixed=False, collision=True, disable_stablization=False, approx="none",
            mass=1.0,
            scale=np.array(spec["scale"], dtype=float),
            position=np.array([0.5, 0, 0.1]),
            orientation=np.array([0, 0, 0, 1]),
        )
        add_update_semantics(obj.prim, obj.name)
        placable_objects.append(obj)
        place_languages.append(spec["language"])
    return placable_objects, place_languages


def remove_placeable_objects(placable_objects, my_world):
    """Remove every prim in `placable_objects` from the scene and the USD stage."""
    for obj in placable_objects:
        my_world.scene.remove_object(obj.name)
        if obj.prim.IsValid():
            prims_utils.delete_prim(obj.prim_path)


# -----------------------------------------------------------------------------
# Per-object pose sampling (Isaac-bound; needs `my_world._backend_utils`)
# -----------------------------------------------------------------------------
def sample_obj_pose(pick_xy, my_world, lay_down=True, initial_pose=np.eye(3)):
    """Sample a (pos, quat) for an object, either lying down or standing on `initial_pose`.

    When ``lay_down=True``, rejection-samples random quaternions until the
    object's local Z axis is within 60° of horizontal (the object is "on its
    side"). Otherwise, applies a random yaw in the world XY plane on top of
    ``initial_pose``.
    """
    if lay_down:
        while True:
            rand_orn = np.random.randn(4)
            rand_orn = rand_orn / (np.linalg.norm(rand_orn) + 1e-8)
            rand_rot = my_world._backend_utils.quats_to_rot_matrices(rand_orn)
            zaxis = rand_rot[:3, 2]
            if abs(zaxis[-1]) < 0.5:
                break
        z = 0.05
    else:
        rand_rot = Rotation.from_rotvec([0, 0, np.random.uniform(0, np.pi*2)]).as_matrix() @ initial_pose
        rand_orn = my_world._backend_utils.rot_matrices_to_quats(rand_rot)
        z = 0.0
    return np.concatenate([pick_xy, [z]]), rand_orn


def sample_place_pose(place_xy, my_world):
    """Sample a final-place gripper pose centered at `place_xy`, gripper pointing straight down.

    Gripper Z is fixed at 0.04 m above the table; orientation is a random yaw
    composed with the canonical "tip pointing down" rotation.
    """
    place_pose = np.eye(4)
    place_pose[:2, 3] = place_xy
    place_pose[2, 3]  = 0.04

    place_pose[:3, :3] = my_world._backend_utils.quats_to_rot_matrices(np.array([0, 1.0, 0, 0]))
    place_pose[:3, :3] = place_pose[:3, :3] @ Rotation.from_rotvec(
        [0, 0, np.pi * 2 * np.random.rand()]).as_matrix()
    return place_pose


# -----------------------------------------------------------------------------
# Whole-scene sampling and application
# -----------------------------------------------------------------------------
def sample_scene_config(
    N_place: int,
    N_pick: int,
    pickable_objects,
    proposals,
    my_world,
    *,
    non_target_objs: Sequence[str],
    place_radius: float,
    extra_rot_x: Sequence[float],
):
    """Sample one `SceneConfig` for the current episode.

    Args:
        N_place / N_pick:   counts of placeable / pickable objects already spawned.
        pickable_objects:   spawned `UsdObject`s used to test `non_target_objs` membership.
        proposals:          grasp-proposal instances (one per pickable) used for ``initial_pose``.
        my_world:           live World, used only for backend quat <-> matrix conversions.
        non_target_objs:    lower-cased prim names that may appear but must not be the pick target.
        place_radius:       min-separation radius (m) for placeable XY sampling.
        extra_rot_x:        per-placeable extra X-axis rotation (degrees) applied after the
                            random yaw (e.g. ``-90`` to flip a bowl upright).
    """
    pick_xys, place_xys = sample_obj_xys2(N_pick, N_place,
                                          pick_raidus=0.07,
                                          place_raidus=place_radius)

    # Random yaw for each placeable, optionally followed by a per-placeable
    # extra rotation about X (e.g. -90° to flip a bowl upright).
    place_obj_poses = np.eye(4)[None].repeat(N_place, axis=0)
    place_obj_poses[:, :2, 3] = place_xys
    place_obj_poses[:,  2, 3] = 0.05
    rand_rotvec = np.zeros((N_place, 3))
    rand_rotvec[:, -1] = np.random.rand(N_place) * np.pi*2
    place_obj_poses[:, :3, :3] = Rotation.from_rotvec(rand_rotvec).as_matrix()
    for i, deg in enumerate(extra_rot_x):
        if deg != 0.0 and i < N_place:
            place_obj_poses[i, :3, :3] = place_obj_poses[i, :3, :3] @ \
                Rotation.from_rotvec([np.deg2rad(deg), 0, 0]).as_matrix()

    place_obj_index    = np.random.randint(0, N_place)
    place_gripper_pose = sample_place_pose(place_xys[place_obj_index], my_world)

    pick_obj_poses = [
        sample_obj_pose(pick_xys[i], my_world, lay_down=False,
                        initial_pose=proposals[i].initial_pose())
        for i in range(N_pick)
    ]
    pick_obj_index = np.random.randint(0, N_pick)
    while pickable_objects[pick_obj_index].name in non_target_objs:
        pick_obj_index = np.random.randint(0, N_pick)
    flip_grasp = np.random.rand() < 0.5

    return SceneConfig(
        place_obj_poses, place_obj_index, place_gripper_pose,
        pick_obj_poses, pick_obj_index, flip_grasp,
    )


def apply_scene_config(
    config: SceneConfig,
    pickable_objects: List[isaac_utils.UsdObject],
    placable_objects: List[isaac_utils.UsdObject],
    my_world,
):
    """Push `config`'s sampled poses to the live prims and step twice to settle."""
    def _apply_pose():
        for i, obj in enumerate(pickable_objects):
            obj.set_world_pose(*config.pick_obj_poses[i])
        for i, obj in enumerate(placable_objects):
            pos = config.place_obj_poses[i][:3, 3]
            orn = my_world._backend_utils.rot_matrices_to_quats(config.place_obj_poses[i][:3, :3])
            obj.set_world_pose(pos, orn)

    for _ in range(2):
        _apply_pose()
        my_world.step()
