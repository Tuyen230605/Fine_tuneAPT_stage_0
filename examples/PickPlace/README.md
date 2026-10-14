# PickPlace Benchmark

A pick-and-place benchmark for evaluating Vision-Language-Action (VLA) policies inside NVIDIA Isaac Sim. A UR5 + Robotiq 85 arm executes language-conditioned pick-and-place tasks on a tabletop scene, and a remote policy server is queried for actions at each control step.

## Evaluation settings

Each setting runs a fixed list of object-combination seeds and records per-episode videos + a final-distance metric.

| Setting | Command | Description |
|---------|---------|-------------|
| **so** — seen object       | `python -m eval.eval_policy --setting so ...` | Object set seen during training, default lighting / ground |
| **uo** — unseen object     | `python -m eval.eval_policy --setting uo ...` | Object set held out from training, default lighting / ground |
| **uc** — unseen container  | `python -m eval.eval_policy --setting uc ...` | Same objects as `so`, but the target container is a held-out mug instead of plate/bowl |
| **uoue** — unseen object + unseen env | `python -m eval.eval_policy --setting uo --novel_bg --random_env --seed N` | Unseen objects plus novel HDR background and randomized ground USD; `eval_all.sh` sweeps seeds 0/20/30/40/50/61/71 |

Each setting's object set, conflicts, language prompts, target containers, episode count, and exclude list are defined in `eval/configs/<setting>.yaml`. To tweak the benchmark (different objects, prompts, etc.), edit the YAML; you should not need to touch `eval/eval_policy.py`.

## Directory layout

```
PickPlace/
├── eval_all.sh                          # top-level driver for all 4 settings
├── eval/
│   ├── eval_policy.py                   # unified benchmark entry point
│   ├── summary.py                       # aggregate per-episode metrics → per-setting SR table
│   └── configs/                         # per-setting YAML configs (so/uo/uc)
├── pp_env/                              # PickPlace-environment package
│   ├── pick_place.py                    # PickPlace state machine
│   ├── scene.py                         # per-episode scene construction + sampling
│   ├── combo.py                         # prompt sampling + combination validity
│   ├── objs/                            # ObjBase subclasses (cans / blocks / ycb / graspnet)
│   └── random_envs/                     # plates, backgrounds, environments
├── utils/                               # task-agnostic helpers
│   ├── sample2d.py                      # collision-free 2D pose sampling
│   ├── poses.py                         # camera + gripper pose sampling
│   ├── align.py                         # SE3 / linear interpolation utilities
│   ├── viz.py                           # OpenCV trajectory + prompt overlays
│   └── video_writer.py                  # OpenCV VideoWriter wrapper
├── isaac_env/                           # Isaac Sim helpers (camera, ground, USD utilities)
│   └── non_isaac/                       # geometry code that does not depend on Isaac
└── assets/
    ├── mesh/                            # USD / OBJ assets (gitignored — see below)
    ├── hdr/                             # environment HDRIs (gitignored — see below)
    └── ur5gp85/                         # UR5 + Robotiq 85 wrapper Python package + USD
```

## Assets

> **TODO**: upload `assets/mesh/` and `assets/hdr/` archives to HuggingFace.

The `assets/mesh/` and `assets/hdr/` directories are not included in the git repository because they total several hundred MB of binary assets. Download them separately and unpack into the repository root so the layout looks like `PickPlace/assets/mesh/...` and `PickPlace/assets/hdr/...`. The exact subdirectories and HDR filenames are referenced by `pp_env/objs/__init__.py`, `pp_env/random_envs/{plates,backgrounds,environments}.py`, and the eval entry points; if any required file is missing the Isaac Sim script will fail when it tries to add a reference to the stage.

## Dependencies

Two separate Python environments are used and **must not be mixed**:

1. **Isaac Sim's bundled Python** — runs everything under `eval/`, `pp_env/`, `utils/`, `isaac_env/`, `assets/ur5gp85/`. Tested against NVIDIA Isaac Sim 4.1. Required Python packages (already shipped with Isaac Sim): `numpy`, `scipy`, `opencv-python`, `matplotlib`, `pillow`, `filelock`, `einops`, `pypose`, `torch`, `trimesh`, `Pyro4`, `pyyaml`.
2. **The model-service environment** — runs the VLA policy via `shm_transport`'s `@expose()` interface. Only `Pyro4`, `numpy`, and whatever the model itself needs.

The Pyro4 nameserver can run in either environment.

## Running an evaluation

Three processes must be started in order:

### 1. Pyro4 nameserver

```bash
pyro4-ns --port 9091
```

### 2. Model service (in the model environment)

The model server must expose its policy via `shm_transport.expose` and register the proxy under a URI such as `control`:

```bash
CUDA_VISIBLE_DEVICES=<GPU_FOR_MODEL> python -m <your_model_module>.remote_service
```

The service must implement at least: `reset()`, `set_prompt(str)`, `set_config(str)`, `add_obs_frame(dict)`, `get_action(sample_num=...) -> (future_ee_poses, future_grippers, future_time, _)`, `get_config() -> {"sample_state_gaps": int}`.

### 3. Evaluation driver (in the Isaac Sim environment)

Run **from inside `examples/PickPlace/`**.

`shm_transport` is **not vendored** here; it lives at the APT repository root (`APT/shm_transport/`). The shell driver (`eval_all.sh`) automatically prepends `APT/` to `PYTHONPATH` so `from shm_transport import ...` resolves. If you launch `eval/eval_policy.py` directly without the driver, do this yourself:

```bash
export PYTHONPATH="$(cd ../.. && pwd):$PYTHONPATH"
```

Full benchmark:

```bash
cd examples/PickPlace
GPU_ID=0 URI=control PORT=9091 SAVE_DIR=./data/exp_results/myrun \
    bash eval_all.sh
```

Or a single setting:

```bash
cd examples/PickPlace
export PYTHONPATH="$(cd ../.. && pwd):$PYTHONPATH"
python -m eval.eval_policy --setting so \
    -i 0 --uri control --port 9091 --no_gui --save \
    --save_dir ./data/exp_results/myrun/so
```

**Important**: do **not** prefix the eval scripts with `CUDA_VISIBLE_DEVICES=`. Each script sets the variable from its own `-i` flag before importing `omni`; an external value gets overridden and may select the wrong device.

### Common flags

| Flag                 | Description |
|----------------------|-------------|
| `--setting`          | One of `so`, `uo`, `uc` (required) — selects `eval/configs/<setting>.yaml` |
| `-i, --index`        | GPU index for Isaac Sim (required; also sets `CUDA_VISIBLE_DEVICES`) |
| `--uri`              | Pyro URI of the policy service (default `control`) |
| `--port`             | Pyro nameserver port (default `9091`) |
| `--no_gui`           | Run Isaac Sim headless |
| `--no_cv_win`        | Disable OpenCV preview windows |
| `--save`             | Save per-episode video + metric |
| `--save_dir DIR`     | Output directory |

Settings whose YAML has `support_env_randomization: true` (currently only `uo`) additionally accept `--random_plate`, `--random_bg`, `--random_env`, `--novel_plate`, `--novel_bg`, `--novel_env`, `--plate`, `--bg`, `--env`, plus `--list_plates` / `--list_bgs` / `--list_envs` to enumerate available options. `eval_all.sh` uses `--novel_bg --random_env --seed N` to produce the `uoue` split.

## Output format

For each run, results are written under `<save_dir>/{so,uo,uc,uoue}/`:

```
<save_dir>/<setting>/
├── videos/
│   ├── 0000.mp4    # one MP4 per episode index
│   ├── 0001.mp4
│   └── ...
└── metrics/
    ├── 0000.json   # { "dist": float }  final XY distance from grasped object to target place pose
    ├── 0001.json
    └── ...
```

`dist` is the L2 distance (meters) between the gripper's commanded place position and the grasped object's actual position when the rollout terminated.

## Reporting results

Once `eval_all.sh` (or one or more `eval_policy.py` runs) has populated `<save_dir>`, aggregate the per-episode metrics into a per-setting success-rate table:

```bash
cd examples/PickPlace
export PYTHONPATH="$(cd ../.. && pwd):$PYTHONPATH"
python -m eval.summary ./data/exp_results/myrun
```

An episode counts as a success when its `dist` is below `--success_threshold` (default **0.11 m**). Per-setting directories with fewer than `--min_episodes` (default 10) recorded JSONs are skipped to avoid noisy partial-run numbers.
