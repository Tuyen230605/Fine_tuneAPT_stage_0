# 🚀 LIBERO Evaluation

This folder runs the APT policy against three LIBERO benchmark families with **one shared codebase**:

| Benchmark      | Upstream simulator                                                          | Suites                                                              | Default trials/task |
|----------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------|---------------------|
| **LIBERO**     | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `libero_object`, `libero_spatial`, `libero_goal`, `libero_10`       | 50                  |
| **LIBERO-PRO** | [Zxy-MLlab/LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO)             | the 4 above × {`_swap`, `_task`} suffixes — **8 suites total**      | 50                  |
| **LIBERO-plus**| [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)             | `libero_object`, `libero_spatial`, `libero_goal`, `libero_10`       | 1                   |

The evaluator (`eval_libero_pred.py`) is identical for all three — the benchmark name controls only which simulator the host conda env loads, which suites the wrapper script iterates, and how many initial-state trials each task gets.

## 📦 0. Environment setup

Each benchmark uses its own conda env so MuJoCo / robosuite / `libero` versions stay isolated.

### LIBERO (vanilla)
```bash
# Follow https://github.com/Lifelong-Robot-Learning/LIBERO   (Python 3.10 recommended)
conda create -n libero python=3.10 -y && conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && cd LIBERO && pip install -e . && cd ..

# Extra packages required by `eval_libero_pred.py`
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4 mujoco==3.2.3

# `experiments.robot.libero.libero_utils` lives in OpenVLA-OFT — clone it next to APT/
git clone https://github.com/openvla-oft/openvla-oft.git
export PYTHONPATH=$(pwd)/openvla-oft:$PYTHONPATH
```

### LIBERO-PRO
Same as LIBERO above but use the PRO fork as the simulator:
```bash
conda create -n libero-pro python=3.10 -y && conda activate libero-pro
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git && cd LIBERO-PRO && pip install -e . && cd ..
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4 mujoco==3.2.3
git clone https://github.com/openvla-oft/openvla-oft.git
export PYTHONPATH=$(pwd)/openvla-oft:$PYTHONPATH
```

### LIBERO-plus
```bash
conda create -n libero-plus python=3.10 -y && conda activate libero-plus
git clone https://github.com/sylvestf/LIBERO-plus.git && cd LIBERO-plus && pip install -e . && cd ..
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4 mujoco==3.2.3
git clone https://github.com/openvla-oft/openvla-oft.git
export PYTHONPATH=$(pwd)/openvla-oft:$PYTHONPATH
```

> The three forks all expose the `libero.libero.benchmark` module with compatible suite names; `eval_libero_pred.py` resolves the suite at runtime via `benchmark.get_benchmark_dict()[args.libero_task_suite]()`.

## 🚀 1. Launch the APT policy server

In a **separate terminal** with the APT conda env activated, start the inference service (see [`../../README.md`](../../README.md#inference) for the full command):

```bash
pyro4-ns -p 9091 &     # naming server
python -m apt.infer.remote_service \
    --ckpt /path/to/apt_vla/ckpt_latest.pt \
    --uri  control \
    --host localhost --port 0 \
    --ensemble 4
```

Take note of the `--uri` (`control` above) and the naming-server port (`9091`).

## 🧪 2. Run the evaluator

Switch back to the LIBERO env terminal, **from the repository root**, then:

```bash
# Vanilla LIBERO — 4 suites × 50 trials each
bash examples/libero/test_libero.sh \
    --benchmark libero \
    --gpu 0 \
    --model_name apt_vla \
    --controller_name control --controller_port 9091

# LIBERO-PRO — 8 suites × 50 trials each
bash examples/libero/test_libero.sh \
    --benchmark libero-pro \
    --gpu 0 \
    --model_name apt_vla \
    --controller_name control --controller_port 9091

# LIBERO-plus — 4 suites × 1 trial each
bash examples/libero/test_libero.sh \
    --benchmark libero-plus \
    --gpu 0 \
    --model_name apt_vla \
    --controller_name control --controller_port 9091
```

### Useful overrides

| Flag                          | Effect                                                                              |
|-------------------------------|-------------------------------------------------------------------------------------|
| `--suites "libero_object libero_goal"` | Run only the listed suites                                              |
| `--num_trials_per_task N`     | Override the benchmark's default trial count                                        |
| `--begin_task_id N`           | Resume mid-suite by skipping the first N tasks                                      |
| `--save_root  PATH`           | Where to drop per-suite CSV success logs                                            |
| `--video_root PATH`           | Where to drop per-episode MP4s                                                      |
| `--no-save`, `--no-video`     | Disable artefact recording                                                          |

## 📂 Output layout

```
<save_root>/<suite>_<suffix>/<model_name>.csv          # one CSV per suite
<video_root>/<suite>_<suffix>/<model_name>/task<id>/ep<id>.mp4
```

Each CSV row records `task_id, episode_id, success, prompt`; the final row appends the suite-level mean success rate.

## 📚 Files in this folder

| File                      | Purpose                                                                       |
|---------------------------|-------------------------------------------------------------------------------|
| `eval_libero_pred.py`     | Connects to the APT policy server and runs the rollout loop                   |
| `align.py`                | SE(3)/SO(3) interpolation between predicted waypoints and env timestep grid   |
| `video_writer.py`         | Wraps `cv2.VideoWriter` and re-encodes with ffmpeg/H.264 for browser playback |
| `test_libero.sh`          | Per-benchmark wrapper (the entry-point above)                                 |
