<p align="center">
  <img src="assets/logo.png" alt="APT" width="40%">
</p>

<div align="center">
<h2 style="border-bottom: none; margin-bottom: 0px ">APT: Action Expert Pretraining<br>Improves Instruction Generalization of Vision-Language-Action Policies</h2>

[Kechun Xu](https://xukechun.github.io/) · [Zhenjie Zhu]() · [Anzhe Chen]() · [Rong Xiong](https://scholar.google.com/citations?user=1hI9bqUAAAAJ&hl=en) · [Yue Wang](https://ywang-zju.github.io/)

<a href="https://arxiv.org/pdf/2606.12366"><img src='https://img.shields.io/badge/Paper-APT-red' alt='Paper PDF'></a>
<a href='https://xukechun.github.io/papers/APT'><img src='https://img.shields.io/badge/Project_Page-APT-green' alt='Project Page'></a>

</div>

**TL; DR**: APT factorizes the VLA policy into a Vision-Action (VA) prior and a language-conditioned VLA likelihood, and pretrains the action expert as the VA prior on vision-action pairs from a frozen VLM. A layer-wise gated fusion mechanism then injects language tokens into the pretrained action expert, preserving the visuomotor prior while enabling instruction following. APT delivers consistent gains on OOD language instructions and compositional tasks.

## 🏆 Highlights

🔍 **Key Findings**: continuous-action VLA policies start from a randomly initialized action expert and learn from imbalanced VLA data, producing noisy gradients that corrupt the VLM backbone and collapse to visual shortcuts.

✨ **Key Insights**:
- **Bayesian factorization** of the VLA policy:

$$
\pi(\mathbf{a}\mid\mathbf{v},\ell)\ \propto\ \pi^{p}(\mathbf{a}\mid\mathbf{v})\cdot L(\ell\mid\mathbf{v},\mathbf{a})
$$

  - **VA prior** $\pi^p(\mathbf{a}\mid\mathbf{v})$ is trained on **balanced** vision-action pairs alone, so the action expert builds coherent visuomotor priors without any language shortcut.
  - **VLA likelihood** $L(\ell\mid\mathbf{v},\mathbf{a})$ then aligns the prior to language instructions, a much easier sub-problem than learning action generation and language grounding jointly.

- **Layer-wise gated fusion** injects each Qwen3-VL intermediate feature into the corresponding action-expert self-attention layer through a learnable sigmoid gate, letting the action expert inherit VLM semantics without overwriting the pretrained visuomotor pathway.

- **Two-stage realization** inside one network. Stage 1 activates only half of the action-expert attention layers and masks language tokens, training a pure VA prior. Stage 2 inserts an interleaved attention layer after each Stage-1 layer, unmasks language, and jointly trains the prior and likelihood under large-scale data.

- **Architecture-agnostic**: the two-stage recipe also boosts $\pi$-style and GR00T-style architectures on OOD language generalization.

## 🧩 Overview

Given VLA datasets with modality imbalance, APT trains the policy in two stages:

- **Stage 1 - VA Prior Pretraining**: the action expert is conditioned solely on visual tokens from a frozen Qwen3-VL backbone and learns $\pi^p(\mathbf{a}\mid\mathbf{v})$.
- **Stage 2 - VLA Likelihood Alignment**: the Stage-1 layers are duplicated with interleaved language-injection layers; the full policy is jointly trained on the same data.

<p align="center">
  <img src="assets/method.png" alt="APT method overview" width="85%">
</p>

## 📁 Project Structure

```
APT/
├── apt/                       # core model + unified trainer (this is the package)
│   ├── vla.py                 # VLA wrapper (VLM + ActionExpert)
│   ├── vlm.py                 # Qwen3-VL encoder bridge
│   ├── action_expert.py       # diffusion-based action expert with gated fusion
│   ├── action_transform.py    # SE(3) ↔ 10-dim action conversions
│   ├── configs.py             # TrainConfig + CONFIGS registry
│   ├── train.py               # unified DDP + DeepSpeed trainer
│   ├── ds_config_zero2.json   # DeepSpeed ZeRO-2 config
│   ├── ds_config_zero3.json   # DeepSpeed ZeRO-3 config
│   ├── encoders/              # Qwen3-VL (LoRA-capable) wrapper
│   ├── layers/                # attention / RoPE / norms / 6D-rotation utils
│   └── infer/                 # planner + remote inference service
├── data_utils/                # HDF5 IO, datasets, video decoding, distributed samplers
├── train_utils/               # EMA implementation
├── infer_utils/               # trajectory ensembler and visualizer
├── shm_transport/             # Pyro4 + shared-memory RPC for remote inference
├── scripts/
│   ├── train.sh               # two-stage pretraining (DDP or DeepSpeed)
│   └── finetune.sh            # task-specific fine-tuning (DDP or DeepSpeed)
├── examples/
│   ├── libero/                # LIBERO / LIBERO-PRO / LIBERO-plus evaluator
│   └── PickPlace/             # Isaac Sim pick-and-place benchmark (UR5 + Robotiq 85)
├── assets/                    # logo / method (PDF source + PNG for README), paper.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## 📘 Usage

### Environment

```bash
conda create -n apt python=3.10 -y
conda activate apt
# Install a PyTorch build that matches your CUDA. The pinned xformers requires
# torch 2.4.1; relax it if you use a different torch version.
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

> **Note on `xformers` / `deepspeed`**: both are version-sensitive. If you do not need DeepSpeed, you can skip installing it - `--backend ddp` works without it. Likewise drop `xformers` if you do not need its kernels.

### Data Preparation

APT consumes trajectories stored as per-episode HDF5 files. Each sample yielded by the dataloader looks like:

```python
{
    "obs_rgbs":             (To, ncam, 3, H, W),  # observation frames
    "prompt_text":          str,                  # task description
    "current_ee_pose":      (nee, 4, 4),          # current EE pose in world frame
    "gt_future_ee_states":  (Ta, nee, 17),        # ground truth pose + gripper
    "history_ee_states":    (Th, nee, 17),
    "obs_norm_xys":         (...),                # per-pixel normalized 2D coords
    "obs_extrinsics":       (To, ncam, 4, 4),
    "valid_ee_mask":        (nee,),
    ...                                           # see data_utils/dataset_base.py
}
```

**Adding a new dataset:**
1. Subclass `H5DatasetMapBase` in `data_utils/datasets.py`.
2. Register the dataset entry in `data_utils/data_loc.py`. The file looks up the host IP (`get_ipv4_address`) and selects the matching dictionary - **edit it to point at your local paths** before launching training.
3. Reference the new class from a `TrainConfig` entry in `apt/configs.py`.

We expose a number of preset configs (see `apt/configs.py` for the full list). Examples:

| Config                          | Purpose                                                    |
|---------------------------------|------------------------------------------------------------|
| `pretrain`                      | Pretrain on Droid + AgiBotWorld + InternA1 + InternM1      |
| `finetune_aloha_pp_storage`     | Real-world ALOHA pick-place + table-storage fine-tuning    |
| `debug`                         | Tiny single-batch config used for smoke tests              |

### Two-stage Pretraining

`scripts/train.sh` drives both stages and accepts either back-end. Common arguments:

| Flag                | Meaning                                                 |
|---------------------|---------------------------------------------------------|
| `--backend`         | `ddp` (torchrun) or `deepspeed`                         |
| `--gpus`            | Comma-separated GPU IDs, e.g. `0,1,2,3`                 |
| `--stage`           | `0` (VA only), `1` (VLA only), or `both`                |
| `--config`          | Config name from `apt/configs.py`                       |
| `--va-name`         | Save name for the Stage-0 checkpoint                    |
| `--vla-name`        | Save name for the Stage-1 checkpoint                    |
| `--va-conti`        | Resume Stage-0 from an existing checkpoint              |
| `--vla-conti`       | Resume Stage-1 from an existing checkpoint              |
| `--bs / --max-iter` | Per-GPU batch size / max iterations                     |

DeepSpeed-only flags (ignored when `--backend ddp`):

| Flag             | Meaning                                            |
|------------------|----------------------------------------------------|
| `--ds-zero 2\|3` | ZeRO stage (selects `ds_config_zero{2,3}.json`)    |
| `--accum N`      | Gradient accumulation steps                        |
| `--vlm-mode`     | `frozen` / `lora` / `full` VLM finetune mode       |
| `--vlm-lr`       | Separate learning rate for VLM parameters          |
| `--gc`           | Enable gradient checkpointing                      |

**DDP (torchrun) + Fix VLM, Stage-0 only:**
```bash
bash scripts/train.sh --backend ddp --gpus 0,1,2,3 --stage 0 \
    --config pretrain \
    --va-name apt_va --vla-name apt_vla
```

**DeepSpeed ZeRO-2 + Full VLM, Stage-1 only:**
```bash
bash scripts/train.sh --backend deepspeed --gpus 0,1,2,3 --stage 1 \
    --config pretrain \
    --va-name apt_va --vla-name apt_vla_vlmft \
    --ds-zero 2 --vlm-mode full --gc --vlm-lr 1e-5
```

**Resume Stage-1 after preemption:**
```bash
bash scripts/train.sh --backend deepspeed --gpus 0,1,2,3 --stage 1 \
    --config pretrain --vla-conti apt_vla_vlmft \
    --ds-zero 2 --vlm-mode full --gc --vlm-lr 1e-5
```

Stage-1 internally calls `VLA.load_from_pretrain(..., load_from_va=True)`, which doubles the Stage-1 attention layers and copies the Stage-0 weights into the odd indices while leaving the inserted (even-index) language-injection layers randomly initialized.

### Task-specific Fine-tuning

`scripts/finetune.sh` mirrors `train.sh` but additionally exposes `--pretrained-ckpt` so you can bootstrap from any pretraining checkpoint. The Stage-1 launch automatically reuses the Stage-0 name (if any) or the pretraining checkpoint as the upstream.

```bash
# Stage-1 only, DeepSpeed ZeRO-2 + Full VLM on Pick-Place from a pretrained VLA checkpoint
bash scripts/finetune.sh --backend deepspeed --gpus 0,1,2,3 --stage 1 \
    --config finetune_pp --pretrained-ckpt apt_vla \
    --vla-name ft_apt_pp \
    --ds-zero 3 --vlm-mode lora --gc --accum 4 --bs 64 --vlm-lr 1e-5

# Stage-1 only, Fix VLM on real ALOHA data from a pretrained VLA checkpoint
bash scripts/finetune.sh --backend ddp --gpus 0,1,2,3 --stage 1 \
    --config finetune_aloha_pp_storage --pretrained-ckpt apt_vla_vlmft \
    --vla-name ft_apt_aloha
```

Checkpoints are written under `./checkpoints/APT/<name>/` and TensorBoard logs under `./logs/APT/<name>/`. Both roots are configurable per run via the optional flags `--ckpt_dir /path/to/ckpts` and `--log_dir /path/to/logs`, e.g. to keep separate experiments on different volumes. Likewise `--dataloader_timeout <seconds>` (default 300) lets you raise the DataLoader worker timeout for slow shared storage.

### Loading existing APT checkpoints

The merged trainer is backwards-compatible with checkpoints produced by the pre-refactor training scripts. A checkpoint is recognised by its file layout:

| Saved by                | Top-level keys in `ckpt_latest.pt`                                          |
|-------------------------|------------------------------------------------------------------------------|
| DDP (`train_dist.py`)   | `weights`, `optimizer`, `scheduler`, `scaler`, `current_iters`, ...          |
| DeepSpeed (`train_deepspeed.py`) | `weights`, `vlm_weights` (if VLM fine-tuned), no embedded optimizer |

The merged `apt.train` accepts both, and you can also switch back-ends across resumes (e.g. resume a DeepSpeed-trained run under DDP). When a checkpoint's optimizer state cannot be re-loaded (e.g. param groups differ because `--vlm-mode` changed), the trainer logs a warning and continues with a fresh optimizer.

**Pre-flight check** - validate any existing checkpoint before launching training:

```bash
# Stage-0 VA checkpoint
python scripts/test_ckpt.py \
    --ckpt /path/to/apt_ckpt_dir/ckpt_latest.pt --train-stage 0

# Same VA checkpoint used to bootstrap Stage-1
python scripts/test_ckpt.py \
    --ckpt /path/to/apt_ckpt_dir/ckpt_latest.pt --train-stage 1 --load-from-va

# Stage-1 VLA checkpoint (already-trained policy)
python scripts/test_ckpt.py \
    --ckpt /path/to/apt_ckpt_dir/ckpt_latest.pt --train-stage 1
```

**Bootstrap a new fine-tuning run from an existing VLA checkpoint (most common):**

```bash
bash scripts/finetune.sh --backend deepspeed --gpus 0,1,2,3 --stage 1 \
    --config finetune_pp \
    --pretrained-ckpt /path/to/apt_ckpt_dir/ckpt_latest.pt \
    --vla-name ft_apt_pp --vlm-mode full
```

`--pretrained-ckpt` accepts either a checkpoint subdir name under `./checkpoints/APT/` **or** an absolute path ending in `.pt`. The trainer starts with `current_iters=0` so the new run gets a clean iteration counter, and saves under `./checkpoints/APT/<vla-name>/`.

### Inference

For local inference, instantiate the planner directly:

```python
from apt.infer.planner import TrajPlanner

planner = TrajPlanner(
    ckpt_path="checkpoints/APT/ft_vla/ckpt_latest.pt",
    device="cuda:0",
    ensemble=4,
    use_ema=False,
)
planner.set_prompt("Pick up the grape and place it on the pink box.")
planner.add_obs_frame(obs_frame)
actions = planner.get_action()
```

To serve the policy as a remote service (e.g. for hardware control), first launch a Pyro4 naming server, then start the service:

```bash
# 1. Naming server (defaults to localhost:9091)
pyro4-ns -p 9091

# 2. Inference service
python -m apt.infer.remote_service \
    --ckpt checkpoints/APT/ft_vla/ckpt_latest.pt \
    --uri apt_control \
    --host localhost --port 0 \
    --ensemble 4
```

The client side uses `shm_transport` (zero-copy shared memory + Pyro4) to call `add_obs_frame`, `set_prompt`, `get_action`, etc.

## 🧪 Evaluation on LIBERO benchmarks

A single evaluator under [`examples/libero/`](examples/libero/) drives all three LIBERO benchmark families against an APT policy server:

| Benchmark      | Simulator                                                                       | Suites                                                            | Default trials/task |
|----------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------|---------------------|
| **LIBERO**     | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `libero_{object,spatial,goal,10}` — 4 suites                      | 50                  |
| **LIBERO-PRO** | [Zxy-MLlab/LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO)                 | the 4 above × {`_swap`, `_task`} — 8 suites                       | 50                  |
| **LIBERO-plus**| [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)                 | `libero_{object,spatial,goal,10}` — 4 suites                      | 1                   |

Quick start (LIBERO conda env, after launching the APT policy server in a separate terminal):

```bash
bash examples/libero/test_libero.sh \
    --benchmark libero \
    --gpu 0 \
    --model_name apt_vla \
    --controller_name control --controller_port 9091
```

See [`examples/libero/README.md`](examples/libero/README.md) for the per-benchmark conda setup, the policy-server launch command, the full list of `test_libero.sh` flags, and the on-disk output layout.

## 🤖 Evaluation on Pick-and-Place in Isaac Sim

An Isaac-Sim-based pick-and-place benchmark lives under [`examples/PickPlace/`](examples/PickPlace/). A UR5 + Robotiq 85 arm executes language-conditioned pick-and-place on a tabletop scene; the APT policy server is queried for actions each control step.

| Setting   | Description                                                                   |
|-----------|-------------------------------------------------------------------------------|
| **so**    | Seen object set, default lighting / ground                                    |
| **uo**    | Held-out object set, default lighting / ground                                |
| **uc**    | Same objects as `so`, but the target container is a held-out mug              |
| **uoue**  | Held-out objects + novel HDR background + randomized ground (seed sweep)      |

Two separate Python environments are involved: the APT env (running the policy server with `shm_transport`) and the Isaac Sim bundled Python (running the benchmark). `shm_transport` lives at the APT root and is picked up automatically by the driver via `PYTHONPATH`. Quick start:

```bash
# 1. APT env (separate terminal) — launch the policy server, see "Inference" above.
# 2. Isaac Sim env — drive all 4 settings:
cd examples/PickPlace
GPU_ID=0 URI=control PORT=9091 SAVE_DIR=./data/exp_results/myrun \
    bash eval_all.sh
```

See [`examples/PickPlace/README.md`](examples/PickPlace/README.md) for the asset preparation, full environment requirements, per-setting flags, and the on-disk output format (`<save_dir>/<setting>/videos/*.mp4` + `metrics/*.json`).

## 📥 Pretrained Checkpoints

All checkpoints live under a single Hugging Face repo: [KechunXu1/apt_models](https://huggingface.co/KechunXu1/apt_models/tree/main).

| Stage                          | Config                          | Datasets                                                      | Hugging Face |
|--------------------------------|---------------------------------|----------------------------------------------------------------|--------------|
| Pretrained VLA policy          | `pretrain` (`--load_from_va`)  | Droid + AgiBotWorld + InternA1 + InternM1                      | [apt_vla](https://huggingface.co/KechunXu1/apt_models/tree/main/apt_vla) |
| LIBERO fine-tuned              | `finetune_libero`               | LIBERO Spatial / Object / Goal / 10                            | [apt_vla_ftlibero](https://huggingface.co/KechunXu1/apt_models/tree/main/apt_vla_ftlibero) |
| Pick-Place fine-tuned          | `finetune_pp`                   | PickPlaceCan                                                   | [apt_vla_ftpp](https://huggingface.co/KechunXu1/apt_models/tree/main/apt_vla_ftpp) |

Download a checkpoint (e.g. the pretrained VLA policy):

```bash
hf download KechunXu1/apt_models --include "apt_vla/*" --local-dir ./checkpoints/APT
```

Then point the inference script at the downloaded checkpoint via `--ckpt ./checkpoints/APT/apt_vla/ckpt_latest.pt` (see the [Inference](#inference) section).

## 🤝 Acknowledgements

This project builds upon [BayesVLA](https://github.com/xukechun/BayesVLA), and [E2VLA](https://github.com/hhcaz/e2vla). We thank these teams for their open-source contributions.

## 📚 Citation

If you find this work useful, please consider citing:

```
@article{xu2026apt,
      title={APT: Action Expert Pretraining Improves Instruction Generalization of Vision-Language-Action Policies},
      author={Xu, Kechun and Zhu, Zhenjie and Chen, Anzhe and Xiong, Rong and Wang, Yue},
      journal={arXiv preprint arXiv:2606.12366},
      year={2026}
    }
```
