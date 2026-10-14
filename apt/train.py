"""
Unified trainer for APT supporting both DDP (torchrun) and DeepSpeed back-ends.

Select the back-end with ``--backend ddp`` (default) or ``--backend deepspeed``.

Launch examples:

  # DDP
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29500 --nproc_per_node 4 \\
      -m apt.train --backend ddp --config pretrain -s my_exp --train_stage 0

  # DeepSpeed (ZeRO-2)
  deepspeed --include localhost:0,1,2,3 --master_port 29500 \\
      --module apt.train --backend deepspeed --ds-config ds_config_zero2.json \\
      --config pretrain -s my_exp --train_stage 0

The two back-ends share argument names, model construction, data loaders,
checkpoint format for actor / VLM weights, EMA and the training loop. Every
back-end-specific decision lives in a ``_Backend`` strategy subclass under
``train_utils.backend``; the ``Trainer`` itself contains no
``if self.backend == ...`` branches.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.amp
import torch.distributed as dist
import torch.multiprocessing as mp
import tyro
from diffusers.optimization import get_scheduler
from torch import Tensor, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_utils.dataset_base import (
    concat_datasets, generate_sample_weights, get_dataloader)
from data_utils.dist_sampler import (
    DistributedMultiplexSampler, DistributedWeightedSampler)
from train_utils.backend import _Backend, make_backend
from train_utils.cpu_affinity import maybe_bind_cpu_affinity
from train_utils.distributed import (
    is_distributed, is_main_process, reduce_metrics)
from train_utils.ema_impl import ExponentialMovingAverage

from . import vla
from .configs import CONFIGS, TrainConfig


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def init_train_config():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", type=str, default="ddp",
                        choices=["ddp", "deepspeed"],
                        help="Distributed back-end. Use 'ddp' with torchrun, "
                             "'deepspeed' with the deepspeed launcher.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-s", dest="save",  type=str, default="",
                        help="experiment name to save under")
    parser.add_argument("-c", dest="conti", type=str, default="",
                        help="experiment name to resume from")
    parser.add_argument("--ds-config", type=str, default="ds_config_zero2.json",
                        help="DeepSpeed JSON config (absolute or relative to "
                             "the apt package). Only used when --backend deepspeed.")
    # DeepSpeed launcher passes --local_rank or --local-rank; argparse needs to
    # accept both spellings transparently so tyro does not choke on it.
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0,
                        help=argparse.SUPPRESS)

    if "-h" in sys.argv or "--help" in sys.argv:
        print("=== argparse help ===")
        parser.print_help()
        print("\n=== tyro help (TrainConfig fields) ===")
        tyro.extras.get_parser(TrainConfig).print_help()
        sys.exit(0)

    args, remaining = parser.parse_known_args()
    cfg = CONFIGS[args.config]
    cfg = tyro.cli(cfg.__class__, default=cfg, args=remaining)

    ds_config_path = args.ds_config
    if not os.path.isabs(ds_config_path):
        ds_config_path = os.path.join(os.path.dirname(__file__), ds_config_path)

    return cfg, args.save, args.conti, args.backend, ds_config_path


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def append(self, val):
        self.sum += val
        self.count += 1

    def avg(self):
        return self.sum / self.count if self.count else 0.0


def count_trainable(m: torch.nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def get_data_loader_for_cfg(cfg: TrainConfig, is_dist: bool):
    """Build a DataLoader (+ optional distributed sampler) from a ``TrainConfig``.

    Two execution paths:

    * ``is_dist=True``: builds a concat-dataset and a distributed sampler
      (multiplex or weighted depending on ``cfg.dataset_weights``); workers
      use the ``spawn`` start method to avoid CUDA-fork crashes.
    * ``is_dist=False``: defers to ``data_utils.dataset_base.get_dataloader``.

    Returns ``(dataloader, sampler_or_None)``.
    """
    if cfg.sample_multiplex > 1:
        assert cfg.dataset_weights is not None, \
            "sample_multiplex requires dataset_weights"

    datasets_list = [D.inst(cfg.train_stage) for D in cfg.dataset_classes]

    if is_dist:
        ds_concat = concat_datasets(
            datasets_list, shuffle=(cfg.dataset_weights is None))
        if cfg.dataset_weights is None:
            sampler = DistributedMultiplexSampler(
                dataset=ds_concat, shuffle=True,
                multiplex=cfg.sample_multiplex)
        else:
            sample_weights = generate_sample_weights(
                datasets_list, cfg.dataset_weights)
            assert len(sample_weights) == len(ds_concat)
            sampler = DistributedWeightedSampler(
                weights=sample_weights,
                num_samples=len(sample_weights) * cfg.sample_multiplex,
                replacement=True)
        # spawn context avoids CUDA-fork worker crashes
        mp_ctx = mp.get_context("spawn") if cfg.workers > 0 else None
        kwargs = dict(
            dataset=ds_concat, batch_size=cfg.bs, sampler=sampler,
            num_workers=cfg.workers, pin_memory=cfg.pin_memory,
            timeout=cfg.dataloader_timeout if cfg.workers > 0 else 0,
        )
        if cfg.workers > 0:
            kwargs["persistent_workers"] = cfg.persistent_workers
            kwargs["prefetch_factor"] = cfg.prefetch_factor
            kwargs["multiprocessing_context"] = mp_ctx
        return DataLoader(**kwargs), sampler
    else:
        sample_weights = (
            generate_sample_weights(datasets_list, cfg.dataset_weights)
            if cfg.dataset_weights is not None else None)
        shuffle = True if cfg.dataset_weights is None else None
        dl = get_dataloader(
            datasets=datasets_list, batch_size=cfg.bs,
            num_workers=cfg.workers, shuffle=shuffle,
            persistent_workers=cfg.persistent_workers,
            pin_memory=cfg.pin_memory,
            prefetch_factor=cfg.prefetch_factor,
            sample_weights=sample_weights,
            sample_multiplex=cfg.sample_multiplex)
        return dl, None


# ──────────────────────────────────────────────────────────────────────────────
# Unified trainer
# ──────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Unified two-stage trainer for APT.

    Holds the model, optimizer, scheduler, EMA, distributed engine and
    checkpoint I/O behind a single object. All back-end-specific decisions
    are delegated to the ``_Backend`` strategy held in ``self.bk``; this
    class contains no ``if self.backend == ...`` branches.
    """

    # Class-level defaults preserved for backward compatibility with any
    # external code that reads them directly; instance attrs below are
    # populated from ``cfg.log_dir`` / ``cfg.ckpt_dir`` in ``__init__``.
    LOG_DIR = "./logs/APT"
    CKPT_DIR = "./checkpoints/APT"

    def __init__(self):
        self.launch_time_str = datetime.now().strftime("%Y%m%d%H%M")
        (self.cfg, save, conti,
         self.backend, self.ds_config_path) = init_train_config()

        # Override class-level paths with the values from cfg (defaults match).
        self.LOG_DIR  = self.cfg.log_dir
        self.CKPT_DIR = self.cfg.ckpt_dir

        # ── Back-end + distributed init ──────────────────────────────────────
        self.bk: _Backend = make_backend(self.backend, self)
        (self.is_dist, self.rank, self.world_size,
         self.local_rank) = self.bk.init_distributed()
        self.model_device = f"cuda:{self.local_rank}" \
            if self.is_dist else "cuda:0"

        maybe_bind_cpu_affinity(self.local_rank, self.cfg)

        if is_main_process():
            print(f"[INFO] Back-end: {self.backend}")
            print("[INFO] Train config:")
            print(self.cfg)
            if self.is_dist:
                print(f"[INFO] Distributed: rank={self.rank} "
                      f"world_size={self.world_size} local_rank={self.local_rank}")

        # ── Build model ──────────────────────────────────────────────────────
        raw_model: vla.VLA = getattr(vla, "vla_" + self.cfg.model.strip())(
            train_stage=self.cfg.train_stage,
            camera_view_dropout=self.cfg.camera_view_dropout,
            vlm_finetune_mode=self.cfg.vlm_finetune_mode,
            use_gradient_checkpointing=self.cfg.use_gradient_checkpointing,
        ).to(self.model_device)

        decay, no_decay = raw_model.parameter_groups()

        # ── Optimizer ────────────────────────────────────────────────────────
        param_groups = self._build_param_groups(raw_model, decay, no_decay)
        optimizer = optim.AdamW(param_groups)
        lr_scheduler = get_scheduler(
            name="constant_with_warmup",
            optimizer=optimizer,
            num_warmup_steps=self.cfg.num_warmup,
        )

        # ── Wrap with back-end ───────────────────────────────────────────────
        (self.engine, self.optimizer, self.scheduler,
         self.scaler, self.zero_stage) = self.bk.wrap_engine(
            raw_model, optimizer, lr_scheduler)
        # Cache once — DDP/DS don't re-wrap during training.
        self.module = self.bk.unwrap(self.engine)

        # ── Data loader ──────────────────────────────────────────────────────
        self.train_loader, self.sampler = get_data_loader_for_cfg(
            self.cfg, is_dist=self.is_dist)

        # ── EMA ──────────────────────────────────────────────────────────────
        # ZeRO-3 shards params across ranks, so a per-rank EMA snapshot is not
        # the true model EMA. Disable in that regime.
        self._ema_enabled = self.cfg.ema_enabled and (self.zero_stage < 3)
        if self.cfg.ema_enabled and self.zero_stage >= 3 and is_main_process():
            print("[WARN] EMA disabled with ZeRO-3 (parameter sharding).")
        trainable_params = [p for p in self.module.parameters() if p.requires_grad]
        self.ema = (ExponentialMovingAverage(trainable_params, decay=self.cfg.ema_decay)
                    if self._ema_enabled else None)

        # ── State tracking ───────────────────────────────────────────────────
        self.save: bool | str = False
        self.writer: Optional[SummaryWriter] = None
        self.best_score: Optional[float] = None
        self.larger_better = False
        self.current_iters = 0
        self.last_ep = -1
        self._is_first_save = True

        # ── Checkpoint load ──────────────────────────────────────────────────
        # When --conti is given as an absolute path ending in .pt (e.g. for
        # loading an existing pretraining checkpoint), self.save would
        # otherwise become that path and os.path.join would resolve future
        # saves back to the source directory — overwriting the source.
        # Derive a safe save subdir name from the parent directory's basename.
        if conti:
            if os.path.isabs(conti) or conti.endswith(".pt"):
                self.save = os.path.basename(
                    os.path.dirname(conti.rstrip("/"))) or "resumed"
                if is_main_process():
                    print(f"[INFO] --conti is an absolute path; new checkpoints "
                          f"will be written to {self.CKPT_DIR}/{self.save}/. "
                          f"Pass -s <name> to override.")
            else:
                self.save = conti
            self._load_checkpoint(conti, is_resume=True)
        elif self.cfg.pretrained_ckpt:
            self._load_checkpoint(self.cfg.pretrained_ckpt, is_resume=False,
                                  load_from_va=self.cfg.load_from_va)

        if save:
            self.save = save

        # ── Optional: torch.compile (back-end may choose to enable or skip) ──
        if self.bk.maybe_compile(self.module) and is_main_process():
            print("[INFO] torch.compile enabled for action expert")

        if is_main_process():
            print("[INFO] Total {:.3f}M trainable parameters".format(
                count_trainable(self.module) / 1e6))

    # ── Optimizer param-group construction ───────────────────────────────────

    def _build_param_groups(self, raw_model, decay, no_decay):
        """Split params into VLM vs rest, applying separate LR / WD if requested."""
        vlm_lr = self.cfg.vlm_lr if self.cfg.vlm_lr is not None else self.cfg.max_lr
        vlm_param_ids = set()
        if self.cfg.vlm_finetune_mode != "frozen":
            vlm_param_ids = {
                id(p) for p in raw_model.vlm.parameters() if p.requires_grad}

        vlm_decay, vlm_no_decay = [], []
        other_decay, other_no_decay = [], []
        for p in decay:
            (vlm_decay if id(p) in vlm_param_ids else other_decay).append(p)
        for p in no_decay:
            (vlm_no_decay if id(p) in vlm_param_ids else other_no_decay).append(p)

        groups = []
        if other_decay:
            groups.append({"params": other_decay,    "lr": self.cfg.max_lr, "weight_decay": self.cfg.wd})
        if other_no_decay:
            groups.append({"params": other_no_decay, "lr": self.cfg.max_lr, "weight_decay": 0.0})
        if vlm_decay:
            groups.append({"params": vlm_decay,      "lr": vlm_lr,          "weight_decay": self.cfg.vlm_wd})
        if vlm_no_decay:
            groups.append({"params": vlm_no_decay,   "lr": vlm_lr,          "weight_decay": 0.0})
        return groups

    # ── Checkpoint I/O ───────────────────────────────────────────────────────

    def _ckpt_path(self, name: str) -> str:
        return name if name.endswith(".pt") else \
            os.path.join(self.CKPT_DIR, name, "ckpt_latest.pt")

    def _load_checkpoint(self, name: str, is_resume: bool,
                         load_from_va: bool = False):
        ckpt_path = self._ckpt_path(name)
        ckpt = torch.load(ckpt_path, map_location=self.model_device,
                          weights_only=False)

        # Resuming should never re-expand the VA prior; only first-time
        # Stage-1 init does that.
        do_va_expand = load_from_va and not is_resume
        self.module.load_from_pretrain(ckpt["weights"], load_from_va=do_va_expand)

        if "vlm_weights" in ckpt:
            incompat = self.module.vlm.load_state_dict(
                ckpt["vlm_weights"], strict=False)
            if is_main_process():
                print(f"[INFO] Loaded VLM weights from checkpoint "
                      f"(missing={len(incompat.missing_keys)}, "
                      f"unexpected={len(incompat.unexpected_keys)})")
                if incompat.unexpected_keys:
                    print(f"[WARN] Unexpected VLM keys (first 3): "
                          f"{incompat.unexpected_keys[:3]}")

        if is_resume:
            self.current_iters = ckpt["current_iters"]
            self.last_ep       = ckpt["last_ep"]
            self.best_score    = ckpt.get("best_score", None)
            self.bk.restore_optimizer_state(ckpt, ckpt_path)
            if "ema" in ckpt and self.ema is not None:
                self.ema.load_state_dict(ckpt["ema"])
            if is_main_process():
                print(f"[INFO] Resumed from {ckpt_path} "
                      f"at iter {self.current_iters}")
        else:
            if is_main_process():
                print(f"[INFO] Loaded pretrained weights from {ckpt_path}")

    # ── Save ─────────────────────────────────────────────────────────────────

    def save_model(self, fname: str, best_score: float, latest_score: float):
        if self.is_dist:
            dist.barrier()

        actor_state = self.bk.gather_state_dict(self.module.actor)
        vlm_state   = self.bk.gather_vlm_trainable(
            self.module.vlm, self.cfg.vlm_finetune_mode)

        # Per-rank artefacts (DS optimizer shards) — every rank writes.
        if self.save:
            ckpt_dir = os.path.join(self.CKPT_DIR, self.save)
            os.makedirs(ckpt_dir, exist_ok=True)
            self.bk.save_extra(ckpt_dir, fname)

        if is_main_process():
            if self.save and self._is_first_save:
                cfg_path = os.path.join(self.CKPT_DIR, self.save,
                                        f"{self.launch_time_str}.json")
                self.cfg.dump(cfg_path)
                self._is_first_save = False

            if self.save:
                ckpt_dir = os.path.join(self.CKPT_DIR, self.save)
                os.makedirs(ckpt_dir, exist_ok=True)
                to_save = {
                    "weights":       actor_state,
                    "current_iters": self.current_iters,
                    "last_ep":       self.last_ep,
                    "lr":            self.scheduler.get_last_lr()[0],
                    "best_score":    best_score,
                    "latest_score":  latest_score,
                }
                to_save.update(self.bk.opt_state_for_ckpt())
                if vlm_state is not None:
                    to_save["vlm_weights"] = vlm_state
                if self._ema_enabled:
                    to_save["ema"] = self.ema.state_dict()
                save_path = os.path.join(ckpt_dir, fname)
                torch.save(to_save, save_path)
                print(f"[INFO] Saved checkpoint → {save_path}")

        if self.is_dist:
            dist.barrier()

    # ── Step ─────────────────────────────────────────────────────────────────

    @staticmethod
    def preprocess_data(data: Dict[str, Tensor], device):
        for k in data:
            if isinstance(data[k], Tensor):
                data[k] = data[k].to(device, non_blocking=True)
        return data

    def compute_metrics(self, data: Dict[str, Tensor]):
        data = self.preprocess_data(data, self.model_device)
        total_loss, metrics = self.engine(
            obs_rgbs=data["obs_rgbs"],
            obs_masks=data.get("obs_masks", None),
            obs_norm_xys=data["obs_norm_xys"],
            obs_extrinsics=data["obs_extrinsics"],
            prompt_text=data["prompt_text"],
            current_ee_pose=data["current_ee_pose"],
            action_ref_pose=data["action_ref_pose"],
            history_ee_states=data["history_ee_states"],
            gt_future_ee_states=data["gt_future_ee_states"],
            valid_ee_mask=data["valid_ee_mask"],
            inference=False,
            fp16=self.cfg.fp16,
            robot_pose_aug=self.cfg.robot_pose_aug,
            camera_drop_prob=data.get("camera_drop_prob", None),
        )
        return total_loss, metrics

    def log_metrics(self, metrics: dict):
        if not is_main_process():
            return
        if not self.save:
            return
        if self.writer is None:
            log_dir = os.path.join(self.LOG_DIR, self.save)
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        self.writer.add_scalar(
            "lr", self.scheduler.get_last_lr()[0], self.current_iters)
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, self.current_iters)

    # ── Main training loop (unified, back-end-agnostic) ─────────────────────

    def fitting(self):
        """Run the unified training loop. Back-end-specific behaviour is
        delegated entirely to ``self.bk``."""
        averages: Dict[str, AverageMeter] = {}
        self.engine.train()
        accum = max(1, self.cfg.gradient_accumulation_steps)
        micro = 0

        while self.current_iters <= self.cfg.max_iterations:
            if self.is_dist and self.sampler is not None:
                self.sampler.set_epoch(self.last_ep + 1)

            loader = self.bk.maybe_prefetch(self.train_loader, self.model_device)

            for data in loader:
                loss, metrics = self.compute_metrics(data)

                if torch.isnan(loss) or torch.isinf(loss):
                    if is_main_process():
                        print("[INFO] NaN / Inf loss, skip")
                    self.bk.on_nan_loss()
                    micro = 0
                    continue

                is_boundary = self.bk.micro_step(
                    loss, micro, accum, self.cfg.grad_clip)

                for k, v in metrics.items():
                    averages.setdefault(k, AverageMeter()).append(v)

                if not is_boundary:
                    micro += 1
                    continue

                micro = 0
                self.current_iters += 1

                if (self.current_iters >= self.cfg.ema_start
                        and self._ema_enabled):
                    self.ema.update()

                self._post_step(averages)

                if self.current_iters > self.cfg.max_iterations:
                    break

            self.last_ep += 1

    def _post_step(self, averages: Dict[str, AverageMeter]):
        """Per-iteration logging / checkpointing logic. Back-end-agnostic."""
        if is_main_process():
            parts = [f"{k} = {v.avg():.3e}" for k, v in averages.items()]
            print("[INFO] {}/{} | {} | lr = {:.3e}".format(
                self.current_iters, self.cfg.max_iterations,
                " | ".join(parts),
                self.scheduler.get_last_lr()[0]))

        if self.current_iters % self.cfg.save_latest_interval == 0:
            avg_m = {k: v.avg() for k, v in averages.items()}
            red_m = reduce_metrics(avg_m, is_dist=self.is_dist)
            latest = red_m.get("total_loss", avg_m.get("total_loss", 0.0))
            save_best = (
                self.best_score is None
                or (self.larger_better and latest > self.best_score)
                or (not self.larger_better and latest < self.best_score))
            if save_best:
                self.best_score = latest
            self.save_model("ckpt_latest.pt", self.best_score, latest)
            if save_best:
                self.save_model("ckpt_best.pt", self.best_score, latest)

        if (self.cfg.save_interval > 0
                and self.current_iters % self.cfg.save_interval == 0):
            avg_m = {k: v.avg() for k, v in averages.items()}
            red_m = reduce_metrics(avg_m, is_dist=self.is_dist)
            latest = red_m.get("total_loss", avg_m.get("total_loss", 0.0))
            self.save_model(
                f"ckpt_{self.current_iters:0>7d}.pt",
                self.best_score, latest)

        if self.current_iters % self.cfg.log_interval == 0:
            log_m = {"train/" + k: v.avg() for k, v in averages.items()}
            red_log = reduce_metrics(log_m, is_dist=self.is_dist)
            self.log_metrics(red_log)
            for m in averages.values():
                m.reset()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    trainer = Trainer()
    try:
        with trainer.bk.fitting_context():
            trainer.fitting()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
