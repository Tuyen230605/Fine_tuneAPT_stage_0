"""Distributed back-end strategies for the APT trainer.

The trainer holds one ``_Backend`` instance and delegates back-end-specific
work to it through the abstract interface below. ``DDPBackend`` covers
``torchrun + DistributedDataParallel``; ``DeepSpeedBackend`` covers
``deepspeed`` ZeRO-{2,3}. Adding a new back-end (e.g. FSDP) is a matter of
implementing another ``_Backend`` subclass and registering it in
``make_backend``.
"""
from __future__ import annotations

import abc
import contextlib
import json
import os
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from .distributed import DataPrefetcher, is_main_process

if TYPE_CHECKING:
    from apt.train import Trainer  # only for type hints; avoids circular import


class _Backend(abc.ABC):
    """Strategy interface for distributed-training back-ends.

    Holds a reference to the parent ``Trainer`` so each method can access
    the live optimizer / scheduler / engine / config without needing them
    threaded through every call.
    """

    name: str

    def __init__(self, trainer: "Trainer"):
        self.t = trainer

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def init_distributed(self) -> Tuple[bool, int, int, int]:
        """Initialise the process group. Returns (is_dist, rank, world_size, local_rank)."""

    @abc.abstractmethod
    def wrap_engine(self, raw_model, optimizer, lr_scheduler):
        """Wrap (model, optimizer, scheduler) for this back-end.

        Returns (engine, optimizer, scheduler, scaler_or_None, zero_stage).
        ``engine`` exposes ``__call__`` returning ``(loss, metrics)`` like
        the raw model, and ``engine.module`` returns the underlying
        ``nn.Module``.
        """

    def unwrap(self, engine) -> torch.nn.Module:
        """Strip wrapping. Default works for DS; DDP overrides."""
        return engine.module

    # ── Optional features ────────────────────────────────────────────────────

    def maybe_prefetch(self, loader, device):
        """Wrap loader in a CUDA-stream prefetcher if appropriate."""
        return loader

    def maybe_compile(self, raw_model) -> bool:
        """Optionally torch.compile parts of the model."""
        return False

    def fitting_context(self):
        """Context manager wrapping the whole training loop. Default: noop."""
        return contextlib.nullcontext()

    # ── Train step ───────────────────────────────────────────────────────────

    @abc.abstractmethod
    def micro_step(self, loss: Tensor, micro_idx: int, accum: int,
                   grad_clip: float) -> bool:
        """Run one micro-step (backward + maybe optimizer step + scheduler).

        Returns True iff this micro-step crossed a gradient-accumulation
        boundary, so the trainer should bump ``current_iters``, run EMA
        update and emit logs.
        """

    def on_nan_loss(self):
        """Cleanup after a NaN/Inf loss. Default: noop."""
        return

    # ── Checkpoint I/O ───────────────────────────────────────────────────────

    @abc.abstractmethod
    def gather_state_dict(self, module: torch.nn.Module) -> Optional[dict]:
        """Return ``module.state_dict()`` on rank-0 (None elsewhere)."""

    @abc.abstractmethod
    def gather_vlm_trainable(self, vlm_module,
                             vlm_finetune_mode: str) -> Optional[dict]:
        """Return the slice of the VLM state-dict to save.

        For ``vlm_finetune_mode == "frozen"`` always returns None.
        """

    @abc.abstractmethod
    def opt_state_for_ckpt(self) -> dict:
        """Optimizer/scheduler/scaler state to embed in the main ckpt file
        (DDP). DS returns {} because per-rank shards are written separately."""

    def save_extra(self, ckpt_dir: str, fname: str):
        """Write per-rank artefacts alongside the main ckpt. Default: noop."""
        return

    @abc.abstractmethod
    def restore_optimizer_state(self, ckpt: dict, ckpt_path: str):
        """Restore optimizer / scheduler / scaler state from a checkpoint."""


class DDPBackend(_Backend):
    """torchrun + DistributedDataParallel."""

    name = "ddp"

    def init_distributed(self):
        if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
            return False, 0, 1, 0
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=world, rank=rank,
        )
        torch.cuda.set_device(local)
        return True, rank, world, local

    def wrap_engine(self, raw_model, optimizer, lr_scheduler):
        if self.t.is_dist:
            # find_unused_parameters=True handles cases where some heads /
            # layers are frozen depending on the stage.
            engine = DDP(
                raw_model,
                device_ids=[self.t.local_rank],
                output_device=self.t.local_rank,
                find_unused_parameters=True,
            )
        else:
            engine = raw_model
        # BF16 has FP32 dynamic range and does not need loss scaling.
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        return engine, optimizer, lr_scheduler, scaler, 0

    def unwrap(self, engine):
        return engine.module if isinstance(engine, DDP) else engine

    def maybe_prefetch(self, loader, device):
        if getattr(self.t.cfg, "use_prefetcher", True):
            return DataPrefetcher(loader, device)
        return loader

    def maybe_compile(self, raw_model):
        if getattr(self.t.cfg, "use_compile", False):
            raw_model.actor = torch.compile(
                raw_model.actor, mode="reduce-overhead", dynamic=True)
            return True
        return False

    def fitting_context(self):
        # Force Flash / mem-efficient SDPA, disable the math fallback so we
        # know flash actually kicked in during DDP training.
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True,
        )

    def micro_step(self, loss, micro_idx, accum, grad_clip) -> bool:
        loss = loss / accum
        is_last_micro = (micro_idx + 1) == accum
        sync_ctx = (contextlib.nullcontext()
                    if (not self.t.is_dist or is_last_micro)
                    else self.t.engine.no_sync())

        with sync_ctx:
            if self.t.scaler.is_enabled():
                self.t.scaler.scale(loss).backward()
            else:
                loss.backward()

        if not is_last_micro:
            return False

        # boundary: apply the accumulated gradients
        if self.t.scaler.is_enabled():
            if grad_clip > 0:
                self.t.scaler.unscale_(self.t.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.t.engine.parameters(), grad_clip)
            self.t.scaler.step(self.t.optimizer)
            self.t.scaler.update()
        else:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.t.engine.parameters(), grad_clip)
            self.t.optimizer.step()

        self.t.optimizer.zero_grad(set_to_none=True)
        self.t.scheduler.step()
        return True

    def on_nan_loss(self):
        self.t.optimizer.zero_grad(set_to_none=True)

    def gather_state_dict(self, module):
        if is_main_process():
            return {k: v.clone() for k, v in module.state_dict().items()}
        return None

    def gather_vlm_trainable(self, vlm_module, vlm_finetune_mode):
        if vlm_finetune_mode == "frozen":
            return None
        if not is_main_process():
            return None
        if vlm_finetune_mode == "full":
            # state_dict preserves tied-weight aliases (e.g. lm_head.weight
            # aliased to embed_tokens.weight) that named_parameters() drops.
            return {k: v.clone() for k, v in vlm_module.state_dict().items()}
        return {n: p.data.clone()
                for n, p in vlm_module.named_parameters() if p.requires_grad}

    def opt_state_for_ckpt(self) -> dict:
        out = {
            "optimizer": self.t.optimizer.state_dict(),
            "scheduler": self.t.scheduler.state_dict(),
        }
        if self.t.scaler is not None:
            out["scaler"] = self.t.scaler.state_dict()
        return out

    def restore_optimizer_state(self, ckpt, ckpt_path):
        # Match the original train_dist.py behaviour: hard-load
        # optimizer / scaler / scheduler from the checkpoint. If the
        # source run had a different param-group structure, this will
        # raise loudly — intentional, since silently dropping optimizer
        # state is far more dangerous than a clear error.
        self.t.optimizer.load_state_dict(ckpt["optimizer"])
        self.t.scaler.load_state_dict(ckpt["scaler"])
        self.t.scheduler.load_state_dict(ckpt["scheduler"])


class DeepSpeedBackend(_Backend):
    """DeepSpeed ZeRO-{2,3}."""

    name = "deepspeed"

    def __init__(self, trainer: "Trainer"):
        super().__init__(trainer)
        import deepspeed
        self.ds = deepspeed

    def init_distributed(self):
        self.ds.init_distributed(dist_backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return True, rank, world, local

    def wrap_engine(self, raw_model, optimizer, lr_scheduler):
        with open(self.t.ds_config_path) as f:
            ds_config: dict = json.load(f)
        zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)

        accum = max(1, self.t.cfg.gradient_accumulation_steps)
        ds_config["gradient_accumulation_steps"] = accum
        ds_config["gradient_clipping"] = (
            self.t.cfg.grad_clip if self.t.cfg.grad_clip > 0 else 1.0)
        ds_config["train_micro_batch_size_per_gpu"] = self.t.cfg.bs
        ds_config["train_batch_size"] = (
            self.t.cfg.bs * accum * self.t.world_size)

        engine, optimizer, _, scheduler = self.ds.initialize(
            model=raw_model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=ds_config,
        )
        if is_main_process():
            print(f"[INFO] DeepSpeed config: {self.t.ds_config_path} "
                  f"(ZeRO stage {zero_stage})")
        return engine, optimizer, scheduler, None, zero_stage

    def micro_step(self, loss, micro_idx, accum, grad_clip) -> bool:
        # Scale loss so the reported magnitude is independent of accum_steps.
        # DeepSpeed handles bf16 internally; engine.step() updates weights
        # only at the accumulation boundary (every accum micro-steps).
        loss = loss / accum
        self.t.engine.backward(loss)
        self.t.engine.step()
        return self.t.engine.is_gradient_accumulation_boundary()

    def gather_state_dict(self, module):
        if self.t.zero_stage == 3:
            with self.ds.zero.GatheredParameters(
                    list(module.parameters()), modifier_rank=0):
                if is_main_process():
                    return {k: v.clone() for k, v in module.state_dict().items()}
            return None
        if is_main_process():
            return {k: v.clone() for k, v in module.state_dict().items()}
        return None

    def gather_vlm_trainable(self, vlm_module, vlm_finetune_mode):
        if vlm_finetune_mode == "frozen":
            return None
        if vlm_finetune_mode == "full":
            gather_params = list(vlm_module.parameters())
        else:
            gather_params = [p for p in vlm_module.parameters() if p.requires_grad]

        def snapshot():
            if vlm_finetune_mode == "full":
                return {k: v.clone() for k, v in vlm_module.state_dict().items()}
            return {n: p.data.clone()
                    for n, p in vlm_module.named_parameters() if p.requires_grad}

        if self.t.zero_stage == 3:
            with self.ds.zero.GatheredParameters(gather_params, modifier_rank=0):
                return snapshot() if is_main_process() else None
        return snapshot() if is_main_process() else None

    def opt_state_for_ckpt(self) -> dict:
        # Per-rank optimizer shards are written separately via save_extra.
        return {}

    def save_extra(self, ckpt_dir: str, fname: str):
        opt_fname = fname.replace(".pt", f"_opt_rank{self.t.rank}.pt")
        opt_path = os.path.join(ckpt_dir, opt_fname)
        torch.save({
            "optimizer": self.t.optimizer.state_dict(),
            "scheduler": self.t.scheduler.state_dict(),
        }, opt_path)

    def restore_optimizer_state(self, ckpt, ckpt_path):
        opt_shard = ckpt_path.replace(".pt", f"_opt_rank{self.t.rank}.pt")
        if not os.path.exists(opt_shard):
            if is_main_process():
                print(f"[WARN] No optimizer shard at {opt_shard}; "
                      "starting with fresh optimizer.")
            return
        try:
            opt_ckpt = torch.load(opt_shard, map_location=self.t.model_device,
                                  weights_only=False)
            ds_sd = opt_ckpt["optimizer"]
            from deepspeed.checkpoint.constants import (
                BASE_OPTIMIZER_STATE, SINGLE_PARTITION_OF_FP32_GROUPS,
                LOSS_SCALER, CLIP_GRAD)
            self.t.optimizer.optimizer.load_state_dict(ds_sd[BASE_OPTIMIZER_STATE])
            saved_fp32 = ds_sd[SINGLE_PARTITION_OF_FP32_GROUPS]
            for current, saved in zip(
                    self.t.optimizer.single_partition_of_fp32_groups, saved_fp32):
                src = saved if saved.numel() <= current.numel() \
                    else saved.narrow(0, 0, current.numel())
                current.data.copy_(src.data)
            if LOSS_SCALER in ds_sd:
                self.t.optimizer.loss_scaler = ds_sd[LOSS_SCALER]
            if CLIP_GRAD in ds_sd:
                self.t.optimizer.clip_grad = ds_sd[CLIP_GRAD]
            if hasattr(self.t.optimizer, "_link_all_hp_params"):
                self.t.optimizer._link_all_hp_params()
            self.t.scheduler.load_state_dict(opt_ckpt["scheduler"])
            if is_main_process():
                print(f"[INFO] Restored optimizer/scheduler from {opt_shard}")
        except Exception as e:
            import traceback
            if is_main_process():
                print(f"[WARN] Could not restore DeepSpeed optimizer state: "
                      f"{type(e).__name__}: {e}")
                traceback.print_exc()
                print("[WARN] Starting with fresh optimizer.")


def make_backend(name: str, trainer: "Trainer") -> _Backend:
    """Construct the back-end strategy for the given name."""
    if name == "ddp":
        return DDPBackend(trainer)
    if name == "deepspeed":
        return DeepSpeedBackend(trainer)
    raise ValueError(f"Unknown backend: {name}")
