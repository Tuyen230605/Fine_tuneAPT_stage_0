"""
Offline compatibility check for an existing APT / BayesVLA-style checkpoint.

Verifies that the checkpoint's ``weights`` state-dict matches what the merged
trainer expects for the requested ``--train-stage``, and (when applicable)
that the Stage-1 VA-expansion mapping is well-formed.

Does not download any HuggingFace assets — runs entirely on the checkpoint
file. Use this as a fast pre-flight before launching training:

    python scripts/test_ckpt.py \
        --ckpt /path/to/va_pretrain/ckpt_latest.pt \
        --train-stage 0

    # Stage-1 bootstrap from a Stage-0 (VA) checkpoint
    python scripts/test_ckpt.py \
        --ckpt /path/to/va_pretrain/ckpt_latest.pt \
        --train-stage 1 --load-from-va

    # Stage-1 resume from a Stage-1 (VLA) checkpoint
    python scripts/test_ckpt.py \
        --ckpt /path/to/vla_pretrain/ckpt_latest.pt \
        --train-stage 1
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import torch


def attn_layer_count(weight_keys):
    pat = re.compile(r"dp_head\.traj_context_attn\.layers\.(\d+)\.")
    return sorted({int(m.group(1)) for k in weight_keys
                   for m in [pat.match(k)] if m})


def layer_params(weight_keys, idx: int):
    prefix = f"dp_head.traj_context_attn.layers.{idx}."
    return {k[len(prefix):] for k in weight_keys if k.startswith(prefix)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to a ckpt_*.pt file")
    ap.add_argument("--train-stage", type=int, default=0, choices=[0, 1])
    ap.add_argument("--load-from-va", action="store_true",
                    help="Set when bootstrapping Stage-1 from a Stage-0 VA ckpt")
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"[FAIL] checkpoint not found: {args.ckpt}")
        sys.exit(2)

    print(f"[INFO] Loading {args.ckpt} ({os.path.getsize(args.ckpt) / 1e6:.1f} MB)")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"[INFO] Top-level keys: {sorted(ck.keys())}")

    if "weights" not in ck:
        print("[FAIL] checkpoint missing 'weights' key — not an APT checkpoint?")
        sys.exit(2)

    keys = set(ck["weights"].keys())
    layers = attn_layer_count(keys)
    print(f"[INFO] actor state has {len(keys)} tensors, "
          f"traj_context_attn layers = {layers}")

    # Expected per-stage layer counts. vla_base sets N=10 in Stage 1, N/2 in
    # Stage 0; sanity-check the file matches what the user claims they have.
    if args.load_from_va:
        if args.train_stage != 1:
            print("[FAIL] --load-from-va is only meaningful with --train-stage 1")
            sys.exit(2)
        if len(layers) != 5:
            print(f"[FAIL] --load-from-va expects a Stage-0 ckpt with 5 attention "
                  f"layers, found {len(layers)}")
            sys.exit(2)
        # Show the s0 → s1 expansion mapping
        print("[INFO] VA-expansion (s0 → s1) mapping that load_from_pretrain will apply:")
        s0 = 0
        for s1 in range(10):
            if s1 % 2 == 1:
                print(f"        va.layer.{s0:>2}  →  vla.layer.{s1:>2}")
                s0 += 1
        # Even-index Stage-1 layers will be left at random init.
        print("[INFO] Stage-1 even-index layers (0,2,4,6,8) keep their fresh init.")
    else:
        expected = 5 if args.train_stage == 0 else 10
        if len(layers) != expected:
            print(f"[FAIL] --train-stage {args.train_stage} expects {expected} "
                  f"attention layers, found {len(layers)} — wrong stage?")
            sys.exit(2)

    if "vlm_weights" in ck:
        print(f"[INFO] checkpoint contains vlm_weights "
              f"({len(ck['vlm_weights'])} tensors). "
              "Run with --vlm-mode lora|full to load them.")
    else:
        print("[INFO] no vlm_weights in checkpoint — VLM stays at HF init "
              "(consistent with --vlm-mode frozen).")

    if "optimizer" in ck:
        print(f"[INFO] checkpoint embeds an optimizer state "
              "(DDP-trained checkpoint).")
    else:
        opt_shard = args.ckpt.replace(".pt", "_opt_rank0.pt")
        if os.path.exists(opt_shard):
            print(f"[INFO] checkpoint has DeepSpeed per-rank optimizer shards "
                  f"({opt_shard} exists).")
        else:
            print("[INFO] no optimizer state found — fine for "
                  "--pretrained_ckpt bootstrap; resume will start fresh.")

    if "current_iters" in ck:
        print(f"[INFO] training metadata: "
              f"current_iters={ck.get('current_iters')} "
              f"last_ep={ck.get('last_ep')} "
              f"best_score={ck.get('best_score')}")

    print("[ OK ] checkpoint is compatible with --train-stage "
          f"{args.train_stage}{' --load-from-va' if args.load_from_va else ''}.")


if __name__ == "__main__":
    main()
