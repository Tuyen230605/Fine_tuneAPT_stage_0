"""Aggregate per-episode metrics into a per-setting success-rate report.

Reads ``<save_dir>/<setting>/metrics/*.json`` produced by ``eval/eval_policy.py``
and prints one row per setting plus an overall total.

Usage:
    python -m eval.summary ./data/exp_results/myrun

Success criterion: an episode counts as a success iff its recorded ``dist``
(final XY distance from the grasped object to the commanded place position)
is below ``--success_threshold`` (default 0.11 m).
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


DEFAULT_SUCCESS_THRESHOLD_M = 0.11
DEFAULT_MIN_EPISODES        = 10


def summarize_setting(setting_dir: Path, success_threshold: float):
    """Read all metric JSONs under ``setting_dir/metrics/``.

    Returns:
        (n_episodes, n_success) — or (0, 0) if no JSONs found.
    """
    files = sorted(glob.glob(str(setting_dir / "metrics" / "*.json")))
    dists = []
    for f in files:
        with open(f) as fp:
            metric = json.load(fp)
        if "dist" in metric:
            dists.append(float(metric["dist"]))
    if not dists:
        return 0, 0
    dists = np.asarray(dists)
    n_success = int(np.sum(dists < success_threshold))
    return len(dists), n_success


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "save_dir", type=Path,
        help="Root directory previously passed to `eval.eval_policy --save_dir`",
    )
    parser.add_argument(
        "--success_threshold", type=float, default=DEFAULT_SUCCESS_THRESHOLD_M,
        help=f"Final XY distance under which an episode counts as a success (meters, "
             f"default {DEFAULT_SUCCESS_THRESHOLD_M})",
    )
    parser.add_argument(
        "--min_episodes", type=int, default=DEFAULT_MIN_EPISODES,
        help=f"Skip settings with fewer completed episodes than this "
             f"(default {DEFAULT_MIN_EPISODES})",
    )
    opt = parser.parse_args()

    if not opt.save_dir.is_dir():
        raise SystemExit(f"save_dir does not exist or is not a directory: {opt.save_dir}")

    setting_dirs = sorted(d for d in opt.save_dir.iterdir() if d.is_dir())
    if not setting_dirs:
        raise SystemExit(f"No per-setting subdirectories found under {opt.save_dir}")

    print(f"{'setting':<10} {'episodes':>10}  {'SR':>8}")
    print("-" * 34)

    total_n, total_success = 0, 0
    for d in setting_dirs:
        n, succ = summarize_setting(d, opt.success_threshold)
        if n < opt.min_episodes:
            print(f"{d.name:<10} {n:>10}  (skipped — fewer than --min_episodes={opt.min_episodes})")
            continue
        sr = succ / n
        print(f"{d.name:<10} {n:>10}  {sr*100:>6.2f}%")
        total_n       += n
        total_success += succ

    if total_n:
        print("-" * 34)
        print(f"{'TOTAL':<10} {total_n:>10}  {total_success / total_n * 100:>6.2f}%")


if __name__ == "__main__":
    main()
