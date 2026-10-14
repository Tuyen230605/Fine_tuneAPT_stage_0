"""NUMA-aware CPU affinity binding for distributed training workers.

When ``cfg.bind_cpu_affinity`` is set, ``maybe_bind_cpu_affinity`` pins the
calling process to one of two CPU groups (one per local-rank half), based on
either the user-supplied ``CPU_AFFINITY_GROUP{0,1}`` env vars or, failing
those, the NUMA topology reported by ``lscpu``.
"""
from __future__ import annotations

import os
import subprocess
from typing import Dict, List, Optional

import torch


def _parse_cpu_affinity(spec: str) -> List[int]:
    """Parse a CPU spec like ``"0-15,32-47"`` into a sorted list of CPU ids."""
    cpus = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def _infer_gpu_group_idx(local_rank: int) -> int:
    """Map a local rank to a 0/1 GPU group, respecting CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        try:
            ids = [int(p.strip()) for p in visible.split(",") if p.strip()]
            if local_rank < len(ids):
                return 0 if ids[local_rank] < 4 else 1
        except ValueError:
            pass
    split = max(1, torch.cuda.device_count() // 2)
    return 0 if local_rank < split else 1


def _get_numa_cpu_groups() -> Optional[List[List[int]]]:
    """Read NUMA-node → CPU mapping from ``lscpu``. Returns None on failure."""
    try:
        out = subprocess.run(["lscpu", "-p=cpu,node"], check=True,
                             capture_output=True, text=True).stdout
    except Exception:
        return None
    groups: Dict[int, list] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cpu_s, node_s = line.split(",", 1)
        if not node_s:
            continue
        groups.setdefault(int(node_s), []).append(int(cpu_s))
    return [sorted(groups[n]) for n in sorted(groups)] if groups else None


def maybe_bind_cpu_affinity(local_rank: int, cfg) -> None:
    """Bind the current process to a CPU group if ``cfg.bind_cpu_affinity``.

    Honours, in order:
      1. ``CPU_AFFINITY_GROUP0`` + ``CPU_AFFINITY_GROUP1`` env vars (explicit override).
      2. NUMA topology from ``lscpu`` (one group per NUMA node).
      3. A 50/50 split of the inherited affinity mask.
    """
    if not getattr(cfg, "bind_cpu_affinity", False):
        return
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return
    current = sorted(os.sched_getaffinity(0))
    if len(current) < 2:
        return
    g0 = os.environ.get("CPU_AFFINITY_GROUP0")
    g1 = os.environ.get("CPU_AFFINITY_GROUP1")
    if g0 and g1:
        cpu_groups = [_parse_cpu_affinity(g0), _parse_cpu_affinity(g1)]
    else:
        cpu_groups = _get_numa_cpu_groups()
        if cpu_groups is None:
            mid = len(current) // 2
            cpu_groups = [current[:mid], current[mid:]]
        else:
            allowed = set(current)
            cpu_groups = [[c for c in g if c in allowed] for g in cpu_groups]
            cpu_groups = [g for g in cpu_groups if g]
    idx = min(_infer_gpu_group_idx(local_rank), len(cpu_groups) - 1)
    target = cpu_groups[idx]
    if not target:
        return
    os.sched_setaffinity(0, target)
    print(f"[INFO] Rank local_rank={local_rank} → CPU group {idx}: "
          f"{target[0]}-{target[-1]} ({len(target)} CPUs)")
