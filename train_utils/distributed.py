"""Distributed-training helpers shared by the trainer and the back-ends.

Three groups of utilities:

* ``is_distributed`` / ``is_main_process`` — small guards used everywhere
  for rank-zero-only side effects (printing, ckpt writing, logging).
* ``reduce_metrics`` — all-reduce a dict of scalar metrics across ranks.
* ``DataPrefetcher`` — wraps a DataLoader iterator with a background CUDA
  stream that overlaps host→device copies with the previous step's compute.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """True iff the default process group is initialised."""
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    """True on rank 0, or always True outside distributed mode."""
    return (not is_distributed()) or dist.get_rank() == 0


def reduce_metrics(metrics: dict, is_dist: bool = False) -> dict:
    """Mean-reduce scalar entries of ``metrics`` across ranks.

    Non-scalar / non-numeric entries are passed through unchanged.
    """
    if not is_dist:
        return metrics
    out = {}
    device = torch.cuda.current_device()
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            t = v.detach().clone().to(device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            out[k] = t.item() / dist.get_world_size()
        elif isinstance(v, (int, float)):
            t = torch.tensor(v, device=device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            out[k] = t.item() / dist.get_world_size()
        else:
            out[k] = v
    return out


class DataPrefetcher:
    """Prefetch the next batch on a background CUDA stream.

    Overlaps the host→device transfer of batch ``i+1`` with the compute on
    batch ``i``. Non-tensor fields are passed through; tensor fields are
    moved with ``non_blocking=True``.
    """

    def __init__(self, loader, device: str):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_data = None
        self._preload()

    def _preload(self):
        try:
            raw = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in raw.items():
                if isinstance(v, torch.Tensor):
                    raw[k] = v.to(self.device, non_blocking=True)
        self.next_data = raw

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_data is None:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self._preload()
        return data
