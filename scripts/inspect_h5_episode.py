"""Print an HDF5 episode's schema without copying its image or video payloads.

Run this on the machine holding the raw data, then share the JSON output when
adding a dataset adapter. Only metadata and small state previews are read.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[:200]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, np.ndarray):
        if value.size > 32:
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        return json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, str):
        return value[:200]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:200]


def describe_attrs(attrs: h5py.AttributeManager) -> dict[str, Any]:
    return {key: json_value(attrs[key]) for key in attrs.keys()}


def describe_dataset(dataset: h5py.Dataset) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "dataset",
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "compression": dataset.compression,
        "attrs": describe_attrs(dataset.attrs),
    }
    # State arrays are small per time step. A first/last preview helps identify
    # units and conventions without reading RGB or compressed video payloads.
    if (
        dataset.ndim >= 1
        and dataset.shape[0] > 0
        and dataset.dtype.kind in "biuf"
        and np.prod(dataset.shape[1:]) <= 32
    ):
        try:
            result["first"] = json_value(dataset[0])
            result["last"] = json_value(dataset[-1])
        except (OSError, ValueError) as exc:
            result["preview_error"] = str(exc)[:200]
    return result


def inspect(path: Path, max_entries: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "entries": {},
    }
    with h5py.File(path, "r") as h5:
        result["root_attrs"] = describe_attrs(h5.attrs)

        def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if len(result["entries"]) >= max_entries:
                result["truncated_at_entries"] = max_entries
                return
            if isinstance(obj, h5py.Dataset):
                result["entries"][name] = describe_dataset(obj)
            else:
                result["entries"][name] = {
                    "kind": "group",
                    "attrs": describe_attrs(obj.attrs),
                }

        h5.visititems(visit)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="A complete raw .h5/.hdf5 episode")
    parser.add_argument("--max-entries", type=int, default=1000)
    args = parser.parse_args()
    if args.max_entries < 1:
        parser.error("--max-entries must be positive")
    if not args.episode.is_file():
        parser.error(f"episode not found: {args.episode}")
    try:
        result = inspect(args.episode, args.max_entries)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({
            "path": str(args.episode),
            "file_size_bytes": os.path.getsize(args.episode),
            "error": str(exc),
        }, indent=2))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
