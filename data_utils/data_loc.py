"""Dataset location lookup.

The set of datasets the trainer knows about is fixed by the ``DSET`` enum.
The mapping from ``DSET`` member → on-disk path is *machine-specific* and
lives in the YAML side-car ``data_loc.yaml`` (or, for personal overrides
that should never be committed, ``data_loc.local.yaml``).

YAML structure::

    hosts:
      "<your-host-ip>":
        LiberoSpatial: /path/to/libero_spatial_no_noops/**/*.h5
        Droid:         /path/to/droid_1.0.1_h5_jpeg
        ...
      "<another-host>":
        ...

Resolution order:

1. ``APT_DATA_LOC`` env var, if set (absolute path to a YAML file).
2. ``data_utils/data_loc.local.yaml`` (gitignored — personal overrides).
3. ``data_utils/data_loc.yaml`` (the checked-in template).

For each candidate the matched host block is the first key prefix-matched
against any of this machine's IPv4 addresses; if none match, the trainer
falls back to the empty mapping (and ``get_loc`` will raise as soon as
any dataset is actually instantiated).

Set ``APT_DATA_LOC_HOST`` to override IP detection (useful in CI / docker).
"""
import os
import socket
from enum import IntEnum, auto
from typing import Dict, List, Optional

import psutil


# ──────────────────────────────────────────────────────────────────────────────
# DSET — the canonical dataset name set
# ──────────────────────────────────────────────────────────────────────────────

class DSET(IntEnum):
    # LIBERO suites
    LiberoSpatial = auto()
    LiberoObject = auto()
    LiberoGoal = auto()
    Libero10 = auto()

    # Large-scale robot trajectory datasets
    Droid = auto()
    AgibotWorldAlpha = auto()
    InternM1Franka = auto()
    InternA1Franka = auto()
    InternA1Lift2 = auto()
    InternA1Genie1 = auto()
    InternA1SplitAloha = auto()

    # Task-specific
    PickPlaceCan = auto()
    AlohaTableStorage = auto()
    AlohaPickPlace = auto()
    AlohaPickPlaceClutter = auto()


_DSET_BY_NAME: Dict[str, DSET] = {m.name: m for m in DSET}


# ──────────────────────────────────────────────────────────────────────────────
# Host / IP detection
# ──────────────────────────────────────────────────────────────────────────────

def get_ipv4_addresses() -> List[str]:
    """Return all IPv4 addresses bound to this host's interfaces."""
    out: List[str] = []
    for _iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                out.append(addr.address)
    return out


def get_ipv4_address() -> str:
    """Pick a representative IPv4 address. Prefer ``172.*`` (lab subnet).

    Override with ``APT_DATA_LOC_HOST`` env var when running in CI/docker
    where the lab IP heuristic doesn't apply.
    """
    override = os.environ.get("APT_DATA_LOC_HOST")
    if override:
        return override
    ips = get_ipv4_addresses()
    for ip in ips:
        if ip.startswith("172."):
            return ip
    raise ValueError(f"No matched ip. Current ips are {ips}")


# ──────────────────────────────────────────────────────────────────────────────
# YAML loading
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_YAML_NAME = "data_loc.yaml"
_LOCAL_YAML_NAME = "data_loc.local.yaml"


def _candidate_yaml_paths() -> List[str]:
    """Yaml files to try, in priority order."""
    candidates: List[str] = []
    env_path = os.environ.get("APT_DATA_LOC")
    if env_path:
        candidates.append(env_path)
    here = os.path.dirname(__file__)
    candidates.append(os.path.join(here, _LOCAL_YAML_NAME))
    candidates.append(os.path.join(here, _DEFAULT_YAML_NAME))
    return candidates


def _match_host_block(yaml_data: dict, host_ip: str) -> Optional[dict]:
    """From a parsed YAML doc, pick the host block matching ``host_ip``.

    Match strategy: exact match first, then any key that ``host_ip``
    starts with (so e.g. ``172.16.5.34`` matches a ``172.16`` block).
    """
    hosts = (yaml_data or {}).get("hosts") or {}
    if host_ip in hosts:
        return hosts[host_ip]
    for key, block in hosts.items():
        if host_ip.startswith(str(key)):
            return block
    return None


def _load_yaml(path: str) -> Optional[dict]:
    try:
        import yaml  # PyYAML, in requirements.txt
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to load data_loc.yaml. "
            "Install via `pip install pyyaml`."
        ) from e
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _resolve_locations(host_ip: str) -> Dict[int, Optional[str]]:
    """Walk the candidate YAML files and return ``{DSET_member.value: path}``."""
    for path in _candidate_yaml_paths():
        if not path or not os.path.isfile(path):
            continue
        block = _match_host_block(_load_yaml(path), host_ip)
        if block is None:
            continue
        loc: Dict[int, Optional[str]] = {}
        for name, value in block.items():
            if name not in _DSET_BY_NAME:
                print(f"[WARN] data_loc YAML at {path} mentions unknown "
                      f"dataset {name!r}; skipping.")
                continue
            loc[_DSET_BY_NAME[name]] = value
        print(f"[INFO] data_loc: loaded {len(loc)} entries from {path} "
              f"for host {host_ip}")
        return loc
    print(f"[WARN] data_loc: no YAML matched host {host_ip}. "
          "Datasets will fail with `No path defined` when instantiated. "
          "Create data_loc.local.yaml or set APT_DATA_LOC.")
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Public surface
# ──────────────────────────────────────────────────────────────────────────────

# dataset_locations is kept for backwards-compatibility with any caller that
# imported it. It contains a single entry — the resolved block for *this* host.
dataset_locations: Dict[str, Dict[int, Optional[str]]] = {}

try:
    _host = get_ipv4_address()
    LOC: Dict[int, Optional[str]] = _resolve_locations(_host)
    dataset_locations[_host] = LOC
except ValueError as _e:
    # No matching IP — leave LOC empty; get_loc() will raise on first use.
    print(f"[WARN] data_loc: {_e}. LOC is empty.")
    LOC = {}
