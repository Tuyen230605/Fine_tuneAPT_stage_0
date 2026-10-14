#!/bin/bash
# Drive the four evaluation settings (so, uo, uc, uoue).
# Override these via environment variables:
#   GPU_ID    GPU index for Isaac Sim (default: 0)
#   URI       Pyro4 URI under which the model service is registered (default: control)
#   PORT      Pyro4 nameserver port (default: 9091)
#   SAVE_DIR  Root output directory; per-setting outputs go under SAVE_DIR/{so,uo,uc,uoue} (default: ./data/exp_results/run)
#
# This script is meant to be run from inside ``examples/PickPlace/``. We
# prepend the APT repository root to PYTHONPATH so that the Isaac Sim
# Python interpreter can import ``shm_transport`` from ``APT/`` instead of
# requiring a vendored copy under this folder.

set -e

GPU_ID="${GPU_ID:-0}"
URI="${URI:-control}"
PORT="${PORT:-9091}"
SAVE_DIR="${SAVE_DIR:-./data/exp_results/run}"

# Make APT/ importable so ``from shm_transport import ...`` resolves.
APT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${APT_ROOT}:${PYTHONPATH:-}"

# seen object
python -m eval.eval_policy --setting so \
    -i "$GPU_ID" --uri "$URI" --port "$PORT" --no_gui --save \
    --save_dir "${SAVE_DIR}/so"

# unseen object
python -m eval.eval_policy --setting uo \
    -i "$GPU_ID" --uri "$URI" --port "$PORT" --no_gui --save \
    --save_dir "${SAVE_DIR}/uo"

# unseen container
python -m eval.eval_policy --setting uc \
    -i "$GPU_ID" --uri "$URI" --port "$PORT" --no_gui --save \
    --save_dir "${SAVE_DIR}/uc"

# unseen object + unseen environment (sweep several seeds)
for SEED in 0 20 30 40 50 61 71; do
    python -m eval.eval_policy --setting uo \
        -i "$GPU_ID" --uri "$URI" --port "$PORT" --no_gui --save \
        --novel_bg --random_env --seed "$SEED" \
        --save_dir "${SAVE_DIR}/uoue"
done
