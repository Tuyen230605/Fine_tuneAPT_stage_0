#!/bin/bash
# Drive ``examples.libero.eval_libero_pred`` across one of three benchmarks:
#
#   --benchmark libero       — 4 suites: libero_{object,spatial,goal,10}              (default 50 trials/task)
#   --benchmark libero-pro   — 8 suites: above × {_swap,_task}                         (default 50 trials/task)
#   --benchmark libero-plus  — 4 suites: libero_{object,spatial,goal,10}              (default  1 trial /task)
#
# Override the suite list with ``--suites "a b c"`` (space-separated).
# Override the per-task trial count with ``--num_trials_per_task N``.
#
# Run-time pre-requisites:
#   1. APT policy server already up (see ../../README.md  →  "Inference").
#   2. ``conda activate`` the LIBERO env that matches the chosen benchmark.

set -e

# ─────── Defaults ───────────────────────────────────────────────────────────
BENCHMARK="${BENCHMARK:-libero}"
GPU="${GPU:-0}"
CONTROLLER_NAME="${CONTROLLER_NAME:-control}"
CONTROLLER_PORT="${CONTROLLER_PORT:-9091}"
MODEL_NAME="${MODEL_NAME:-apt}"
BEGIN_TASK_ID="${BEGIN_TASK_ID:-0}"
SUITES=""                       # if empty, auto-fill from BENCHMARK
NUM_TRIALS_PER_TASK=""          # if empty, auto-fill from BENCHMARK
SAVE_ROOT="${SAVE_ROOT:-results}"
VIDEO_ROOT="${VIDEO_ROOT:-videos}"
SUFFIX="${SUFFIX:-all}"
SAVE_FLAG="--save"
VIDEO_FLAG="--video"
ALL_FLAG="--all"
CONTI_SAVE_FLAG="--conti_save"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]
Options:
  --benchmark {libero|libero-pro|libero-plus}
                                  Benchmark family [default: $BENCHMARK]
  --gpu ID                        GPU device ID [default: $GPU]
  --controller_name NAME          shm_transport controller name [default: $CONTROLLER_NAME]
  --controller_port PORT          shm_transport naming-server port [default: $CONTROLLER_PORT]
  --model_name NAME               Label used in output paths [default: $MODEL_NAME]
  --num_trials_per_task N         Trials per task (default: 50 for libero/libero-pro, 1 for libero-plus)
  --begin_task_id N               Skip first N tasks [default: $BEGIN_TASK_ID]
  --suites "s1 s2 ..."            Override the auto-derived suite list
  --save_root  PATH               CSV log root [default: $SAVE_ROOT]
  --video_root PATH               MP4 root [default: $VIDEO_ROOT]
  --suffix STR                    Output subdir suffix [default: $SUFFIX]
  --no-save                       Do not write CSV
  --no-video                      Do not write MP4
  --no-all                        Only run the first trial per task
  --no-conti                      Overwrite CSV (default appends)
  -h, --help                      Show this help and exit
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark)           BENCHMARK="$2";           shift 2 ;;
        --gpu)                 GPU="$2";                 shift 2 ;;
        --controller_name)     CONTROLLER_NAME="$2";     shift 2 ;;
        --controller_port)     CONTROLLER_PORT="$2";     shift 2 ;;
        --model_name)          MODEL_NAME="$2";          shift 2 ;;
        --num_trials_per_task) NUM_TRIALS_PER_TASK="$2"; shift 2 ;;
        --begin_task_id)       BEGIN_TASK_ID="$2";       shift 2 ;;
        --suites)              SUITES="$2";              shift 2 ;;
        --save_root)           SAVE_ROOT="$2";           shift 2 ;;
        --video_root)          VIDEO_ROOT="$2";          shift 2 ;;
        --suffix)              SUFFIX="$2";              shift 2 ;;
        --no-save)             SAVE_FLAG="";             shift ;;
        --no-video)            VIDEO_FLAG="";            shift ;;
        --no-all)              ALL_FLAG="";              shift ;;
        --no-conti)            CONTI_SAVE_FLAG="";       shift ;;
        -h|--help)             usage ;;
        *) echo "Unknown argument: $1"; echo "Use --help for usage"; exit 1 ;;
    esac
done

# ─────── Benchmark presets ──────────────────────────────────────────────────
case "$BENCHMARK" in
    libero)
        DEFAULT_SUITES="libero_object libero_spatial libero_goal libero_10"
        DEFAULT_TRIALS=50
        ;;
    libero-pro)
        DEFAULT_SUITES=$(printf "%s_%s " \
            "libero_object" "libero_spatial" "libero_goal" "libero_10" \
            | sed -e 's/\([a-z0-9]\) /\1_swap /g' \
            | awk '{
                for (i=1;i<=NF;i++) printf "%s ", $i;
                gsub(/_swap/, "_task");
                for (i=1;i<=NF;i++) printf "%s ", $i;
            }')
        # The shell juggling above is fragile — fall back to the explicit list.
        DEFAULT_SUITES="libero_object_swap libero_object_task libero_spatial_swap libero_spatial_task libero_goal_swap libero_goal_task libero_10_swap libero_10_task"
        DEFAULT_TRIALS=50
        ;;
    libero-plus)
        DEFAULT_SUITES="libero_object libero_spatial libero_goal libero_10"
        DEFAULT_TRIALS=1
        ;;
    *)
        echo "Error: --benchmark must be one of {libero, libero-pro, libero-plus}"
        exit 1
        ;;
esac
[[ -z "$SUITES"              ]] && SUITES="$DEFAULT_SUITES"
[[ -z "$NUM_TRIALS_PER_TASK" ]] && NUM_TRIALS_PER_TASK="$DEFAULT_TRIALS"

# ─────── Echo config ────────────────────────────────────────────────────────
echo "=========================================="
echo "Libero evaluation:"
echo "  benchmark:           $BENCHMARK"
echo "  suites:              $SUITES"
echo "  num_trials_per_task: $NUM_TRIALS_PER_TASK"
echo "  gpu:                 $GPU"
echo "  controller:          $CONTROLLER_NAME @ port $CONTROLLER_PORT"
echo "  model_name:          $MODEL_NAME"
echo "  begin_task_id:       $BEGIN_TASK_ID"
echo "  save_root:           $SAVE_ROOT"
echo "  video_root:          $VIDEO_ROOT"
echo "  suffix:              $SUFFIX"
echo "=========================================="
echo ""

# ─────── Run ────────────────────────────────────────────────────────────────
for LIBERO_TASK_SUITE in $SUITES; do
    CUDA_VISIBLE_DEVICES=$GPU python -m examples.libero.eval_libero_pred \
        --libero_task_suite "$LIBERO_TASK_SUITE" \
        --model_name        "$MODEL_NAME" \
        --controller_name   "$CONTROLLER_NAME" \
        --controller_port   "$CONTROLLER_PORT" \
        --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
        --begin_task_id     "$BEGIN_TASK_ID" \
        --save_root         "$SAVE_ROOT" \
        --video_root        "$VIDEO_ROOT" \
        --suffix            "$SUFFIX" \
        $SAVE_FLAG $VIDEO_FLAG $ALL_FLAG $CONTI_SAVE_FLAG
done
