#!/bin/bash
# APT task-specific fine-tuning script supporting both DDP and DeepSpeed back-ends.
#
# Stage 0 (optional) fine-tunes the VA prior on the task data; Stage 1 then
# fine-tunes the full VLA policy on the same data. Either stage may also
# bootstrap from a pre-existing pretraining checkpoint via --pretrained-ckpt.
#
# Usage examples:
#
#   # DDP, both stages, starting from a pretrained VLA checkpoint
#   ./scripts/finetune.sh --backend ddp --gpus 0,1,2,3 \
#       --config finetune_libero --pretrained-ckpt my_vla_pretrain \
#       --va-name ft_va --vla-name ft_vla
#
#   # DeepSpeed ZeRO-3 + LoRA + gradient checkpointing, Stage-1 only
#   ./scripts/finetune.sh --backend deepspeed --gpus 0,1,2,3 --stage 1 \
#       --config finetune_libero --pretrained-ckpt my_vla_pretrain \
#       --vla-name ft_lora \
#       --ds-zero 3 --vlm-mode lora --gc --vlm-lr 5e-6 --accum 4 --bs 8

set -e

# ── Defaults ──────────────────────────────────────────────────────────────────
BACKEND="${BACKEND:-ddp}"
GPU_IDS="${GPU_IDS:-0,1}"
CONFIG_NAME="${CONFIG_NAME:-finetune_libero}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
VA_SAVE_NAME="${VA_SAVE_NAME:-}"
VLA_SAVE_NAME="${VLA_SAVE_NAME:-}"
VA_CONTI_NAME="${VA_CONTI_NAME:-}"
VLA_CONTI_NAME="${VLA_CONTI_NAME:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
BS="${BS:-64}"
MAX_ITER="${MAX_ITER:-25000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
STAGE="${STAGE:-both}"
# DeepSpeed-specific
DS_ZERO="${DS_ZERO:-2}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
VLM_MODE="${VLM_MODE:-frozen}"
VLM_LR="${VLM_LR:-}"
USE_GC="${USE_GC:-false}"

# ── Argument parser ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend)         BACKEND="$2";         shift 2 ;;
        --gpus)            GPU_IDS="$2";         shift 2 ;;
        --stage)           STAGE="$2";           shift 2 ;;
        --config)          CONFIG_NAME="$2";     shift 2 ;;
        --pretrained-ckpt) PRETRAINED_CKPT="$2"; shift 2 ;;
        --va-name)         VA_SAVE_NAME="$2";    shift 2 ;;
        --vla-name)        VLA_SAVE_NAME="$2";   shift 2 ;;
        --va-conti)        VA_CONTI_NAME="$2";   shift 2 ;;
        --vla-conti)       VLA_CONTI_NAME="$2";  shift 2 ;;
        --port)            MASTER_PORT="$2";     shift 2 ;;
        --bs)              BS="$2";              shift 2 ;;
        --max-iter)        MAX_ITER="$2";        shift 2 ;;
        --save-interval)   SAVE_INTERVAL="$2";   shift 2 ;;
        --ds-zero)         DS_ZERO="$2";         shift 2 ;;
        --accum)           ACCUM_STEPS="$2";     shift 2 ;;
        --vlm-mode)        VLM_MODE="$2";        shift 2 ;;
        --vlm-lr)          VLM_LR="$2";          shift 2 ;;
        --gc)              USE_GC="true";        shift   ;;
        -h|--help)
            cat <<EOF
Usage: $0 [OPTIONS]

Shared options:
  --backend ddp|deepspeed     Distributed back-end [default: $BACKEND]
  --gpus GPU_IDS              GPU IDs, e.g. '0,1,2,3' [default: $GPU_IDS]
  --stage 0|1|both            Stage to run [default: $STAGE]
  --config CONFIG_NAME        Fine-tuning config (see apt/configs.py) [default: $CONFIG_NAME]
  --pretrained-ckpt NAME      Pretraining checkpoint to bootstrap from
  --va-name NAME              Stage-0 save name
  --vla-name NAME             Stage-1 save name
  --va-conti NAME             Stage-0 resume checkpoint
  --vla-conti NAME            Stage-1 resume checkpoint
  --port PORT                 Master port [default: $MASTER_PORT]
  --bs BATCH_SIZE             Per-GPU batch size [default: $BS]
  --max-iter N                Max iterations [default: $MAX_ITER]
  --save-interval N           Save interval [default: $SAVE_INTERVAL]

DeepSpeed-only options (ignored when --backend ddp):
  --ds-zero 2|3               ZeRO stage [default: $DS_ZERO]
  --accum N                   Gradient accumulation steps [default: $ACCUM_STEPS]
  --vlm-mode frozen|lora|full VLM finetune mode [default: $VLM_MODE]
  --vlm-lr FLOAT              VLM learning rate (empty = same as max_lr)
  --gc                        Enable gradient checkpointing
EOF
            exit 0
            ;;
        *) echo "Unknown option: $1"; echo "Use --help for usage information"; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [[ ! "$BACKEND" =~ ^(ddp|deepspeed)$ ]]; then
    echo "Error: --backend must be 'ddp' or 'deepspeed'"; exit 1
fi
if [[ ! "$STAGE" =~ ^(0|1|both)$ ]]; then
    echo "Error: --stage must be '0', '1', or 'both'"; exit 1
fi
if [[ "$BACKEND" == "deepspeed" && ! "$DS_ZERO" =~ ^(2|3)$ ]]; then
    echo "Error: --ds-zero must be 2 or 3"; exit 1
fi
if [[ ( "$STAGE" == "0" || "$STAGE" == "both" ) && -z "$VA_SAVE_NAME"  && -z "$VA_CONTI_NAME"  ]]; then
    echo "Error: Stage 0 requires --va-name <name> (or --va-conti <name>)"; exit 1
fi
if [[ ( "$STAGE" == "1" || "$STAGE" == "both" ) && -z "$VLA_SAVE_NAME" && -z "$VLA_CONTI_NAME" ]]; then
    echo "Error: Stage 1 requires --vla-name <name> (or --vla-conti <name>)"; exit 1
fi

NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# ── Build extra args ──────────────────────────────────────────────────────────
EXTRA_ARGS="--vlm_finetune_mode $VLM_MODE"
if [[ -n "$VLM_LR" ]];          then EXTRA_ARGS="$EXTRA_ARGS --vlm_lr $VLM_LR"; fi
if [[ "$USE_GC" == "true" ]];   then EXTRA_ARGS="$EXTRA_ARGS --use_gradient_checkpointing"; fi
if [[ "$ACCUM_STEPS" -gt 1 ]];  then EXTRA_ARGS="$EXTRA_ARGS --gradient_accumulation_steps $ACCUM_STEPS"; fi

DS_ARGS=""
if [[ "$BACKEND" == "deepspeed" ]]; then
    DS_ARGS="--ds-config ds_config_zero${DS_ZERO}.json"
fi

# ── Print config ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "APT Fine-tuning Configuration:"
echo "  Back-end:        $BACKEND"
echo "  GPUs:            $GPU_IDS ($NUM_GPUS GPUs)"
echo "  Config:          $CONFIG_NAME"
echo "  Stage(s):        $STAGE"
[[ -n "$PRETRAINED_CKPT" ]] && echo "  Pretrained ckpt: $PRETRAINED_CKPT"
[[ -n "$VA_SAVE_NAME"    ]] && echo "  Stage-0 name:    $VA_SAVE_NAME"
[[ -n "$VA_CONTI_NAME"   ]] && echo "  Stage-0 resume:  $VA_CONTI_NAME"
[[ -n "$VLA_SAVE_NAME"   ]] && echo "  Stage-1 name:    $VLA_SAVE_NAME"
[[ -n "$VLA_CONTI_NAME"  ]] && echo "  Stage-1 resume:  $VLA_CONTI_NAME"
echo "  Master port:     $MASTER_PORT"
echo "  Batch size:      $BS"
echo "  Max iterations:  $MAX_ITER"
echo "  Save interval:   $SAVE_INTERVAL"
if [[ "$BACKEND" == "deepspeed" ]]; then
    echo "  ZeRO stage:      $DS_ZERO"
    echo "  Accum steps:     $ACCUM_STEPS  (eff. bs = $BS x $ACCUM_STEPS x $NUM_GPUS)"
    echo "  VLM mode:        $VLM_MODE"
    [[ -n "$VLM_LR"       ]] && echo "  VLM lr:          $VLM_LR"
    [[ "$USE_GC" == "true" ]] && echo "  Grad checkpoint: enabled"
fi
echo "=========================================="
echo ""

# ── Launcher helper ───────────────────────────────────────────────────────────
launch() {
    local stage_cmd="$1"
    local stage_id="$2"
    local extra="$3"  # per-stage extra args

    if [[ "$BACKEND" == "ddp" ]]; then
        CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun \
            --master_port $MASTER_PORT \
            --nproc_per_node $NUM_GPUS \
            -m apt.train \
            --backend ddp \
            --config $CONFIG_NAME \
            $stage_cmd \
            --train_stage $stage_id \
            --bs $BS \
            --max_iterations $MAX_ITER \
            --save_interval $SAVE_INTERVAL \
            $extra $EXTRA_ARGS
    else
        deepspeed \
            --include localhost:$GPU_IDS \
            --master_port $MASTER_PORT \
            --module apt.train \
            --backend deepspeed \
            $DS_ARGS \
            --config $CONFIG_NAME \
            $stage_cmd \
            --train_stage $stage_id \
            --bs $BS \
            --max_iterations $MAX_ITER \
            --save_interval $SAVE_INTERVAL \
            $extra $EXTRA_ARGS
    fi
}

# ── Stage 0: VA fine-tuning ───────────────────────────────────────────────────
if [[ "$STAGE" == "0" || "$STAGE" == "both" ]]; then
    echo "=========================================="
    echo "STAGE 0: Fine-tuning (VA model)"
    echo "=========================================="
    if [[ -n "$VA_CONTI_NAME" ]]; then
        echo "Resuming from checkpoint: $VA_CONTI_NAME"
        STAGE0_CMD="-c $VA_CONTI_NAME"
        STAGE0_EXTRA=""
    else
        echo "Starting new training, saving to: $VA_SAVE_NAME"
        STAGE0_CMD="-s $VA_SAVE_NAME"
        STAGE0_EXTRA=""
        if [[ -n "$PRETRAINED_CKPT" ]]; then
            STAGE0_EXTRA="--pretrained_ckpt $PRETRAINED_CKPT"
        fi
    fi
    launch "$STAGE0_CMD" 0 "$STAGE0_EXTRA"
    echo "Stage 0 completed successfully!"
    echo ""
fi

# ── Stage 1: VLA fine-tuning ──────────────────────────────────────────────────
if [[ "$STAGE" == "1" || "$STAGE" == "both" ]]; then
    echo "=========================================="
    echo "STAGE 1: Fine-tuning (VLA model)"
    echo "=========================================="

    # Pick the upstream checkpoint to adapt from (priority: va-conti > va-name > pretrained-ckpt).
    if [[ -n "$VA_CONTI_NAME" ]]; then
        VLA_CKPT="$VA_CONTI_NAME"
    elif [[ -n "$VA_SAVE_NAME" ]]; then
        VLA_CKPT="$VA_SAVE_NAME"
    elif [[ -n "$PRETRAINED_CKPT" ]]; then
        VLA_CKPT="$PRETRAINED_CKPT"
    else
        VLA_CKPT=""
    fi

    if [[ -n "$VLA_CONTI_NAME" ]]; then
        echo "Resuming Stage-1 from checkpoint: $VLA_CONTI_NAME"
        STAGE1_CMD="-c $VLA_CONTI_NAME"
        STAGE1_EXTRA=""
    else
        echo "Adapting from upstream checkpoint: $VLA_CKPT"
        STAGE1_CMD="-s $VLA_SAVE_NAME"
        STAGE1_EXTRA=""
        if [[ -n "$VLA_CKPT" ]]; then
            STAGE1_EXTRA="--pretrained_ckpt $VLA_CKPT"
            # When bootstrapping directly from a Stage-0 VA prior, expand layers.
            if [[ -n "$VA_CONTI_NAME" || -n "$VA_SAVE_NAME" ]]; then
                STAGE1_EXTRA="$STAGE1_EXTRA --load_from_va"
            fi
        fi
    fi
    launch "$STAGE1_CMD" 1 "$STAGE1_EXTRA"
    echo "Stage 1 completed successfully!"
    echo ""
fi

echo "=========================================="
echo "All fine-tuning stages completed!"
echo "=========================================="
