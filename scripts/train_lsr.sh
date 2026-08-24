#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="0"
DATA_ROOT="$REPO_ROOT/data/OpenImages"
CONFIG="$REPO_ROOT/lsr_training/configs/swinir-liif-latent-sdxl-v3.yaml"
OUTPUT_DIR="$REPO_ROOT/outputs/lsr/swinir-liif-latent-sdxl-v3"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -h|--help)
            echo "Usage: $0 [--gpus 0,1] [--data-root PATH] [--config PATH] [--output-dir PATH] [-- TRAIN_ARGS...]"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
NPROC_PER_NODE="${#GPU_LIST[@]}"

cd "$REPO_ROOT"
python3 scripts/check_dataset.py --data-root "$DATA_ROOT" --samples 4

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" \
    -m lsr_training.train \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}"
