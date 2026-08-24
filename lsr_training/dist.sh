#!/usr/bin/env bash
set -euo pipefail

echo "lsr_training/dist.sh has been replaced by scripts/train_lsr.sh." >&2
echo "Run from the repository root, for example:" >&2
echo "  bash scripts/train_lsr.sh --gpus 0 --data-root data/OpenImages" >&2
exit 2
