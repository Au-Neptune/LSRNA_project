#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python run_lsrna.py \
    --prompt "A realistic photo taken from behind of a police officer riding a blue police scooter down a cobblestone street. The officer is wearing a light blue shirt and a white helmet. The scooter has a blue flashing light on a pole. The street is paved with cobblestones. Bright sunny day, hard shadows, high resolution, street photography." \
    --negative_prompt "blurry, ugly, duplicate, poorly drawn, deformed, mosaic" \
    --height 2048 \
    --width 2048 \
    --seed 0 \
    --lsr-checkpoint "lsr/swinir-liif-latent-sdxl.pth" \
    --rna-min-std 0.0 \
    --rna-max-std 0.8 \
    --inversion-depth 30 \
    --output-dir "outputs/generation/example"
