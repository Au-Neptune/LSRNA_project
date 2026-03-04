#!/usr/bin/env bash

source /home/m11215122/miniconda3/etc/profile.d/conda.sh
conda activate lsrna

CUDA_VISIBLE_DEVICES=0 python main.py \
    --prompt "A realistic photo taken from behind of a police officer riding a blue police scooter down a cobblestone street. The officer is wearing a light blue shirt and a white helmet. The scooter has a blue flashing light on a pole. The street is paved with cobblestones. Bright sunny day, hard shadows, high resolution, street photography." \
    --negative_prompt "blurry, ugly, duplicate, poorly drawn, deformed, mosaic" \
    --height 2048 \
    --width 2048 \
    --seed 0 \
    --lsr_path "lsr/swinir-liif-latent-sdxl.pth" \
    --rna_min_std 0.0 \
    --rna_max_std 0.8 \
    --inversion_depth 30 \
    --save_dir "results" \
    #--low_vram
