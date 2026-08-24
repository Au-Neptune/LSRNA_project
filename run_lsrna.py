#!/usr/bin/env python3
"""Run the final LSRNA-DemoFusion pipeline from a trained LSR checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image

from pipeline_lsrna_demofusion_sdxl import DemoFusionLSRNASDXLPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--negative-prompt",
        default="blurry, ugly, duplicate, poorly drawn, deformed, mosaic",
    )
    parser.add_argument("--image-lr", help="Optional low-resolution reference image.")
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lsr-checkpoint", required=True)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--output-dir", default="outputs/generation")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--view-batch-size", type=int, default=8)
    parser.add_argument("--stride-ratio", type=float, default=0.5)
    parser.add_argument("--rna-min-std", type=float, default=0.0)
    parser.add_argument("--rna-max-std", type=float, default=1.2)
    parser.add_argument("--inversion-depth", type=int, default=30)
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    if not torch.cuda.is_available():
        raise RuntimeError("LSRNA inference requires a CUDA GPU.")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive.")
    if max(args.height, args.width) < 1024 or max(args.height, args.width) % 1024 != 0:
        raise ValueError("The larger target dimension must be a positive multiple of 1024.")
    scale = max(args.height, args.width) // 1024
    if args.height % (scale * 8) != 0 or args.width % (scale * 8) != 0:
        raise ValueError("Both target dimensions divided by the LSR scale must be divisible by 8.")
    if not 0 < args.stride_ratio <= 1:
        raise ValueError("--stride-ratio must be in (0, 1].")
    if not 1 <= args.inversion_depth <= 50:
        raise ValueError("--inversion-depth must be between 1 and 50.")

    checkpoint = Path(args.lsr_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LSR checkpoint not found: {checkpoint}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    expected_outputs = (output_dir / "reference.png", output_dir / "final.png")
    if not args.overwrite and any(path.exists() for path in expected_outputs):
        raise FileExistsError(f"Output already exists in {output_dir}; pass --overwrite to replace it.")
    if args.image_lr and not Path(args.image_lr).expanduser().is_file():
        raise FileNotFoundError(f"Input image not found: {args.image_lr}")
    return checkpoint, output_dir


def load_reference_image(path: str, height: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    scale = max(height, width) // 1024
    reference_size = (width // scale, height // scale)
    image = Image.open(path).convert("RGB").resize(reference_size, Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=dtype)


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.Generator(device="cuda").manual_seed(seed)


def main() -> None:
    args = parse_args()
    checkpoint, output_dir = validate_args(args)
    generator = seed_everything(args.seed)

    scheduler = DDIMScheduler.from_pretrained(args.model, subfolder="scheduler")
    pipe = DemoFusionLSRNASDXLPipeline.from_pretrained(
        args.model,
        scheduler=scheduler,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.vae.enable_tiling()

    image_lr = None
    if args.image_lr:
        image_lr = load_reference_image(args.image_lr, args.height, args.width, torch.float16)

    images = pipe(
        args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
        view_batch_size=args.view_batch_size,
        stride_ratio=args.stride_ratio,
        lsr_path=str(checkpoint),
        cosine_scale_1=3,
        cosine_scale_2=1,
        cosine_scale_3=1,
        sigma=0.8,
        rna_min_std=args.rna_min_std,
        rna_max_std=args.rna_max_std,
        inversion_depth=args.inversion_depth,
        low_vram=args.low_vram,
        image_lr=image_lr,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    images[0].save(output_dir / "reference.png")
    images[-1].save(output_dir / "final.png")
    run_config = vars(args).copy()
    run_config["lsr_checkpoint"] = str(checkpoint)
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved reference and final images to {output_dir}")


if __name__ == "__main__":
    main()
