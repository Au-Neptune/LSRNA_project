# LSRNA

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://3587jjh.github.io/LSRNA/)
[![arXiv](https://img.shields.io/badge/arXiv-2503.18446-b31b1b)](https://arxiv.org/abs/2503.18446)

Official code for **Latent Space Super-Resolution for Higher-Resolution Image
Generation with Diffusion Models**. This handoff version includes a prepared
dataset bundle, a portable LSR training entrypoint, and an independent final
LSRNA-DemoFusion pipeline.

![LSRNA overview](figures/teaser.jpg)

## Reproduction overview

The prepared dataset already contains the paired SDXL latents used to train the
LSR module. After extracting it, no crawling, RGB cropping, LR generation, or
VAE encoding is required.

```text
lsrna_train_latents_v1.zip
        │ extract
        ▼
data/OpenImages/{HR_sdxl_latent,LR_sdxl_latent,valid}
        │ scripts/train_lsr.sh
        ▼
outputs/lsr/<experiment>/iter_last.pth
        │ run_lsrna.py
        ▼
outputs/generation/<run>/{reference.png,final.png,run_config.json}
```

## Requirements

- Linux
- Python 3.10
- NVIDIA GPU with a CUDA-compatible PyTorch installation
- Enough local storage for the extracted latent dataset (approximately 98 GB)
- Hugging Face access to `stabilityai/stable-diffusion-xl-base-1.0`

The paper-scale settings are expensive. Start with the dataset check and a
short smoke run before launching the one-million-iteration configuration.

## Installation

```bash
conda create -n lsrna python=3.10
conda activate lsrna
pip install -r requirements.txt
```

Verify the important entrypoints:

```bash
python scripts/check_dataset.py --help
python run_lsrna.py --help
bash scripts/train_lsr.sh --help
```

To create a clean code handoff ZIP from the current working tree (excluding
`.git`, local datasets, outputs, dataset archives, and historical generated
images/config snapshots):

```bash
bash scripts/package_code.sh
```

## Prepared dataset

The handoff is split into independent archives:

| Archive | Contents | Needed for |
|---|---|---|
| `lsrna_train_latents_v1.zip` | HR latent, LR latent X2/X3/X4, validation RGB | LSR training |
| `lsrna_eval_data_v1.zip` | test/validation images and captions | Final evaluation |
| `lsrna_rgb_intermediates_v1.zip` | cropped HR and RGB LR X2/X3/X4 | Optional preprocessing inspection |
| `lsrna_smoke_data_v1.zip` | 8 paired latents and 2 validation images | Fast CPU/GPU smoke checks |

Each archive is accompanied by a `.sha256` checksum, a manifest, and a dataset
validation report. The RGB archive is not required for training.

### Extract and verify

From the repository root:

```bash
mkdir -p data
sha256sum -c lsrna_train_latents_v1.zip.sha256
unzip -q lsrna_train_latents_v1.zip -d data

python scripts/check_dataset.py \
    --data-root data/OpenImages \
    --samples 16
```

Expected layout:

```text
data/OpenImages/
├── HR_sdxl_latent/
├── LR_sdxl_latent/
│   ├── X2/
│   ├── X3/
│   └── X4/
└── valid/
    ├── HR/
    └── HR_resized/
```

The released training bundle contains 200,765 matching `.npy` files in each of
the four latent directories. `check_dataset.py` verifies filenames and samples
latent shapes before training.

### CPU reproduction smoke test

The smoke test exercises dataset loading, SwinIR-LIIF forward/backward,
optimizer and scheduler updates, checkpoint saving, and checkpoint loading by
the inference-side `lsr` package. It does not replace the CUDA SDXL pipeline
test.

```bash
unzip -q lsrna_smoke_data_v1.zip -d data/smoke

python scripts/check_dataset.py \
    --data-root data/smoke/OpenImages \
    --samples 8

python scripts/smoke_reproduction.py \
    --data-root data/smoke/OpenImages \
    --output-dir outputs/smoke_cpu \
    --iterations 2
```

To recreate the small dataset from a prepared full dataset:

```bash
python scripts/make_smoke_dataset.py \
    --source data/OpenImages \
    --output data/smoke/OpenImages \
    --count 8 \
    --valid-count 2
```

To build the same bundle from the original experiment machine:

```bash
bash scripts/package_dataset.sh \
    --data-root /path/to/OpenImages \
    --output-dir artifacts/datasets \
    --bundle train
```

Use `--bundle eval`, `--bundle rgb`, or `--bundle full` for the other variants.
The packager refuses to overwrite an existing archive and tests the ZIP before
writing its checksum.

> OpenImages files can carry image-specific licenses. Confirm redistribution
> terms before publishing these archives. Internal handoff should still retain
> dataset attribution and the supplied manifests.

## Train the LSR module

### Single GPU

```bash
bash scripts/train_lsr.sh \
    --gpus 0 \
    --data-root data/OpenImages \
    --config lsr_training/configs/swinir-liif-latent-sdxl-v3.yaml \
    --output-dir outputs/lsr/swinir
```

The script validates the dataset first and launches training with `torchrun`.
It does not activate Conda internally, so activate the intended environment
before running it. The training seed defaults to `0` and can be changed with
`-- --seed N`.

For a two-iteration GPU smoke test that still writes `iter_last.pth`:

```bash
bash scripts/train_lsr.sh \
    --gpus 0 \
    --data-root data/OpenImages \
    --output-dir outputs/lsr/smoke \
    -- --first-k 8 --batch-size 1 --num-workers 0 --max-iterations 2
```

### Multiple GPUs

```bash
bash scripts/train_lsr.sh \
    --gpus 0,1,2,3 \
    --data-root data/OpenImages \
    --config lsr_training/configs/swinir-liif-latent-sdxl-v3.yaml \
    --output-dir outputs/lsr/swinir
```

The configured total batch size and worker count must both be divisible by the
number of GPUs.

### Resume

Training starts fresh unless `--resume` is explicitly passed. Resume loads
`<output-dir>/iter_last.pth`.

```bash
bash scripts/train_lsr.sh \
    --gpus 0 \
    --data-root data/OpenImages \
    --output-dir outputs/lsr/swinir \
    -- --resume
```

Available backbones:

- `swinir-liif-latent-sdxl-v3.yaml` (default)
- `dat-liif-latent-sdxl.yaml`
- `hat-liif-latent-sdxl.yaml`
- `drct-liif-latent-sdxl.yaml`

Checkpoints contain the model, optimizer, scheduler, and iteration. The same
`iter_last.pth` format is accepted directly by the final pipeline.

## Run the final LSRNA pipeline

`run_lsrna.py` is independent of the training and evaluation scripts. It saves
the 1K reference, final target, and all run arguments in one output directory.

### Text-to-image

```bash
CUDA_VISIBLE_DEVICES=0 python run_lsrna.py \
    --prompt "A well-worn baseball glove and ball sitting on fresh-cut grass." \
    --negative-prompt "blurry, ugly, duplicate, poorly drawn, deformed, mosaic" \
    --height 2048 \
    --width 2048 \
    --seed 0 \
    --lsr-checkpoint outputs/lsr/swinir/iter_last.pth \
    --rna-min-std 0.0 \
    --rna-max-std 1.2 \
    --inversion-depth 30 \
    --output-dir outputs/generation/baseball
```

For a quick inference check, use the included checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python run_lsrna.py \
    --prompt "A small red cabin beside a lake at sunrise." \
    --height 2048 \
    --width 2048 \
    --lsr-checkpoint lsr/swinir-liif-latent-sdxl.pth \
    --output-dir outputs/generation/smoke
```

### Image-to-image upscaling

```bash
CUDA_VISIBLE_DEVICES=0 python run_lsrna.py \
    --prompt "A detailed mountain landscape reflected in a lake." \
    --image-lr path/to/reference.png \
    --height 2048 \
    --width 2048 \
    --seed 0 \
    --lsr-checkpoint outputs/lsr/swinir/iter_last.pth \
    --output-dir outputs/generation/mountain
```

The input image is converted to RGB and resized to the pipeline's 1K reference
resolution while preserving the requested target aspect ratio.

Important inference parameters:

| Argument | Default | Meaning |
|---|---:|---|
| `--height`, `--width` | 2048 | Target dimensions; the larger dimension must be divisible by 1024 |
| `--rna-min-std` | 0.0 | Minimum region-wise noise |
| `--rna-max-std` | 1.2 | Maximum region-wise noise; larger values add more detail/noise |
| `--inversion-depth` | 30 | RNA inversion depth from 1 to 50 |
| `--stride-ratio` | 0.5 | Patch stride relative to patch size |
| `--view-batch-size` | 8 | Number of views evaluated together |
| `--low-vram` | off | Offload components to reduce peak VRAM |
| `--overwrite` | off | Allow replacement of existing reference/final images |

## Outputs and troubleshooting

Training writes its resolved config and checkpoints below `--output-dir`.
Inference writes:

```text
outputs/generation/example/
├── reference.png
├── final.png
└── run_config.json
```

- **Dataset directory missing:** confirm extraction produced
  `data/OpenImages/HR_sdxl_latent`, not an extra nested directory.
- **Latent mismatch:** rerun `scripts/check_dataset.py` and compare the archive
  SHA-256 before training.
- **CUDA out of memory:** add `--low-vram`, reduce `--view-batch-size`, or begin
  with 2048×2048 output.
- **Hugging Face download failure:** authenticate with `huggingface-cli login`
  and confirm access to the configured SDXL model.
- **Resume checkpoint missing:** use the identical `--output-dir` that created
  `iter_last.pth`, then pass `-- --resume`.

## Citation

```bibtex
@inproceedings{jeong2025latent,
  title={Latent space super-resolution for higher-resolution image generation with diffusion models},
  author={Jeong, Jinho and Han, Sangmin and Kim, Jinwoo and Kim, Seon Joo},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={2355--2365},
  year={2025}
}
```

## Acknowledgements

This repository is based on
[DemoFusion](https://github.com/PRIS-CV/DemoFusion) and
[LIIF](https://github.com/yinboc/liif).
