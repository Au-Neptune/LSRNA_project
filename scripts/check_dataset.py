#!/usr/bin/env python3
"""Validate a prepared LSRNA dataset without modifying it."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


LATENT_DIRS = (
    "HR_sdxl_latent",
    "LR_sdxl_latent/X2",
    "LR_sdxl_latent/X3",
    "LR_sdxl_latent/X4",
)
VALID_DIRS = ("valid/HR", "valid/HR_resized")


def directory_files(path: Path, suffix: str | None = None) -> set[str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing directory: {path}")
    return {
        entry.name
        for entry in os.scandir(path)
        if entry.is_file() and (suffix is None or entry.name.lower().endswith(suffix))
    }


def sample_latents(data_root: Path, names: list[str], sample_count: int) -> list[dict]:
    if sample_count <= 0 or not names:
        return []
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required when --samples is greater than zero.") from exc

    indexes = sorted({round(i * (len(names) - 1) / max(sample_count - 1, 1)) for i in range(sample_count)})
    results = []
    for index in indexes:
        name = names[index]
        arrays = {}
        for relative_dir in LATENT_DIRS:
            array = np.load(data_root / relative_dir / name, mmap_mode="r")
            if array.ndim != 3 or array.shape[-1] != 4:
                raise ValueError(f"Unexpected latent shape at {relative_dir}/{name}: {array.shape}")
            arrays[relative_dir] = array

        hr_shape = arrays["HR_sdxl_latent"].shape
        for scale in (2, 3, 4):
            lr_shape = arrays[f"LR_sdxl_latent/X{scale}"].shape
            if hr_shape[0] != lr_shape[0] * scale or hr_shape[1] != lr_shape[1] * scale:
                raise ValueError(
                    f"Scale mismatch for {name} at X{scale}: HR={hr_shape}, LR={lr_shape}"
                )
        results.append(
            {
                "name": name,
                "dtype": str(arrays["HR_sdxl_latent"].dtype),
                "hr_shape": list(hr_shape),
                "lr_shapes": {
                    f"X{scale}": list(arrays[f"LR_sdxl_latent/X{scale}"].shape)
                    for scale in (2, 3, 4)
                },
            }
        )
    return results


def validate(data_root: Path, sample_count: int) -> dict:
    data_root = data_root.expanduser().resolve()
    latent_names = {
        relative_dir: directory_files(data_root / relative_dir, ".npy")
        for relative_dir in LATENT_DIRS
    }
    hr_names = latent_names["HR_sdxl_latent"]
    if not hr_names:
        raise ValueError(f"No .npy files found in {data_root / 'HR_sdxl_latent'}")

    mismatches = {}
    for relative_dir, names in latent_names.items():
        missing = sorted(hr_names - names)
        extra = sorted(names - hr_names)
        if missing or extra:
            mismatches[relative_dir] = {
                "missing_count": len(missing),
                "extra_count": len(extra),
                "missing_examples": missing[:10],
                "extra_examples": extra[:10],
            }
    if mismatches:
        raise ValueError(f"Latent filename mismatch: {json.dumps(mismatches, indent=2)}")

    validation_counts = {}
    for relative_dir in VALID_DIRS:
        files = directory_files(data_root / relative_dir)
        validation_counts[relative_dir] = len(files)
        if not files:
            raise ValueError(f"No validation images found in {data_root / relative_dir}")

    sorted_names = sorted(hr_names)
    return {
        "schema_version": 1,
        "data_root_name": data_root.name,
        "latent_file_count_per_scale": len(hr_names),
        "latent_directories": {key: len(value) for key, value in latent_names.items()},
        "validation_directories": validation_counts,
        "sample_validation": sample_latents(data_root, sorted_names, sample_count),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/OpenImages")
    parser.add_argument(
        "--samples", type=int, default=8,
        help="Number of evenly distributed latent pairs whose shapes and dtypes are inspected.",
    )
    parser.add_argument("--report", help="Optional path for a JSON validation report.")
    args = parser.parse_args()

    try:
        report = validate(Path(args.data_root), args.samples)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Dataset validation failed: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
