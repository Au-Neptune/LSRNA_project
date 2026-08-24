#!/usr/bin/env python3
"""Create a small, self-contained dataset from a prepared LSRNA dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


LATENT_DIRS = (
    "HR_sdxl_latent",
    "LR_sdxl_latent/X2",
    "LR_sdxl_latent/X3",
    "LR_sdxl_latent/X4",
)
VALID_DIRS = ("valid/HR", "valid/HR_resized")


def copy_selected(source: Path, output: Path, relative_dir: str, names: list[str]) -> None:
    destination = output / relative_dir
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(source / relative_dir / name, destination / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Prepared OpenImages root.")
    parser.add_argument("--output", default="data/smoke/OpenImages")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--valid-count", type=int, default=2)
    args = parser.parse_args()

    if args.count < 1 or args.valid_count < 1:
        parser.error("--count and --valid-count must be at least 1")

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")

    latent_names = sorted(path.name for path in (source / LATENT_DIRS[0]).glob("*.npy"))[: args.count]
    if len(latent_names) != args.count:
        raise ValueError(f"Requested {args.count} latent pairs, found {len(latent_names)}")
    for relative_dir in LATENT_DIRS:
        missing = [name for name in latent_names if not (source / relative_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {relative_dir} pairs: {missing}")
        copy_selected(source, output, relative_dir, latent_names)

    validation = {}
    for relative_dir in VALID_DIRS:
        names = sorted(path.name for path in (source / relative_dir).iterdir() if path.is_file())[: args.valid_count]
        if len(names) != args.valid_count:
            raise ValueError(f"Requested {args.valid_count} files from {relative_dir}, found {len(names)}")
        copy_selected(source, output, relative_dir, names)
        validation[relative_dir] = names

    manifest = {
        "source_name": source.name,
        "latent_pairs": latent_names,
        "validation_images": validation,
    }
    (output / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {len(latent_names)} latent pairs at {output}")


if __name__ == "__main__":
    main()
