#!/usr/bin/env python3
"""Run a two-iteration CPU smoke test and verify checkpoint inference loading."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lsr
from lsr_training import datasets, models
from lsr_training.utils import make_optim_sched


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/smoke/OpenImages")
    parser.add_argument(
        "--config",
        default="lsr_training/configs/swinir-liif-latent-sdxl-v3.yaml",
    )
    parser.add_argument("--output-dir", default="outputs/smoke_cpu")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")

    seed_everything(args.seed)
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_spec = copy.deepcopy(config["train_dataset"]["dataset"])
    dataset_spec["args"]["hr_path"] = str(data_root / "HR_sdxl_latent")
    dataset_spec["args"]["lr_path"] = str(data_root / "LR_sdxl_latent")
    raw_dataset = datasets.make(dataset_spec)
    dataset = datasets.make(
        copy.deepcopy(config["train_dataset"]["wrapper"]),
        args={"dataset": raw_dataset},
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model_spec = copy.deepcopy(config["model"])
    optimizer_spec = copy.deepcopy(config["optimizer"])
    scheduler_spec = copy.deepcopy(config["lr_scheduler"])
    model = models.make(model_spec).cpu().train()
    optimizer, scheduler = make_optim_sched(
        model.parameters(), optimizer_spec, scheduler_spec
    )
    loss_fn = nn.L1Loss()

    losses = []
    iterator = iter(loader)
    last_batch = None
    for iteration in range(1, args.iterations + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad()
        prediction = model(batch["lr"], batch["coord"], batch["cell"])
        loss = loss_fn(prediction, batch["hr"])
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        last_batch = batch
        print(f"iteration={iteration} loss={loss.item():.6f}")

    model_spec["sd"] = model.state_dict()
    optimizer_spec["sd"] = optimizer.state_dict()
    scheduler_spec["sd"] = scheduler.state_dict()
    checkpoint = {
        "model": model_spec,
        "optimizer": optimizer_spec,
        "lr_scheduler": scheduler_spec,
        "iter": args.iterations,
    }
    checkpoint_path = output_dir / "iter_last.pth"
    torch.save(checkpoint, checkpoint_path)

    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    inference_model = lsr.models.make(loaded["model"], load_sd=True).cpu().eval()
    model.eval()
    with torch.no_grad():
        train_output = model(last_batch["lr"], last_batch["coord"], last_batch["cell"])
        inference_output = inference_model(
            last_batch["lr"], last_batch["coord"], last_batch["cell"]
        )
    max_abs_diff = (train_output - inference_output).abs().max().item()
    if max_abs_diff > 1e-6:
        raise RuntimeError(f"Checkpoint reload output mismatch: {max_abs_diff}")

    report = {
        "status": "passed",
        "device": "cpu",
        "dataset_pairs": len(dataset),
        "iterations": args.iterations,
        "losses": losses,
        "checkpoint": str(checkpoint_path),
        "checkpoint_reload_max_abs_diff": max_abs_diff,
        "final_output_shape": list(inference_output.shape),
    }
    report_path = output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
