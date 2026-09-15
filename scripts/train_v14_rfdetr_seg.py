#!/usr/bin/env python3
"""GPU runner for VACA V1.4 RF-DETR instance segmentation fine-tuning.

Expected dataset layout follows current RF-DETR COCO training docs:
  DATASET/
    train/_annotations.coco.json
    valid/_annotations.coco.json
    test/_annotations.coco.json   # optional but recommended

Images referenced by each COCO JSON must exist in that split directory.
This runner intentionally refuses CPU training for benchmark runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=432)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for VACA V1.4 benchmark training; refusing CPU benchmark.")

    train_json = args.dataset_dir / "train" / "_annotations.coco.json"
    valid_json = args.dataset_dir / "valid" / "_annotations.coco.json"
    if not train_json.exists() or not valid_json.exists():
        raise SystemExit(f"Missing RF-DETR COCO split JSON: {train_json} / {valid_json}")

    from rfdetr import RFDETRSegMedium

    model = RFDETRSegMedium()
    model.train(
        dataset_dir=str(args.dataset_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        resolution=args.resolution,
        output_dir=str(args.output_dir),
    )


if __name__ == "__main__":
    main()
