#!/usr/bin/env python3
"""Convert CattleEyeView YOLO polygon labels to COCO instance segmentation.

Each non-empty YOLO line is preserved as ONE independent cow instance.
No union/merge of polygons is performed. This is deliberate: VACA V1.4
must train true instance segmentation rather than binary foreground masks.

Label format expected per line:
    <class_id> x1 y1 x2 y2 ... xn yn
with normalized [0,1] coordinates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


def polygon_area_xy(points: list[float]) -> float:
    """Shoelace area for flattened [x1,y1,...] polygon in pixel units."""
    if len(points) < 6 or len(points) % 2:
        return 0.0
    xy = list(zip(points[0::2], points[1::2]))
    return abs(sum(x1*y2 - x2*y1 for (x1,y1),(x2,y2) in zip(xy, xy[1:]+xy[:1]))) / 2.0


def bbox_xywh(points: list[float]) -> list[float]:
    xs = points[0::2]
    ys = points[1::2]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return [xmin, ymin, xmax-xmin, ymax-ymin]


def parse_label(path: Path, width: int, height: int, class_id: int = 0) -> list[dict]:
    instances = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        toks = raw.split()
        try:
            cls = int(float(toks[0]))
            coords = [float(v) for v in toks[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid numeric token") from exc
        if cls != class_id:
            continue
        if len(coords) < 6 or len(coords) % 2:
            raise ValueError(f"{path}:{line_no}: polygon needs >=3 x/y pairs")
        pix = []
        for i in range(0, len(coords), 2):
            x = min(1.0, max(0.0, coords[i])) * width
            y = min(1.0, max(0.0, coords[i+1])) * height
            pix.extend([x, y])
        area = polygon_area_xy(pix)
        if not math.isfinite(area) or area <= 1.0:
            continue
        instances.append({"segmentation": [pix], "bbox": bbox_xywh(pix), "area": area})
    return instances


def iter_labels(label_dir: Path) -> Iterable[Path]:
    yield from sorted(label_dir.glob("*.txt"), key=lambda p: p.stem)


def convert(label_dir: Path, output_json: Path, width: int, height: int,
            image_prefix: str = "", image_ext: str = ".jpg", class_id: int = 0) -> dict:
    images, annotations = [], []
    ann_id = 1
    for image_id, label_path in enumerate(iter_labels(label_dir), 1):
        stem = label_path.stem
        file_name = f"{image_prefix}{stem}{image_ext}"
        images.append({"id": image_id, "file_name": file_name, "width": width, "height": height,
                       "frame_index": int(stem) if stem.isdigit() else stem})
        for inst in parse_label(label_path, width, height, class_id=class_id):
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": inst["segmentation"],
                "bbox": inst["bbox"],
                "area": inst["area"],
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {
            "description": "CattleEyeView polygons converted for VACA true instance segmentation",
            "vaca_contract": "one YOLO polygon line = one independent cow instance; never union masks",
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "cow", "supercategory": "animal"}],
        "images": images,
        "annotations": annotations,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return coco


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--image-prefix", default="")
    ap.add_argument("--image-ext", default=".jpg")
    ap.add_argument("--class-id", type=int, default=0)
    args = ap.parse_args()
    coco = convert(args.labels, args.output, args.width, args.height,
                   args.image_prefix, args.image_ext, args.class_id)
    print(f"images={len(coco['images'])} instances={len(coco['annotations'])} output={args.output}")

if __name__ == "__main__":
    main()
