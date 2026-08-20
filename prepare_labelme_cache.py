#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
from detectron2.structures import BoxMode
from tqdm import tqdm


def parse_class_names(value: str) -> List[str]:
    names = [x.strip() for x in value.replace(",", " ").split() if x.strip()]
    if not names:
        raise ValueError("class_names 不能为空")
    return names


def create_data_pairs_labelme(
    dataset_root: str,
    split: str,
    img_exts=(".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"),
) -> List[Tuple[str, str]]:
    split_dir = Path(dataset_root) / split
    if not split_dir.exists():
        print(f"[WARN] split dir not found: {split_dir}")
        return []

    img_paths = []
    for ext in img_exts:
        img_paths.extend(split_dir.glob(f"*{ext}"))
    img_paths = sorted(set(img_paths))

    pairs = []
    for img_path in img_paths:
        json_path = img_path.with_suffix(".json")
        if json_path.is_file():
            pairs.append((str(img_path), str(json_path)))

    print(f"[OK] {split}: matched {len(pairs)} image-json pairs")
    return pairs


def _shape_points_to_polygon(points):
    poly = []
    for x, y in points:
        poly.extend([float(x), float(y)])
    return poly


def _circle_points_to_polygon(points, num_points=72):
    if not points or len(points) < 2:
        return None

    (cx, cy) = points[0]
    (px, py) = points[1]
    cx, cy, px, py = float(cx), float(cy), float(px), float(py)

    radius = math.hypot(px - cx, py - cy)
    if radius <= 1e-6:
        return None

    poly = []
    for k in range(num_points):
        theta = 2.0 * math.pi * k / num_points
        poly.extend([
            float(cx + radius * math.cos(theta)),
            float(cy + radius * math.sin(theta)),
        ])

    return poly if len(poly) >= 6 else None


def _clip_poly_to_image(poly, width, height):
    clipped = []
    for i in range(0, len(poly), 2):
        x = max(0.0, min(float(poly[i]), float(width - 1)))
        y = max(0.0, min(float(poly[i + 1]), float(height - 1)))
        clipped.extend([x, y])
    return clipped


def get_labelme_instance_dicts(
    dataset_root: str,
    split: str,
    class_names: List[str],
    circle_num_points: int = 72,
) -> List[Dict[str, Any]]:
    name2id = {name: i for i, name in enumerate(class_names)}
    pairs = create_data_pairs_labelme(dataset_root, split)

    dataset_dicts = []
    for image_id, (img_path, json_path) in enumerate(
        tqdm(pairs, desc=f"Converting {split}", unit="img", ncols=100)
    ):
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"cv2.imread failed: {img_path}")
        height, width = image.shape[:2]

        record = {
            "file_name": img_path,
            "image_id": image_id,
            "height": height,
            "width": width,
        }

        with open(json_path, "r", encoding="utf-8") as f:
            anno = json.load(f)

        objs = []
        for shape in anno.get("shapes", []):
            label = shape.get("label")
            if label not in name2id:
                continue

            shape_type = shape.get("shape_type", "polygon")
            points = shape.get("points")
            poly = None

            if shape_type == "polygon":
                if points and len(points) >= 3:
                    poly = _shape_points_to_polygon(points)
            elif shape_type == "rectangle":
                if points and len(points) >= 2:
                    (x1, y1), (x2, y2) = points[0], points[1]
                    poly = [
                        float(x1), float(y1),
                        float(x2), float(y1),
                        float(x2), float(y2),
                        float(x1), float(y2),
                    ]
            elif shape_type == "circle":
                poly = _circle_points_to_polygon(points, num_points=circle_num_points)

            if poly is None or len(poly) < 6:
                continue

            poly = _clip_poly_to_image(poly, width, height)
            xs = poly[0::2]
            ys = poly[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            if x_max <= x_min or y_max <= y_min:
                continue

            objs.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": int(name2id[label]),
                "iscrowd": 0,
            })

        record["annotations"] = objs
        dataset_dicts.append(record)

    print(f"[OK] {split}: built {len(dataset_dicts)} records")
    return dataset_dicts


def save_json_gz(data: Any, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"[OVERWRITE] existing file will be replaced: {out_path}")
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    size = len(data) if hasattr(data, "__len__") else "unknown"
    print(f"[OK] saved: {out_path} ({size} records)")


def parse_args():
    parser = argparse.ArgumentParser("Prepare Detectron2 LabelMe cache")
    parser.add_argument("--dataset-root", "--DATASET_ROOT", required=True)
    parser.add_argument("--cache-dir", "--CACHE_DIR", required=True)
    parser.add_argument(
        "--class-names",
        "--class_names",
        required=True,
        help='Comma or space separated class names, e.g. "agg,block,rod"',
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="dataset splits to convert",
    )
    parser.add_argument("--circle-num-points", type=int, default=72)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="overwrite existing cache files; enabled by default",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    class_names = parse_class_names(args.class_names)

    dataset_root = str(Path(args.dataset_root).resolve())
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] cache dir: {cache_dir}")
    print("[INFO] existing cache files with the same names will be overwritten")

    for split in args.splits:
        dicts = get_labelme_instance_dicts(
            dataset_root=dataset_root,
            split=split,
            class_names=class_names,
            circle_num_points=args.circle_num_points,
        )
        save_json_gz(dicts, str(cache_dir / f"{split}_dicts.json.gz"))

    meta = {
        "class_names": class_names,
        "circle_num_points": args.circle_num_points,
        "dataset_root": dataset_root,
        "splits": args.splits,
    }
    save_json_gz(meta, str(cache_dir / "meta.json.gz"))


if __name__ == "__main__":
    main()
