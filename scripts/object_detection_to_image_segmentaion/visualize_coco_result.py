"""
Visualize one image from a COCO detection or segmentation annotation file.

This helper is useful for quickly checking the original object-detection labels
or the converted SAM2 segmentation result on top of the source image.

Examples:
    python -m scripts.object_detection_to_image_segmentaion.visualize_coco_result ^
      --annotations C:\\Users\\B50137\\Downloads\\hiod_coco\\content\\data\\hiod_coco\\train\\_annotations.coco.json ^
      --image-root C:\\Users\\B50137\\Downloads\\hiod_coco\\content\\data\\hiod_coco\\train ^
      --image-name image_1003_jpg.rf.599632984c37efa7d4c06fc3dfff5cf1.jpg ^
      --output outputs\\viz_detection.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_coco(ann_path: Path) -> dict:
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(image_root: Path, file_name: str) -> Path:
    direct = image_root / file_name
    if direct.exists():
        return direct

    by_name = image_root / Path(file_name).name
    if by_name.exists():
        return by_name

    stem = Path(file_name).stem
    for ext in IMAGE_EXTENSIONS:
        candidate = image_root / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found under {image_root}: {file_name}")


def anns_by_image_id(annotations: List[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for ann in annotations:
        grouped.setdefault(ann["image_id"], []).append(ann)
    return grouped


def make_color(seed: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed)
    color = rng.integers(64, 255, size=3, dtype=np.int32)
    return int(color[0]), int(color[1]), int(color[2])


def draw_polygon(canvas: np.ndarray, polygon: List[float], color: tuple[int, int, int]) -> None:
    if len(polygon) < 6 or len(polygon) % 2 != 0:
        return

    points = np.array(polygon, dtype=np.float32).reshape(-1, 2).astype(np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [points], color)
    canvas[:] = cv2.addWeighted(canvas, 1.0, overlay, 0.25, 0.0)
    cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=2)


def visualize_image(
    image_bgr: np.ndarray,
    annotations: List[dict],
    categories_by_id: Dict[int, dict],
) -> np.ndarray:
    canvas = image_bgr.copy()

    for ann in annotations:
        category = categories_by_id.get(ann["category_id"], {})
        class_name = str(category.get("name", ann["category_id"]))
        color = make_color(ann["category_id"] * 101 + ann["id"] * 17)

        bbox = ann.get("bbox", None)
        if bbox:
            x, y, w, h = [float(v) for v in bbox]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} id={ann['id']}"
            cv2.putText(
                canvas,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        segmentation = ann.get("segmentation", [])
        if isinstance(segmentation, list):
            for polygon in segmentation:
                if isinstance(polygon, list):
                    draw_polygon(canvas, polygon, color)

    return canvas


def pick_image_info(coco: dict, image_name: str | None, image_id: int | None) -> dict:
    images = coco.get("images", [])
    if image_id is not None:
        for image_info in images:
            if int(image_info["id"]) == int(image_id):
                return image_info
        raise KeyError(f"image_id {image_id} not found in annotations")

    if image_name is not None:
        image_name_only = Path(image_name).name
        for image_info in images:
            if Path(image_info["file_name"]).name == image_name_only:
                return image_info
        raise KeyError(f"image_name {image_name_only} not found in annotations")

    if not images:
        raise RuntimeError("No images found in COCO annotations")
    return images[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one image from a COCO annotation file")
    parser.add_argument("--annotations", required=True, help="Path to _annotations.coco.json")
    parser.add_argument("--image-root", required=True, help="Directory containing the image files")
    parser.add_argument("--image-name", default=None, help="Specific image file name to visualize")
    parser.add_argument("--image-id", type=int, default=None, help="Specific COCO image id to visualize")
    parser.add_argument("--output", default="output/coco_visualization.jpg", help="Where to save the rendered image")
    parser.add_argument("--show", action="store_true", help="Show the visualization in an OpenCV window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ann_path = Path(args.annotations)
    image_root = Path(args.image_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coco = load_coco(ann_path)
    image_info = pick_image_info(coco, image_name=args.image_name, image_id=args.image_id)
    image_path = resolve_image_path(image_root, image_info["file_name"])

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    grouped = anns_by_image_id(coco.get("annotations", []))
    categories_by_id = {cat["id"]: cat for cat in coco.get("categories", [])}
    rendered = visualize_image(
        image_bgr=image_bgr,
        annotations=grouped.get(image_info["id"], []),
        categories_by_id=categories_by_id,
    )

    ok = cv2.imwrite(str(output_path), rendered)
    if not ok:
        raise RuntimeError(f"Failed to save visualization to: {output_path}")

    print(f"Saved visualization to: {output_path}")
    print(f"Image: {image_path}")
    print(f"Image ID: {image_info['id']}")
    print(f"Annotations: {len(grouped.get(image_info['id'], []))}")

    if args.show:
        cv2.imshow("COCO Visualization", rendered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
