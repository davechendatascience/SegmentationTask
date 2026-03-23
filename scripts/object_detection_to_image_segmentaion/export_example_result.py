"""
Export one example as:
1. original image
2. bbox visualization image
3. single-image COCO json from a segmentation annotation file (with bbox preserved)

Example:
    python -m scripts.object_detection_to_image_segmentaion.export_example_result ^
      --detection-annotations C:\\Users\\B50137\\Downloads\\hiod_coco\\content\\data\\hiod_coco\\train\\_annotations.coco.json ^
      --image-root C:\\Users\\B50137\\Downloads\\hiod_coco\\content\\data\\hiod_coco\\train ^
      --image-name image_1003_jpg.rf.599632984c37efa7d4c06fc3dfff5cf1.jpg ^
      --output-dir output\\example_export

If you also have converted segmentation annotations:
    python -m scripts.object_detection_to_image_segmentaion.export_example_result ^
      --detection-annotations data\\hiod_coco\\train\\_annotations.coco.json ^
      --segmentation-annotations data\\hiod_sam2_seg\\train\\_annotations.coco.json ^
      --image-root data\\hiod_coco\\train ^
      --image-name some_image.jpg ^
      --output-dir output\\example_export
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_coco(ann_path: Path) -> dict:
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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


def draw_bbox_image(
    image_bgr: np.ndarray,
    annotations: List[dict],
    categories_by_id: Dict[int, dict],
) -> np.ndarray:
    canvas = image_bgr.copy()
    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox:
            continue

        x, y, w, h = [float(v) for v in bbox]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        color = make_color(ann["id"] * 17 + ann["category_id"] * 101)
        class_name = str(categories_by_id.get(ann["category_id"], {}).get("name", ann["category_id"]))

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            canvas,
            f"{class_name} id={ann['id']}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def pick_image_info(coco: dict, image_name: str | None, image_id: int | None) -> dict:
    images = coco.get("images", [])
    if image_id is not None:
        for image_info in images:
            if int(image_info["id"]) == int(image_id):
                return image_info
        raise KeyError(f"image_id {image_id} not found")

    if image_name is not None:
        target_name = Path(image_name).name
        for image_info in images:
            if Path(image_info["file_name"]).name == target_name:
                return image_info
        raise KeyError(f"image_name {target_name} not found")

    if not images:
        raise RuntimeError("No images found in annotations")
    return images[0]


def build_single_image_coco(coco: dict, image_info: dict) -> dict:
    grouped = anns_by_image_id(coco.get("annotations", []))
    selected_annotations = grouped.get(image_info["id"], [])
    category_ids = sorted({ann["category_id"] for ann in selected_annotations})
    selected_categories = [
        cat for cat in coco.get("categories", [])
        if cat["id"] in category_ids
    ]

    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": [image_info],
        "annotations": selected_annotations,
        "categories": selected_categories,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one example image, bbox image, and segmentation COCO json")
    parser.add_argument("--detection-annotations", required=True, help="Path to original detection _annotations.coco.json")
    parser.add_argument("--image-root", required=True, help="Directory containing source images")
    parser.add_argument("--segmentation-annotations", default=None, help="Optional converted segmentation _annotations.coco.json")
    parser.add_argument("--image-name", default=None, help="Image file name to export")
    parser.add_argument("--image-id", type=int, default=None, help="COCO image id to export")
    parser.add_argument("--output-dir", default="output/example_export", help="Directory to write exported files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    det_ann_path = Path(args.detection_annotations)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_coco = load_coco(det_ann_path)
    image_info = pick_image_info(detection_coco, image_name=args.image_name, image_id=args.image_id)
    image_path = resolve_image_path(image_root, image_info["file_name"])

    original_image_out = output_dir / f"{Path(image_info['file_name']).stem}_original{image_path.suffix}"
    bbox_image_out = output_dir / f"{Path(image_info['file_name']).stem}_bbox{image_path.suffix}"

    shutil.copy2(image_path, original_image_out)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    det_grouped = anns_by_image_id(detection_coco.get("annotations", []))
    categories_by_id = {cat["id"]: cat for cat in detection_coco.get("categories", [])}
    bbox_image = draw_bbox_image(
        image_bgr=image_bgr,
        annotations=det_grouped.get(image_info["id"], []),
        categories_by_id=categories_by_id,
    )
    ok = cv2.imwrite(str(bbox_image_out), bbox_image)
    if not ok:
        raise RuntimeError(f"Failed to write bbox visualization: {bbox_image_out}")

    print(f"Original image: {original_image_out}")
    print(f"BBox image: {bbox_image_out}")

    if args.segmentation_annotations:
        seg_ann_path = Path(args.segmentation_annotations)
        segmentation_coco = load_coco(seg_ann_path)
        seg_image_info = pick_image_info(segmentation_coco, image_name=image_info["file_name"], image_id=image_info["id"])
        single_image_coco = build_single_image_coco(segmentation_coco, seg_image_info)
        seg_json_out = output_dir / f"{Path(image_info['file_name']).stem}_segmentation_with_bbox.json"
        save_json(seg_json_out, single_image_coco)
        print(f"Segmentation COCO json: {seg_json_out}")
    else:
        print("Segmentation COCO json: skipped (provide --segmentation-annotations to export it)")


if __name__ == "__main__":
    main()
