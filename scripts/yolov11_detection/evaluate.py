"""
Evaluate a trained YOLOv11 object detection model.

Usage:
    python -m scripts.yolov11_detection.evaluate --model output/yolov11_detection/exp/weights/best.pt --data data.yaml
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def _load_simple_data_yaml(yaml_path: Path) -> dict:
    data: dict = {}
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def _resolve_coco_annotation_from_data_yaml(data_yaml: Path, split: str) -> Path | None:
    data_cfg = _load_simple_data_yaml(data_yaml)
    yolo_root = Path(str(data_cfg.get("path", data_yaml.parent))).resolve()
    coco_root = yolo_root.parent
    split_mapping = {"train": "train", "val": "valid", "test": "test"}
    coco_split = split_mapping.get(split, split)
    ann_path = coco_root / coco_split / "_annotations.coco.json"
    return ann_path if ann_path.exists() else None


def _print_object_size_distribution(ann_path: Path) -> None:
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    small = 0
    medium = 0
    large = 0

    for ann in data.get("annotations", []):
        if "area" in ann:
            area = float(ann["area"])
        else:
            bbox = ann.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            area = float(bbox[2]) * float(bbox[3])

        if area < 32 * 32:
            small += 1
        elif area < 96 * 96:
            medium += 1
        else:
            large += 1

    total = small + medium + large
    if total == 0:
        print("\n=== Object Size Distribution ===")
        print("No annotations found in the resolved COCO file.")
        return

    print("\n=== Object Size Distribution ===")
    print(f"small : {small} ({small / total:.2%})")
    print(f"medium: {medium} ({medium / total:.2%})")
    print(f"large : {large} ({large / total:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 object detection model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")
    parser.add_argument("--imgsz", type=int, required=True, help="Resize image size used during evaluation")
    parser.add_argument("--show-size-distribution", action="store_true", help="Print COCO small/medium/large object counts before evaluation")
    args = parser.parse_args()

    model = YOLO(args.model)

    if args.show_size_distribution:
        ann_path = _resolve_coco_annotation_from_data_yaml(Path(args.data), args.split)
        if ann_path is not None:
            _print_object_size_distribution(ann_path)
        else:
            print("\n=== Object Size Distribution ===")
            print("COCO annotation file could not be resolved from data.yaml.")

    print(f"Evaluating model on {args.split} set with imgsz={args.imgsz}...")
    results = model.val(
        data=args.data,
        split=args.split,
        batch=args.batch_size,
        device=args.device,
        imgsz=args.imgsz,
        save_json=True,
        save_txt=True,
        save_conf=True,
        save=True,
    )

    print("\nEvaluation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    print(f"Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
