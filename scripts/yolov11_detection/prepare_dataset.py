"""
Prepare a YOLOv11 detection dataset from the original COCO detection export.

Usage:
    python -m scripts.yolov11_detection.prepare_dataset
"""
import argparse

from .config import DataConfig
from .dataset import build_yolo_dataset_from_coco_detection


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLOv11 detection dataset from COCO annotations")
    parser.add_argument("--data-root", type=str, default=None, help="Root of the original COCO detection dataset")
    parser.add_argument("--output-root", type=str, default=None, help="Output directory for YOLO detection files")
    parser.add_argument(
        "--preserve-category-ids",
        action="store_true",
        default=True,
        help="Use original COCO category ids as YOLO class indices instead of reindexing to 0..N-1 (default: enabled)",
    )
    parser.add_argument(
        "--reindex-category-ids",
        dest="preserve_category_ids",
        action="store_false",
        help="Reindex category ids to contiguous 0..N-1 class indices",
    )
    args = parser.parse_args()

    data_cfg = DataConfig()
    if args.data_root:
        data_cfg.data_root = args.data_root
    if args.output_root:
        data_cfg.yolo_dataset_dir = args.output_root

    build_yolo_dataset_from_coco_detection(
        coco_root=data_cfg.data_root,
        output_root=data_cfg.yolo_dataset_dir,
        preserve_category_ids=args.preserve_category_ids,
    )


if __name__ == "__main__":
    main()
