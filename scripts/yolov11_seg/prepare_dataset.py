"""
Prepare a YOLO-seg dataset view from the shared COCO dataset.

Usage:
    python -m scripts.yolov11_seg.prepare_dataset
"""
import argparse
from pathlib import Path

from .config import DataConfig
from .dataset import build_yolo_dataset_from_coco


def main():
    parser = argparse.ArgumentParser(
        description="Build YOLO labels and data.yaml from the shared COCO dataset"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="COCO dataset root, e.g. data/hospital_coco",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Directory for YOLO images/labels/data.yaml. Defaults to <data-root>/yolo",
    )
    parser.add_argument(
        "--output-yaml",
        type=str,
        default="data.yaml",
        help="Filename for the generated YAML manifest",
    )
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

    output_root = args.output_root or str(Path(data_cfg.data_root) / "yolo")
    yaml_path = build_yolo_dataset_from_coco(
        coco_root=data_cfg.data_root,
        output_root=output_root,
        output_yaml=args.output_yaml,
        preserve_category_ids=args.preserve_category_ids,
    )

    print(f"Prepared dataset YAML: {yaml_path}")


if __name__ == "__main__":
    main()
