"""
下載 COCO 2017 指定類別，並輸出成明確的 COCO 資料夾格式。
Download selected COCO 2017 classes and export them in a clear COCO folder layout.

Usage:
    python -m scripts.tools.download_coco2017_by_class --output-root data/coco_filtered

Example:
    python -m scripts.tools.download_coco2017_by_class \
        --output-root data/coco_filtered \
        --classes bowl laptop "cell phone" remote book backpack handbag suitcase toothbrush

輸出結構 / Output layout:
    <output-root>/
      train/
        _annotations.coco.json
        *.jpg
      valid/
        _annotations.coco.json
        *.jpg
      test/
        _annotations.coco.json
        *.jpg
      _fiftyone_cache/
        ...
"""

import argparse
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

# 將 FiftyOne 的 split 名稱轉成專案較常用的資料夾名稱。
# Map FiftyOne split names to the folder names we want to export.
SPLIT_EXPORT_NAMES = {
    "train": "train",
    "validation": "valid",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    """解析命令列參數。Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download COCO 2017 samples with FiftyOne and export them in COCO folder format"
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where the exported COCO-format dataset should be written",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "bowl",
            "laptop",
            "cell phone",
            "remote",
            "book",
            "backpack",
            "handbag",
            "suitcase",
            "toothbrush",
        ],
        help='Class names to filter, for example: --classes bowl laptop "cell phone" remote',
    )
    return parser.parse_args()


def export_split(dataset: fo.Dataset, output_root: Path, split_name: str) -> None:
    """將單一 split 匯出為 COCO 格式。Export one split in COCO format."""
    export_split_name = SPLIT_EXPORT_NAMES[split_name]
    split_dir = output_root / export_split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # 匯出成每個 split 一個資料夾，並固定 annotation 檔名為 _annotations.coco.json。
    # Export each split into its own folder with a fixed COCO annotation filename.
    dataset.export(
        export_dir=str(split_dir),
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(split_dir),
        labels_path=str(split_dir / "_annotations.coco.json"),
        export_media="copy",
    )
    print(f"Exported split: {split_name} -> {split_dir}")


def download_split(split_name: str, cache_root: Path, classes: list[str]) -> fo.Dataset:
    """下載單一 split 到 FiftyOne cache。Download one split into the FiftyOne cache."""
    # train split 保留 shuffle，validation/test 則維持原順序。
    # Keep shuffling for train, while preserving the default order for validation/test.
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split_name,
        shuffle=(split_name == "train"),
        classes=classes,
        label_types=["detections", "segmentations"],
        dataset_dir=str(cache_root),
    )
    print(f"Downloaded split: {split_name}")
    return dataset


def main() -> None:
    """程式進入點。Program entry point."""
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # 先下載到內部 cache，再匯出成專案可直接使用的 COCO 結構。
    # Download into an internal cache first, then export into a project-friendly COCO layout.
    cache_root = output_root / "_fiftyone_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # 依序處理 train / validation / test，並將 validation 匯出為 valid。
    # Process train / validation / test in order, exporting validation as valid.
    for split_name in ("train", "validation", "test"):
        dataset = download_split(split_name, cache_root, args.classes)
        export_split(dataset, output_root, split_name)


if __name__ == "__main__":
    main()
