"""
下載 COCO 2017 指定類別，並輸出成明確的 COCO 資料夾格式。
Download selected COCO 2017 classes and export split annotations without copying images.

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
      valid/
        _annotations.coco.json
      test/
        _annotations.coco.json

下載來源說明 / Download source:
    COCO 原始資料會由 FiftyOne 下載到其預設 cache 位置，
    再匯出成 <output-root> 下的 COCO 格式資料夾。
    The raw COCO files are downloaded by FiftyOne into its default cache,
    and then exported into the COCO-format folders under <output-root>.
"""

import argparse
import json
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
        description="Download COCO 2017 samples with FiftyOne and export COCO annotations that point to the FiftyOne cache"
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


def rewrite_annotation_file_names(dataset: fo.Dataset, labels_path: Path) -> None:
    """將 COCO annotations 的 file_name 改成對應的 cache 影像路徑。Rewrite file_name entries to point to cached images."""
    file_name_to_path = {}
    for sample in dataset.iter_samples(progress=True):
        sample_path = Path(sample.filepath)
        file_name_to_path[sample_path.name] = sample_path.as_posix()

    with labels_path.open("r", encoding="utf-8") as f:
        coco_data = json.load(f)

    missing_files: list[str] = []
    for image in coco_data.get("images", []):
        exported_name = Path(image["file_name"]).name
        cached_path = file_name_to_path.get(exported_name)
        if cached_path is None:
            missing_files.append(exported_name)
            continue
        image["file_name"] = cached_path

    if missing_files:
        preview = ", ".join(missing_files[:5])
        raise ValueError(
            "Failed to map some exported COCO image names back to the FiftyOne cache: "
            f"{preview}"
        )

    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False)


def export_split(dataset: fo.Dataset, output_root: Path, split_name: str) -> None:
    """將單一 split 匯出為只含 annotations 的 COCO 格式。Export one split as COCO annotations only."""
    export_split_name = SPLIT_EXPORT_NAMES[split_name]
    split_dir = output_root / export_split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    labels_path = split_dir / "_annotations.coco.json"

    # 只輸出 annotations，不複製圖片；後續再把 file_name 重寫成 cache 內的實際路徑。
    # Export annotations only without copying images, then rewrite file_name to cached image paths.
    dataset.export(
        export_dir=str(split_dir),
        dataset_type=fo.types.COCODetectionDataset,
        labels_path=str(labels_path),
        export_media=False,
    )
    rewrite_annotation_file_names(dataset, labels_path)
    print(f"Exported split: {split_name} -> {split_dir}")


def download_split(split_name: str, classes: list[str]) -> fo.Dataset:
    """下載單一 split。Download one split via FiftyOne zoo."""
    # train split 保留 shuffle，validation/test 則維持原順序。
    # Keep shuffling for train, while preserving the default order for validation/test.
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split_name,
        shuffle=(split_name == "train"),
        classes=classes,
        label_types=["detections", "segmentations"],
    )
    print(f"Downloaded split: {split_name}")
    return dataset


def main() -> None:
    """程式進入點。Program entry point."""
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # COCO 原始資料由 FiftyOne 管理下載位置，這裡只負責匯出成專案要用的格式。
    # FiftyOne manages the raw download location; this script exports the project-ready format.
    for split_name in ("train", "validation", "test"):
        dataset = download_split(split_name, args.classes)
        export_split(dataset, output_root, split_name)


if __name__ == "__main__":
    main()
