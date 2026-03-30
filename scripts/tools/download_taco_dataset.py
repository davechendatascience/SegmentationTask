"""
Download TACO images from Flickr and auto-orient them during save.

Usage:
    python -m scripts.tools.download_taco_dataset

Before running this script, prepare the dataset archive manually by following:
    scripts/tools/README.md
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from scripts.tools.auto_orient_tool import auto_orient_and_strip

DEFAULT_DATASET_DIR = "data/taco_dataset"
DEFAULT_NUM_WORKERS = 8
SPLITS = ("train", "valid", "test")
ANNOTATION_FILENAMES = ("_annotation.coco.json", "_annotations.coco.json")
README_PATH = Path("scripts/tools/README.md")
TMP_SUFFIX = ".download_tmp"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download TACO images from Flickr and auto-orient them during save. "
            "Prepare the base dataset manually first as documented in "
            "scripts/tools/README.md."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Directory containing the extracted TACO dataset.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for each request.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of worker threads for parallel downloads.",
    )
    return parser.parse_args()


def resolve_dataset_dir(dataset_dir: Path) -> Path:
    """
    Resolve the actual dataset root directory automatically.

    Supported layouts:
    1. data/taco_dataset/train
    2. data/taco_dataset/taco_dataset/train
    """
    direct_split_dirs = [dataset_dir / split for split in SPLITS]
    if all(path.exists() for path in direct_split_dirs):
        return dataset_dir

    nested_dir = dataset_dir / dataset_dir.name
    nested_split_dirs = [nested_dir / split for split in SPLITS]
    if nested_dir.exists() and all(path.exists() for path in nested_split_dirs):
        return nested_dir

    return dataset_dir


def ensure_dataset_exists(dataset_dir: Path) -> None:
    """Validate that the extracted dataset already exists on disk."""
    resolved_dataset_dir = resolve_dataset_dir(dataset_dir)
    split_dirs = [resolved_dataset_dir / split for split in SPLITS]
    if all(path.exists() for path in split_dirs):
        return

    raise FileNotFoundError(
        "TACO dataset was not found. Please download and extract it manually first. "
        f"See {README_PATH.as_posix()} for the required commands. "
        f"Expected dataset directory: {dataset_dir}"
    )


def find_split_annotation_paths(dataset_dir: Path) -> list[Path]:
    """Find annotation files for train/valid/test."""
    annotation_paths: list[Path] = []
    for split in SPLITS:
        split_dir = dataset_dir / split
        annotation_path = None

        for filename in ANNOTATION_FILENAMES:
            candidate = split_dir / filename
            if candidate.exists():
                annotation_path = candidate
                break

        if annotation_path is None:
            raise FileNotFoundError(
                f"Annotation file not found for split '{split}' under {split_dir}"
            )
        annotation_paths.append(annotation_path)

    return annotation_paths


def build_temp_path(destination_path: Path) -> Path:
    """Return the temporary download path for an image."""
    suffix = destination_path.suffix or ".jpg"
    return destination_path.with_name(
        f"{destination_path.stem}{TMP_SUFFIX}{suffix}"
    )


def download_image_to_temp(
    image_url: str,
    temp_path: Path,
    timeout: float,
) -> None:
    """Download one image into its temporary file."""
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(image_url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with temp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def print_progress(prefix: str, current_count: int, total_count: int) -> None:
    """Print a simple progress bar."""
    bar_size = 30
    filled = int(bar_size * current_count / total_count) if total_count else 0
    sys.stdout.write(
        "{}[{}{}] - {}/{}\r".format(
            prefix,
            "=" * filled,
            "." * (bar_size - filled),
            current_count,
            total_count,
        )
    )
    sys.stdout.flush()


def download_single_image(image: dict, dataset_dir: Path, timeout: float) -> Path | None:
    """Download one missing image into a temporary file."""
    file_name = image["file_name"]
    image_url = image.get("flickr_url") or image.get("flickr_640_url")
    destination_path = dataset_dir / file_name
    temp_path = build_temp_path(destination_path)

    if destination_path.exists():
        return None

    if temp_path.exists():
        return temp_path

    if not image_url:
        raise ValueError(f"Missing Flickr URL for image: {file_name}")

    download_image_to_temp(
        image_url=image_url,
        temp_path=temp_path,
        timeout=timeout,
    )
    return temp_path


def orient_single_image(temp_path: Path, destination_path: Path) -> None:
    """Auto-orient one downloaded temporary image and then remove it."""
    try:
        auto_orient_and_strip(temp_path, destination_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def process_annotation_file(dataset_path: Path, timeout: float, num_workers: int) -> None:
    """Process one annotation file in download and auto-orient phases."""
    dataset_dir = dataset_path.parent

    print(f"Processing annotation file: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    images = annotations.get("images", [])
    total_count = len(images)
    max_workers = max(1, min(num_workers, total_count if total_count else 1))
    temp_targets: list[tuple[Path, Path]] = []

    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {
            executor.submit(download_single_image, image, dataset_dir, timeout): image
            for image in images
        }

        for future in as_completed(future_to_image):
            image = future_to_image[future]
            temp_path = future.result()
            if temp_path is not None:
                temp_targets.append((temp_path, dataset_dir / image["file_name"]))
            completed_count += 1
            print_progress(
                prefix=f"{dataset_dir.name} download: ",
                current_count=completed_count,
                total_count=total_count,
            )

    sys.stdout.write("\n")

    if temp_targets:
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_target = {
                executor.submit(orient_single_image, temp_path, destination_path): (
                    temp_path,
                    destination_path,
                )
                for temp_path, destination_path in temp_targets
            }

            for future in as_completed(future_to_target):
                future.result()
                completed_count += 1
                print_progress(
                    prefix=f"{dataset_dir.name} orient: ",
                    current_count=completed_count,
                    total_count=len(temp_targets),
                )

        sys.stdout.write("\n")

    sys.stdout.write(f"{dataset_dir.name}: Finished\n")


def main() -> None:
    """Program entry point."""
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)

    ensure_dataset_exists(dataset_dir)
    dataset_dir = resolve_dataset_dir(dataset_dir)
    annotation_paths = find_split_annotation_paths(dataset_dir)

    print(
        "Note. If the connection is interrupted, running this command again "
        "will continue from the remaining files."
    )

    for annotation_path in annotation_paths:
        process_annotation_file(
            annotation_path,
            timeout=args.timeout,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
