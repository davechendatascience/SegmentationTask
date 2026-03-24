
"""
Check image resolutions in a folder and count how many meet Full HD (1920x1080).

This script recursively scans all images under the given directory and
categorizes them into:
- High resolution (>= 1920x1080, including rotated images)
- Low resolution (< 1920x1080)
- Unreadable images (corrupted or unsupported)

Supported image formats:
    .jpg, .jpeg, .png, .bmp, .tif, .tiff

Example:
    python -m scripts.tools.check_image_resolution \
      --input-dir data/images

Output:
    >=1920x1080: 120
    <1920x1080: 45
    Unreadable: 3

Notes:
    - Images are checked in both orientations (width x height and height x width),
      so rotated images (e.g., 1080x1920) are also counted as high resolution.
    - Useful for dataset quality control before training (e.g., ensuring sufficient
      resolution for small object detection tasks).
"""
import argparse
from pathlib import Path
from PIL import Image
def main() -> None:
    parser = argparse.ArgumentParser(description="Check the resolution of images in folder")
    parser.add_argument("--input-dir", required=True, help="Image input folder path")
    parser.add_argument("--first", required=True, help="Image input folder path")
    parser.add_argument("--second", required=True, help="Image input folder path")
    args = parser.parse_args()
    root = Path(args.input_dir)  
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    hi = 0
    lo = 0
    bad = []

    for path in root.rglob("*"):
        if path.suffix.lower() not in exts:
            continue
        try:
            with Image.open(path) as img:
                w, h = img.size
            if w >= args.first and h >= args.second:
                hi += 1
            elif h >= args.first and w >=  args.second:
                hi += 1
            else:
                lo += 1
        except Exception as e:
            bad.append((str(path), str(e)))

    print(f">={args.first}x{args.second}: {hi}")
    print(f"<{args.first}x{args.second}: {lo}")
    print(f"Unreadable: {len(bad)}")

if __name__ == "__main__":
    main()
