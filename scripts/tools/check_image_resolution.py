
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
    =====================
    Result Save in data/images/hi.txt data/images/lo.txt 

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
    args = parser.parse_args()
    root = Path(args.input_dir)  
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    hi = 0
    lo = 0
    bad = []
    hi_list = []
    lo_list = []
    hi_list.append("path\tw\th\n")
    hi_list.append("path\tw\th\n")
    for path in root.rglob("*"):
        if path.suffix.lower() not in exts:
            continue
        try:
            with Image.open(path) as img:
                w, h = img.size
            if w >= args.first and h >= args.second:
                hi += 1
                hi_list.append(f"{path}\t{w}\t{h}\n")
            elif h >= args.first and w >=  args.second:
                hi += 1
                hi_list.append(f"{path}\t{w}\t{h}\n")
            else:
                lo += 1
                lo_list.append(f"{path}\t{w}\t{h}\n")
        except Exception as e:
            bad.append((str(path), str(e)))

    print(f">={args.first}x{args.second}: {hi}")
    print(f"<{args.first}x{args.second}: {lo}")
    print(f"Unreadable: {len(bad)}")


    hi_path = f'{args.input_dir}/hi.txt'
    lo_path = f"{args.input_dir}/lo.txt"      
    f = open(hi_path, 'w')
    f.writelines(hi_list)
    f.close()
    f = open(lo_path, 'w')
    f.writelines(lo_list)
    f.close()
    print("=====================")
    print(f"Result Save in {args.input_dir}/hi.txt {args.input_dir}/lo.txt")
if __name__ == "__main__":
    main()
