"""
Visualize Mask2Former predictions on the test set.
Saves overlay images (original + colored masks + labels) to output_dir.

Usage:
    python -m scripts.mask2former_seg.visualize
    python -m scripts.mask2former_seg.visualize --checkpoint output/mask2former/best_model --split test --max_images 50
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from .config import DataConfig, ModelConfig
from .dataset import load_coco


# Distinct colors for up to 30 classes
PALETTE = [
    (255, 56,  56 ), (255, 157, 151), (255, 112, 31 ), (255, 178, 29 ),
    (207, 210, 49 ), (72,  249, 10 ), (146, 204, 23 ), (61,  219, 134),
    (26,  147, 52 ), (0,   212, 187), (44,  153, 168), (0,   194, 255),
    (52,  69,  147), (100, 115, 255), (0,   24,  236), (132, 56,  255),
    (82,  0,   133), (203, 56,  255), (255, 149, 200), (255, 55,  199),
    (180, 180, 180), (200, 150, 100), (100, 200, 150), (150, 100, 200),
    (50,  200, 200), (200, 50,  100), (100, 50,  200), (200, 100, 50 ),
    (50,  100, 200), (200, 200, 50 ),
]


def overlay_masks(image: Image.Image, seg_np: np.ndarray, segments_info: list,
                  idx_to_class: dict, alpha: float = 0.45) -> Image.Image:
    """Draw colored instance masks + labels on a PIL image."""
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for seg_info in segments_info:
        seg_id   = seg_info["id"]
        label_id = seg_info["label_id"]
        score    = seg_info["score"]

        if label_id == 0:  # background
            continue

        real_idx = label_id - 1
        class_name = idx_to_class.get(real_idx, f"cls{real_idx}")
        color = PALETTE[real_idx % len(PALETTE)]

        # Colored mask
        mask = (seg_np == seg_id)
        if not mask.any():
            continue

        mask_img = Image.fromarray((mask * int(alpha * 255)).astype(np.uint8), mode="L")
        color_layer = Image.new("RGBA", img.size, color + (180,))
        overlay.paste(color_layer, mask=mask_img)

        # Label at centroid
        ys, xs = np.where(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        label_text = f"{class_name} {score:.2f}"
        draw.text((cx, cy), label_text, fill=(255, 255, 255, 255), font=font)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result


def visualize(
    data_cfg: DataConfig = None,
    model_cfg: ModelConfig = None,
    checkpoint: str = None,
    split: str = "test",
    threshold: float = 0.4,
    max_images: int = None,
    output_dir: str = None,
):
    data_cfg  = data_cfg  or DataConfig()
    model_cfg = model_cfg or ModelConfig()

    checkpoint = checkpoint or os.path.join(model_cfg.output_dir, "best_model")
    output_dir = output_dir or os.path.join(model_cfg.output_dir, "visualizations", split)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {checkpoint}")
    processor = Mask2FormerImageProcessor.from_pretrained(checkpoint)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint)
    model.to(device)
    model.eval()

    ann_path = Path(data_cfg.data_root) / split / "_annotations.coco.json"
    coco_data = load_coco(str(ann_path))
    sorted_cats = sorted(coco_data["categories"], key=lambda c: c["id"])
    idx_to_class = {i: cat["name"] for i, cat in enumerate(sorted_cats)}
    img_dir = Path(data_cfg.data_root) / split

    images = coco_data["images"]
    if max_images:
        images = random.sample(images, min(max_images, len(images)))

    print(f"Visualizing {len(images)} images → {output_dir}")

    with torch.no_grad():
        for img_info in tqdm(images, desc="Visualizing"):
            img_path = img_dir / img_info["file_name"]
            pil_image = Image.open(str(img_path)).convert("RGB")
            orig_w, orig_h = pil_image.size

            inputs = processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs, threshold=threshold, target_sizes=[(orig_h, orig_w)]
            )

            pred    = results[0]
            seg_map = pred.get("segmentation")

            if seg_map is None:
                # Save original with "no detections" label
                vis = pil_image.copy()
                d = ImageDraw.Draw(vis)
                d.text((10, 10), "No detections", fill=(255, 0, 0))
            else:
                seg_np = seg_map.cpu().numpy()
                vis = overlay_masks(pil_image, seg_np, pred.get("segments_info", []),
                                    idx_to_class, alpha=0.45)

            # Save with same filename
            stem = Path(img_info["file_name"]).stem
            out_path = os.path.join(output_dir, f"{stem}_pred.jpg")
            vis.save(out_path, quality=92)

    print(f"\nDone. Images saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str,   default=None)
    parser.add_argument("--split",       type=str,   default="test")
    parser.add_argument("--data_root",   type=str,   default=None)
    parser.add_argument("--threshold",   type=float, default=0.4)
    parser.add_argument("--max_images",  type=int,   default=None,
                        help="Limit number of images (None = all)")
    parser.add_argument("--output_dir",  type=str,   default=None)
    args = parser.parse_args()

    data_cfg = DataConfig()
    if args.data_root:
        data_cfg.data_root = args.data_root

    visualize(
        data_cfg=data_cfg,
        checkpoint=args.checkpoint,
        split=args.split,
        threshold=args.threshold,
        max_images=args.max_images,
        output_dir=args.output_dir,
    )
