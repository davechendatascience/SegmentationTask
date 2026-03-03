"""
Visualize SAM2-UNet predictions on the test set.
Saves overlay images (original + colored masks + labels) to output_dir.

Usage:
    python -m scripts.sam2_seg.visualize
    python -m scripts.sam2_seg.visualize --checkpoint output/sam2/best_model --split test --max_images 50
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .config import DataConfig, ModelConfig
from .dataset import load_coco, IMAGENET_MEAN, IMAGENET_STD
from .evaluate import load_model, preprocess_image, semantic_to_instances


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


def overlay_masks(
    image: Image.Image,
    instances: list,
    idx_to_class: dict,
    alpha: float = 0.45,
) -> Image.Image:
    """Draw colored instance masks + labels on a PIL image."""
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for inst in instances:
        cls_idx = inst["class_idx"]
        score = inst["score"]
        mask = inst["mask"]

        class_name = idx_to_class.get(cls_idx, f"cls{cls_idx}")
        color = PALETTE[cls_idx % len(PALETTE)]

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
    model, ckpt = load_model(checkpoint, device)
    num_classes = ckpt["num_classes"]
    image_size = ckpt.get("image_size", 1024)

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

            input_tensor = preprocess_image(pil_image, image_size).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_tensor)

            # Resize to original dimensions
            logits_orig = F.interpolate(
                logits, size=(orig_h, orig_w),
                mode="bilinear", align_corners=False,
            )
            logits_np = logits_orig[0].cpu().float().numpy()
            pred_classes = logits_np.argmax(axis=0)

            instances = semantic_to_instances(pred_classes, logits_np, num_classes)

            if instances:
                vis = overlay_masks(pil_image, instances, idx_to_class, alpha=0.45)
            else:
                vis = pil_image.copy()
                d = ImageDraw.Draw(vis)
                d.text((10, 10), "No detections", fill=(255, 0, 0))

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
        max_images=args.max_images,
        output_dir=args.output_dir,
    )
