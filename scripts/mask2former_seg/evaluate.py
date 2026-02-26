"""
Evaluate Mask2Former on the hospital test set and report COCO mAP50.

Loads images at their ORIGINAL resolution (no crop) so predicted masks
are in the same coordinate space as the COCO ground-truth annotations.

Usage:
    python -m scripts.mask2former_seg.evaluate
    python -m scripts.mask2former_seg.evaluate --checkpoint output/mask2former/best_model
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from .config import DataConfig, ModelConfig
from .dataset import load_coco


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert binary mask (H, W) to COCO RLE format."""
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def evaluate(
    data_cfg: DataConfig = None,
    model_cfg: ModelConfig = None,
    checkpoint: str = None,
    split: str = "test",
    threshold: float = 0.5,
):
    data_cfg  = data_cfg  or DataConfig()
    model_cfg = model_cfg or ModelConfig()

    checkpoint = checkpoint or os.path.join(model_cfg.output_dir, "best_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {checkpoint}")
    processor = Mask2FormerImageProcessor.from_pretrained(checkpoint)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint)
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------ #
    # Load COCO annotations                                                #
    # ------------------------------------------------------------------ #
    ann_path = Path(data_cfg.data_root) / split / "_annotations.coco.json"
    coco_data = load_coco(str(ann_path))

    # Map 0-indexed sorted position → COCO category_id
    sorted_cats = sorted(coco_data["categories"], key=lambda c: c["id"])
    idx_to_cat_id = {i: cat["id"] for i, cat in enumerate(sorted_cats)}
    print(f"Classes ({len(sorted_cats)}): {[c['name'] for c in sorted_cats]}")

    images_by_id = {img["id"]: img for img in coco_data["images"]}
    img_dir = Path(data_cfg.data_root) / split

    coco_gt = COCO(str(ann_path))
    coco_results = []
    skipped = 0

    with torch.no_grad():
        for img_info in tqdm(coco_data["images"], desc=f"Evaluating [{split}]"):
            image_id = img_info["id"]
            img_path = img_dir / img_info["file_name"]

            # Load ORIGINAL resolution image — no crop/resize by us
            pil_image = Image.open(str(img_path)).convert("RGB")
            orig_w, orig_h = pil_image.size

            # Processor handles resizing; target_sizes restores to original dims
            inputs = processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                target_sizes=[(orig_h, orig_w)],   # restore to original image size
            )

            pred   = results[0]
            seg_map = pred.get("segmentation")

            if seg_map is None:
                skipped += 1
                continue

            seg_np = seg_map.cpu().numpy()

            for seg_info in pred.get("segments_info", []):
                seg_id   = seg_info["id"]
                label_id = seg_info["label_id"]  # 0=background, 1..N=real classes
                score    = seg_info["score"]

                if label_id == 0:          # skip background
                    continue

                binary_mask = (seg_np == seg_id).astype(np.uint8)
                if binary_mask.sum() == 0:
                    continue

                # Undo the +1 offset applied in train collate_fn
                real_class_idx = label_id - 1
                cat_id = idx_to_cat_id.get(real_class_idx, -1)
                if cat_id == -1:
                    skipped += 1
                    continue

                rle = mask_to_rle(binary_mask)
                coco_results.append({
                    "image_id":    image_id,
                    "category_id": cat_id,
                    "segmentation": rle,
                    "score":       float(score),
                })

    print(f"\nTotal predictions: {len(coco_results)}  |  skipped: {skipped}")

    if not coco_results:
        print("No predictions — try lowering --threshold (currently {threshold})")
        return

    # ------------------------------------------------------------------ #
    # COCO evaluation                                                      #
    # ------------------------------------------------------------------ #
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map50_95 = coco_eval.stats[0]
    map50    = coco_eval.stats[1]
    print(f"\n{'='*50}")
    print(f"  mAP50:       {map50*100:.2f}%  (target: > 70%)")
    print(f"  mAP50:95:    {map50_95*100:.2f}%")
    print(f"{'='*50}")

    return {"mAP50": map50, "mAP50_95": map50_95}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split",      type=str, default="test")
    parser.add_argument("--data_root",  type=str, default=None)
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()

    data_cfg = DataConfig()
    if args.data_root:
        data_cfg.data_root = args.data_root

    evaluate(
        data_cfg=data_cfg,
        checkpoint=args.checkpoint,
        split=args.split,
        threshold=args.threshold,
    )
