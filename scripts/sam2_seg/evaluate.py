"""
Evaluate SAM2-UNet on the hospital test set and report COCO mAP50.

Converts semantic segmentation predictions into per-instance detections
via connected-component analysis, then uses pycocotools COCOeval.

Usage:
    python -m scripts.sam2_seg.evaluate
    python -m scripts.sam2_seg.evaluate --checkpoint output/sam2/best_model
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy import ndimage
from tqdm import tqdm

from .config import DataConfig, ModelConfig
from .dataset import load_coco, IMAGENET_MEAN, IMAGENET_STD
from .segmentation_model import SAM2SegModel


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert binary mask (H, W) to COCO RLE format."""
    rle = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def semantic_to_instances(
    pred_classes: np.ndarray,
    logits_np: np.ndarray,
    num_classes: int,
) -> list:
    """
    Convert a semantic prediction into a list of instance detections.

    Uses connected-component analysis to split each class region into
    individual instances. Confidence score is the mean softmax probability
    within each component.

    Args:
        pred_classes: [H, W] int array of predicted class indices (0=bg)
        logits_np: [C, H, W] float array of raw logits
        num_classes: total number of classes including background

    Returns:
        List of dicts: {"class_idx": int, "mask": np.ndarray, "score": float}
    """
    probs = _softmax(logits_np)  # [C, H, W]
    instances = []

    for cls_id in range(1, num_classes):  # skip background (0)
        cls_mask = (pred_classes == cls_id).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue

        # Connected components for this class
        labelled, n_components = ndimage.label(cls_mask)

        for comp_id in range(1, n_components + 1):
            comp_mask = (labelled == comp_id).astype(np.uint8)
            if comp_mask.sum() < 10:  # skip tiny fragments
                continue

            # Confidence = mean probability of this class within the component
            score = float(probs[cls_id][comp_mask > 0].mean())

            instances.append({
                "class_idx": cls_id - 1,  # convert back to 0-indexed (no bg)
                "mask": comp_mask,
                "score": score,
            })

    return instances


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numpy softmax over axis 0."""
    exp = np.exp(logits - logits.max(axis=0, keepdims=True))
    return exp / exp.sum(axis=0, keepdims=True)


def load_model(checkpoint_dir: str, device: torch.device):
    """Load a trained SAM2SegModel from a checkpoint directory."""
    ckpt_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(ckpt_path):
        # Try training_state.pt (epoch checkpoints)
        ckpt_path = os.path.join(checkpoint_dir, "training_state.pt")

    ckpt = torch.load(ckpt_path, map_location=device)

    model = SAM2SegModel.build(
        model_name=ckpt.get("model_name", "facebook/sam2.1-hiera-large"),
        num_classes=ckpt["num_classes"],
        use_adapters=ckpt.get("use_adapters", True),
        adapter_dim=ckpt.get("adapter_dim", 64),
        image_size=ckpt.get("image_size", 1024),
    )

    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt


def preprocess_image(pil_image: Image.Image, image_size: int) -> torch.Tensor:
    """Preprocess a PIL image to a normalised tensor [1, 3, H, W]."""
    img = pil_image.resize((image_size, image_size), Image.BILINEAR)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return tensor


def evaluate(
    data_cfg: DataConfig = None,
    model_cfg: ModelConfig = None,
    checkpoint: str = None,
    split: str = "test",
):
    data_cfg  = data_cfg  or DataConfig()
    model_cfg = model_cfg or ModelConfig()

    checkpoint = checkpoint or os.path.join(model_cfg.output_dir, "best_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {checkpoint}")
    model, ckpt = load_model(checkpoint, device)
    num_classes = ckpt["num_classes"]
    image_size = ckpt.get("image_size", 1024)

    # ------------------------------------------------------------------ #
    # Load COCO annotations                                                #
    # ------------------------------------------------------------------ #
    ann_path = Path(data_cfg.data_root) / split / "_annotations.coco.json"
    coco_data = load_coco(str(ann_path))

    # Map 0-indexed sorted position → COCO category_id
    sorted_cats = sorted(coco_data["categories"], key=lambda c: c["id"])
    idx_to_cat_id = {i: cat["id"] for i, cat in enumerate(sorted_cats)}
    print(f"Classes ({len(sorted_cats)}): {[c['name'] for c in sorted_cats]}")

    img_dir = Path(data_cfg.data_root) / split
    coco_gt = COCO(str(ann_path))
    coco_results = []
    skipped = 0

    with torch.no_grad():
        for img_info in tqdm(coco_data["images"], desc=f"Evaluating [{split}]"):
            image_id = img_info["id"]
            img_path = img_dir / img_info["file_name"]

            pil_image = Image.open(str(img_path)).convert("RGB")
            orig_w, orig_h = pil_image.size

            # Preprocess
            input_tensor = preprocess_image(pil_image, image_size).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_tensor)  # [1, C, H, W]

            # Resize logits to original image size
            logits_orig = F.interpolate(
                logits, size=(orig_h, orig_w),
                mode="bilinear", align_corners=False,
            )

            logits_np = logits_orig[0].cpu().float().numpy()  # [C, H, W]
            pred_classes = logits_np.argmax(axis=0)  # [H, W]

            # Extract instances via connected components
            instances = semantic_to_instances(pred_classes, logits_np, num_classes)

            for inst in instances:
                cat_id = idx_to_cat_id.get(inst["class_idx"], -1)
                if cat_id == -1:
                    skipped += 1
                    continue

                rle = mask_to_rle(inst["mask"])
                coco_results.append({
                    "image_id":    image_id,
                    "category_id": cat_id,
                    "segmentation": rle,
                    "score":       inst["score"],
                })

    print(f"\nTotal predictions: {len(coco_results)}  |  skipped: {skipped}")

    if not coco_results:
        print(f"No predictions — model may not be trained yet")
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
    args = parser.parse_args()

    data_cfg = DataConfig()
    if args.data_root:
        data_cfg.data_root = args.data_root

    evaluate(
        data_cfg=data_cfg,
        checkpoint=args.checkpoint,
        split=args.split,
    )
