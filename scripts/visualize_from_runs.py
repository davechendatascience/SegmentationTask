import argparse
import glob
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core_design_2.dataset_block import ADE20kPanopticDataset, get_transforms, IMAGE_SIZE


def load_ade_label_names(data_dir):
    labels_path = os.path.join(data_dir, "objectInfo150.txt")
    if not os.path.exists(labels_path):
        return {}
    id2label = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("idx"):
                continue
            parts = line.split("\t")
            if len(parts) < 5 or not parts[0].isdigit():
                continue
            label_id = int(parts[0]) - 1
            label_name = parts[4].strip().replace("_", " ")
            id2label[label_id] = label_name
    return id2label


def load_ade_palette(data_dir, split):
    ann_dir = os.path.join(
        data_dir,
        "annotations",
        "training" if split == "train" else "validation",
    )
    mask_paths = sorted(glob.glob(os.path.join(ann_dir, "*.png")))
    if not mask_paths:
        return None
    with Image.open(mask_paths[0]) as img:
        palette = img.getpalette()
    if not palette:
        return None
    colors = []
    for i in range(0, len(palette), 3):
        colors.append(tuple(palette[i : i + 3]))
    return colors


def get_palette_color(palette, label_id):
    if not palette:
        rng = np.random.RandomState(label_id)
        return tuple((rng.rand(3) * 255).astype(np.uint8))
    palette_idx = label_id + 1
    if palette_idx < len(palette):
        return palette[palette_idx]
    return palette[0] if palette else (255, 255, 255)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize predictions from a runs/ checkpoint.")
    parser.add_argument("--run-dir", type=str, default="runs/mask2former_ade20k/best")
    parser.add_argument("--data-dir", type=str, default="ADEChallengeData2016")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="runs/visualizations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--score-thresh", type=float, default=0.05)
    parser.add_argument("--mask-thresh", type=float, default=0.3)
    return parser.parse_args()


@torch.no_grad()
def visualize_sample(
    model,
    dataset,
    idx,
    device,
    score_thresh,
    mask_thresh,
    out_path,
    meta_path,
    id2label,
    palette,
):
    model.eval()
    item = dataset[idx]
    image_tensor = item["pixel_values"].unsqueeze(0).to(device)
    pixel_mask = item["pixel_mask"].unsqueeze(0).to(device)
    gt_masks = item["mask_labels"]
    gt_labels = item["class_labels"]

    out = model(pixel_values=image_tensor, pixel_mask=pixel_mask)
    logits = out.class_queries_logits[0]
    masks = out.masks_queries_logits[0]

    scores = logits.softmax(-1)[:, :-1].max(-1).values
    labels = logits.softmax(-1)[:, :-1].max(-1).indices
    keep = scores > score_thresh
    if keep.sum() == 0:
        k = min(10, scores.numel())
        topk = scores.topk(k).indices
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep[topk] = True

    final_masks = masks[keep]
    final_scores = scores[keep]
    final_labels = labels[keep]

    if len(final_masks) > 0:
        H, W = image_tensor.shape[-2:]
        final_masks = (
            F.interpolate(final_masks.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)
            .squeeze(1)
            .sigmoid()
            > mask_thresh
        )

    img_disp = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-6)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_disp)
    axs[0].set_title("Input")
    axs[0].axis("off")

    gt_overlay = np.zeros_like(img_disp)
    if len(gt_masks) > 0:
        for m, label_id in zip(gt_masks, gt_labels):
            color = np.array(get_palette_color(palette, int(label_id.item()))) / 255.0
            gt_overlay[m.numpy() > 0.5] = color
    axs[1].imshow(img_disp)
    axs[1].imshow(gt_overlay, alpha=0.5)
    axs[1].set_title(f"GT ({len(gt_masks)})")
    axs[1].axis("off")

    pred_overlay = np.zeros_like(img_disp)
    if len(final_masks) > 0:
        for m, label_id in zip(final_masks, final_labels):
            color = np.array(get_palette_color(palette, int(label_id.item()))) / 255.0
            pred_overlay[m.cpu().numpy()] = color
    axs[2].imshow(img_disp)
    axs[2].imshow(pred_overlay, alpha=0.5)
    axs[2].set_title(f"Pred ({len(final_masks)})")
    axs[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    id2label = id2label or getattr(model.config, "id2label", None) or {}
    instances = []
    if len(final_masks) > 0:
        for i in range(len(final_masks)):
            label_id = int(final_labels[i].item())
            label_name = id2label.get(label_id, f"class_{label_id}")
            mask_area = int(final_masks[i].sum().item())
            instances.append(
                {
                    "label_id": label_id,
                    "label_name": label_name,
                    "score": float(final_scores[i].item()),
                    "mask_area": mask_area,
                }
            )

    meta = {
        "index": idx,
        "num_instances": len(instances),
        "instances": instances,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    os.makedirs(args.out_dir, exist_ok=True)

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.run_dir,
        local_files_only=True,
    ).to(device)

    dataset = ADE20kPanopticDataset(
        args.data_dir,
        split=args.split,
        transform=get_transforms(IMAGE_SIZE, train=False),
    )
    id2label = load_ade_label_names(args.data_dir)
    palette = load_ade_palette(args.data_dir, args.split)

    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), k=min(args.num_samples, len(dataset)))
    for idx in indices:
        out_path = os.path.join(args.out_dir, f"{args.split}_idx_{idx}.png")
        meta_path = os.path.join(args.out_dir, f"{args.split}_idx_{idx}.json")
        visualize_sample(
            model,
            dataset,
            idx,
            device,
            score_thresh=args.score_thresh,
            mask_thresh=args.mask_thresh,
            out_path=out_path,
            meta_path=meta_path,
            id2label=id2label,
            palette=palette,
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

