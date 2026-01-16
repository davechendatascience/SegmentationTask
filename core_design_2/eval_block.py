import glob
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchmetrics.detection import PanopticQuality
from tqdm import tqdm

SCORE_THRESH = 0.05
VIS_MASK_THRESH = 0.3
MAX_DETECTIONS = 100


def _build_panoptic_from_masks(masks, labels, scores=None, score_thresh=0.05, mask_thresh=0.5):
    if masks.numel() == 0:
        return None
    if scores is not None:
        keep = scores > score_thresh
        masks = masks[keep]
        labels = labels[keep]
        scores = scores[keep]
        if masks.numel() == 0:
            return None
    if scores is not None:
        order = scores.argsort(descending=True)
        masks = masks[order]
        labels = labels[order]
    masks = masks.sigmoid() if masks.dtype.is_floating_point else masks
    masks = masks > mask_thresh
    return masks, labels


def _make_panoptic_tensor(masks, labels, height, width):
    unknown = 255
    device = masks.device if masks is not None else torch.device("cpu")
    panoptic = torch.full((height, width, 2), unknown, dtype=torch.int64, device=device)
    panoptic[..., 1] = 0
    if masks is None:
        return panoptic
    occupied = torch.zeros((height, width), dtype=torch.bool, device=device)
    instance_id = 1
    for m, label in zip(masks, labels):
        m = m.bool()
        m = m & (~occupied)
        if not m.any():
            continue
        panoptic[m, 0] = int(label)
        panoptic[m, 1] = instance_id
        occupied[m] = True
        instance_id += 1
    return panoptic


@torch.no_grad()
def evaluate_model(model, dataloader, device, use_cpu_metric=True):
    model.eval()
    metric_device = torch.device("cpu") if use_cpu_metric else torch.device(device)
    metric = PanopticQuality(
        things=set(range(150)),
        stuffs=set(),
        allow_unknown_preds_category=True,
        return_sq_and_rq=True,
    ).to(metric_device)
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        mask_labels = batch["mask_labels"]
        class_labels = batch["class_labels"]

        out = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        logits = out.class_queries_logits.detach().cpu()
        masks = out.masks_queries_logits.detach().cpu()
        del out

        preds = []
        targets_pan = []
        for i in range(logits.shape[0]):
            scores = logits[i].softmax(-1)[:, :-1].max(-1).values
            labels = logits[i].softmax(-1)[:, :-1].max(-1).indices
            keep = scores > SCORE_THRESH

            tgt_masks = mask_labels[i]
            tgt_labels = class_labels[i]
            target_h, target_w = tgt_masks.shape[-2:]

            pred_masks = masks[i]
            if pred_masks.shape[-2:] != (target_h, target_w):
                pred_masks = (
                    F.interpolate(
                        pred_masks.unsqueeze(1),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(1)
                )

            pred_pack = _build_panoptic_from_masks(pred_masks, labels, scores=scores, score_thresh=SCORE_THRESH)
            pred_pan = _make_panoptic_tensor(
                pred_pack[0], pred_pack[1], target_h, target_w
            ) if pred_pack is not None else _make_panoptic_tensor(None, None, target_h, target_w)

            tgt_pack = _build_panoptic_from_masks(tgt_masks, tgt_labels, scores=None)
            tgt_pan = _make_panoptic_tensor(
                tgt_pack[0], tgt_pack[1], target_h, target_w
            ) if tgt_pack is not None else _make_panoptic_tensor(None, None, target_h, target_w)

            preds.append(pred_pan)
            targets_pan.append(tgt_pan)

        preds = torch.stack(preds, dim=0).to(metric_device)
        targets_pan = torch.stack(targets_pan, dim=0).to(metric_device)
        metric.update(preds, targets_pan)
    result = metric.compute()
    return {"pq": result[0], "sq": result[1], "rq": result[2]}


def _load_ade_palette(dataset):
    root_dir = getattr(dataset, "root_dir", None)
    split = getattr(dataset, "split", None)
    if not root_dir or not split:
        return None
    ann_dir = os.path.join(root_dir, "annotations", split)
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


def _get_palette_color(palette, label_id):
    if not palette:
        rng = np.random.RandomState(label_id)
        return tuple((rng.rand(3) * 255).astype(np.uint8))
    palette_idx = label_id + 1
    if palette_idx < len(palette):
        return palette[palette_idx]
    return palette[0] if palette else (255, 255, 255)


@torch.no_grad()
def visualize_prediction(model, dataset, idx, device):
    model.eval()
    palette = _load_ade_palette(dataset)
    item = dataset[idx]
    image_tensor = item["pixel_values"].unsqueeze(0).to(device)
    pixel_mask = item["pixel_mask"].unsqueeze(0).to(device)

    out = model(pixel_values=image_tensor, pixel_mask=pixel_mask)
    logits = out.class_queries_logits[0]
    masks = out.masks_queries_logits[0]

    scores = logits.softmax(-1)[:, :-1].max(-1).values
    labels = logits.softmax(-1)[:, :-1].max(-1).indices
    keep = scores > SCORE_THRESH
    if keep.sum() == 0:
        k = min(10, scores.numel())
        topk = scores.topk(k).indices
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep[topk] = True
    elif keep.sum() > MAX_DETECTIONS:
        topk = scores.topk(MAX_DETECTIONS).indices
        keep = torch.zeros_like(scores, dtype=torch.bool)
        keep[topk] = True

    final_masks = masks[keep]
    final_scores = scores[keep]
    final_labels = labels[keep]

    if len(final_masks) > 0:
        H, W = image_tensor.shape[-2:]
        final_masks = F.interpolate(final_masks.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        final_masks = final_masks.sigmoid() > VIS_MASK_THRESH

    img_disp = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-6)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_disp)
    axs[0].set_title("Input")
    axs[0].axis("off")

    overlay = np.zeros_like(img_disp)
    if len(final_masks) > 0:
        for m, label_id in zip(final_masks, final_labels):
            color = np.array(_get_palette_color(palette, int(label_id.item()))) / 255.0
            overlay[m.cpu().numpy()] = color
    axs[1].imshow(img_disp)
    axs[1].imshow(overlay, alpha=0.5)
    axs[1].set_title(f"Predictions ({len(final_masks)})")
    axs[1].axis("off")
    plt.show()

