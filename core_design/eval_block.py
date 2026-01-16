import glob
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchmetrics.detection import PanopticQuality
from tqdm import tqdm

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
def evaluate_model(model, dataloader, device, return_debug=False, use_cpu_metric=True):
    model.eval()
    metric_device = torch.device("cpu") if use_cpu_metric else torch.device(device)
    metric = PanopticQuality(
        things=set(range(150)),
        stuffs=set(),
        allow_unknown_preds_category=True,
        return_sq_and_rq=True,
    ).to(metric_device)
    debug = {
        "avg_max_score": 0.0,
        "avg_noobj": 0.0,
        "avg_mask_logit": 0.0,
        "avg_keep_frac": 0.0,
        "batches": 0,
    }
    
    print("Running evaluation...")
    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values, targets = batch
        pixel_values = pixel_values.to(device)

        outputs = model(pixel_values)
        outputs_logits = outputs["pred_logits"].detach().cpu()
        outputs_masks = outputs["pred_masks"].detach().cpu()
        del outputs

        preds = []
        targets_pan = []
        for i, t in enumerate(targets):
            tgt_masks = t["masks"]
            tgt_labels = t["class_labels"]
            target_h, target_w = tgt_masks.shape[-2:]

            logits = outputs_logits[i]
            masks_logits = outputs_masks[i]
            prob = logits.softmax(-1)
            scores, labels = prob[:, :-1].max(-1)
            no_obj = prob[:, -1]
            keep = scores > 0.05

            if masks_logits.shape[-2:] != (target_h, target_w):
                masks_logits = F.interpolate(
                    masks_logits.unsqueeze(1),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            pred_masks = masks_logits
            pred_pack = _build_panoptic_from_masks(pred_masks, labels, scores=scores, score_thresh=0.05)
            pred_pan = _make_panoptic_tensor(
                pred_pack[0], pred_pack[1], target_h, target_w
            ) if pred_pack is not None else _make_panoptic_tensor(None, None, target_h, target_w)

            tgt_pack = _build_panoptic_from_masks(tgt_masks, tgt_labels, scores=None)
            tgt_pan = _make_panoptic_tensor(
                tgt_pack[0], tgt_pack[1], target_h, target_w
            ) if tgt_pack is not None else _make_panoptic_tensor(None, None, target_h, target_w)

            preds.append(pred_pan)
            targets_pan.append(tgt_pan)

            debug["avg_max_score"] += float(scores.mean().detach().cpu())
            debug["avg_noobj"] += float(no_obj.mean().detach().cpu())
            debug["avg_mask_logit"] += float(masks_logits.mean().detach().cpu())
            debug["avg_keep_frac"] += float(keep.float().mean().detach().cpu())
            debug["batches"] += 1

        preds = torch.stack(preds, dim=0).to(metric_device)
        targets_pan = torch.stack(targets_pan, dim=0).to(metric_device)
        metric.update(preds, targets_pan)
        
    result = metric.compute()
    if return_debug and debug["batches"] > 0:
        scale = 1.0 / debug["batches"]
        debug = {
            "avg_max_score": debug["avg_max_score"] * scale,
            "avg_noobj": debug["avg_noobj"] * scale,
            "avg_mask_logit": debug["avg_mask_logit"] * scale,
            "avg_keep_frac": debug["avg_keep_frac"] * scale,
        }
        return {"pq": result[0], "sq": result[1], "rq": result[2]}, debug
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


def visualize_prediction(model, dataset, idx, device):
    model.eval()
    palette = _load_ade_palette(dataset)
    
    # Load raw item for display
    # We need transforms for model input, but want raw for display
    # Re-access dataset item
    item_dict = dataset[idx]
    image_tensor = item_dict['pixel_values'].unsqueeze(0).to(device) # [1, 3, H, W]
    
    # Ground Truth
    gt_masks = item_dict['masks']
    gt_labels = item_dict['class_labels']
    
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Decode Prediction
    logits = outputs['pred_logits'][0]
    pred_masks = outputs['pred_masks'][0]
    
    prob = logits.softmax(-1)
    scores, labels = prob[:, :-1].max(-1)
    
    keep = scores > 0.05
    final_masks = pred_masks[keep]
    final_scores = scores[keep]
    final_labels = labels[keep]
    
    # Upsample
    if len(final_masks) > 0:
        H, W = image_tensor.shape[-2:]
        final_masks = F.interpolate(final_masks.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
        final_masks = final_masks.sigmoid() > 0.5
    
    # --- Plotting ---
    # Denormalize Image
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    img_disp = (item_dict['pixel_values'].cpu() * std + mean).permute(1, 2, 0).numpy()
    img_disp = np.clip(img_disp, 0, 1)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original
    axs[0].imshow(img_disp)
    axs[0].set_title("Input Image")
    axs[0].axis('off')
    
    # 2. Ground Truth Overlay
    combined_gt = np.zeros_like(img_disp)
    if len(gt_masks) > 0:
        for m, label_id in zip(gt_masks, gt_labels):
            color = np.array(_get_palette_color(palette, int(label_id.item()))) / 255.0
            m = m.numpy()
            combined_gt[m > 0.5] = color
    
    axs[1].imshow(img_disp)
    axs[1].imshow(combined_gt, alpha=0.5)
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')
    
    # 3. Prediction Overlay
    combined_pred = np.zeros_like(img_disp)
    if len(final_masks) > 0:
        for m, label_id in zip(final_masks, final_labels):
            color = np.array(_get_palette_color(palette, int(label_id.item()))) / 255.0
            m = m.cpu().numpy()
            combined_pred[m > 0.5] = color
            
    axs[2].imshow(img_disp)
    axs[2].imshow(combined_pred, alpha=0.5)
    axs[2].set_title(f"Prediction ({len(final_masks)} objects)")
    axs[2].axis('off')
    
    plt.show()
