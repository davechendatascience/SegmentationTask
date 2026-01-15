import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchmetrics.detection.mean_ap import MeanAveragePrecision

@torch.no_grad()
def evaluate_model(model, dataloader, device, return_debug=False):
    model.eval()
    metric = MeanAveragePrecision(iou_type="segm")
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
        
        # Function to ensure targets are in format expected by torchmetrics
        # Torchmetrics expects:
        # - targets: list of dicts with 'masks' (bool or uint8), 'labels', 'boxes' (optional for segm but good to have)
        formatted_targets = []
        for t in targets:
             # Masks: [N, H, W] -> Boolean
            masks_bool = t['masks'].to(device) > 0.5
            formatted_targets.append({
                "masks": masks_bool,
                "labels": t['class_labels'].to(device)
            })

        outputs = model(pixel_values)
        
        # Process Outputs
        # pred_logits: [B, Q, K+1]
        # pred_masks: [B, Q, H, W]
        preds = []
        for i in range(len(formatted_targets)):
            logits = outputs['pred_logits'][i]
            masks_logits = outputs['pred_masks'][i]
            
            # Probabilities and labels
            prob = logits.softmax(-1) # [Q, K+1]
            scores, labels = prob[:, :-1].max(-1) # Exclude 'no-object' class
            no_obj = prob[:, -1]
            
            # Filter low confidence
            # DETR models have low confidence initially. 
            # 0.5 is too high for early epochs and kills mAP (cuts off PR curve).
            keep = scores > 0.05 
            
            if keep.sum() == 0:
                # No predictions
                preds.append({
                    "masks": torch.zeros((0, *masks_logits.shape[-2:]), dtype=torch.bool, device=device),
                    "scores": torch.tensor([], device=device),
                    "labels": torch.tensor([], device=device)
                })
                continue

            filtered_scores = scores[keep]
            filtered_labels = labels[keep]
            filtered_masks = masks_logits[keep]
            
            # Upsample masks to target resolution (if needed, usually done by metric but let's match target)
            # Assuming target resolution is 512x512 (same as input)
            # Model outputs low-res or 512x512 depending on decoder upsample. 
            # Our LightMask2Former outputs 16x downsampled or similar if not upsampled at end.
            # Let's force upsample to IMAGE_SIZE (512)
            target_H, target_W = formatted_targets[i]['masks'].shape[-2:]
            
            filtered_masks = F.interpolate(filtered_masks.unsqueeze(1), size=(target_H, target_W), mode="bilinear", align_corners=False).squeeze(1)
            filtered_masks = filtered_masks.sigmoid() > 0.5
            
            preds.append({
                "masks": filtered_masks,
                "scores": filtered_scores,
                "labels": filtered_labels
            })

            # Debug stats (per-image)
            debug["avg_max_score"] += float(scores.mean().detach().cpu())
            debug["avg_noobj"] += float(no_obj.mean().detach().cpu())
            debug["avg_mask_logit"] += float(masks_logits.mean().detach().cpu())
            debug["avg_keep_frac"] += float(keep.float().mean().detach().cpu())
            debug["batches"] += 1
            
        metric.update(preds, formatted_targets)
        
    result = metric.compute()
    if return_debug and debug["batches"] > 0:
        scale = 1.0 / debug["batches"]
        debug = {
            "avg_max_score": debug["avg_max_score"] * scale,
            "avg_noobj": debug["avg_noobj"] * scale,
            "avg_mask_logit": debug["avg_mask_logit"] * scale,
            "avg_keep_frac": debug["avg_keep_frac"] * scale,
        }
        return result, debug
    return result

def visualize_prediction(model, dataset, idx, device):
    model.eval()
    
    # Load raw item for display
    # We need transforms for model input, but want raw for display
    # Re-access dataset item
    item_dict = dataset[idx]
    image_tensor = item_dict['pixel_values'].unsqueeze(0).to(device) # [1, 3, H, W]
    
    # Ground Truth
    gt_masks = item_dict['masks']
    
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
        for i, m in enumerate(gt_masks):
            color = np.random.rand(3)
            # mask is float 0-1
            m = m.numpy()
            combined_gt[m > 0.5] = color
    
    axs[1].imshow(img_disp)
    axs[1].imshow(combined_gt, alpha=0.5)
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')
    
    # 3. Prediction Overlay
    combined_pred = np.zeros_like(img_disp)
    if len(final_masks) > 0:
        for i, m in enumerate(final_masks):
            color = np.random.rand(3)
            m = m.cpu().numpy()
            combined_pred[m > 0.5] = color
            
    axs[2].imshow(img_disp)
    axs[2].imshow(combined_pred, alpha=0.5)
    axs[2].set_title(f"Prediction ({len(final_masks)} objects)")
    axs[2].axis('off')
    
    plt.show()
