import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

SCORE_THRESH = 0.05
VIS_MASK_THRESH = 0.3
MAX_DETECTIONS = 100


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    steps = 0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        mask_labels = [m.to(device) for m in batch["mask_labels"]]
        class_labels = [c.to(device) for c in batch["class_labels"]]
        out = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        total_loss += float(out.loss.detach().cpu())
        steps += 1
    return {"val_loss": total_loss / max(1, steps)}


@torch.no_grad()
def visualize_prediction(model, dataset, idx, device):
    model.eval()
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
        for m in final_masks:
            color = np.random.rand(3)
            overlay[m.cpu().numpy()] = color
    axs[1].imshow(img_disp)
    axs[1].imshow(overlay, alpha=0.5)
    axs[1].set_title(f"Predictions ({len(final_masks)})")
    axs[1].axis("off")
    plt.show()

