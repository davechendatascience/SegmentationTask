import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    # Note: For PQ, you might need 'panopticapi' or specific torchmetrics version.
    # We will focus on AP (Instance Segmentation) first as requested.
except ImportError:
    print("Please install torchmetrics: pip install torchmetrics")
    MeanAveragePrecision = None

# Import our custom blocks
# In Colab, if these are in one cell, just call them directly. 
# If separate files, import them.
from core_design.dataset_block import ADE20kPanopticDataset, get_transforms, collafe_fn, IMAGE_SIZE
from core_design.model_block import CLIPPanopticModel
from core_design.eval_block import evaluate_model, visualize_prediction
from core_design.loss_block import SetCriterion, HungarianMatcher

# --- Hyperparameters ---
LR_BACKBONE = 1e-5
LR_DECODER = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_M2F_CLASS_LOSS = True
USE_POINT_SAMPLING = True
EOS_COEF = 0.1
RUN_VALIDATION = False
RETURN_DEBUG_STATS = True
VISUALIZE_SAMPLE = False

def train_one_epoch(model, criterion, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Unpack
        pixel_values, targets = batch
        pixel_values = pixel_values.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward
        outputs = model(pixel_values)
        
        # Loss
        loss, loss_dict = criterion(outputs, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item(), "cls": loss_dict['loss_ce'].item(), "mask": loss_dict['loss_mask'].item()})
        
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    metric = MeanAveragePrecision() if MeanAveragePrecision else None
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values, targets = batch
        pixel_values = pixel_values.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(pixel_values)
        
        if metric:
            # Prepare for torchmetrics
            # preds: list of dicts(boxes, scores, labels, masks)
            preds = []
            for i in range(len(targets)):
                # Filter low scores
                prob = outputs["pred_logits"][i].softmax(-1) # [Q, Classes+1]
                scores, labels = prob[:, :-1].max(-1)
                
                # Keep top K or threshold
                keep = scores > 0.5
                
                # Get Masks (Sigmoid)
                masks = outputs["pred_masks"][i][keep].sigmoid() > 0.5
                
                # Resize masks to original size if needed (metric handles it?)
                # Torchmetrics MAP expects boolean masks
                
                preds.append({
                    "masks": masks, 
                    "scores": scores[keep], 
                    "labels": labels[keep]
                })
            
            # Update metric
            # Note: MAP expects 'boxes' usually, but 'masks' supported in newer versions for segregation
            # If standard MAP doesn't support only masks, we calculate IOU manually.
            # Assuming recent torchmetrics supports input_type="segm" or similar implied by 'masks' key.
            pass # Placeholder for actual update call to avoid version errors in this block
            
    if metric:
        # return metric.compute()
        return {"AP": 0.0} # Placeholder
    return {}

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Data
    # Assuming the dataset is ready
    train_ds = ADE20kPanopticDataset(split="train", transform=get_transforms(IMAGE_SIZE, train=True))
    val_ds = ADE20kPanopticDataset(split="validation", transform=get_transforms(IMAGE_SIZE, train=False))
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collafe_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collafe_fn, num_workers=2)
    
    # 2. Model
    base_model = CLIPPanopticModel(num_classes=150).to(DEVICE)
    model = base_model
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    
    # 3. Loss
    matcher = HungarianMatcher(
        use_torch_lap=True,
        use_point_sampling=USE_POINT_SAMPLING,
    )
    weight_dict = {'loss_ce': 2.0, 'loss_mask': 5.0, 'loss_dice': 5.0} # Standard Mask2Former weights
    criterion = SetCriterion(
        num_classes=150,
        matcher=matcher,
        weight_dict=weight_dict,
        use_mask2former_cls=USE_M2F_CLASS_LOSS,
        use_point_sampling=USE_POINT_SAMPLING,
        eos_coef=EOS_COEF,
    ).to(DEVICE)
    
    # 4. Optimizer
    # Separate param groups for backbone (LoRA) and decoder
    param_dicts = [
        {"params": [p for n, p in base_model.named_parameters() if "backbone" in n and p.requires_grad], "lr": LR_BACKBONE},
        {"params": [p for n, p in base_model.named_parameters() if "decoder" in n and p.requires_grad], "lr": LR_DECODER},
    ]
    optimizer = optim.AdamW(param_dicts, weight_decay=WEIGHT_DECAY)
    
    # 5. Training Loop
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, DEVICE)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validation
        if RUN_VALIDATION:
            if RETURN_DEBUG_STATS:
                metrics, debug = evaluate_model(model, val_loader, DEVICE, return_debug=True)
                print(f"Validation Metrics: {metrics}")
                print(f"Debug: {debug}")
            else:
                metrics = evaluate_model(model, val_loader, DEVICE)
                print(f"Validation Metrics: {metrics}")

        if VISUALIZE_SAMPLE:
            rand_idx = np.random.randint(0, len(val_ds))
            visualize_prediction(model, val_ds, rand_idx, DEVICE)
        
    # Save
    torch.save(base_model.state_dict(), "clip_panoptic_lora.pth")
    print("Model saved!")

if __name__ == "__main__":
    main()
