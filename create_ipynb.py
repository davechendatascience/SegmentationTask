import argparse
import json
import os

# Define the structure of the notebook
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_cell(cell_type, source):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }
    # Remove last newline for cleaner look
    if cell["source"] and cell["source"][-1] == "\n":
         cell["source"][-1] = ""
    notebook["cells"].append(cell)

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# --- Content Assembly ---

parser = argparse.ArgumentParser()
parser.add_argument("--core", type=str, default="core_design")
args = parser.parse_args()
core_dir = args.core

# Cell 1: Header
add_cell("markdown", "# Panoptic Segmentation with CLIP + Mask2Former (LoRA)\n\nThis notebook implements a panoptic segmentation pipeline using a CLIP backbone (fine-tuned with LoRA) and a lightweight Mask2Former-style decoder.\n\n**Note**: The dataset is downloaded manually from MIT SceneParsing to ensure stability.")

# Cell 2: Setup
add_cell("code", "!pip install transformers datasets albumentations peft torchmetrics scipy")
if core_dir == "core_design":
    add_cell("code", "!pip install torch-linear-assignment --no-build-isolation")

# Cell 3: Imports (Hardcoded to ensure order)
imports_code = """import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import glob
from torch.utils.data import Dataset, DataLoader
# from datasets import load_dataset # Eliminated
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import CLIPVisionModel, CLIPConfig
from peft import LoraConfig, get_peft_model
from scipy.optimize import linear_sum_assignment
from tqdm.notebook import tqdm

# Configuration
IMAGE_SIZE = 512
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
NUM_CLASSES = 150 
"""
add_cell("code", imports_code)

if core_dir == "core_design_2":
    add_cell("markdown", "## Mask2Former (HF) Training\nThis notebook cell contains the training blocks from `core_design_2`.")
    aug_code = read_file("core_design_2/augmentation_block.py")
    add_cell("code", "# --- Augmentation ---\n" + aug_code)
    dataset_code = read_file("core_design_2/dataset_block.py")
    dataset_code = dataset_code.replace("from core_design_2.augmentation_block import get_train_transforms, get_eval_transforms\n", "")
    add_cell("code", "# --- Dataset Implementation ---\n" + dataset_code)
    model_code = read_file("core_design_2/model_block.py")
    add_cell("code", "# --- Model Architecture ---\n" + model_code)
    eval_code = read_file("core_design_2/eval_block.py")
    add_cell("code", "# --- Evaluation & Visualization ---\n" + eval_code)
    loss_code = read_file("core_design_2/loss_block.py")
    add_cell("code", "# --- Loss (Built-in) ---\n" + loss_code)
    train_code = read_file("core_design_2/train_block.py")
    add_cell("code", "# --- Training Loop ---\n" + train_code + "\n\n# Run training\n# main()")
else:
    # Cell 4: Augmentation Block
    aug_code = read_file(f"{core_dir}/augmentation_block.py")
    add_cell("code", "# --- Augmentation ---\n" + aug_code)

    # Cell 5: Dataset Block
    dataset_code = read_file(f"{core_dir}/dataset_block.py")
    dataset_code = dataset_code.replace(f"from {core_dir}.augmentation_block import get_train_transforms, get_eval_transforms\n", "")
    add_cell("code", "# --- Dataset Implementation ---\n" + dataset_code)

    # Cell 6: Model Block
    model_code = read_file(f"{core_dir}/model_block.py")
    add_cell("code", "# --- Model Architecture ---\n" + model_code)

    # Cell 7: Evaluation Block
    eval_code = read_file(f"{core_dir}/eval_block.py")
    add_cell("code", "# --- Evaluation & Visualization ---\n" + eval_code)

    # Cell 8: Loss Block
    loss_code = read_file(f"{core_dir}/loss_block.py")
    add_cell("code", "# --- Loss & Matcher ---\n" + loss_code)

    # Cell 9: Training Block (Modified Main)
    training_execution_code = """# --- Training Loop Execution ---
print(f"Using device: {DEVICE}")

# 1. Data
# This will trigger download if not found
train_ds = ADE20kPanopticDataset(split="train", transform=get_transforms(IMAGE_SIZE))
val_ds = ADE20kPanopticDataset(split="validation", transform=get_transforms(IMAGE_SIZE))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collafe_fn, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collafe_fn, num_workers=2)

# Configuration
IMAGE_SIZE = 512
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
NUM_CLASSES = 150 
LORA_RANK = 16 # Adjustable LoRA Rank

# 2. Model
model = CLIPPanopticModel(num_classes=NUM_CLASSES, lora_rank=LORA_RANK)
model.to(DEVICE)

# Optimization: PyTorch 2.0 Compilation
# This fuses kernels for JAX-like performance
if hasattr(torch, "compile"):
    print("Compiling model with torch.compile...")
    model = torch.compile(model)
torch.backends.cudnn.benchmark = True

# 3. Loss
matcher = HungarianMatcher(use_torch_lap=True)
# Weights updated to prioritize Hierarchical Loss (Self-Supervised + Semantic)
weight_dict = {
    'loss_ce': 0.1,          # Reduced focal loss (let hierarchy drive)
    'loss_parent': 1.0,      # Coarse semantic grouping
    'loss_kl': 2.0,          # Soft target matching
    'loss_mask': 5.0,        # Shape
    'loss_dice': 5.0,        # Overlap
    'loss_boundary': 2.0,    # Edges
    'loss_consistency': 1.0  # Multi-scale
} 
# Note: criterion will autodownload CLIP for hierarchy on init
criterion = SetCriterion(num_classes=NUM_CLASSES, matcher=matcher, weight_dict=weight_dict, device=DEVICE).to(DEVICE)

# 4. Optimizer
param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 1e-5},
    {"params": [p for n, p in model.named_parameters() if "decoder" in n and p.requires_grad], "lr": 1e-4},
]
optimizer = optim.AdamW(param_dicts, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler() # Mixed Precision Scaler

# 5. Loop
EPOCHS = 5
for epoch in range(EPOCHS):
    print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
    
    # Train One Epoch Inline
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch in pbar:
        pixel_values, targets = batch
        pixel_values = pixel_values.to(DEVICE)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        # Mixed Precision Training
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pixel_values)
            loss, loss_dict = criterion(outputs, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
        
    print(f"Average Loss: {total_loss / len(train_loader):.4f}")
    
    # Validation & Visualization Step (Every Epoch)
    if (epoch + 1) % 1 == 0:
        # Visualize Prediction on a random sample
        print("Visualizing random sample...")
        rand_idx = np.random.randint(0, len(val_ds))
        visualize_prediction(model, val_ds, rand_idx, DEVICE)

torch.save(model.state_dict(), "clip_panoptic_lora.pth")
print("Model saved!")
"""
    add_cell("code", training_execution_code)


# Output File
with open("panoptic_segmentation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook 'panoptic_segmentation.ipynb' refreshed successfully.")
