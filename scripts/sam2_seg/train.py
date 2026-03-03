"""
Train SAM2-UNet for segmentation on the hospital COCO dataset.

Usage:
    python -m scripts.sam2_seg.train
    python -m scripts.sam2_seg.train --resume output/sam2/checkpoint_epoch5
    python -m scripts.sam2_seg.train --epochs 20 --batch_size 1
"""
import argparse
import contextlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import AugConfig, DataConfig, ModelConfig, TrainConfig
from .dataset import HospitalCOCOSegDataset, collate_fn
from .segmentation_model import SAM2SegModel


# --------------------------------------------------------------------------- #
# Loss functions                                                               #
# --------------------------------------------------------------------------- #

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Per-class Dice loss.

    Args:
        logits: [B, C, H, W] raw model output
        targets: [B, H, W] integer class labels
    """
    probs = F.softmax(logits, dim=1)  # [B, C, H, W]
    targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()  # [B,C,H,W]

    dims = (0, 2, 3)  # sum over batch and spatial
    intersection = (probs * targets_one_hot).sum(dims)
    cardinality = probs.sum(dims) + targets_one_hot.sum(dims)

    dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
    return 1.0 - dice.mean()


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ce_weight: float = 1.0,
    dice_w: float = 1.0,
) -> torch.Tensor:
    """Combined cross-entropy + dice loss."""
    ce = F.cross_entropy(logits, targets)
    dl = dice_loss(logits, targets, num_classes)
    return ce_weight * ce + dice_w * dl


# --------------------------------------------------------------------------- #
# Training loop                                                                #
# --------------------------------------------------------------------------- #

def train(
    data_cfg: DataConfig = None,
    model_cfg: ModelConfig = None,
    train_cfg: TrainConfig = None,
    aug_cfg: AugConfig = None,
    resume: str = None,
):
    data_cfg  = data_cfg  or DataConfig()
    model_cfg = model_cfg or ModelConfig()
    train_cfg = train_cfg or TrainConfig()
    aug_cfg   = aug_cfg   or AugConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(model_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Datasets & DataLoaders                                               #
    # ------------------------------------------------------------------ #
    train_dataset = HospitalCOCOSegDataset(
        data_cfg.data_root, split="train", image_size=data_cfg.image_size, augment=True
    )
    val_dataset = HospitalCOCOSegDataset(
        data_cfg.data_root, split="valid", image_size=data_cfg.image_size, augment=False
    )

    num_classes = train_dataset.num_classes + 1  # +1 for background
    model_cfg.num_classes = num_classes
    print(f"Number of classes (incl. background): {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model = SAM2SegModel.build(
        model_name=model_cfg.sam2_model_name,
        checkpoint_path=model_cfg.sam2_checkpoint,
        config_path=model_cfg.sam2_config,
        num_classes=num_classes,
        use_adapters=model_cfg.use_adapters,
        adapter_dim=model_cfg.adapter_dim,
        image_size=data_cfg.image_size,
    )
    model.to(device)

    # ------------------------------------------------------------------ #
    # Optimizer & Scheduler                                                #
    # ------------------------------------------------------------------ #
    adapter_lr = train_cfg.learning_rate * train_cfg.adapter_lr_factor
    param_groups = model.get_param_groups(
        adapter_lr=adapter_lr,
        decoder_lr=train_cfg.learning_rate,
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.weight_decay,
    )

    total_steps = max(1, len(train_loader) * train_cfg.num_epochs // train_cfg.grad_accum_steps)
    warmup_steps = train_cfg.lr_warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def get_amp_ctx():
        if train_cfg.bf16:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        elif train_cfg.fp16:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------ #
    # Resume from checkpoint                                               #
    # ------------------------------------------------------------------ #
    start_epoch = 0
    best_val_loss = float("inf")

    if resume and Path(resume).exists():
        print(f"Resuming from: {resume}")
        ckpt = torch.load(os.path.join(resume, "training_state.pt"), map_location=device)
        # Load only adapter + decoder state (backbone is frozen and loaded fresh)
        model_state = ckpt.get("model_state", {})
        # Filter to only load trainable parts
        current_state = model.state_dict()
        for k, v in model_state.items():
            if k in current_state and current_state[k].shape == v.shape:
                current_state[k] = v
        model.load_state_dict(current_state)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # ------------------------------------------------------------------ #
    # Training epochs                                                      #
    # ------------------------------------------------------------------ #
    global_step = 0

    for epoch in range(start_epoch, train_cfg.num_epochs):
        model.train()
        # Keep backbone in eval mode (frozen BatchNorm etc.)
        model.backbone.eval()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(device)
            targets = batch["semantic_masks"].to(device)

            with get_amp_ctx():
                logits = model(images)
                loss = compute_loss(
                    logits, targets, num_classes,
                    ce_weight=train_cfg.ce_weight,
                    dice_w=train_cfg.dice_weight,
                )
                loss = loss / train_cfg.grad_accum_steps

            loss.backward()

            if (batch_idx + 1) % train_cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            current_loss = loss.item() * train_cfg.grad_accum_steps
            epoch_loss += current_loss
            pbar.set_postfix(loss=f"{current_loss:.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"\n  Epoch {epoch+1} avg train loss: {avg_train_loss:.4f}")

        # -------------------------------------------------------------- #
        # Validation                                                       #
        # -------------------------------------------------------------- #
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                images = batch["images"].to(device)
                targets = batch["semantic_masks"].to(device)

                with get_amp_ctx():
                    logits = model(images)
                    loss = compute_loss(
                        logits, targets, num_classes,
                        ce_weight=train_cfg.ce_weight,
                        dice_w=train_cfg.dice_weight,
                    )
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"  Epoch {epoch+1} val loss: {avg_val_loss:.4f}")

        # -------------------------------------------------------------- #
        # Checkpoint                                                       #
        # -------------------------------------------------------------- #
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(model_cfg.output_dir, "best_model")
            Path(best_path).mkdir(parents=True, exist_ok=True)
            # Save only trainable parameters + config
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "model_name": model_cfg.sam2_model_name,
                "use_adapters": model_cfg.use_adapters,
                "adapter_dim": model_cfg.adapter_dim,
                "image_size": data_cfg.image_size,
            }, os.path.join(best_path, "model.pt"))
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

        if (epoch + 1) % train_cfg.save_every_epochs == 0:
            ckpt_path = os.path.join(model_cfg.output_dir, f"checkpoint_epoch{epoch+1}")
            Path(ckpt_path).mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "num_classes": num_classes,
                "model_name": model_cfg.sam2_model_name,
                "use_adapters": model_cfg.use_adapters,
                "adapter_dim": model_cfg.adapter_dim,
                "image_size": data_cfg.image_size,
            }, os.path.join(ckpt_path, "training_state.pt"))
            print(f"  Saved checkpoint: {ckpt_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model: {model_cfg.output_dir}/best_model")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    train_cfg = TrainConfig()
    data_cfg  = DataConfig()
    model_cfg = ModelConfig()

    if args.epochs:
        train_cfg.num_epochs = args.epochs
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.data_root:
        data_cfg.data_root = args.data_root
    if args.image_size:
        data_cfg.image_size = args.image_size
    if args.lr:
        train_cfg.learning_rate = args.lr

    train(data_cfg=data_cfg, model_cfg=model_cfg, train_cfg=train_cfg, resume=args.resume)
