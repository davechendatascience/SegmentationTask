"""
Train Mask2Former for instance segmentation on the hospital COCO dataset.

Usage:
    python -m scripts.mask2former_seg.train
    python -m scripts.mask2former_seg.train --resume output/mask2former/checkpoint_epoch5
"""
import argparse
import contextlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from .config import AugConfig, DataConfig, ModelConfig, TrainConfig
from .dataset import HospitalCOCODataset


# --------------------------------------------------------------------------- #
# Collate function                                                              #
# --------------------------------------------------------------------------- #

def make_collate_fn(processor):
    """
    Build batches using Mask2FormerImageProcessor's correct v5 API:
      - segmentation_maps: list of (H, W) uint16 maps where each pixel = instance_id
        (0 = background, 1 = instance 1, 2 = instance 2, ...)
      - instance_id_to_semantic_id: list of dicts {instance_id -> class_id (0-indexed)}
    The processor then returns 'mask_labels' and 'class_labels' as expected by the model.
    """

    def collate_fn(batch):
        images = [item["pixel_values"] for item in batch]
        all_masks = [item["masks"] for item in batch]
        all_labels = [item["class_labels"] for item in batch]
        image_ids = [item["image_id"] for item in batch]

        segmentation_maps = []
        id_to_semantic_maps = []

        for masks, labels in zip(all_masks, all_labels):
            h, w = masks[0].shape if masks else (512, 512)
            seg_map = np.zeros((h, w), dtype=np.int32)  # 0 = background
            # Key fix: processor iterates ALL unique pixel values including 0 (background).
            # We must include 0 in the id_to_semantic dict, mapped to 0 (a dummy class).
            # The processor will create a mask for background but the model treats class 0 as ignored.
            id_to_semantic = {0: 0}

            for inst_id, (mask, class_id) in enumerate(zip(masks, labels), start=1):
                seg_map[mask > 0] = inst_id
                id_to_semantic[inst_id] = class_id + 1  # +1 because 0 is reserved for background

            segmentation_maps.append(seg_map)
            id_to_semantic_maps.append(id_to_semantic)

        encodings = processor(
            images=images,
            segmentation_maps=segmentation_maps,
            instance_id_to_semantic_id=id_to_semantic_maps,
            return_tensors="pt",
        )
        encodings["image_ids"] = image_ids
        return encodings

    return collate_fn


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
    train_dataset = HospitalCOCODataset(
        data_cfg.data_root, split="train", image_size=data_cfg.image_size, augment=True
    )
    val_dataset = HospitalCOCODataset(
        data_cfg.data_root, split="valid", image_size=data_cfg.image_size, augment=False
    )

    num_classes = train_dataset.num_classes
    print(f"Number of classes: {num_classes}")

    # ------------------------------------------------------------------ #
    # Model & Processor                                                    #
    # ------------------------------------------------------------------ #
    print(f"Loading checkpoint: {model_cfg.checkpoint}")
    processor = Mask2FormerImageProcessor.from_pretrained(
        model_cfg.checkpoint,
        do_reduce_labels=False,  # Keep all class IDs as-is (0-indexed)
    )

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        model_cfg.checkpoint,
        num_labels=num_classes + 1,  # +1 for background class (index 0)
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    collate_fn = make_collate_fn(processor)

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
    # Optimizer & Scheduler                                                #
    # ------------------------------------------------------------------ #
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": train_cfg.learning_rate * train_cfg.backbone_lr_factor},
            {"params": head_params,     "lr": train_cfg.learning_rate},
        ],
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

    amp_ctx = (
        contextlib.nullcontext()
        if not (train_cfg.bf16 or not train_cfg.fp16)
        else torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if train_cfg.bf16
        else torch.amp.autocast("cuda", dtype=torch.float16)
    )
    # Simpler: just always use bfloat16 if bf16=True
    def get_amp_ctx():
        if train_cfg.bf16:
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------ #
    # Resume from checkpoint                                               #
    # ------------------------------------------------------------------ #
    start_epoch = 0
    best_val_loss = float("inf")

    if resume and Path(resume).exists():
        print(f"Resuming from: {resume}")
        ckpt = torch.load(os.path.join(resume, "training_state.pt"), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # ------------------------------------------------------------------ #
    # Training epochs                                                      #
    # ------------------------------------------------------------------ #
    global_step = 0

    for epoch in range(start_epoch, train_cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask   = batch.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device)

            mask_labels  = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]

            with get_amp_ctx():
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
                loss = outputs.loss / train_cfg.grad_accum_steps

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
                pixel_values = batch["pixel_values"].to(device)
                pixel_mask   = batch.get("pixel_mask")
                if pixel_mask is not None:
                    pixel_mask = pixel_mask.to(device)
                mask_labels  = [m.to(device) for m in batch["mask_labels"]]
                class_labels = [c.to(device) for c in batch["class_labels"]]

                with get_amp_ctx():
                    outputs = model(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        mask_labels=mask_labels,
                        class_labels=class_labels,
                    )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"  Epoch {epoch+1} val loss: {avg_val_loss:.4f}")

        # -------------------------------------------------------------- #
        # Checkpoint                                                       #
        # -------------------------------------------------------------- #
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(model_cfg.output_dir, "best_model")
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
            print(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

        if (epoch + 1) % train_cfg.save_every_epochs == 0:
            ckpt_path = os.path.join(model_cfg.output_dir, f"checkpoint_epoch{epoch+1}")
            model.save_pretrained(ckpt_path)
            processor.save_pretrained(ckpt_path)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                os.path.join(ckpt_path, "training_state.pt"),
            )
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
    args = parser.parse_args()

    train_cfg = TrainConfig()
    data_cfg  = DataConfig()
    if args.epochs:
        train_cfg.num_epochs = args.epochs
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.data_root:
        data_cfg.data_root = args.data_root

    train(data_cfg=data_cfg, train_cfg=train_cfg, resume=args.resume)



