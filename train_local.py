import argparse
import importlib
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP+Mask2Former locally")
    parser.add_argument("--data-root", type=str, default="./ADEChallengeData2016")
    parser.add_argument("--core", type=str, default="core_design")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=min(8, (os.cpu_count() or 2)))
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-decoder", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-path", type=str, default="clip_panoptic_lora.pth")
    parser.add_argument("--use-m2f-class-loss", action="store_true", default=True)
    parser.add_argument("--no-m2f-class-loss", action="store_false", dest="use_m2f_class_loss")
    parser.add_argument("--use-point-sampling", action="store_true", default=True)
    parser.add_argument("--no-point-sampling", action="store_false", dest="use_point_sampling")
    parser.add_argument("--eos-coef", type=float, default=0.1)
    return parser.parse_known_args()


def train_one_epoch(model, criterion, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        pixel_values, targets = batch
        pixel_values = pixel_values.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(pixel_values)
        loss, loss_dict = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    return total_loss / max(1, len(dataloader))


def set_precision(device):
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        torch.backends.cudnn.benchmark = True


def main():
    args, extra = parse_args()
    if args.core == "core_design_2":
        train_block = importlib.import_module("core_design_2.train_block")
        return train_block.main(extra)
    dataset_block = importlib.import_module(f"{args.core}.dataset_block")
    model_block = importlib.import_module(f"{args.core}.model_block")
    loss_block = importlib.import_module(f"{args.core}.loss_block")
    eval_block = importlib.import_module(f"{args.core}.eval_block")
    device = args.device
    print(f"Using device: {device}")
    train_ds = dataset_block.ADE20kPanopticDataset(
        root_dir=args.data_root,
        split="train",
        transform=dataset_block.get_transforms(dataset_block.IMAGE_SIZE),
    )
    val_ds = dataset_block.ADE20kPanopticDataset(
        root_dir=args.data_root,
        split="validation",
        transform=dataset_block.get_transforms(dataset_block.IMAGE_SIZE),
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=True,
        prefetch_factor=2,
    )
    train_loader = DataLoader(train_ds, shuffle=True, collate_fn=dataset_block.collafe_fn, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, collate_fn=dataset_block.collafe_fn, **loader_kwargs)

    base_model = model_block.CLIPPanopticModel(num_classes=150).to(device)
    model = base_model
    if hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    matcher = loss_block.HungarianMatcher(
        use_torch_lap=True,
        use_point_sampling=args.use_point_sampling,
    )
    weight_dict = {
        "loss_ce": 2.0,
        "loss_mask": 5.0,
        "loss_dice": 5.0,
    }
    criterion = loss_block.SetCriterion(
        num_classes=150,
        matcher=matcher,
        weight_dict=weight_dict,
        use_mask2former_cls=args.use_m2f_class_loss,
        use_point_sampling=args.use_point_sampling,
        eos_coef=args.eos_coef,
    ).to(device)

    backbone_params = []
    decoder_params = []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            decoder_params.append(param)
    param_dicts = [
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": decoder_params, "lr": args.lr_decoder},
    ]
    optimizer = optim.AdamW(param_dicts, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            metrics = eval_block.evaluate_model(model, val_loader, device)
            print(f"Validation Metrics: {metrics}")

    torch.save(base_model.state_dict(), args.save_path)
    print(f"Model saved to: {args.save_path}")


if __name__ == "__main__":
    main()

