import argparse
import importlib
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import Mask2FormerForUniversalSegmentation


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate validation metrics for a core design.")
    parser.add_argument("--core", type=str, default="core_design", choices=["core_design", "core_design_2"])
    parser.add_argument("--data-dir", type=str, default="ADEChallengeData2016")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default="runs/core_design_ade20k/best/clip_panoptic_lora.pth")
    parser.add_argument("--run-dir", type=str, default="runs/mask2former_ade20k/best")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def eval_core_design(args):
    dataset_block = importlib.import_module("core_design.dataset_block")
    model_block = importlib.import_module("core_design.model_block")
    eval_block = importlib.import_module("core_design.eval_block")

    dataset = dataset_block.ADE20kPanopticDataset(
        root_dir=args.data_dir,
        split="validation",
        transform=dataset_block.get_transforms(dataset_block.IMAGE_SIZE, train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset_block.collafe_fn,
        pin_memory=args.device.startswith("cuda"),
    )

    model = model_block.CLIPPanopticModel(num_classes=150).to(args.device)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state, strict=False)

    metrics = eval_block.evaluate_model(model, loader, args.device)
    print("Validation Metrics:")
    print(metrics)


def eval_core_design_2(args):
    dataset_block = importlib.import_module("core_design_2.dataset_block")
    eval_block = importlib.import_module("core_design_2.eval_block")

    dataset = dataset_block.ADE20kPanopticDataset(
        args.data_dir,
        split="validation",
        transform=dataset_block.get_transforms(dataset_block.IMAGE_SIZE, train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset_block.collate_fn,
        pin_memory=args.device.startswith("cuda"),
    )

    if not os.path.exists(args.run_dir):
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.run_dir,
        local_files_only=not args.allow_download,
    ).to(args.device)

    metrics = eval_block.evaluate_model(model, loader, args.device)
    print("Validation Metrics:")
    print(metrics)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    if args.core == "core_design":
        eval_core_design(args)
    else:
        eval_core_design_2(args)


if __name__ == "__main__":
    main()

