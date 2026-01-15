import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core_design_2.dataset_block import ADE20kPanopticDataset, collate_fn, get_transforms, IMAGE_SIZE
from core_design_2.model_block import build_model, build_processor
from core_design_2.eval_block import evaluate_model, visualize_prediction


def _build_label_maps(supervise_background: bool):
    num_labels = 150 + (1 if supervise_background else 0)
    if supervise_background:
        id2label = {0: "background"}
        for i in range(150):
            id2label[i + 1] = f"class_{i}"
        label2id = {v: k for k, v in id2label.items()}
    else:
        id2label = {i: f"class_{i}" for i in range(150)}
        label2id = {v: k for k, v in id2label.items()}
    return num_labels, id2label, label2id


def main(cli_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="ADEChallengeData2016")
    ap.add_argument("--model-id", type=str, default="facebook/mask2former-swin-base-coco-panoptic")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--output-dir", type=str, default="runs/mask2former_ade20k")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--supervise-background", action="store_true")
    ap.add_argument("--freeze-mode", type=str, default="class_and_mask_decoder", choices=["none", "class_only", "class_and_mask_decoder"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--visualize", action="store_true")
    args = ap.parse_args(cli_args)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = build_processor(args.model_id, local_files_only=args.local_files_only)
    num_labels, id2label, label2id = _build_label_maps(args.supervise_background)
    base_model = build_model(
        args.model_id,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        local_files_only=args.local_files_only,
    ).to(device)
    model = base_model
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    model.train()

    if args.freeze_mode != "none":
        for p in base_model.parameters():
            p.requires_grad = False
        for p in base_model.class_predictor.parameters():
            p.requires_grad = True
        if args.freeze_mode == "class_and_mask_decoder":
            for p in base_model.model.transformer_module.decoder.parameters():
                p.requires_grad = True
            for p in base_model.model.transformer_module.decoder.mask_predictor.parameters():
                p.requires_grad = True

    if args.download_only:
        processor.save_pretrained(out_dir / "processor_cache_check")
        base_model.save_pretrained(out_dir / "model_cache_check")
        print(f"Downloaded/loaded model+processor: {args.model_id}")
        return 0

    rescale_factor = getattr(processor, "rescale_factor", None) if getattr(processor, "do_rescale", True) else None
    train_ds = ADE20kPanopticDataset(
        args.data_dir,
        "train",
        transform=get_transforms(args.size, train=True),
    )
    val_ds = ADE20kPanopticDataset(
        args.data_dir,
        "validation",
        transform=get_transforms(args.size, train=False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn,
        pin_memory=device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_fn,
        pin_memory=device.startswith("cuda"),
    )

    trainable_params = [p for p in base_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise SystemExit("No trainable parameters (check --freeze-mode).")
    opt = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.fp16 and device.startswith("cuda")))

    def run_epoch(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()
        total = 0.0
        steps = 0
        it = tqdm(loader, desc=("train" if train else "val"), leave=False)
        for batch in it:
            if args.max_steps is not None and steps >= int(args.max_steps):
                break
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            mask_labels = [m.to(device) for m in batch["mask_labels"]]
            class_labels = [c.to(device) for c in batch["class_labels"]]

            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=bool(args.fp16 and device.startswith("cuda"))):
                    out = model(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        mask_labels=mask_labels,
                        class_labels=class_labels,
                    )
                    loss = out.loss
                if train:
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()

            lv = float(loss.detach().cpu())
            total += lv
            steps += 1
            it.set_postfix(loss=f"{lv:.4f}")
        return total / max(1, steps)

    best_val = float("inf")
    for epoch in range(int(args.epochs)):
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            base_model.save_pretrained(out_dir / "best")
            processor.save_pretrained(out_dir / "best")

        if args.visualize:
            visualize_prediction(model, val_ds, 0, device)

    base_model.save_pretrained(out_dir / "last")
    processor.save_pretrained(out_dir / "last")
    print(f"Saved checkpoints under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

