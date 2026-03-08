"""
Train YOLOv11 segmentation model using Ultralytics with Roboflow dataset.

Usage:
    python -m scripts.yolov11_seg.train
"""
import argparse
import os
from pathlib import Path

from ultralytics import YOLO

from .config import DataConfig, ModelConfig, TrainConfig
from .dataset import download_roboflow_dataset, setup_data_yaml


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on Roboflow dataset")
    parser.add_argument("--download-data", action="store_true",
                       help="Download dataset from Roboflow")
    parser.add_argument("--data-yaml", type=str, default=None,
                       help="Path to data.yaml (if not downloading)")
    parser.add_argument("--model", type=str, default=None,
                       help="Model name/size (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (overrides config)")
    args = parser.parse_args()

    # Load configurations
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    # Override with command line args
    if args.model:
        model_cfg.model_name = args.model
    if args.epochs:
        model_cfg.epochs = args.epochs
    if args.batch_size:
        model_cfg.batch_size = args.batch_size

    # Handle dataset
    if args.download_data:
        print("Downloading dataset from Roboflow...")
        data_dir = download_roboflow_dataset(
            workspace=data_cfg.roboflow_workspace,
            project=data_cfg.roboflow_project,
            version=data_cfg.roboflow_version,
            api_key=data_cfg.roboflow_api_key
        )
        data_yaml = setup_data_yaml(data_dir)
    else:
        if args.data_yaml:
            data_yaml = args.data_yaml
        else:
            data_yaml = data_cfg.data_yaml_path

        if not Path(data_yaml).exists():
            raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    print(f"Using data.yaml: {data_yaml}")

    # Load model
    if model_cfg.pretrained:
        model = YOLO(model_cfg.model_name)
    else:
        model = YOLO(model_cfg.model_name).load()  # Load from scratch

    # Training configuration
    train_args = {
        "data": str(data_yaml),
        "epochs": model_cfg.epochs,
        "batch": model_cfg.batch_size,
        "patience": model_cfg.patience,
        "device": train_cfg.device,
        "lr0": train_cfg.lr0,
        "lrf": train_cfg.lrf,
        "momentum": train_cfg.momentum,
        "weight_decay": train_cfg.weight_decay,
        "hsv_h": train_cfg.hsv_h,
        "hsv_s": train_cfg.hsv_s,
        "hsv_v": train_cfg.hsv_v,
        "degrees": train_cfg.degrees,
        "translate": train_cfg.translate,
        "scale": train_cfg.scale,
        "shear": train_cfg.shear,
        "perspective": train_cfg.perspective,
        "flipud": train_cfg.flipud,
        "fliplr": train_cfg.fliplr,
        "mosaic": train_cfg.mosaic,
        "mixup": train_cfg.mixup,
        "save_period": train_cfg.save_period,
        "project": train_cfg.project,
        "name": train_cfg.name,
        "val": train_cfg.val,
        "split": train_cfg.split,
    }

    # Train the model
    print("Starting training...")
    results = model.train(**train_args)

    print(f"Training completed. Results saved to {results.save_dir}")

    # Save best model path
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()