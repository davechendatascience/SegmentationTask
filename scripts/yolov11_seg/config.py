"""
Configuration for YOLOv11 instance segmentation training with Roboflow dataset.

Uses Ultralytics YOLOv11 for training on COCO-format segmentation data.
"""
import torch
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    # Roboflow dataset configuration
    roboflow_workspace: str = "your-workspace"
    roboflow_project: str = "your-project"
    roboflow_version: int = 1
    roboflow_api_key: str = None  # Set via environment variable or config

    # Local dataset paths (after download)
    data_yaml_path: str = "data/roboflow_hospital/data.yaml"
    image_size: int = 640  # YOLOv11 default


@dataclass
class ModelConfig:
    # YOLOv11 model
    model_name: str = "yolo11n-seg.pt"  # nano version for segmentation
    pretrained: bool = True

    # Training parameters
    epochs: int = 100
    batch_size: int = 16
    patience: int = 50  # early stopping

    # Output directory
    output_dir: str = "output/yolov11"


@dataclass
class TrainConfig:
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # Augmentation
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0

    # Logging
    save_period: int = 10
    project: str = "yolov11_seg"
    name: str = "exp"

    # Validation
    val: bool = True
    split: str = "val"