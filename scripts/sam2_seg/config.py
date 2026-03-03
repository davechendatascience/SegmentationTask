"""
Configuration for SAM2 hospital segmentation pipeline.
Mirrors mask2former_seg/config.py structure.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    data_root: str = "data/hospital_coco"
    image_size: int = 1024  # SAM2 native resolution


@dataclass
class ModelConfig:
    # SAM2 model identifier (HuggingFace or local path)
    sam2_model_name: str = "facebook/sam2.1-hiera-large"
    # Alternatively, use local checkpoint + config
    sam2_checkpoint: Optional[str] = None
    sam2_config: Optional[str] = None

    # Adapter settings
    use_adapters: bool = True
    adapter_dim: int = 64

    # Number of classes (set automatically from dataset)
    num_classes: int = 26  # hospital_coco default (excluding background)

    # Output directory
    output_dir: str = "output/sam2"


@dataclass
class TrainConfig:
    # Batch configuration
    batch_size: int = 2
    grad_accum_steps: int = 4       # Effective batch size = 8
    num_epochs: int = 40
    num_workers: int = 0            # 0 to avoid multiprocessing issues

    # Optimizer
    learning_rate: float = 1e-4     # Decoder LR
    adapter_lr_factor: float = 0.1  # Adapter LR = learning_rate * adapter_lr_factor
    weight_decay: float = 0.01

    # LR schedule
    lr_warmup_steps: int = 100

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Loss weights
    ce_weight: float = 1.0
    dice_weight: float = 1.0

    # Logging & checkpointing
    log_every: int = 10             # batches
    save_every_epochs: int = 5

    # Mixed precision
    fp16: bool = False
    bf16: bool = True


@dataclass
class AugConfig:
    # Training augmentations
    random_crop_size: int = 1024
    min_scale: float = 0.8
    max_scale: float = 1.25
    flip_prob: float = 0.5
    color_jitter: bool = True
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2
    hue: float = 0.1
