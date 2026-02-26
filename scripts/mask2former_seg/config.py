"""
Configuration for Mask2Former hospital segmentation pipeline.
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    data_root: str = "data/hospital_coco"
    image_size: int = 512

@dataclass
class ModelConfig:
    # COCO pretrained Mask2Former checkpoint
    checkpoint: str = "facebook/mask2former-swin-large-coco-instance"
    # Output directory
    output_dir: str = "output/mask2former"

@dataclass
class TrainConfig:
    # Batch configuration
    batch_size: int = 2
    grad_accum_steps: int = 4   # Effective batch size = 8
    num_epochs: int = 40
    num_workers: int = 0        # Set to 0 to avoid multiprocessing issues

    # Optimizer
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    backbone_lr_factor: float = 0.1  # Lower LR for frozen backbone layers

    # LR schedule
    lr_warmup_steps: int = 100

    # Gradient clipping
    max_grad_norm: float = 0.01

    # Logging & checkpointing
    log_every: int = 10         # batches
    save_every_epochs: int = 5

    # Mixed precision
    fp16: bool = False          # bfloat16 on Ampere+ GPUs
    bf16: bool = True

@dataclass  
class AugConfig:
    # Training augmentations
    random_crop_size: int = 512
    min_scale: float = 0.8
    max_scale: float = 1.25
    flip_prob: float = 0.5
    color_jitter: bool = True
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.2
    hue: float = 0.1
