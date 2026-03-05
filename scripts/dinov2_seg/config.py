"""Configuration for DINOv2 + Query Decoder instance segmentation."""
from dataclasses import dataclass


@dataclass
class DataConfig:
    data_root: str = "data/hospital_coco"
    image_size: int = 518  # DINOv2 native: 518 = 37 * 14 patches


@dataclass
class ModelConfig:
    # DINOv2 backbone
    dinov2_model: str = "dinov2_vitl14_reg"
    d_model: int = 256

    # Query decoder
    decoder_type: str = "query"
    num_queries: int = 30
    decoder_layers: int = 3
    nhead: int = 8
    dim_ff: int = 1024

    # Output
    output_dir: str = "output/dinov2"
    num_classes: int = 27  # set at runtime


@dataclass
class TrainConfig:
    num_epochs: int = 40
    batch_size: int = 4  # smaller images → larger batch
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    num_workers: int = 2
    lr_warmup_steps: int = 500
    save_every_epochs: int = 5

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # Loss weights
    cls_weight: float = 2.0
    mask_bce_weight: float = 5.0
    mask_dice_weight: float = 5.0
    no_object_weight: float = 0.1

    # Backbone LR (for projection layers only)
    backbone_proj_lr_factor: float = 0.1


@dataclass
class AugConfig:
    pass
