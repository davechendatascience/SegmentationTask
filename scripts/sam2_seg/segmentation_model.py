"""
Top-level segmentation model combining SAM2 backbone + adapters + UNet decoder.

Usage:
    model = SAM2SegModel.build(model_name="facebook/sam2.1-hiera-large", num_classes=27)
    logits = model(images)  # [B, num_classes, H, W]
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .sam2_backbone import SAM2Backbone
from .adapters import MultiStageAdapters
from .unet_decoder import UNetDecoder


class SAM2SegModel(nn.Module):
    """
    SAM2 backbone + optional adapters + UNet decoder for segmentation.

    The backbone is frozen. Only adapters and decoder are trained.
    """

    def __init__(
        self,
        backbone: SAM2Backbone,
        decoder: UNetDecoder,
        adapters: Optional[MultiStageAdapters] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.adapters = adapters
        self.decoder = decoder

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W] ImageNet-normalised float32

        Returns:
            logits: [B, num_classes, H, W]
        """
        features = self.backbone(images)

        if self.adapters is not None:
            features = self.adapters(features)

        logits = self.decoder(features)
        return logits

    def get_param_groups(self, adapter_lr: float, decoder_lr: float) -> list:
        """
        Build optimizer parameter groups with separate learning rates.

        Args:
            adapter_lr: learning rate for adapter parameters
            decoder_lr: learning rate for decoder parameters

        Returns:
            List of param group dicts for the optimizer
        """
        groups = []

        if self.adapters is not None:
            adapter_params = list(self.adapters.parameters())
            if adapter_params:
                groups.append({"params": adapter_params, "lr": adapter_lr})

        decoder_params = list(self.decoder.parameters())
        if decoder_params:
            groups.append({"params": decoder_params, "lr": decoder_lr})

        return groups

    @classmethod
    def build(
        cls,
        model_name: str = "facebook/sam2.1-hiera-large",
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        num_classes: int = 27,
        use_adapters: bool = True,
        adapter_dim: int = 64,
        image_size: int = 1024,
    ) -> "SAM2SegModel":
        """
        Factory method: build the full segmentation model.

        1. Create backbone and run dummy forward to discover feature shapes
        2. Create adapters (if enabled)
        3. Create decoder wired to the backbone's feature shapes
        """
        print(f"Building SAM2 segmentation model...")
        print(f"  Model: {model_name}")
        print(f"  Classes: {num_classes}")
        print(f"  Adapters: {use_adapters} (dim={adapter_dim})")

        # 1) Backbone
        backbone = SAM2Backbone(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            freeze=True,
        )

        # Discover feature shapes via dummy forward
        print(f"  Probing backbone feature shapes...")
        feat_info = backbone.get_feature_info(image_size)
        print(f"  Feature stages:")
        for name, (c, h, w) in feat_info.items():
            print(f"    {name}: C={c}, H={h}, W={w}")

        # 2) Adapters
        adapters = None
        if use_adapters:
            stage_channels = {name: shape[0] for name, shape in feat_info.items()}
            adapters = MultiStageAdapters(stage_channels, adapter_dim)
            n_adapter_params = sum(p.numel() for p in adapters.parameters())
            print(f"  Adapter params: {n_adapter_params:,}")

        # 3) Decoder
        stage_specs = [
            (name, shape[0], shape[1], shape[2])
            for name, shape in feat_info.items()
        ]
        decoder = UNetDecoder(
            stage_channels=stage_specs,
            num_classes=num_classes,
            input_size=image_size,
        )
        n_decoder_params = sum(p.numel() for p in decoder.parameters())
        print(f"  Decoder params: {n_decoder_params:,}")

        model = cls(backbone=backbone, decoder=decoder, adapters=adapters)

        # Summary
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {total_all:,} ({total_trainable:,} trainable)")

        return model
