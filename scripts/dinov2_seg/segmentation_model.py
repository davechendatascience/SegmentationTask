"""
DINOv2 + Query Decoder segmentation model.

Architecture:
    DINOv2 (frozen) → multi-scale projections (trainable) → QueryInstanceDecoder
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .dinov2_backbone import build_dinov2_backbone
from .query_decoder import QueryInstanceDecoder


class DINOv2SegModel(nn.Module):
    """DINOv2 backbone + query-based instance segmentation decoder."""

    def __init__(self, backbone, decoder):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(images)
        return self.decoder(features)

    def get_param_groups(self, proj_lr: float, decoder_lr: float):
        """Separate param groups for backbone projections vs decoder."""
        proj_params = []
        decoder_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("backbone."):
                proj_params.append(param)
            else:
                decoder_params.append(param)

        groups = []
        if proj_params:
            groups.append({"params": proj_params, "lr": proj_lr})
        if decoder_params:
            groups.append({"params": decoder_params, "lr": decoder_lr})

        return groups

    @staticmethod
    def build(
        model_name: str = "dinov2_vitl14_reg",
        d_model: int = 256,
        num_classes: int = 27,
        num_queries: int = 30,
        decoder_layers: int = 3,
        nhead: int = 8,
        dim_ff: int = 1024,
        image_size: int = 518,
        **kwargs,
    ) -> "DINOv2SegModel":
        """Build a DINOv2 segmentation model."""
        print("Building DINOv2 segmentation model...")
        print(f"  Model: {model_name}")
        print(f"  Classes: {num_classes}, Queries: {num_queries}")

        backbone = build_dinov2_backbone(model_name=model_name, d_model=d_model)

        # Probe actual feature shapes
        print("  Probing backbone feature shapes...")
        backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, image_size, image_size)
            feats = backbone(dummy)
            stage_channels = []
            for name in sorted(feats.keys()):
                f = feats[name]
                print(f"    {name}: C={f.shape[1]}, H={f.shape[2]}, W={f.shape[3]}")
                stage_channels.append((name, f.shape[1], f.shape[2], f.shape[3]))

        # Build query decoder
        decoder = QueryInstanceDecoder(
            stage_channels=stage_channels,
            num_classes=num_classes,
            d_model=d_model,
            num_queries=num_queries,
            num_layers=decoder_layers,
            nhead=nhead,
            dim_ff=dim_ff,
        )

        model = DINOv2SegModel(backbone, decoder)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
        print(f"  Total params: {total:,} ({trainable:,} trainable)")

        return model
