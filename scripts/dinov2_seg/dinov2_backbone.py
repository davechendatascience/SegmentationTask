"""
DINOv2 backbone wrapper for dense prediction.

Extracts features from 4 intermediate ViT layers and creates synthetic
multi-scale features via conv projections (single-scale → 3 scales).

DINOv2 is self-supervised on 142M images and proven for dense tasks
(depth estimation, segmentation, matching).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class DINOv2Backbone(nn.Module):
    """
    Frozen DINOv2 backbone with multi-scale feature projection.

    DINOv2 ViT outputs single-scale features from intermediate layers.
    We project + downsample to create 3 scales for the pixel decoder:
        stage0 (deep):    [B, 256, H/4, W/4]  — smallest spatial
        stage1 (mid):     [B, 256, H/2, W/2]  — medium
        stage2 (shallow): [B, 256, H, W]       — largest spatial (native)

    where H, W = input_size // patch_size (e.g., 518//14 = 37).
    """

    MODELS = {
        "dinov2_vits14_reg": {"dim": 384, "layers": 12},
        "dinov2_vitb14_reg": {"dim": 768, "layers": 12},
        "dinov2_vitl14_reg": {"dim": 1024, "layers": 24},
        "dinov2_vitg14_reg": {"dim": 1536, "layers": 40},
        # Non-register variants
        "dinov2_vits14": {"dim": 384, "layers": 12},
        "dinov2_vitb14": {"dim": 768, "layers": 12},
        "dinov2_vitl14": {"dim": 1024, "layers": 24},
        "dinov2_vitg14": {"dim": 1536, "layers": 40},
    }

    def __init__(
        self,
        model_name: str = "dinov2_vitl14_reg",
        d_model: int = 256,
        n_layers: int = 4,
    ):
        """
        Args:
            model_name: DINOv2 variant from torch.hub
            d_model: output channel dimension for all scales
            n_layers: number of intermediate layers to extract
        """
        super().__init__()
        self.model_name = model_name
        self.d_model = d_model
        self.n_layers = n_layers

        info = self.MODELS[model_name]
        self.feat_dim = info["dim"]

        # Load frozen DINOv2
        print(f"  Loading DINOv2: {model_name} (dim={self.feat_dim})")
        self.vit = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True
        )
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False

        # Fuse n_layers intermediate features → single feature map
        self.feat_proj = nn.Sequential(
            nn.Conv2d(self.feat_dim * n_layers, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # Create synthetic multi-scale features from the fused features:
        # stage2 (shallow): native resolution (e.g., 37×37)
        # stage1 (mid):     stride-2 downsample (e.g., 19×19)
        # stage0 (deep):    stride-2 again (e.g., 10×10)
        self.down1 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

    @torch.no_grad()
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract and reshape intermediate DINOv2 features."""
        features = self.vit.get_intermediate_layers(
            images, n=self.n_layers, reshape=True
        )
        # features: list of [B, C, H, W] tensors (all same resolution)
        return torch.cat(features, dim=1)  # [B, C*n_layers, H, W]

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W] ImageNet-normalized

        Returns:
            dict with "stage0", "stage1", "stage2" feature maps at 3 scales
        """
        # Extract frozen DINOv2 features
        raw = self._extract_features(images)  # [B, C*n, H, W]

        # Project to d_model channels (trainable)
        feat = self.feat_proj(raw)  # [B, 256, H, W]  (stage2 = shallowest)

        # Create multi-scale via downsampling
        feat_mid = self.down1(feat)   # [B, 256, H/2, W/2]  (stage1)
        feat_deep = self.down2(feat_mid)  # [B, 256, H/4, W/4]  (stage0)

        return {
            "stage0": feat_deep,   # deepest, smallest spatial
            "stage1": feat_mid,    # middle
            "stage2": feat,        # shallowest, largest spatial
        }

    def get_stage_channels(self) -> List[Tuple[str, int, int, int]]:
        """Return stage info for PixelDecoder: (name, channels, H, W)."""
        # Approximate spatial dims (will be exact at runtime)
        # For 518px input with patch_size=14: 518//14 = 37
        return [
            ("stage0", self.d_model, 10, 10),   # ~H/4
            ("stage1", self.d_model, 19, 19),   # ~H/2
            ("stage2", self.d_model, 37, 37),   # native
        ]


def build_dinov2_backbone(
    model_name: str = "dinov2_vitl14_reg",
    d_model: int = 256,
) -> DINOv2Backbone:
    """Build a frozen DINOv2 backbone with multi-scale projections."""
    backbone = DINOv2Backbone(model_name=model_name, d_model=d_model)

    # Count params
    total = sum(p.numel() for p in backbone.parameters())
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"  DINOv2 backbone: {frozen:,} frozen + {trainable:,} trainable params")

    return backbone
