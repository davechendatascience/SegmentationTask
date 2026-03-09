import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

class SAM3Backbone(nn.Module):
    def __init__(self, model_name: str = "facebook/sam3", checkpoint_path: Optional[str] = None, config_path: Optional[str] = None, freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.encoder = self._load_encoder(model_name, checkpoint_path, config_path)
        
        # 如果需要凍結參數
        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def _load_encoder(self, model_name, checkpoint_path, config_path):
        """Load the SAM3 encoder using official SAM3 API."""
        try:
            # 優先使用本地 checkpoint
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Loading SAM3 encoder from local checkpoint: {checkpoint_path}")
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                model = self._build_sam3_model()
                model.load_state_dict(state_dict)
                encoder = model.image_encoder if hasattr(model, 'image_encoder') else model.encoder
                return encoder
            
            # 使用官方 SAM3 API 載入
            if model_name == "facebook/sam3":
                print(f"Loading SAM3 encoder using official API: {model_name}")
                from sam3.model_builder import build_sam3_image_model
                
                model = build_sam3_image_model()
                encoder = model.image_encoder if hasattr(model, 'image_encoder') else model.encoder
                return encoder
            
            # 回退到通用載入
            else:
                print(f"Loading generic model: {model_name}")
                from transformers import AutoModel
                model = AutoModel.from_pretrained(model_name)
                return model
            
        except Exception as e:
            print(f"Error loading SAM3 encoder: {e}")
            print("Trying fallback: simple Vision Transformer")
            
            try:
                import timm
                encoder = timm.create_model('vit_large_patch14_clip_336', pretrained=True)
                return encoder
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                raise
    
    def _build_sam3_model(self):
        """Build a basic SAM3 model structure for loading checkpoints."""
        try:
            from sam3.model_builder import build_sam3_image_model
            return build_sam3_image_model()
        except ImportError:
            print("SAM3 package not available. Using placeholder architecture...")
            import timm
            return timm.create_model('vit_large_patch14_clip_336', pretrained=False)
    
    def _get_input_size(self) -> int:
        """Get the expected input size for the encoder."""
        try:
            # 對於 timm 模型
            if hasattr(self.encoder, 'default_cfg'):
                input_size = self.encoder.default_cfg.get('input_size', [3, 224, 224])
                if isinstance(input_size, (list, tuple)) and len(input_size) >= 2:
                    return input_size[-1]  # 假設 H == W
            
            # 對於 transformers 模型
            if hasattr(self.encoder, 'config'):
                if hasattr(self.encoder.config, 'image_size'):
                    return self.encoder.config.image_size
            
            # 對於 SAM3/SAM2 模型
            if hasattr(self.encoder, 'image_size'):
                return self.encoder.image_size
            
            # 預設值
            return 224
            
        except Exception as e:
            print(f"Could not determine input size: {e}")
            return 224
    
    def get_feature_info(self, image_size: Optional[int] = None) -> Dict[str, tuple]:
        """Get feature information by running a dummy forward pass.
        
        Args:
            image_size: Size of the input image. If None, uses model's expected size.
            
        Returns:
            Dictionary with feature stage names as keys and (C, H, W) tuples as values.
        """
        # 如果沒有指定 image_size，使用模型期望的大小
        if image_size is None:
            image_size = self._get_input_size()
        
        # 創建虛擬輸入
        dummy_input = torch.randn(1, 3, image_size, image_size).to(next(self.parameters()).device)
        
        # 運行前向傳播
        with torch.no_grad():
            features = self.forward(dummy_input)
        
        # 提取特徵資訊
        feature_info = {}
        if isinstance(features, dict):
            for stage_name, feat in features.items():
                if isinstance(feat, torch.Tensor):
                    C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
                    feature_info[stage_name] = (C, H, W)
        elif isinstance(features, (list, tuple)):
            for i, feat in enumerate(features):
                if isinstance(feat, torch.Tensor):
                    C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
                    feature_info[f"stage{i}"] = (C, H, W)
        else:
            # 單一 tensor
            C, H, W = features.shape[1], features.shape[2], features.shape[3]
            feature_info["stage0"] = (C, H, W)
        
        return feature_info
    
    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from the encoder output.
        
        Args:
            x: Encoder output (dict, list, tuple, or tensor).
            
        Returns:
            Dictionary of feature tensors.
        """
        if isinstance(x, dict):
            # 如果已經是字典，直接返回
            return x
        elif isinstance(x, (list, tuple)):
            # 如果是列表/元組，轉換為字典
            return {f"stage{i}": feat for i, feat in enumerate(x)}
        else:
            # 單一 tensor
            return {"stage0": x}
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Dictionary of feature tensors.
        """
        if self.freeze:
            with torch.no_grad():
                encoder_output = self.encoder(x)
        else:
            encoder_output = self.encoder(x)
        
        # 提取特徵
        features = self._extract_features(encoder_output)
        
        # Detach 以避免圖形問題（如果凍結）
        if self.freeze:
            features = {k: v.detach() for k, v in features.items()}
        
        return features