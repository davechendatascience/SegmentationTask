import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPConfig
from peft import LoraConfig, get_peft_model

class CLIPBackbone(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16", use_lora=True, lora_rank=16):
        super().__init__()
        # Load standard CLIP Vision Model
        # SPEED OPTIMIZATION: Default output_hidden_states=False is much faster
        self.base_model = CLIPVisionModel.from_pretrained(model_name)
        
        if use_lora:
            print(f"Injecting LoRA adapters with rank={lora_rank}...")
            peft_config = LoraConfig(
                r=lora_rank, 
                lora_alpha=lora_rank*2, 
                target_modules=["q_proj", "v_proj"], 
                lora_dropout=0.1, 
                bias="none",
                modules_to_save=[], 
            )
            self.base_model = get_peft_model(self.base_model, peft_config)
            self.base_model.print_trainable_parameters()

    def forward(self, x):
        # x: [B, 3, H, W]
        outputs = self.base_model(pixel_values=x, interpolate_pos_encoding=True)
        last_hidden = outputs.last_hidden_state
        patch_tokens = last_hidden[:, 1:, :] 
        B, L, D = patch_tokens.shape
        H = W = int(L**0.5) 
        
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, D, H, W)
                    
        return feature_map

class SimpleFPN(nn.Module):
    """
    ViTDet-style Simple Feature Pyramid.
    Builds a pyramid from a single high-level feature map.
    """
    def __init__(self, in_channels=768, hidden_dim=256):
        super().__init__()
        
        self.simfpn0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, hidden_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2),
        )
        self.simfpn1 = nn.Sequential(
             nn.ConvTranspose2d(in_channels, hidden_dim, kernel_size=2, stride=2),
        )
        self.simfpn2 = nn.Sequential(
            nn.Identity(), 
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        )
        self.simfpn3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.simfpn3_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, 768, H/16, W/16]
        p2 = self.simfpn0(x) # 1/4
        p3 = self.simfpn1(x) # 1/8
        p4 = self.simfpn2(x) # 1/16
        p5 = self.simfpn3(x) # 1/32
        p5 = self.simfpn3_proj(p5)
        
        return [p2, p3, p4, p5] 

class LightMask2Former(nn.Module):
    def __init__(self, in_channels=256, num_queries=100, num_classes=150, hidden_dim=256):
        super().__init__()
        
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=1024)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # P3 and P4 projection layers? 
        # Actually we assume FPN outputs are all 'hidden_dim' channels, so we can reuse logic.
        
    def forward_mask_prediction(self, mask_embed, feature_map):
        # mask_embed: [B, Q, C]
        # feature_map: [B, C, H, W]
        B, C, H, W = feature_map.shape
        pixel_embed_flat = feature_map.flatten(2) # [B, C, HW]
        pred_masks = torch.bmm(mask_embed, pixel_embed_flat)
        pred_masks = pred_masks.reshape(B, self.num_queries, H, W)
        return pred_masks

    def forward(self, features):
        # features: [P2, P3, P4, P5]
        # Main Scale: P2
        pixel_embed = features[0] 
        B, C, H, W = pixel_embed.shape
        
        # Flatten [H*W, B, C]
        pixel_embed_flat = pixel_embed.flatten(2).permute(2, 0, 1)
        
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1) # [Q, B, C]
        
        # Decode
        out_queries = self.transformer_decoder(tgt=queries, memory=pixel_embed_flat)
        out_queries = out_queries.permute(1, 0, 2) # [B, Q, C]
        
        pred_logits = self.class_head(out_queries)
        mask_embed = self.mask_head(out_queries) # [B, Q, C]
        
        # Main Prediction (P2)
        pred_masks = self.forward_mask_prediction(mask_embed, features[0])
        
        # Auxiliary Predictions (P3, P4) for Consistency Loss
        # We use the SAME mask embeddings, just projected onto coarser feature maps.
        # This forces the feature maps to be consistent.
        pred_masks_p3 = self.forward_mask_prediction(mask_embed, features[1])
        pred_masks_p4 = self.forward_mask_prediction(mask_embed, features[2])
        
        return {
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
            "aux_outputs": [
                {"pred_masks": pred_masks_p3}, # Scale 1/8
                {"pred_masks": pred_masks_p4}  # Scale 1/16
            ]
        }

class CLIPPanopticModel(nn.Module):
    def __init__(self, num_classes=150, lora_rank=64):
        super().__init__()
        self.backbone = CLIPBackbone(lora_rank=lora_rank)
        self.pixel_decoder = SimpleFPN(in_channels=768, hidden_dim=256)
        self.decoder = LightMask2Former(in_channels=256, hidden_dim=256, num_classes=num_classes)
        
    def forward(self, x):
        backbone_feature = self.backbone(x)
        fpn_features = self.pixel_decoder(backbone_feature)
        outputs = self.decoder(fpn_features)
        return outputs
