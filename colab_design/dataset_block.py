import os
import torch
import numpy as np
import cv2
import glob
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import CLIPProcessor

# --- Configuration Block ---
# Adjust these based on your specific Colab environment and requirements
DATASET_NAME = "scene_parse_150" # HuggingFace dataset name for ADE20k
IMAGE_SIZE = 512  # Resize images to this dimension (Square)
CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
BATCH_SIZE = 4
NUM_WORKERS = 2

class ADE20kPanopticDataset(Dataset):
    """
    Dataset class for ADE20k Panoptic Segmentation.
    
    Note: The official HuggingFace 'scene_parse_150' dataset provides semantic and instance masks.
    For true 'panoptic' format, we typically combine these.
    However, for this simplified implementation, we will treat it as a collection of binary masks 
    and class labels, which is what Mask2Former expects.
    """
    def __init__(self, root_dir="./ADEChallengeData2016", split="train", transform=None):
        self.root_dir = root_dir
        self.split = "training" if split == "train" else "validation"
        self.transform = transform
        
        # Check if dataset exists, if not download
        if not os.path.exists(self.root_dir):
            self.download_ade20k()
            
        self.image_dir = os.path.join(self.root_dir, "images", self.split)
        self.mask_dir = os.path.join(self.root_dir, "annotations", self.split)
        
        self.images = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        self.masks = sorted(glob.glob(os.path.join(self.mask_dir, "*.png")))
        
        print(f"Found {len(self.images)} images in {self.image_dir}")

    def download_ade20k(self):
        print("Downloading ADE20k dataset (this may take a while)...")
        # Direct link to ADE20k
        url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
        zip_path = "ADEChallengeData2016.zip"
        
        if not os.path.exists(zip_path):
            os.system(f"wget {url} -O {zip_path}")
            
        print("Unzipping...")
        os.system(f"unzip -q {zip_path}")
        print("Download complete.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Annotation in ADE20k zip: Int masks
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Convert back to numpy if albumentations converted to tensor
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        # Mask2Former expects:
        # - pixel_values: (C, H, W) -> Normalized image
        # - pixel_mask: (H, W) -> Padding mask (optional)
        # - mask_labels: list of binary masks (N, H, W)
        # - class_labels: list of class ids (N)
        
        # Process Mask into Binary Masks + Labels
        unique_ids = np.unique(mask)
        # Remove background/ignore index if present (usually 0 or 255)
        unique_ids = unique_ids[unique_ids != 0] 
        
        masks = []
        labels = []
        
        for uid in unique_ids:
            # Create binary mask for this instance/class
            binary_mask = (mask == uid).astype(np.float32)
            masks.append(binary_mask)
            labels.append(uid - 1) # ADE20k IDs are 1-150. We need 0-149 for model.
            
        if len(masks) > 0:
            masks = torch.tensor(np.stack(masks), dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            # Handle standard case with no objects (rare in ADE20k)
            masks = torch.zeros((0, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
            labels = torch.tensor([], dtype=torch.long)

        # Normalize image for CLIP
        # CLIP Expects:
        # mean = [0.48145466, 0.4578275, 0.40821073]
        # std  = [0.26862954, 0.26130258, 0.27577711]
        # Validated against CLIPProcessor defaults
        
        return {
            "pixel_values": image, 
            "masks": masks, 
            "class_labels": labels,
            "original_size": (image.shape[1], image.shape[2]) # H, W after transform (or keep original before)
        }

def get_transforms(image_size=512):
    # CLIP Normalization constants
    mean = (0.48145466, 0.4578275, 0.40821073)
    std  = (0.26862954, 0.26130258, 0.27577711)
    
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

def collafe_fn(batch):
    # Custom collate because masks have variable channel (N instances)
    pixel_values = torch.stack([x['pixel_values'] for x in batch])
    
    targets = []
    for x in batch:
        targets.append({
            "masks": x['masks'],
            "class_labels": x['class_labels']
        })
        
    return pixel_values, targets

# --- Usage Example ---
# dataset = ADE20kPanopticDataset(split="train", transform=get_transforms(IMAGE_SIZE))
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collafe_fn)
