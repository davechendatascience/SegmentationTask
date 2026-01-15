import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from core_design_2.augmentation_block import get_train_transforms, get_eval_transforms

IMAGE_SIZE = 640


class ADE20kPanopticDataset(Dataset):
    def __init__(self, root_dir="./ADEChallengeData2016", split="train", transform=None):
        self.root_dir = root_dir
        self.split = "training" if split == "train" else "validation"
        self.transform = transform

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Missing ADE20k root at {self.root_dir}")

        self.image_dir = os.path.join(self.root_dir, "images", self.split)
        self.mask_dir = os.path.join(self.root_dir, "annotations", self.split)
        self.images = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        self.masks = sorted(glob.glob(os.path.join(self.mask_dir, "*.png")))
        if not self.images or not self.masks:
            raise FileNotFoundError(f"Missing ADE20k images/masks under {self.root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int32)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids != 0]

        masks = []
        labels = []
        for uid in unique_ids:
            binary_mask = (mask == uid).astype(np.float32)
            masks.append(binary_mask)
            labels.append(uid - 1)

        if len(masks) > 0:
            mask_labels = torch.tensor(np.stack(masks), dtype=torch.float32)
            class_labels = torch.tensor(labels, dtype=torch.int64)
        else:
            mask_labels = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.float32)
            class_labels = torch.tensor([], dtype=torch.int64)

        pixel_mask = torch.ones((image.shape[1], image.shape[2]), dtype=torch.uint8)

        return {
            "pixel_values": image,
            "pixel_mask": pixel_mask,
            "mask_labels": mask_labels,
            "class_labels": class_labels,
        }


def get_transforms(image_size=IMAGE_SIZE, train=True):
    if train:
        return get_train_transforms(image_size)
    return get_eval_transforms(image_size)


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    pixel_mask = torch.stack([b["pixel_mask"] for b in batch], dim=0)
    mask_labels = [b["mask_labels"] for b in batch]
    class_labels = [b["class_labels"] for b in batch]
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }

