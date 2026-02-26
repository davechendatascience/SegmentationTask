"""
COCO Dataset for Mask2Former instance segmentation.

Converts COCO polygon annotations to the format expected by
Mask2FormerImageProcessor:
  - masks: list of binary uint8 masks (H, W)
  - class_labels: list of int category ids (zero-indexed)
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def load_coco(ann_path: str):
    with open(ann_path) as f:
        return json.load(f)


def polygons_to_binary_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Convert COCO polygon(s) or RLE to a binary mask (H, W) uint8."""
    if isinstance(segmentation, list):
        # Polygon format
        rles = coco_mask.frPyObjects(segmentation, height, width)
        rle = coco_mask.merge(rles)
    elif isinstance(segmentation, dict):
        # RLE format
        rle = segmentation
    else:
        return np.zeros((height, width), dtype=np.uint8)

    binary = coco_mask.decode(rle).astype(np.uint8)
    return binary


def build_id_remap(categories: List[dict]) -> Dict[int, int]:
    """
    Create mapping from COCO category_id → 0-indexed class id.
    Returns (remap_dict, num_classes).
    """
    sorted_cats = sorted(categories, key=lambda c: c["id"])
    return {cat["id"]: idx for idx, cat in enumerate(sorted_cats)}


# --------------------------------------------------------------------------- #
# Augmentations                                                                #
# --------------------------------------------------------------------------- #

def get_train_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size + 64),
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.8
            ),
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
        ]
    )


def get_val_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=image_size),
            A.CenterCrop(height=image_size, width=image_size),
        ]
    )


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

class HospitalCOCODataset(Dataset):
    """
    Loads a COCO-format segmentation dataset and returns:
      {
        "pixel_values": PIL.Image (RGB),
        "masks":        List[np.ndarray] shape (H, W), binary uint8
        "class_labels": List[int]  (0-indexed)
      }
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 512,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size

        # Find annotation file
        ann_path = self.data_dir / split / "_annotations.coco.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation not found: {ann_path}")

        coco = load_coco(str(ann_path))
        self.images = {img["id"]: img for img in coco["images"]}
        self.categories = coco["categories"]
        self.id_remap = build_id_remap(self.categories)
        self.num_classes = len(self.categories)

        # Group annotations by image_id
        self.ann_by_image: Dict[int, List[dict]] = {}
        for ann in coco.get("annotations", []):
            self.ann_by_image.setdefault(ann["image_id"], []).append(ann)

        # Only keep images that have at least one annotation
        self.image_ids = [
            img_id
            for img_id in self.images
            if img_id in self.ann_by_image
        ]

        self.transforms = (
            get_train_transforms(image_size) if augment else get_val_transforms(image_size)
        )

        print(
            f"[{split}] {len(self.image_ids)} images, "
            f"{len(self.categories)} classes: {[c['name'] for c in self.categories]}"
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        anns = self.ann_by_image[img_id]

        # Load image
        img_path = self.data_dir / self.split / img_info["file_name"]
        image = np.array(Image.open(str(img_path)).convert("RGB"))
        h, w = image.shape[:2]

        # Build masks and labels
        masks = []
        class_labels = []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            seg = ann.get("segmentation", [])
            if not seg:
                continue
            mask = polygons_to_binary_mask(seg, h, w)  # (H, W)
            if mask.sum() == 0:
                continue
            masks.append(mask)
            class_labels.append(self.id_remap[ann["category_id"]])

        if not masks:
            # Fallback: blank mask with first class (rare edge case)
            masks = [np.zeros((h, w), dtype=np.uint8)]
            class_labels = [0]

        # Apply augmentations via albumentations (supports multiple masks)
        transformed = self.transforms(
            image=image,
            masks=masks,
        )
        image_aug = transformed["image"]    # (H, W, 3) uint8 numpy
        masks_aug = transformed["masks"]    # list of (H, W) uint8 numpy

        # Filter out masks that became empty after augmentation
        final_masks, final_labels = [], []
        for m, l in zip(masks_aug, class_labels):
            if m.sum() > 0:
                final_masks.append(m)
                final_labels.append(l)

        if not final_masks:
            final_masks = [np.zeros(image_aug.shape[:2], dtype=np.uint8)]
            final_labels = [0]

        return {
            "pixel_values": Image.fromarray(image_aug),  # PIL for processor
            "masks": final_masks,                         # list of np (H, W)
            "class_labels": final_labels,                 # list of int
            "image_id": img_id,
        }
