import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=640, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(image_size=640, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

