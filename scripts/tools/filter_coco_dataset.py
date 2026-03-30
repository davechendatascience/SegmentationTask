import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    classes=[
        "bowl", "laptop", "cell phone", "remote",
        "book", "backpack", "handbag", "suitcase", "toothbrush"
    ],
    label_types=["detections"],
    max_samples=500
)

dataset.export(
    export_dir="coco_filtered",
    dataset_type=fo.types.COCODetectionDataset
)