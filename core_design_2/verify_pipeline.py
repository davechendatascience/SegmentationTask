import torch
from core_design_2.model_block import build_model, build_processor


def test_pipeline():
    model_id = "facebook/mask2former-swin-base-coco-panoptic"
    processor = build_processor(model_id, local_files_only=False)
    num_labels = 150
    id2label = {i: f"class_{i}" for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}
    model = build_model(model_id, num_labels, id2label, label2id).cpu()
    model.eval()

    pixel_values = torch.randn(1, 3, 640, 640)
    pixel_mask = torch.ones(1, 640, 640, dtype=torch.uint8)
    mask_labels = [torch.zeros((0, 640, 640), dtype=torch.float32)]
    class_labels = [torch.zeros((0,), dtype=torch.int64)]

    with torch.no_grad():
        out = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
    print(f"Loss: {out.loss.item():.4f}")


if __name__ == "__main__":
    test_pipeline()

