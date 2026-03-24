"""
Evaluate trained YOLOv11 segmentation model.

Usage:
    python -m scripts.yolov11_seg.evaluate --model output/yolov11/exp/weights/best.pt --data data.yaml
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model weights")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to evaluate on (val/test)")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run evaluation on")
    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)

    # Run evaluation
    print(f"Evaluating model on {args.split} set...")
    results = model.val(
        data=args.data,
        split=args.split,
        batch=args.batch_size,
        device=args.device,
        save_json=True,
        save_txt=True,
        save_conf=True,
        save=True  # Save prediction results
    )

    # Print key metrics
    print("\nEvaluation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    if hasattr(results, 'seg'):
        print(f"Segmentation mAP50: {results.seg.map50:.4f}")
        print(f"Segmentation mAP50-95: {results.seg.map:.4f}")

    print(f"Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()
