"""
Visualize YOLOv11 segmentation predictions.

Usage:
    python -m scripts.yolov11_seg.visualize --model output/yolov11/exp/weights/best.pt --source data/test/images --max-images 10
"""
import argparse
import random
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLOv11 predictions")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model weights")
    parser.add_argument("--source", type=str, required=True,
                       help="Path to images or video for prediction")
    parser.add_argument("--max-images", type=int, default=10,
                       help="Maximum number of images to process")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on")
    parser.add_argument("--save", action="store_true",
                       help="Save prediction results")
    parser.add_argument("--show", action="store_true",
                       help="Show results in window")
    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)

    # Get image paths
    source_path = Path(args.source)
    if source_path.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(source_path.glob(f"**/*{ext}")))
        image_paths = sorted(image_paths)
        if args.max_images > 0:
            image_paths = random.sample(image_paths, min(args.max_images, len(image_paths)))
    else:
        image_paths = [source_path]

    print(f"Processing {len(image_paths)} images...")

    # Run predictions
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        device=args.device,
        save=args.save,
        show=args.show,
        max_det=100,  # Maximum detections per image
        retina_masks=True,  # High-quality masks
    )

    # Print summary
    total_detections = sum(len(result.boxes) for result in results)
    print(f"Total detections: {total_detections}")

    if args.save:
        print(f"Results saved to: runs/predict/")


if __name__ == "__main__":
    main()