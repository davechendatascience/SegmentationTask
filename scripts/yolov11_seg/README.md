# YOLOv11 Segmentation Training with Roboflow

This module provides training scripts for YOLOv11 instance segmentation using Roboflow datasets.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Roboflow API key:
```bash
export ROBOFLOW_API_KEY="your-api-key-here"
```

Or update `config.py` with your API key.

## Configuration

Edit `config.py` to set:
- Roboflow workspace, project, and version
- Model size (yolo11n-seg.pt, yolo11s-seg.pt, etc.)
- Training parameters

## Training

### Option 1: Download and train
```bash
python -m scripts.yolov11_seg.train --download-data
```

### Option 2: Use existing dataset
```bash
python -m scripts.yolov11_seg.train --data-yaml path/to/data.yaml
```

## Evaluation

```bash
python -m scripts.yolov11_seg.evaluate --model output/yolov11/exp/weights/best.pt --data data.yaml
```

## Visualization

```bash
python -m scripts.yolov11_seg.visualize --model output/yolov11/exp/weights/best.pt --source data/test/images --max-images 10
```

## Notes

- YOLOv11 uses YOLOv8 format for segmentation
- Make sure your Roboflow dataset is exported in YOLOv8 format
- Adjust class names in `dataset.py` based on your dataset