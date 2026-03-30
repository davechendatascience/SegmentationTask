# YOLOv11 Segmentation Training with Roboflow

This module provides training scripts for YOLOv11 instance segmentation using the
same COCO dataset root as the Mask2Former pipeline.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the shared COCO dataset used by Mask2Former under `data/hospital_coco`.

### Colab

If you want to run this on Google Colab instead of Docker, the simplest flow is:

```bash
git clone <your-repo-url>
cd SegmentationTask
pip install -r scripts/yolov11_seg/requirements.txt
python -m scripts.mask2former_seg.download_dataset \
  --output_dir /content/data/hospital_coco \
  --credentials /content/roboflow_credentials.json
python -m scripts.yolov11_seg.prepare_dataset \
  --data-root /content/data/hospital_coco
python -m scripts.yolov11_seg.train \
  --data-yaml /content/data/hospital_coco/yolo/data.yaml \
  --workers 2
```

You can also skip the credentials JSON and pass Roboflow values directly:

```bash
python -m scripts.mask2former_seg.download_dataset \
  --output_dir /content/data/hospital_coco \
  --api-key "$ROBOFLOW_API_KEY" \
  --workspace "<workspace>" \
  --project "<project>" \
  --version "<version>"
```

## Configuration

Edit `config.py` to set:
- Shared dataset root (`data_root`)
- Model size (yolo11n-seg.pt, yolo11s-seg.pt, etc.)
- Training parameters

## Training

### 1. Prepare YOLO data.yaml and labels

```bash
python -m scripts.yolov11_seg.prepare_dataset
```

### Use a different shared COCO dataset root

```bash
python -m scripts.yolov11_seg.prepare_dataset --data-root path/to/hospital_coco
```

For Colab, a typical path is `/content/data/hospital_coco`.

This step generates `data/hospital_coco/yolo/data.yaml` plus YOLO segmentation
labels from the same COCO annotations used by Mask2Former.

### 2. Train from the prepared YAML
```bash
python -m scripts.yolov11_seg.train \
  --data-yaml data/hospital_coco/yolo/data.yaml \
  --imgsz 640
```

### Save directly into a chosen experiment directory

```bash
python -m scripts.yolov11_seg.train \
  --data-yaml data/hospital_coco/yolo/data.yaml \
  --imgsz 640 \
  --output-dir output/yolov11_seg/exp1
```

### Or still train directly from the shared COCO dataset
```bash
python -m scripts.yolov11_seg.train \
  --data-root path/to/hospital_coco \
  --imgsz 640
```

### Fine-tune from pretrained YOLOv11 segmentation weights
```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")
model.train(data="data/hospital_coco/yolo/data.yaml", epochs=100, imgsz=640)
```

## Evaluation

```bash
python -m scripts.yolov11_seg.evaluate \
  --model output/yolov11_seg/exp1/weights/best.pt \
  --data data/hospital_coco/yolo/data.yaml \
  --imgsz 640
```

## Visualization

This script is mainly used to quickly inspect whether predicted masks, boxes, and
class labels look reasonable on sample images.
這支腳本主要用來快速檢查 prediction mask、bbox 和類別名稱在樣本影像上是否合理。

```bash
python -m scripts.yolov11_seg.visualize \
  --model output/yolov11_seg/exp1/weights/best.pt \
  --source data/test/images \
  --max-images 10 \
  --save
```

Notes:

- `--source` can be a single image, an image folder, or a video path.
- `--save` stores rendered prediction images in the Ultralytics default output folder.
- `--show` opens a window to preview the rendered results interactively.
- `retina_masks=True` is enabled in this script so saved masks are drawn with higher quality.

中文補充：

- `--source` 可以是單張圖片、圖片資料夾或影片路徑。
- `--save` 會把視覺化結果存到 Ultralytics 預設輸出資料夾。
- `--show` 會開啟視窗互動式預覽結果。
- 這支腳本已啟用 `retina_masks=True`，輸出的 segmentation mask 邊界會比較細緻。

## Notes

- YOLOv11 uses YOLOv8 format for segmentation
- This repo converts the Roboflow COCO export into YOLO labels automatically
- In Colab, increase `--workers` from `0` if the runtime is stable enough
