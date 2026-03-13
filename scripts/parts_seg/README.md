# Parts Segmentation

`parts_seg` 提供零件分割（part segmentation）相關的資料準備、訓練、評估與視覺化腳本，目前主要涵蓋：

- `Pascal-Part`
- `ADE20KPart234`
- `PartImageNet++` 的資料下載與 dataset helper

目前目錄內有三條主要流程：

1. `SAM2+YOLO` part instance segmentation
2. `Ultralytics YOLOv11-seg` part segmentation
3. `YOLOv11-seg + hierarchy`，在 part 預測之外加入 object-level 輔助 supervision

## 目錄說明

- `dataset.py`: Pascal-Part dataset loader
- `dataset_ade20k234.py`: ADE20KPart234 dataset loader
- `dataset_partimagenetpp.py`: PartImageNet++ dataset helper
- `export_yolo_seg_format.py`: 將 ADE20KPart234 匯出為 Ultralytics YOLO segmentation 格式
- `train_yolo.py`: 使用 `SAM2+YOLO` 訓練 part segmentation
- `evaluate_yolo.py`: 評估 `SAM2+YOLO`
- `visualize_yolo.py`: 視覺化 `SAM2+YOLO` 預測結果
- `train_yolo11_seg.py`: 使用 Ultralytics `YOLOv11-seg` 訓練
- `train_yolo_seg_hierarchy.py`: 使用 hierarchy 版本的 `YOLOv11-seg` 訓練
- `evaluate_yolo_seg_hierarchy.py`: 評估 hierarchy 版本模型
- `visualize_yolo_seg_hierarchy.py`: 視覺化 hierarchy 版本模型
- `download_partimagenetpp.py`: 下載 PartImageNet++ annotation
- `download_imagenet1k.py`: 透過 Kaggle 下載 ImageNet-1K

## 環境需求

建議使用 Python 3.10+ 與可用的 CUDA 環境。

常見依賴包含：

```bash
pip install torch torchvision torchaudio
pip install numpy pillow scipy tqdm albumentations
pip install pycocotools opencv-python
pip install ultralytics
```

若要使用 PartImageNet++，還需要：

```bash
pip install huggingface_hub datasets
```

若要下載 ImageNet-1K，還需要：

```bash
pip install kaggle
```

## 資料集格式

### Pascal-Part

預設路徑為 `data/pascal_part`，資料結構應為：

```text
data/pascal_part/
├─ VOCdevkit/VOC2010/JPEGImages/*.jpg
├─ VOCdevkit/VOC2010/ImageSets/Main/{train,val}.txt
└─ Annotations_Part/*.mat
```

### ADE20KPart234

預設路徑為 `data/ADE20KPart234`，資料結構應為：

```text
data/ADE20KPart234/
├─ images/training/*.jpg
├─ images/validation/*.jpg
├─ annotations_detectron2_part/training/*.png
├─ annotations_detectron2_part/validation/*.png
├─ ade20k_instance_train.json
└─ ade20k_instance_val.json
```

## 流程一：SAM2+YOLO

### 訓練

```bash
python -m scripts.parts_seg.train_yolo
python -m scripts.parts_seg.train_yolo --dataset pascal --data_root data/pascal_part
python -m scripts.parts_seg.train_yolo --dataset ade20k234 --data_root data/ADE20KPart234 --image_size 512
```

預設輸出目錄為 `output/parts_seg_yolo`。

### 評估

```bash
python -m scripts.parts_seg.evaluate_yolo \
  --checkpoint output/parts_seg_yolo/latest_model.pt \
  --dataset ade20k234
```

### 視覺化

```bash
python -m scripts.parts_seg.visualize_yolo \
  --checkpoint output/parts_seg_yolo/latest_model.pt \
  --dataset ade20k234 \
  --max_images 20
```

### 注意事項

`train_yolo.py`、`evaluate_yolo.py`、`visualize_yolo.py` 會依賴 `scripts.sam2_yolo_seg`。目前這個模組不在此資料夾內，若你的 workspace 沒有對應實作，這條流程將無法直接執行。

## 流程二：YOLOv11-seg

這條流程使用 Ultralytics 的 segmentation checkpoint，例如 `yolo11n-seg.pt`、`yolo11s-seg.pt`。

### 1. 匯出成 YOLO segmentation 格式

```bash
python -m scripts.parts_seg.export_yolo_seg_format \
  --data_root data/ADE20KPart234 \
  --output_dir data/ADE20KPart234_yolo_seg
```

輸出結構如下：

```text
data/ADE20KPart234_yolo_seg/
├─ images/train
├─ images/val
├─ labels/train
├─ labels/val
└─ data.yaml
```

### 2. 開始訓練

```bash
python -m scripts.parts_seg.train_yolo11_seg \
  --data_yaml data/ADE20KPart234_yolo_seg/data.yaml \
  --epochs 50
```

常用參數：

- `--model`: 預訓練模型，預設 `yolo11n-seg.pt`
- `--imgsz`: 輸入尺寸，預設 `640`
- `--batch`: batch size，預設 `8`
- `--workers`: 預設 `0`，可避免部分環境的 DataLoader / shared memory 問題
- `--project`: 輸出根目錄，預設 `output/parts_yolo11_seg`

## 流程三：YOLOv11-seg + hierarchy

這條流程會保留每個 part instance 對應的 object id，並在訓練時加入 object auxiliary loss。

### 1. 匯出 hierarchy 格式資料

```bash
python -m scripts.parts_seg.export_yolo_seg_format \
  --data_root data/ADE20KPart234 \
  --output_dir data/ADE20KPart234_yolo_seg \
  --preserve_hierarchy
```

啟用後，每張影像除了 `.txt` 標註外，還會額外輸出：

```text
labels/train/xxx.hierarchy.json
labels/val/xxx.hierarchy.json
```

### 2. 訓練 hierarchy 模型

```bash
python -m scripts.parts_seg.train_yolo_seg_hierarchy \
  --data_yaml data/ADE20KPart234_yolo_seg/data.yaml \
  --epochs 30 \
  --batch 4
```

預設輸出目錄為 `output/parts_yolo_seg_hierarchy`。

### 3. 評估

```bash
python -m scripts.parts_seg.evaluate_yolo_seg_hierarchy \
  --checkpoint output/parts_yolo_seg_hierarchy/train/weights/best.pt \
  --data_yaml data/ADE20KPart234_yolo_seg/data.yaml
```

### 4. 視覺化

```bash
python -m scripts.parts_seg.visualize_yolo_seg_hierarchy \
  --checkpoint output/parts_yolo_seg_hierarchy/train/weights/best.pt \
  --data_yaml data/ADE20KPart234_yolo_seg/data.yaml \
  --max_images 20
```

## PartImageNet++ 輔助工具

目前 `parts_seg` 內有 PartImageNet++ 的 annotation / image 準備工具與 dataset helper，但沒有直接對應的完整訓練腳本。

### 下載 PartImageNet++ annotation

```bash
python -m scripts.parts_seg.download_partimagenetpp
python -m scripts.parts_seg.download_partimagenetpp --data_dir data/PartImageNetPP
```

### 下載 ImageNet-1K

```bash
python -m scripts.parts_seg.download_imagenet1k
python -m scripts.parts_seg.download_imagenet1k --output_dir data/ImageNet-1K
```

下載 ImageNet 前請先：

1. 註冊 Kaggle 帳號
2. 加入 `imagenet-object-localization-challenge`
3. 設定 `kaggle.json`

## 輸出位置

常見輸出目錄：

- `output/parts_seg_yolo`: SAM2+YOLO checkpoint
- `output/parts_yolo11_seg`: YOLOv11-seg 訓練結果
- `output/parts_yolo_seg_hierarchy`: hierarchy 版本訓練結果與權重

## 建議起手式

如果你要先確認整條流程能順利跑通，最建議從 `ADE20KPart234 -> YOLOv11-seg` 開始：

```bash
python -m scripts.parts_seg.export_yolo_seg_format \
  --data_root data/ADE20KPart234 \
  --output_dir data/ADE20KPart234_yolo_seg

python -m scripts.parts_seg.train_yolo11_seg \
  --data_yaml data/ADE20KPart234_yolo_seg/data.yaml \
  --model yolo11n-seg.pt \
  --epochs 10 \
  --imgsz 512 \
  --batch 4
```

若之後需要 object-part 關係，再切到 `--preserve_hierarchy` 與 `train_yolo_seg_hierarchy.py`。
