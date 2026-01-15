# Core Design 2 Pipeline (Mask2Former HF)

This is a temporary working note for the `core_design_2` pipeline. Expect updates as the design evolves.

## Architecture
- **Model**: `Mask2FormerForUniversalSegmentation` (HF Transformers).
- **Backbone/decoder**: built into the pretrained Mask2Former checkpoint.
- **No extra heads**: we use the native Mask2Former outputs and loss.

## Data & Augmentation
- **Dataset**: ADE20k (images + grayscale masks).
- **Transforms**: `core_design_2/augmentation_block.py`
  - Train: resize/pad, flip, jitter, normalize.
  - Eval: resize/pad, normalize.
- **Collation**: `core_design_2/dataset_block.py` returns `pixel_values`, `pixel_mask`, `mask_labels`, `class_labels`.

## Training
- **Loss**: built‑in HF Mask2Former loss (Hungarian matching inside).
- **Optimizer**: AdamW on trainable params.
- **Freeze modes**:
  - `none`: full fine‑tuning.
  - `class_only`: only class predictor.
  - `class_and_mask_decoder`: class predictor + mask decoder.
- **Compile**: `torch.compile` is enabled by default; we save `base_model` (not compiled wrapper).

## Evaluation / Visualization
- `core_design_2/eval_block.py` provides a simple val‑loss evaluation and visualization helper.

## Entrypoints
- `core_design_2/train_block.py`
- `core_design_2/mask2former_train.py` (thin wrapper)
- `train_local.py --core core_design_2`

