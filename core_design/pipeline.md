# Core Design Pipeline (CLIP + Light Mask2Former)

This is a temporary working note for the `core_design` pipeline. Expect updates as the design evolves.

## Architecture
- **Backbone**: CLIP ViT (`openai/clip-vit-base-patch16`) with LoRA adapters.
- **Pixel decoder**: SimpleFPN (ViTDet‑style) to build multi‑scale features.
- **Decoder**: Lightweight Mask2Former‑style transformer decoder with fixed queries.
- **Heads**: class logits + mask embeddings.

## Data & Augmentation
- **Dataset**: ADE20k (images + grayscale masks).
- **Transforms**: `core_design/augmentation_block.py`
  - Train: resize, flip, jitter, blur, normalize.
  - Eval: resize + normalize.

## Training
- **Losses**: focal classification + mask BCE + dice.
- **Matcher**: Hungarian matching (linear assignment). GPU solver optional via `torch-linear-assignment`.
- **Optimizer**: AdamW with separate LR for backbone and decoder.
- **Compile**: `torch.compile` is enabled; the compiled model is *not* saved. We save the base model state dict.

## Evaluation / Visualization
- Basic evaluation and visualization in `core_design/eval_block.py`.
- Visualization thresholds can be tuned for early‑epoch outputs.

## Entrypoints
- `core_design/train_block.py` (Colab or script style)
- `train_local.py --core core_design`

