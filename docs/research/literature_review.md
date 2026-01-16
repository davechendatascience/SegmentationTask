# Literature Review: Beyond Mask2Former

This document summarizes relevant literature and actionable directions for advancing this project beyond Mask2Former. The focus is on panoptic segmentation, open-vocabulary extensions, and training strategies that can improve performance or generalization.

## Core Panoptic Foundations

- **Panoptic FPN** (2019): top-down panoptic fusion; useful baseline for "things vs stuff" trade-offs.
  - https://arxiv.org/abs/1901.02446
- **Panoptic-DeepLab** (2019): bottom-up panoptic approach; strong stuff segmentation and efficient inference.
  - https://arxiv.org/abs/1911.10194

## Set-Prediction and Transformer Lineage

- **DETR** (2020): set prediction for detection; the conceptual foundation for query-based segmentation.
  - https://arxiv.org/abs/2005.12872
- **Deformable DETR** (2020): sparse attention improves convergence and speed.
  - https://arxiv.org/abs/2010.04159
- **MaskFormer** (2021): first unified transformer for semantic/instance/panoptic via mask classification.
  - https://arxiv.org/abs/2107.06278
- **Mask2Former** (2021): masked attention + multi-scale pixel decoder; current core baseline.
  - https://arxiv.org/abs/2112.01527
- **Mask DINO** (2022): denoising training for stronger matching and stability; often improves over Mask2Former.
  - https://arxiv.org/abs/2206.02777

## Universal / Task-Conditioned Segmentation

- **OneFormer** (2022): task conditioning inside one architecture (semantic/instance/panoptic).
  - https://arxiv.org/abs/2211.06220

## Alternative Panoptic Formulations

- **MaX-DeepLab** (2020): end-to-end panoptic with mask transformers; distinct mask formation and supervision.
  - https://arxiv.org/abs/2012.00759

## Open-Vocabulary / Foundation Models

- **CLIP** (2021): text-image alignment backbone; enables open-vocabulary classification of masks.
  - https://arxiv.org/abs/2103.00020
- **Segment Anything (SAM)** (2023): promptable mask foundation model; useful for proposals or pseudo-labels.
  - https://arxiv.org/abs/2304.02643

## Data-Centric Scaling

- **SegGen** (2023): synthetic data generation improves segmentation results on ADE20K and COCO.
  - https://arxiv.org/abs/2311.03355

## Efficiency-Oriented Alternatives

- **FastInst** (2023): efficient instance/panoptic segmentation; competitive trade-offs for speed.
  - https://arxiv.org/abs/2303.08594

---

## Novel Directions to Explore (Actionable)

### 1) Open-Vocabulary Panoptic with CLIP-Aligned Queries
**Idea:** predict class-agnostic masks; classify via text embeddings.

**Implementation sketch:**
- Replace class head logits with cosine similarity to CLIP text embeddings.
- Use prompt ensembling ("a photo of a {class}") for stability.
- Train with known classes, evaluate zero-shot on novel classes.

**Why it matters:** generalizes beyond ADE20K label space and is publishable.

### 2) SAM Proposals + Panoptic Fusion
**Idea:** use SAM for high-recall masks; a lightweight head assigns class and thing/stuff.

**Implementation sketch:**
- Stage 1: SAM masks (frozen).
- Stage 2: classifier + panoptic assembly (Mask2Former-style or Panoptic-FPN).

**Why it matters:** sharp masks and reduced training burden for mask quality.

### 3) Denoising Query Training (Mask DINO-Style)
**Idea:** inject noisy copies of GT masks into the query set with denoising loss.

**Implementation sketch:**
- Add denoising branch in decoder.
- Denoising losses encourage stable matching and faster convergence.

**Why it matters:** reduces collapse cases and improves query stability.

### 4) Dual-Path Things/Stuff Queries
**Idea:** separate query sets for things vs stuff, fuse late.

**Implementation sketch:**
- Two query pools; stuff queries focus on large regions.
- Late fusion for panoptic assembly.

**Why it matters:** improves stuff coherence and reduces fragmented stuff masks.

### 5) Lightweight Hierarchy Distillation
**Idea:** use hierarchical supervision on embeddings instead of full KL over logits.

**Implementation sketch:**
- Hierarchy loss on query embeddings or class prototypes.
- Avoids VRAM-heavy KL losses.

**Why it matters:** keeps class-relationship benefits without memory blow-ups.

### 6) Semi-Supervised / Pseudo-Label Bootstrapping
**Idea:** generate pseudo-labels with a teacher (Mask2Former or SAM), retrain.

**Implementation sketch:**
- Periodically update pseudo labels with a frozen teacher checkpoint.
- Train with a mix of clean labels and pseudo labels.

**Why it matters:** practical accuracy gains without extra labeling.

---

## Mapping to This Repo

- **core_design** (CLIP + LightMask2Former): best for open-vocabulary and CLIP-aligned heads.
- **core_design_2** (HF Mask2Former): best for denoising query training and fast experiments.

If desired, a `core_design_3` can be added for a dedicated open-vocabulary pipeline or SAM-based proposal fusion.

