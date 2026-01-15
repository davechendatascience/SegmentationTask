# Mask2Former Full‑Loss Port Plan (Core Design 1)

This document tracks the step‑by‑step port of the **full Mask2Former loss** into `core_design`.
We will keep changes incremental and measurable.

## Goal
Align `core_design` loss with Mask2Former’s training scheme:
- Hungarian matching with class + mask costs
- **Softmax CE** with no‑object class
- **Point‑sampled** mask BCE + dice
- **Deep supervision** (loss at each decoder layer)

---

## Step 0 — Baseline (Done)
**Status:** ✅ Complete  
**Change:** Switched classification loss to Mask2Former‑style softmax CE with no‑object class.  
**Code:** `core_design/loss_block.py` (`use_mask2former_cls`)

---

## Step 1 — Add Decoder Aux Outputs
**Goal:** Make the decoder return per‑layer outputs:
```python
{
  "pred_logits": ...,
  "pred_masks": ...,
  "aux_outputs": [
     {"pred_logits": ..., "pred_masks": ...},
     ...
  ]
}
```
**Files:**
- `core_design/model_block.py` (decoder outputs)

**Notes:**  
Mask2Former applies losses to intermediate decoder layers to stabilize training.

---

## Step 2 — Deep Supervision in Loss
**Goal:** Compute class + mask losses for each aux output and aggregate.
**Files:**
- `core_design/loss_block.py`
  - Add `loss_labels` + `loss_masks` for aux outputs
  - Control weights via `weight_dict` (e.g., `loss_ce`, `loss_mask`, `loss_dice`)

**Notes:**  
Usually the same weights are applied to aux outputs (or scaled down).

---

## Step 3 — Point‑Sampled Mask Loss
**Goal:** Replace full‑resolution BCE/dice with point sampling for efficiency:
- Sample points based on uncertainty
- Compute BCE + dice only at sampled points

**Files:**
- `core_design/loss_block.py`

**Notes:**  
Mask2Former uses point sampling to reduce memory and stabilize training.

---

## Step 4 — Matching Cost Updates
**Goal:** Ensure Hungarian matching uses **class + mask + dice costs** consistent with Mask2Former.
**Files:**
- `core_design/loss_block.py` → `HungarianMatcher`

**Notes:**  
We may reuse the current matcher but update mask cost to point‑sampled cost if needed.

---

## Step 5 — Config / Weight Alignment
**Goal:** Validate weights match Mask2Former defaults:
- `loss_ce`, `loss_mask`, `loss_dice`
- `eos_coef` (no‑object weighting) if applicable

**Files:**
- `core_design/train_block.py`
- `train_local.py`

---

## Step 6 — Validation & Visual Checks
**Goal:** Verify:
- Loss decreases across all components
- Visual quality improves
- No over‑suppression by no‑object class

**Files:**
- `core_design/eval_block.py`
- `core_design/pipeline.md` (log updates)

---

## Notes / Risks
- Mask2Former loss is **tightly coupled** to its decoder architecture.
- Deep supervision requires extra memory.
- Point‑sampling and cost updates are delicate (must be consistent).

---

## Next Action
Proceed with **Step 1**: add aux outputs in the decoder.

