# Experimental Design

## Overview

This study systematically compares molecular feature representations and neural
network architectures for binary drug–excipient incompatibility prediction. All
experiments follow a fixed protocol to ensure fair comparison.

---

## 1. Data Protocol

### 1.1 Source

Single CSV file: `data/raw/drug_excipient_data.csv`  
Columns: `drug_cid`, `excipient_cid`, `drug_smiles`, `excipient_smiles`, `output_value`

### 1.2 Label Mapping

| Source string | Internal integer | Class name |
|---------------|-----------------|------------|
| `"Imcompatible"` (sic) | 1 | INCOMPATIBLE (positive class) |
| `"Compatible"` | 0 | COMPATIBLE |

The misspelling is preserved exactly as in the source file. Any unlisted label
value causes script 01 to abort with an error.

### 1.3 Pair Key

An ordered drug–excipient pair is uniquely identified by:
`(RDKit canonical drug SMILES, RDKit canonical excipient SMILES)`.

Canonicalisation is used **only** for building the pair key. No other chemical
standardisation (salt stripping, tautomers, charge neutralisation) is applied.

### 1.4 Cleaning Funnel (sequential, each row counted once)

1. **Missing required values**: rows with blank `drug_smiles`, `excipient_smiles`, or `output_value` are removed.
2. **Invalid SMILES**: rows where RDKit cannot parse either SMILES are removed.
3. **Conflicting labels**: pairs appearing with more than one label — **all occurrences** are removed.
4. **Exact duplicate pairs**: pairs with the same label appearing more than once — all but the first occurrence are removed.

### 1.5 Splitting

| Split | Fraction | Use |
|-------|----------|-----|
| Development | 90% | All training, validation, model selection |
| Hold-out test | 10% | Not used in this project phase |

Development set: 5-fold stratified cross-validation.  
Seed: 42 (fixed, not a hyperparameter).

---

## 2. Feature Representations

See [`experiment_registry.md`](experiment_registry.md) for the full keyword table.

### 2.1 Feature Generation

All feature matrices are computed from the RDKit canonical SMILES of each unique
molecule and cached as `.npy` files with JSON metadata. The cache is invalidated
if the molecule set or configuration changes.

### 2.2 Scaling

Only the `RD` (descriptors) family is scaled. Z-score statistics are fitted on
the **training fold** of each CV fold; the validation fold is transformed with
the training statistics. Binary fingerprints are never scaled.

---

## 3. Training Protocol

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 1e-4 |
| Batch size | 128 |
| Max epochs | 200 |
| Early stopping patience | 20 epochs |
| Early stopping monitor | inner-val BCE loss |
| Inner val fraction | 10% of training fold |
| Class weights | balanced (w_c = n / (n_classes × n_c)) |
| Grad clip | None (configurable) |
| LR scheduler | None (configurable; `plateau` available) |

### 3.1 Early Stopping

An **inner validation split** (10% of the training fold, stratified) is carved
out for early stopping. The CV held-out fold is **never** used for stopping
decisions or checkpoint selection. Heldout CV fold loss/AP is logged every epoch
as a diagnostic, not used for any decision.

---

## 4. Evaluation

### 4.1 Metrics

| Metric | Role |
|--------|------|
| Average Precision (AP/AUPRC) | **Primary** — model selection across representations and architectures |
| AUROC | Secondary — comparability with DE-INTERACT baseline |
| F2 score | Threshold-dependent; reflects recall emphasis (high FN cost) |
| ECE | Calibration quality |
| MCC | Balanced accuracy in imbalanced settings |

### 4.2 Threshold Selection

The operating threshold is selected by maximising F2 (β=2) on the pooled
out-of-fold (OOF) predictions. The test set is **never used** for threshold
selection.

### 4.3 Out-of-Fold Pooling

OOF predictions from all folds are pooled for evaluation. When multiple seeds
are used, OOF scores for the same sample are averaged across seeds.

---

## 5. Experimental Stages

### Stage 1 — Representation Screening

Fixed architecture: CONCAT  
Varied: all 15 non-empty subsets of {PC, MG, MK, RD}  
Goal: identify the best representation(s) by AP before varying the architecture.

### Stage 2 — Architecture Comparison

Fixed representation(s): chosen manually from Stage 1 results  
Varied: CONCAT, INTERACT, ATTN  
Goal: identify the best architecture for the chosen representation.

---

## 6. What This Design Does NOT Do

* **No test set evaluation** in this phase (it is completely untouched).
* **No hyperparameter search** — all parameters are fixed across representations and architectures for a fair comparison.
* **No oversampling** — class imbalance is addressed only through class-weighted loss.
* **No chemical standardisation** beyond what is needed to build the pair key.
