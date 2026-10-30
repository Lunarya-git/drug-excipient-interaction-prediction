# Drug–Excipient Incompatibility Prediction — DE-INTERACT Research Repository

> **Research code — work in progress.**  
> This repository implements a systematic, reproducible study to benchmark and
> improve molecular feature representations and neural architectures for
> predicting drug–excipient incompatibilities.

---

## Table of Contents
1. [Background](#background)  
2. [Dataset](#dataset)  
3. [Project structure](#project-structure)  
4. [Experimental design](#experimental-design)  
5. [Quick start](#quick-start)  
6. [Running the pipeline](#running-the-pipeline)  
7. [Configuration reference](#configuration-reference)  
8. [Adding new experiments](#adding-new-experiments)  
9. [Reproducibility](#reproducibility)  
10. [Citation](#citation)

---

## Background

Drug–excipient incompatibility is a critical formulation challenge. The
**DE-INTERACT** paper (see `docs/Baseline-DEINTERACT paper.pdf`) introduced a
supervised machine-learning approach using PubChem (CACTVS) substructure
fingerprints and a concatenated ANN to predict incompatibility from SMILES
strings alone. This repository extends that baseline in a controlled experimental
framework:

* **Stage 1** — fix the architecture and compare all non-empty subsets (15) of
  four molecular fingerprint/descriptor families.  
* **Stage 2** — fix the best representation(s) from Stage 1 and compare three
  architectures: the CONCAT baseline, an explicit-interaction fusion network
  (INTERACT), and a lightweight Transformer over feature-group tokens (ATTN).

---

## Dataset

| Property | Value |
|----------|-------|
| File | `data/raw/drug_excipient_data.csv` |
| Columns | `drug_cid`, `excipient_cid`, `drug_smiles`, `excipient_smiles`, `output_value` |
| Labels | `"Compatible"` (0) / `"Imcompatible"` (1, sic — original spelling preserved) |
| Positive class | INCOMPATIBLE (label = 1) |

The raw file is **never modified**. Script 01 writes cleaned and split data into
`data/processed/` and `data/splits/`.

---

## Project Structure

```
drug-excipient-interaction-prediction/
├── configs/
│   ├── data.yaml            # dataset path, column names, label mapping
│   ├── features.yaml        # molecular feature families (PC, MG, MK, RD)
│   ├── training.yaml        # training loop, hardware, architectures
│   └── experiments.yaml     # stage definitions and representation lists
│
├── data/
│   ├── raw/                 # original CSV (read-only)
│   ├── processed/           # cleaned dataset, audit log, report
│   ├── features/            # cached feature matrices (.npy + .meta.json)
│   └── splits/              # CV fold definitions (JSON)
│
├── src/
│   ├── data/                # loading, cleaning, splitting, leakage checks
│   ├── features/            # generators (PC, MG, MK, RD), cache store
│   ├── models/              # CONCAT, INTERACT, ATTN architectures
│   ├── training/            # trainer loop, fold runner, early stopping
│   ├── evaluation/          # metrics (AP, AUROC, F-beta, ECE), reporting
│   ├── visualization/       # PR/ROC curves, learning curves, bar charts
│   └── utils/               # config, logging, IO, paths
│
├── scripts/
│   ├── 01_prepare_data.py          # clean + split
│   ├── 02_build_features.py        # generate + cache features
│   ├── 03_compare_representations.py  # Stage 1
│   └── 04_compare_architectures.py    # Stage 2
│
├── outputs/                 # all generated artefacts (gitignored)
├── tests/                   # pytest unit tests
├── docs/                    # methodology, experiment registry, literature survey
├── requirements.txt
└── README.md
```

---

## Experimental Design

See [`docs/experimental-design.md`](docs/experimental-design.md) for the full
protocol. In brief:

* **Pair key**: Ordered (drug, excipient) pair identified by RDKit canonical SMILES.  
* **Cleaning funnel**: missing values → invalid SMILES → conflicting labels → exact duplicates.  
* **Splitting**: 90% development / 10% hold-out test (stratified). 5-fold stratified CV on development.  
* **Early stopping**: inner validation split (10% of training fold), never touches the CV fold.  
* **Imbalance**: class-weighted BCE loss (balanced mode).  
* **Primary metric**: average precision (AUPRC). Threshold selected by maximising F2 on OOF predictions.

### Feature families

| Code | Name | Bits/Dims | Backend |
|------|------|-----------|---------|
| PC   | PubChem (CACTVS) substructure fingerprint | 881 | scikit-fingerprints |
| MG   | Morgan / ECFP4 | 2048 | RDKit |
| MK   | MACCS keys | 167 | RDKit |
| RD   | RDKit physicochemical descriptors | 19 | RDKit |

> **Note on PC fingerprints**: `scikit-fingerprints` (`skfp.fingerprints.PubChemFingerprint`)
> is a Python/RDKit implementation of the PubChem 881-bit key set based on the CDK
> `PubchemFingerprinter` with Andrew Dalke's fixes. It is **not** the PaDEL executable
> and bit-for-bit identity with PaDEL output is not claimed.

---

## Quick Start

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install PyTorch (CUDA recommended)

Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and
select your CUDA version. Example for CUDA 12.6:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

For CPU-only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the configuration (dry-run)

```bash
python scripts/01_prepare_data.py --dry-run
```

---

## Running the Pipeline

Run scripts sequentially. Each script writes outputs that the next one reads.

```bash
# Step 1: clean the dataset and create CV splits
python scripts/01_prepare_data.py

# Step 2: generate and cache molecular features (may take several minutes)
python scripts/02_build_features.py

# Step 3: Stage 1 — representation screening (long, GPU recommended)
python scripts/03_compare_representations.py

# Step 4: edit experiments.yaml, then run Stage 2
python scripts/04_compare_architectures.py
```

### Dry-run any script

```bash
python scripts/03_compare_representations.py --dry-run
```

### Run a subset of representations

```bash
python scripts/03_compare_representations.py --representations PC PCMG
```

### Override a config value at the command line

```bash
python scripts/03_compare_representations.py --override training.max_epochs=50 training.seeds=[42,0]
```

### Re-generate only one feature family

```bash
python scripts/02_build_features.py --families PC --force
```

---

## Configuration Reference

All configuration lives in `configs/`. The four YAML files are merged into one
dictionary by `src/utils/config.py`:

| File | Contents |
|------|----------|
| `data.yaml` | Raw data path, column mappings, label mapping, pair key, split settings |
| `features.yaml` | Feature family definitions, n_jobs, nonfinite policy |
| `training.yaml` | Training loop, hardware, architectures (CONCAT / INTERACT / ATTN) |
| `experiments.yaml` | Stage definitions, output root, representation and architecture lists |

Any leaf value can be overridden at the command line:

```bash
--override data.pair_key=raw_smiles
--override training.optimizer.lr=0.0001
--override features.families.MG.n_bits=1024
```

---

## Adding New Experiments

### New feature family

1. Add a new 2-letter code block under `features.families` in `configs/features.yaml`.
2. If the `type` matches an existing generator (`morgan`, `maccs`, …) no code changes are needed.
3. For a new generator type, add a `@register_generator("new_type")` function in
   `src/features/generators.py`.

### New architecture

1. Add a new key under `models:` in `configs/training.yaml`.
2. Implement a `PairModel` subclass in `src/models/` and register it with
   `@register_model("type_name")`.
3. Add it to the `architectures` list in `experiments.stages.architecture_comparison`.

---

## Reproducibility

* **Raw data** is never modified. Its SHA-256 is recorded in every report.  
* **Cleaned dataset** SHA-256 is embedded in the split metadata and feature cache manifest.  
* **Splits** are fixed once by script 01 and versioned in `data/splits/`.  
* **Feature cache** is invalidated automatically when the molecule set or configuration changes.  
* **Seeds** are set for Python, NumPy, and PyTorch in every run.  
* **Git commit hash** is recorded in experiment metadata when the repo is a git repository.

---

## Citation

If you use this code or build upon it, please cite the baseline paper:

```
DE-INTERACT: Drug-Excipient Interaction Prediction Using Machine Learning.
[Author list — see docs/Baseline-DEINTERACT paper.pdf]
```
