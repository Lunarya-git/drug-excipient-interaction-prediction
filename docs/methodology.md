# Methodology

## Overview

This document describes the methodology used in this research project. It
complements the [experimental design](experimental-design.md) document, which
focuses on the protocol, and adds rationale for the choices made.

---

## 1. Problem Framing

Drug–excipient incompatibility prediction is treated as a **binary
classification problem**: given a drug SMILES and an excipient SMILES, predict
whether the pair is `INCOMPATIBLE` (positive class, label = 1) or `COMPATIBLE`
(label = 0).

**Why binary?** The original dataset provides two-class annotations. Treating
this as binary classification allows direct comparison with the DE-INTERACT
baseline.

**Why SMILES?** Structure-based prediction from SMILES alone is the most
practically useful setting: it requires no wet-lab data and can screen
candidate pairs before formulation experiments.

---

## 2. Molecular Representations

### 2.1 PubChem (CACTVS) Substructure Fingerprint — PC

The 881-bit PubChem CACTVS substructure key set encodes the presence of
predefined chemical substructures (element counts, rings, functional groups,
SMARTS patterns). It was chosen by the DE-INTERACT baseline for its
interpretability and alignment with pharmaceutical substructure knowledge.

**Backend**: `scikit-fingerprints` (`skfp.fingerprints.PubChemFingerprint`), a
Python/RDKit reimplementation of the CDK PubchemFingerprinter. This avoids a
Java dependency (PaDEL) while following the same 881-key specification.

**Caveat**: bit-for-bit identity with PaDEL output is not guaranteed. Any
comparison with published PaDEL numbers should account for this.

### 2.2 Morgan / ECFP4 — MG

Circular Morgan fingerprints (radius 2, 2048 bits) capture local chemical
environments and are the most widely used hashed fingerprints in
cheminformatics. They complement PubChem keys because they hash arbitrary
substructures rather than matching predefined patterns.

### 2.3 MACCS Keys — MK

167-bit MACCS keys are a classical, human-curated set of substructure queries.
Compared with Morgan, they are lower-dimensional and highly interpretable.

### 2.4 RDKit Physicochemical Descriptors — RD

19 scalar descriptors covering molecular weight, lipophilicity (LogP),
hydrogen-bond capacity, topological polar surface area, ring counts, and formal
charge. Unlike binary fingerprints, these capture continuous physico-chemical
properties and are z-score scaled at the fold level.

---

## 3. Architecture Choices

### 3.1 CONCAT (baseline)

Concatenate `[drug_features, excipient_features]` and pass through a 3-layer
ReLU MLP. This is the controlled reproduction of the DE-INTERACT architecture:
the combination function is implicit (learned inside the MLP) and the two roles
are distinguished only by position in the concatenated vector.

### 3.2 INTERACT

Encode drug and excipient with separate MLPs into a fixed-size embedding. Fuse
using four explicit interaction functions:

```
fused = [h_d, h_e, h_d ⊙ h_e, |h_d − h_e|]
```

Hadamard product and absolute difference are inductive biases toward detecting
synergistic and differential effects between drug and excipient. A final MLP
classifies from `fused`. When `shared_encoder=True`, a Siamese encoder is used.

### 3.3 ATTN (Transformer over feature tokens)

Instead of operating on raw fingerprint bits, the ATTN model compresses each
molecule into a small set of **feature-group tokens** (e.g. the 7 PubChem key
sections) using learned linear projections. A role embedding distinguishes drug
from excipient tokens. A CLS token and both sets of molecule tokens attend
jointly through a shallow Transformer encoder.

This design is motivated by the observation that cross-molecule attention should
operate on meaningful chemical concepts, not individual bits. The sequence length
is at most a few dozen tokens, so the model stays lightweight.

---

## 4. Training Choices

### 4.1 Class-Weighted Loss

With ~7% positive-class prevalence, naive unweighted BCE would be dominated by
the majority class. Balanced class weights (`w_c = n / (n_classes × n_c)`)
upweight the incompatible class proportionally.

### 4.2 Inner-Fold Validation for Early Stopping

A 10% inner validation split is carved out of each training fold for early
stopping. The CV held-out fold is never used for stopping decisions, preserving
the unbiasedness of OOF predictions.

### 4.3 Average Precision as Primary Metric

For imbalanced data, AUROC can be misleadingly high. Average Precision (the
area under the precision-recall curve) is more discriminating when the positive
class is rare and the cost of missed positives is high (a missed incompatibility
can cause patient harm).

### 4.4 F2 Threshold Selection

The operating threshold is selected by maximising F2 (β=2) on OOF predictions.
β=2 weights recall twice as heavily as precision, reflecting the asymmetric cost
of a false negative (missed incompatibility) vs. a false positive (unnecessary
exclusion of a compatible excipient).

---

## 5. Reproducibility Safeguards

* Raw data is read-only; SHA-256 is recorded in all reports.
* Cleaned dataset SHA-256 is embedded in split metadata and the feature cache manifest.
* Feature cache is invalidated automatically when the molecule set or configuration changes.
* All random seeds are set for Python, NumPy, and PyTorch.
* Deterministic algorithms are enabled (`torch.use_deterministic_algorithms(warn_only=True)`).
