# Experiment Registry

This file documents every representation keyword and architecture keyword used
in the experiments. Update it when you add new families or architectures.

---

## Representation Keywords

A representation keyword is formed by concatenating 2-letter family codes in
the order they appear in the combined molecule vector.

### Feature Families

| Code | Full Name | Type | Dim | Notes |
|------|-----------|------|-----|-------|
| PC   | PubChem (CACTVS) substructure fingerprint | `pubchem_fingerprint` | 881 | scikit-fingerprints |
| MG   | Morgan / ECFP4 | `morgan` | 2048 | RDKit, radius 2 |
| MK   | MACCS keys | `maccs` | 167 | RDKit |
| RD   | RDKit physicochemical descriptors | `rdkit_descriptors` | 19 | z-score scaled, clipped ±10 |

### All 15 Representation Keywords (Stage 1)

**Singles (4)**
| Keyword | Families |
|---------|----------|
| PC | PubChem |
| MG | Morgan |
| MK | MACCS |
| RD | Descriptors |

**Pairs (6)**
| Keyword | Families |
|---------|----------|
| PCMG | PubChem + Morgan |
| PCMK | PubChem + MACCS |
| PCRD | PubChem + Descriptors |
| MGMK | Morgan + MACCS |
| MGRD | Morgan + Descriptors |
| MKRD | MACCS + Descriptors |

**Triples (4)**
| Keyword | Families |
|---------|----------|
| PCMGMK | PubChem + Morgan + MACCS |
| PCMGRD | PubChem + Morgan + Descriptors |
| PCMKRD | PubChem + MACCS + Descriptors |
| MGMKRD | Morgan + MACCS + Descriptors |

**All four (1)**
| Keyword | Families |
|---------|----------|
| PCMGMKRD | All four |

---

## Architecture Keywords

| Keyword | Class | Description |
|---------|-------|-------------|
| CONCAT | `ConcatANN` | Concatenated [drug, exc] → 3-layer ReLU MLP |
| INTERACT | `InteractANN` | Separate encoders → [h_d, h_e, h_d·h_e, \|h_d−h_e\|] → MLP |
| ATTN | `AttentionPairModel` | Transformer over feature-group tokens with a CLS token |

---

## Experiment IDs

An experiment ID is `<representation_keyword>-<architecture_keyword>`.

Examples:
- `PC-CONCAT` — PubChem fingerprints, concatenated MLP baseline
- `PCMG-INTERACT` — PubChem + Morgan, interaction fusion
- `PCMGMKRD-ATTN` — All four families, Transformer

---

## Stage Results (fill after running)

### Stage 1 — Representation Screening

See `outputs/metrics/representation_screening/stage1_summary.csv`.

### Stage 2 — Architecture Comparison

See `outputs/metrics/architecture_comparison/stage2_summary.csv`.
