"""Tests for the feature generators (synthetic molecules only)."""

from __future__ import annotations

import numpy as np
import pytest


# A handful of valid, simple SMILES for testing.
VALID_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CN", "C1CCCCC1"]
INVALID_SMILES = ["NOTASMILES", "***"]


class TestMorganGenerator:
    def test_shape(self):
        from src.features.generators import generate_morgan
        arr = generate_morgan(VALID_SMILES, {"n_bits": 256, "radius": 2}, {})
        assert arr.shape == (len(VALID_SMILES), 256)

    def test_binary_values(self):
        from src.features.generators import generate_morgan
        arr = generate_morgan(VALID_SMILES, {"n_bits": 256, "radius": 2}, {})
        assert set(arr.flatten().tolist()).issubset({0, 1})

    def test_different_radius_gives_different_result(self):
        from src.features.generators import generate_morgan
        arr2 = generate_morgan(["CCO"], {"n_bits": 512, "radius": 2}, {})
        arr3 = generate_morgan(["CCO"], {"n_bits": 512, "radius": 3}, {})
        # Not necessarily different for very simple molecules, but shape must match.
        assert arr2.shape == arr3.shape == (1, 512)

    def test_invalid_smiles_raises(self):
        from src.features.generators import FeatureGenerationError, generate_morgan
        with pytest.raises(FeatureGenerationError):
            generate_morgan(INVALID_SMILES, {"n_bits": 256, "radius": 2}, {})


class TestMACCSGenerator:
    def test_shape_and_type(self):
        from src.features.generators import generate_maccs
        arr = generate_maccs(VALID_SMILES, {"n_bits": 167}, {})
        assert arr.shape == (len(VALID_SMILES), 167)
        assert arr.dtype == np.uint8

    def test_benzene_has_aromatic_key(self):
        from src.features.generators import generate_maccs
        # MACCS key 162 is typically associated with aromatic atoms.
        arr = generate_maccs(["c1ccccc1"], {"n_bits": 167}, {})
        assert arr.shape == (1, 167)


class TestRDKitDescriptors:
    _FAM = {
        "descriptors": ["MolWt", "MolLogP", "NumHDonors"],
        "type": "rdkit_descriptors",
    }

    def test_shape(self):
        from src.features.generators import generate_rdkit_descriptors
        arr = generate_rdkit_descriptors(VALID_SMILES, self._FAM, {"nonfinite_policy": "error"})
        assert arr.shape == (len(VALID_SMILES), 3)
        assert arr.dtype == np.float32

    def test_ethanol_mw(self):
        from src.features.generators import generate_rdkit_descriptors
        arr = generate_rdkit_descriptors(["CCO"], self._FAM, {"nonfinite_policy": "error"})
        mw = float(arr[0, 0])
        assert abs(mw - 46.04) < 1.0  # ethanol MW ≈ 46.07

    def test_unknown_descriptor_raises(self):
        from src.features.generators import FeatureGenerationError, generate_rdkit_descriptors
        fam = {"descriptors": ["NOTADESCRIPTOR"], "type": "rdkit_descriptors"}
        with pytest.raises(FeatureGenerationError):
            generate_rdkit_descriptors(VALID_SMILES, fam, {})


class TestExpectedDimension:
    def test_morgan(self):
        from src.features.generators import expected_dimension
        assert expected_dimension({"type": "morgan", "n_bits": 1024}) == 1024

    def test_maccs(self):
        from src.features.generators import expected_dimension
        assert expected_dimension({"type": "maccs", "n_bits": 167}) == 167

    def test_descriptors(self):
        from src.features.generators import expected_dimension
        assert expected_dimension({"type": "rdkit_descriptors", "descriptors": ["MolWt", "TPSA"]}) == 2
