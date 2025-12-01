import pytest
import numpy as np


# The tokenization mapping (will be moved to constants.py)
RNA_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}


class TestTokenization:
    """Tests for RNA sequence tokenization."""

    def test_rna_to_int_mapping_basic(self):
        """Test that basic RNA bases map to correct integers."""
        assert RNA_TO_INT['A'] == 0
        assert RNA_TO_INT['C'] == 1
        assert RNA_TO_INT['G'] == 2
        assert RNA_TO_INT['U'] == 3

    def test_t_maps_to_same_as_u(self):
        """Test that T (DNA) maps to same value as U (RNA)."""
        assert RNA_TO_INT['T'] == RNA_TO_INT['U']
        assert RNA_TO_INT['T'] == 3

    def test_tokenize_sequence(self, sample_rna_sequences):
        """Test tokenizing complete sequences."""
        seq = sample_rna_sequences[0]  # "ACGU"
        tokens = [RNA_TO_INT[c] for c in seq]
        assert tokens == [0, 1, 2, 3]

    def test_tokenize_homopolymer(self):
        """Test tokenizing homopolymer sequences."""
        for base, expected in [('A', 0), ('C', 1), ('G', 2), ('U', 3)]:
            seq = base * 10
            tokens = [RNA_TO_INT[c] for c in seq]
            assert all(t == expected for t in tokens)

    def test_all_bases_covered(self):
        """Test that all expected bases are in the mapping."""
        expected_bases = {'A', 'C', 'G', 'U', 'T'}
        assert set(RNA_TO_INT.keys()) == expected_bases

    def test_tokenize_to_numpy_array(self, sample_rna_sequences):
        """Test converting tokenized sequence to numpy array."""
        seq = sample_rna_sequences[0]  # "ACGU"
        tokens = np.array([RNA_TO_INT[c] for c in seq], dtype=np.int64)

        assert tokens.dtype == np.int64
        assert tokens.shape == (4,)
        np.testing.assert_array_equal(tokens, [0, 1, 2, 3])
