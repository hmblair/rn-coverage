import pytest
import subprocess
import tempfile
import os
import h5py
import numpy as np

# Path to the rn-coverage script
BIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'bin')
RN_COVERAGE = os.path.join(BIN_DIR, 'rn-coverage')


class TestPredictCLI:
    """Tests for the predict CLI command."""

    def test_predict_no_args_shows_usage(self):
        """Test that predict with no args shows usage message."""
        result = subprocess.run(
            [RN_COVERAGE, 'predict'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 1
        assert 'Usage:' in result.stdout
        assert 'rn-coverage predict' in result.stdout

    def test_predict_missing_file_shows_error(self):
        """Test that predict with non-existent file shows error."""
        result = subprocess.run(
            [RN_COVERAGE, 'predict', 'nonexistent.h5'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 1
        assert 'does not exist' in result.stdout

    def test_predict_missing_output_dir_arg(self):
        """Test that -o without directory shows error."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
            temp_path = f.name

        try:
            # Create a minimal HDF5 file
            with h5py.File(temp_path, 'w') as f:
                f.create_dataset('sequence', data=np.array([[0, 1, 2, 3]]))

            result = subprocess.run(
                [RN_COVERAGE, 'predict', temp_path, '-o'],
                capture_output=True,
                text=True
            )
            assert result.returncode == 1
            assert 'requires an output directory' in result.stdout
        finally:
            os.remove(temp_path)

    def test_predict_config_missing_file(self):
        """Test that --config without file shows usage."""
        result = subprocess.run(
            [RN_COVERAGE, 'predict', '--config'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 1
        assert 'Usage:' in result.stdout


class TestTokenizeCLI:
    """Tests for the tokenize CLI command."""

    def test_tokenize_no_args_shows_usage(self):
        """Test that tokenize with no args shows usage."""
        result = subprocess.run(
            [RN_COVERAGE, 'tokenize'],
            capture_output=True,
            text=True
        )
        # The tokenize script should show usage or error
        assert result.returncode != 0 or 'usage' in result.stdout.lower() or 'usage' in result.stderr.lower()


class TestMainCLI:
    """Tests for the main CLI entry point."""

    def test_no_command_shows_usage(self):
        """Test that no command shows available commands."""
        result = subprocess.run(
            [RN_COVERAGE],
            capture_output=True,
            text=True
        )
        assert result.returncode == 1
        assert 'Usage:' in result.stdout
        assert 'tokenize' in result.stdout
        assert 'predict' in result.stdout

    def test_invalid_command_shows_usage(self):
        """Test that invalid command shows available commands."""
        result = subprocess.run(
            [RN_COVERAGE, 'invalid'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 1
        assert 'Usage:' in result.stdout
