import pytest
import torch
import numpy as np
import tempfile
import h5py


@pytest.fixture
def sample_rna_sequences():
    """Sample RNA sequences for testing."""
    return [
        "ACGU",
        "AAAA",
        "UUUU",
        "ACGUACGU",
    ]


@pytest.fixture
def sample_tensors():
    """Sample tensors for dataset testing."""
    x = torch.randn(100, 10)
    y = torch.randn(100, 2)
    return (x, y)


@pytest.fixture
def temp_hdf5_file():
    """Create a temporary HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        path = f.name

    # Create sample data
    with h5py.File(path, 'w') as f:
        f.create_dataset('sequence', data=np.random.randint(0, 4, (10, 50)))
        f.create_dataset('target', data=np.random.randn(10, 50))

    yield path

    # Cleanup
    import os
    if os.path.exists(path):
        os.remove(path)
