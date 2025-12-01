import pytest
import torch
import numpy as np
import tempfile
import xarray as xr


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
def temp_netcdf_file():
    """Create a temporary netCDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
        path = f.name

    # Create sample data
    data = {
        'sequence': (['batch', 'nucleotide'], np.random.randint(0, 4, (10, 50))),
        'target': (['batch', 'nucleotide'], np.random.randn(10, 50)),
    }
    ds = xr.Dataset(data)
    ds.to_netcdf(path, engine='h5netcdf')

    yield path

    # Cleanup
    import os
    if os.path.exists(path):
        os.remove(path)
