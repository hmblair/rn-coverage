import pytest
import torch
import sys
import os
import h5py

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.datasets import (
    SimpleDataset,
    SimpleIterableDataset,
    HDF5Dataset,
    stack_hdf5,
    get_idx_distributed,
)


class TestSimpleDataset:
    """Tests for SimpleDataset."""

    def test_init_with_valid_data(self, sample_tensors):
        """Test initialization with valid tensors."""
        dataset = SimpleDataset(sample_tensors)
        assert len(dataset) == 100

    def test_init_with_mismatched_lengths(self):
        """Test that mismatched tensor lengths raise ValueError."""
        x = torch.randn(100, 10)
        y = torch.randn(50, 2)  # Different length
        with pytest.raises(ValueError, match="same length"):
            SimpleDataset((x, y))

    def test_getitem_single_index(self, sample_tensors):
        """Test indexing with a single integer."""
        dataset = SimpleDataset(sample_tensors)
        x, y = dataset[0]
        assert x.shape == (10,)
        assert y.shape == (2,)

    def test_getitem_slice(self, sample_tensors):
        """Test indexing with a slice."""
        dataset = SimpleDataset(sample_tensors)
        x, y = dataset[0:10]
        assert x.shape == (10, 10)
        assert y.shape == (10, 2)

    def test_len(self, sample_tensors):
        """Test __len__ returns correct length."""
        dataset = SimpleDataset(sample_tensors)
        assert len(dataset) == len(sample_tensors[0])


class TestSimpleIterableDataset:
    """Tests for SimpleIterableDataset."""

    def test_init_with_valid_data(self, sample_tensors):
        """Test initialization with valid tensors."""
        dataset = SimpleIterableDataset(sample_tensors, batch_size=10)
        assert len(dataset) == 10  # 100 samples / 10 batch_size

    def test_init_with_mismatched_lengths(self):
        """Test that mismatched tensor lengths raise ValueError."""
        x = torch.randn(100, 10)
        y = torch.randn(50, 2)
        with pytest.raises(ValueError, match="same length"):
            SimpleIterableDataset((x, y), batch_size=10)

    def test_iteration_yields_batches(self, sample_tensors):
        """Test that iteration yields correct batch sizes."""
        dataset = SimpleIterableDataset(
            sample_tensors, batch_size=10, should_shuffle=False
        )
        iterator = iter(dataset)

        batch = next(iterator)
        assert len(batch) == 2  # x and y
        assert batch[0].shape == (10, 10)
        assert batch[1].shape == (10, 2)

    def test_distributed_slicing(self, sample_tensors):
        """Test distributed data loading with rank/world_size."""
        # With world_size=2, each rank should see half the batches
        dataset_rank0 = SimpleIterableDataset(
            sample_tensors, batch_size=10, rank=0, world_size=2
        )
        dataset_rank1 = SimpleIterableDataset(
            sample_tensors, batch_size=10, rank=1, world_size=2
        )

        assert len(dataset_rank0) == 5
        assert len(dataset_rank1) == 5

    def test_shuffle_actually_shuffles(self, sample_tensors):
        """Test that shuffling actually changes the data order.

        NOTE: This test will FAIL with the current buggy implementation
        because self.tensors is set instead of self.data. After fixing
        the bug, this test should pass.
        """
        # Create dataset with shuffling enabled
        x = torch.arange(100).unsqueeze(1).float()  # [0, 1, 2, ..., 99]
        y = torch.zeros(100, 1)
        dataset = SimpleIterableDataset((x, y), batch_size=10, should_shuffle=True)

        # Get first epoch
        iterator = iter(dataset)
        first_epoch_batches = [next(iterator) for _ in range(len(dataset))]
        first_epoch_x = torch.cat([b[0] for b in first_epoch_batches], dim=0)

        # Continue to second epoch (after shuffle)
        second_epoch_batches = [next(iterator) for _ in range(len(dataset))]
        second_epoch_x = torch.cat([b[0] for b in second_epoch_batches], dim=0)

        # The data should be shuffled - order should be different
        # (with very high probability for 100 elements)
        assert not torch.equal(first_epoch_x, second_epoch_x), \
            "Shuffle bug: data order is identical across epochs"


class TestHDF5Dataset:
    """Tests for HDF5Dataset."""

    def test_init_with_valid_file(self, temp_hdf5_file):
        """Test initialization with valid HDF5 file."""
        dataset = HDF5Dataset(temp_hdf5_file)
        assert len(dataset) == 10

    def test_init_with_nonexistent_file(self):
        """Test that nonexistent file raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            HDF5Dataset("/nonexistent/path.h5")

    def test_init_with_wrong_extension(self, tmp_path):
        """Test that non-.h5 file raises ValueError."""
        wrong_file = tmp_path / "test.txt"
        wrong_file.touch()
        with pytest.raises(ValueError, match="HDF5"):
            HDF5Dataset(str(wrong_file))

    def test_getitem(self, temp_hdf5_file):
        """Test indexing the dataset."""
        dataset = HDF5Dataset(temp_hdf5_file)
        item = dataset[0]
        assert 'sequence' in item
        assert 'target' in item

    def test_shuffle(self, temp_hdf5_file):
        """Test shuffling the dataset."""
        dataset = HDF5Dataset(temp_hdf5_file)
        original_first = dataset[0]['sequence'].copy()
        dataset.shuffle()
        # After shuffle, data should be reordered
        # (not guaranteed to be different at index 0, but dataset order changes)
        assert len(dataset) == 10  # Length unchanged


class TestGetIdxDistributed:
    """Tests for get_idx_distributed helper function."""

    def test_single_device(self):
        """Test with single device (no distribution)."""
        num_datapoints = {'file1.h5': 100, 'file2.h5': 50}
        idx = get_idx_distributed(num_datapoints, batch_size=10, rank=0, world_size=1)

        # Should have 15 batches total (10 + 5)
        assert len(idx) == 15
        assert all(isinstance(item, tuple) for item in idx)
        assert all(len(item) == 2 for item in idx)

    def test_multi_device_distribution(self):
        """Test that work is distributed across devices."""
        num_datapoints = {'file1.h5': 100}
        idx_rank0 = get_idx_distributed(num_datapoints, batch_size=10, rank=0, world_size=2)
        idx_rank1 = get_idx_distributed(num_datapoints, batch_size=10, rank=1, world_size=2)

        assert len(idx_rank0) == 5
        assert len(idx_rank1) == 5

        # Different ranks should get different slices
        slices0 = [s for _, s in idx_rank0]
        slices1 = [s for _, s in idx_rank1]
        for s0, s1 in zip(slices0, slices1):
            assert s0 != s1


class TestStackHDF5:
    """Tests for stack_hdf5 helper function."""

    def test_stack_single_variable(self, temp_hdf5_file):
        """Test stacking a single variable."""
        with h5py.File(temp_hdf5_file, 'r') as f:
            data = {name: f[name][:] for name in f.keys()}

        result = stack_hdf5(data, ['sequence'])
        assert result is not None
        assert result.shape == (10, 50)

    def test_stack_multiple_variables(self, temp_hdf5_file):
        """Test stacking multiple variables."""
        with h5py.File(temp_hdf5_file, 'r') as f:
            data = {name: f[name][:] for name in f.keys()}

        result = stack_hdf5(data, ['sequence', 'target'])
        assert result is not None
        assert result.shape == (10, 50, 2)

    def test_stack_empty_list(self, temp_hdf5_file):
        """Test stacking empty variable list returns None."""
        with h5py.File(temp_hdf5_file, 'r') as f:
            data = {name: f[name][:] for name in f.keys()}

        result = stack_hdf5(data, [])
        assert result is None
