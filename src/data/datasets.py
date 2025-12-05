# datasets.py

from __future__ import annotations
import os
from typing import Callable, Union, Sequence, Iterable
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, IterableDataset
from .constants import (
    H5_EXTENSION,
    DEFAULT_BATCH_DIMENSION,
)
from .utils import validate_equal_lengths
import random


class SimpleDataset(Dataset):
    """
    A simple PyTorch dataset, that allows for indexing a sequence of tensors.

    Parameters
    ----------
    data : Sequence
        The data to be loaded.
    """

    def __init__(self, data: Sequence) -> None:
        super().__init__()
        validate_equal_lengths(data, 'All tensors must have the same length.')
        self.data = data

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return len(self.data[0])

    def __getitem__(
            self,
            idx: Union[int, list, slice],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return all tensors at the given index.

        Parameters
        ----------
        idx : int | list | slice
            The index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The input and target tensors.
        """
        return tuple(array[idx] for array in self.data)


class SimpleIterableDataset(IterableDataset):
    """
    A simple PyTorch iterable dataset, that allows for iterating over a sequence
    of tensors in batches, optionally shuffling the data.

    Parameters
    ----------
    data : Sequence
        The tensors to iterate over.
    batch_size : int
        The batch size.
    rank : int
        The rank of the current process.
    world_size : int
        The total number of processes.
    should_shuffle : bool
        Whether to shuffle the data.
    """

    def __init__(
            self,
            data: tuple[torch.Tensor],
            batch_size: int,
            rank: int = 0,
            world_size: int = 1,
            should_shuffle: bool = True,
    ) -> None:
        super().__init__()
        validate_equal_lengths(data, 'All tensors must have the same length.')
        self.data = data
        self.batch_size = batch_size
        self.slices = [
            slice(i, i+batch_size)
            for i in range(0, len(self.data[0]), batch_size)
        ]
        self.slices = self.slices[rank::world_size]
        self.should_shuffle = should_shuffle

    def __iter__(self) -> Iterable[tuple[torch.Tensor]]:
        """
        Return an iterator over the dataset.
        """
        while True:
            for batch_slice in self.slices:
                yield tuple(array[batch_slice] for array in self.data)
            if self.should_shuffle:
                ix = torch.randperm(len(self.data[0]))
                self.data = tuple(array[ix] for array in self.data)

    def __len__(self) -> int:
        """
        Return the length of the dataset.
        """
        return len(self.slices)


def get_idx_distributed(
        num_datapoints: dict[str, int],
        batch_size: int,
        rank: int,
        world_size: int,
) -> list[tuple[str, slice]]:
    """
    Get the indices for an iterable dataset in a distributed setting. The rank
    and world size are used to distribute the dataset across multiple devices.

    Parameters
    ----------
    num_datapoints : dict[str, int]
        Each dataset, and the corresponding number of datapoints in that
        dataset.
    batch_size : int
        The batch size to use.
    rank : int
        The rank of the current device.
    world_size : int
        The number of devices that the dataset will be distributed across.

    Returns
    -------
    list[tuple[str, slice]]
        A list of tuples, where the first element is the name of the dataset,
        and the second element is a slice that can be used to index the dataset.
    """

    # construct slices for each dataset
    idx = [
        (name, slice(i, min(i + batch_size, num)))
        for name, num in num_datapoints.items()
        for i in range(0, num, batch_size)
    ]

    overflow = len(idx) % world_size
    if overflow:
        idx = idx[:-overflow]

    # return the slices for the current rank
    return idx[rank::world_size]


def stack_hdf5(data: dict[str, np.ndarray], variables: list[str]) -> np.ndarray | None:
    """
    Stack variables from a dictionary of numpy arrays into a single array.
    If the list of variables is empty, return None.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Dictionary mapping variable names to numpy arrays.
    variables : list[str]
        List of variable names to stack.

    Returns
    -------
    np.ndarray | None
        The stacked array, or None if the list of variables is empty.
    """

    # if the list of variables is empty, return None
    if not variables:
        return None

    # stack the variables into a numpy array
    stack = np.stack([data[name] for name in variables], axis=-1)

    # if the stack has a single dimension, remove it
    if stack.shape[-1] == 1:
        stack = stack[..., 0]
    return stack


class HDF5Dataset(Dataset):
    """
    A wrapper class for an HDF5 file, allowing for behaviour similar to a
    PyTorch dataset, as well as shuffling the dataset along the batch dimension.

    Parameters
    ----------
    path : str
        The path to the HDF5 file.
    """

    def __init__(
            self: HDF5Dataset,
            path: str,
    ) -> None:

        # verify that the path exists, and points to an HDF5 file
        if not os.path.exists(path):
            raise ValueError(f'The path "{path}" does not exist.')
        if not path.endswith(H5_EXTENSION):
            raise ValueError(
                f'The path "{path}" does not point to an HDF5 file.')

        self.path = path

        # load all data into memory
        with h5py.File(path, 'r') as f:
            self.data = {name: f[name][:] for name in f.keys()}
            # get batch size from first dataset
            first_key = list(f.keys())[0]
            self.num_datapoints = f[first_key].shape[0]

    def __len__(self: HDF5Dataset) -> int:
        """
        Return the number of datapoints in the dataset.
        """
        return self.num_datapoints

    def __getitem__(self: HDF5Dataset, idx: int | slice) -> dict[str, np.ndarray]:
        """
        Return the data at the specified index or slice.
        """
        return {name: arr[idx] for name, arr in self.data.items()}

    def shuffle(self: HDF5Dataset) -> None:
        """
        Shuffle the dataset along the batch dimension.
        """

        # get a random permutation of the indices
        idx = np.random.permutation(self.num_datapoints)

        # shuffle the dataset
        for name in self.data:
            self.data[name] = self.data[name][idx]


class HDF5IterableDataset(IterableDataset):
    """
    A PyTorch iterable dataset that loads data from one or more HDF5 files
    in batches. Each batch is returned as two dictionaries, with
    the first containing any input variables and the second containing any
    target variables.

    Parameters
    ----------
    paths : list[str]
        The paths to the HDF5 files.
    batch_size : int
        The maximum batch size to use.
    input_variables : list[tuple[list[str], str]]
        The names of the variables that will be returned in the first dictionary.
        Each element of the list should be a tuple, where the first element is a
        list of variable names, all of which will be stacked into a single numpy
        array, and the second element is the name of the key in the dictionary
        that will be yielded. Defaults to an empty list.
    target_variables : list[tuple[list[str], str]]
        The names of the variables that will be returned in the second dictionary.
        It has the same structure as input_variables. Defaults to an empty list.
    rank : int
        The rank of the current device. Defaults to 0.
    world_size : int
        The number of devices that this dataset will be distributed across.
        Defaults to 1.
    should_shuffle : bool
        Whether the dataset should be shuffled. Defaults to False.
    transforms : list[Callable]
        List of transform functions to apply. Defaults to an empty list.
    """

    def __init__(
            self: HDF5IterableDataset,
            paths: list[str],
            batch_size: int,
            input_variables: list[tuple[list[str], str]] = [],
            target_variables: list[tuple[list[str], str]] = [],
            rank: int = 0,
            world_size: int = 1,
            should_shuffle: bool = False,
            transforms: list[Callable] = [],
    ) -> None:

        # verify that at least one path is specified
        if not paths:
            raise ValueError('At least one path must be specified.')

        # verify that at least one variable is specified
        if not input_variables + target_variables:
            raise ValueError(
                'At least one input or target variable must be specified.')

        # store the input and target variables
        self.input_variables = input_variables
        self.target_variables = target_variables

        # store whether the dataset should be shuffled
        self.should_shuffle = should_shuffle

        # store the batch size, rank, and world size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size

        # open the datasets
        self.datasets = {
            path: HDF5Dataset(path=path)
            for path in paths
        }

        # get the length of each dataset
        self.num_datapoints = {
            path: dataset.num_datapoints
            for path, dataset in self.datasets.items()
        }

        # get the slices for each dataset
        self.idx = get_idx_distributed(
            num_datapoints=self.num_datapoints,
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
        )

    def __iter__(self: HDF5IterableDataset) -> Iterable[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
        """
        Iterate over the dataset in batches, shuffling the dataset along the
        batch dimension if specified. The input and target variables of each
        batch are stacked into numpy arrays and yielded. If the list of target
        variables is empty, the targets are set to None.
        """

        while True:
            # get the slices for each dataset
            idx = get_idx_distributed(
                num_datapoints=self.num_datapoints,
                batch_size=self.batch_size,
                rank=self.rank,
                world_size=self.world_size,
            )

            # shuffle the datasets if specified
            if self.should_shuffle:
                self.shuffle()
                random.shuffle(idx)

            # loop over the indices
            for path, ix in idx:
                batch = self.datasets[path][ix]
                x = {out_name: stack_hdf5(batch, in_names)
                     for in_names, out_name in self.input_variables}
                y = {out_name: stack_hdf5(batch, in_names)
                     for in_names, out_name in self.target_variables}
                yield x, y

    def __len__(self: HDF5IterableDataset) -> int:
        """
        Return the number of batches in the dataset.
        """
        return len(self.idx)

    def shuffle(self: HDF5IterableDataset) -> None:
        """
        Shuffle each dataset along the batch dimension.
        """
        for dataset in self.datasets.values():
            dataset.shuffle()
