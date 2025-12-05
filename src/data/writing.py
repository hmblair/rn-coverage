# writing.py

from typing import Any, Optional, Sequence, Callable
import os
from abc import ABCMeta, abstractmethod
import numpy as np
import h5py
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter

class DistributedPredictionWriter(BasePredictionWriter, metaclass=ABCMeta):
    """
    Abstract writer class for saving predictions to file when making predictions
    on a distributed system.

    Parameters
    ----------
    write_interval : str
        Interval at which to write the predictions, either 'batch' or 'epoch'.
        Defaults to 'batch'.

    Attributes
    ----------
    write_interval : str
        Interval at which to write the predictions.
    """
    def __init__(self, write_interval : str = 'batch'):
        super().__init__(write_interval)
        self.model_name = None


    def setup(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            stage : Optional[str] = None,
            ) -> None:
        """
        Sets the model name attribute.
        """
        self.model_name = getattr(pl_module, 'name', pl_module.__class__.__name__)
        return super().setup(trainer, pl_module, stage)


    @abstractmethod
    def write(self, prediction : np.ndarray) -> None:
        """
        Write the prediction to a file or other output medium. An abstract method
        that must be implemented by a subclass.

        Parameters
        ----------
        prediction : np.ndarray
            The prediction to be written.
        """
        return


    def _gather_tensor(self, tensor : torch.Tensor) -> torch.Tensor:
        if torch.distributed.is_initialized():
            gathered_predictions = [
                torch.zeros_like(tensor).contiguous()
                for _ in range(torch.distributed.get_world_size())
                ]
            torch.distributed.all_gather(
                gathered_predictions,
                tensor.contiguous()
                )
        else:
            gathered_predictions = [tensor]

        return gathered_predictions


    def _gather_and_write(
            self,
            trainer : pl.Trainer,
            pl_module : pl.LightningModule,
            prediction : torch.Tensor,
            batch_indices : Sequence[int],
            batch : Any,
            batch_idx : int,
            dataloader_idx : int,
            ) -> None:
        """
        Gathers the predictions from all distributed processes and writes them
        to a file on the root process.
        """
        # gather the predictions from all distributed processes
        gathered_predictions = self._gather_tensor(prediction)
        if trainer.global_rank == 0:
            # get the data name and model name
            data_names = trainer.datamodule.data_names['predict']
            data_name = data_names[dataloader_idx]
            # write the predictions to a file
            prediction = torch.cat(gathered_predictions, dim=0).cpu().numpy()
            self.write(prediction, data_name)


    def write_on_batch_end(self, *args, **kwargs) -> None:
        """
        Writes the output to a file at the end of a batch.
        """
        self._gather_and_write(*args, **kwargs)


    def write_on_epoch_end(self, *args, **kwargs) -> None:
        """
        Writes the output to a file at the end of the epoch.
        """
        self._gather_and_write(*args, **kwargs)



class HDF5PredictionWriter(DistributedPredictionWriter):
    """
    A writer class for saving predictions to an HDF5 file when making
    predictions on a distributed system.

    Parameters
    ----------
    path : str
        The directory to save the predictions to.
    variable_name : str
        The name of the HDF5 dataset to save the predictions to.
    """
    def __init__(
            self,
            path : str,
            variable_name : str = 'predictions',
            *args, **kwargs,
            ) -> None:
        super().__init__(*args, **kwargs)
        print('Initialising writer...')
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.variable_name = variable_name


    def write(
            self,
            prediction : np.ndarray,
            data_name : str,
            ) -> None:
        """
        Writes the output to an HDF5 file. Appends to existing file if present.

        Parameters
        ----------
        prediction : np.ndarray
            The model output.
        data_name : str
            The name of the data, used to name the HDF5 file.
        """
        # get the directory to save the HDF5 file to, and create it if it
        # does not exist.
        dir = self.path
        if not os.path.exists(dir):
            os.mkdir(dir)

        # get the path to the HDF5 file
        file = os.path.join(dir, f'{data_name}.h5')

        if not os.path.exists(file):
            # Create new file with resizable dataset
            with h5py.File(file, 'w') as f:
                # maxshape=None allows unlimited resizing on first axis
                maxshape = (None,) + prediction.shape[1:]
                f.create_dataset(
                    self.variable_name,
                    data=prediction,
                    maxshape=maxshape,
                    chunks=True,
                )
        else:
            # Append to existing file
            with h5py.File(file, 'a') as f:
                dataset = f[self.variable_name]
                current_size = dataset.shape[0]
                new_size = current_size + prediction.shape[0]

                # Resize dataset to accommodate new data
                dataset.resize(new_size, axis=0)

                # Write new data at the end
                dataset[current_size:new_size] = prediction
