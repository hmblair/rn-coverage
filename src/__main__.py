from pytorch_lightning.cli import LightningCLI
from tqdm import tqdm
import os
import torch

PRECISION = "medium"
torch.set_float32_matmul_precision(PRECISION)

tqdm._instances.clear()

if __name__ == "__main__":
    LightningCLI(save_config_callback=None)
