from pytorch_lightning.cli import LightningCLI
from tqdm import tqdm
import os
import torch
import sys

PRECISION = "medium"
torch.set_float32_matmul_precision(PRECISION)

CKPT = os.environ.get("RN_COV_CKPT", "")
if not CKPT:
    raise RuntimeError(
        "The RN_COV_CKPT environment variable must point to the "
        "pre-trained weights."
    )

tqdm._instances.clear()

if __name__ == "__main__":
    LightningCLI(save_config_callback=None)
