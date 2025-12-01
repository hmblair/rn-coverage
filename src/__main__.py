import os

import torch
from pytorch_lightning.cli import LightningCLI
from tqdm import tqdm

# Configuration from environment variables with sensible defaults
WANDB_DIR = os.environ.get("WANDB_DIR", "logs/wandb")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "rn-coverage")
MATMUL_PRECISION = os.environ.get("TORCH_MATMUL_PRECISION", "medium")

# Setup wandb logging directory
os.makedirs(WANDB_DIR, exist_ok=True)
os.environ.setdefault('WANDB_DIR', WANDB_DIR)
os.environ.setdefault('WANDB_PROJECT', WANDB_PROJECT)

# Set matrix multiplication precision for performance
torch.set_float32_matmul_precision(MATMUL_PRECISION)

# Require pre-trained checkpoint path
CKPT = os.environ.get("RN_COV_CKPT", "")
if not CKPT:
    raise RuntimeError(
        "The RN_COV_CKPT environment variable must point to the "
        "pre-trained weights."
    )

# Clear any lingering tqdm instances to prevent duplicate progress bars
# when running in interactive environments
tqdm._instances.clear()

if __name__ == "__main__":
    LightningCLI(save_config_callback=None)
