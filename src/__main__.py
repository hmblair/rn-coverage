import os
from pathlib import Path
import urllib.request

import torch
from pytorch_lightning.cli import LightningCLI
from tqdm import tqdm

# Configuration from environment variables with sensible defaults
WANDB_DIR = os.environ.get("WANDB_DIR", "logs/wandb")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "rn-coverage")
MATMUL_PRECISION = os.environ.get("TORCH_MATMUL_PRECISION", "medium")

# Checkpoint configuration
CACHE_DIR = Path.home() / ".cache" / "rn-coverage"
CHECKPOINT_FILENAME = "rn-coverage.pt"
CHECKPOINT_URL = (
    "https://www.dropbox.com/scl/fi/m539j9s7ylzdx95obkryh/RibonanzaNet-Filter.pt"
    "?rlkey=t1j2igmo2y1n3912wk7wetql4&dl=1"
)


def ensure_checkpoint() -> str:
    """
    Ensure the model checkpoint is available, downloading if necessary.

    Returns
    -------
    str
        Path to the checkpoint file.
    """
    # Check if explicitly set via environment variable
    ckpt_path = os.environ.get("RN_COV_CKPT", "")
    if ckpt_path:
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found at RN_COV_CKPT={ckpt_path}")
        return ckpt_path

    # Check cache directory
    cached_path = CACHE_DIR / CHECKPOINT_FILENAME
    if cached_path.exists():
        return str(cached_path)

    # Download to cache
    print(f"Downloading model checkpoint to {cached_path}...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
        if not hasattr(_progress_hook, "pbar"):
            _progress_hook.pbar = tqdm(total=total_size, unit="B", unit_scale=True)
        downloaded = block_num * block_size
        _progress_hook.pbar.update(block_size)
        if downloaded >= total_size:
            _progress_hook.pbar.close()

    urllib.request.urlretrieve(CHECKPOINT_URL, cached_path, reporthook=_progress_hook)
    print("Download complete.")
    return str(cached_path)


# Setup wandb logging directory
os.makedirs(WANDB_DIR, exist_ok=True)
os.environ.setdefault('WANDB_DIR', WANDB_DIR)
os.environ.setdefault('WANDB_PROJECT', WANDB_PROJECT)

# Set matrix multiplication precision for performance
torch.set_float32_matmul_precision(MATMUL_PRECISION)

# Ensure checkpoint is available and set environment variable
CKPT = ensure_checkpoint()
os.environ["RN_COV_CKPT"] = CKPT

# Clear any lingering tqdm instances to prevent duplicate progress bars
# when running in interactive environments
tqdm._instances.clear()

if __name__ == "__main__":
    LightningCLI(save_config_callback=None)
