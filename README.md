# Overview

Inference for the `rn-coverage` model described in the paper \[WIP\]. Used for filtering DNA/RNA sequences prior to MaP-seq experiments based on their predicted read counts, as well as for more advanced cases such as sub-pooling and barcode rebalancing. An example of using `rn-coverage` for barcode rebalancing can be found at [this link](https://drive.google.com/drive/folders/1su8oOGtnxpzIJm9vHg5tydZrnm9gQwJs?usp=drive_link).

# Installation

First, clone the repository
```
git clone https://github.com/hmblair/rn-coverage
cd rn-coverage
```
and install the dependencies however you want (shown here using a `venv`).
```
python3 -m venv ./venv
source venv/bin/activate
pip3 install -r requirements.txt
```
Don't forget to add the `./bin` directory to your path.

The model checkpoint will be downloaded automatically on first run to `~/.cache/rn-coverage/`. Alternatively, set `RN_COV_CKPT` to use a manually downloaded checkpoint.

# Usage

## Tokenization

Making predictions with `rn-coverage` requires tokenizing the sequences of interest. This can be done with the `tokenize` subcommand, for example
```
rn-coverage tokenize test.fasta tokens.h5
```
Both text and FASTA files are accepted.

## Coverage Prediction

Coverage prediction is done via the `predict` subcommand. The simplest usage is:
```
rn-coverage predict tokens.h5
```
This will write predictions to the `predictions/` directory. To specify a different output directory:
```
rn-coverage predict tokens.h5 -o output/
```

For advanced use cases, you can provide a config file:
```
rn-coverage predict --config config.yaml
```
A minimal configuration file is as below. If GPUs are available, they will be used automatically, otherwise it will fall back to running on the CPU.
```yaml
trainer:
  devices: 1
  callbacks:
    - class_path: src.data.writing.HDF5PredictionWriter
      init_args:
        path: predictions

data:
  init_args:
    batch_size: 1
    paths:
      predict:
        - data/tokens.h5
```
The output `predictions/tokens.h5` will contain a single $`n \times 2`$ dataset `reads`, inside which are the predicted reads for 2A3 and DMS experiments. This `.h5` file can be opened with `h5py`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RN_COV_CKPT` | `~/.cache/rn-coverage/rn-coverage.pt` | Path to the pre-trained model checkpoint |
| `WANDB_DIR` | `logs/wandb` | Directory for Weights & Biases logs |
| `WANDB_PROJECT` | `rn-coverage` | Weights & Biases project name |
| `TORCH_MATMUL_PRECISION` | `medium` | PyTorch matrix multiplication precision |

# Development

## Running Tests

```
pytest tests/ -v
```

## Project Structure

```
src/
├── data/           # Data loading and preprocessing
│   ├── constants.py    # Shared constants (tokenization, defaults)
│   ├── datasets.py     # Dataset classes
│   ├── datamodules.py  # PyTorch Lightning data modules
│   └── utils.py        # Utility functions
├── training/       # Training and inference
│   ├── modules.py      # Lightning modules
│   ├── hooks.py        # Training hooks
│   ├── finetuning/     # LoRA and layer unfreezing
│   └── optimisation/   # Loss functions and LR schedulers
├── ribonanzanet.py # RibonanzaNet model (external)
└── __main__.py     # CLI entry point
```
