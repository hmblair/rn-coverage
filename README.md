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

## Fine-tuning

Fine-tuning is done via the `train` subcommand:
```
rn-coverage train config.yaml
```

### Data Format

Training data should be HDF5 files (`.h5`) containing:
- `sequence`: tokenized RNA sequences (integers 0-3 for A, C, G, U/T)
- Target variables (e.g., `2A3`, `DMS`): the values to predict

Which datasets to use is configured via `input_variables` and `target_variables`:
```yaml
data:
  class_path: src.data.datamodules.HDF5DataModule
  init_args:
    input_variables:
      - [[sequence], x]           # read 'sequence' dataset, output as 'x'
    target_variables:
      - [[2A3, DMS], target]      # stack '2A3' and 'DMS', output as 'target'
    paths:
      train:
        - data/train.h5
      validate:
        - data/val.h5
```

### Example Configuration

See `examples/training/config.yaml` for a complete example. Key settings include:
- `trainer.max_epochs`: number of training epochs
- `trainer.callbacks`: checkpointing, early stopping, layer unfreezing schedule
- `optimizer.lr`: learning rate
- `data.init_args.batch_size`: batch size

The `FineTuningScheduler` callback progressively unfreezes RibonanzaNet layers during training, which can improve fine-tuning stability.

## CLI Reference

The CLI is built on [PyTorch Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html). Available subcommands:

| Command | Description |
|---------|-------------|
| `rn-coverage train <config>` | Fine-tune the model |
| `rn-coverage predict <input.h5>` | Run inference (simple mode) |
| `rn-coverage predict --config <config>` | Run inference (config mode) |
| `rn-coverage tokenize <input> <output.h5>` | Tokenize sequences |

For advanced usage, you can also invoke the CLI directly:
```
python -m src fit --config config.yaml      # training
python -m src validate --config config.yaml # validation only
python -m src predict --config config.yaml  # inference
```

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
