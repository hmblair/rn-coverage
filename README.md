# Overview

Inference for the `rn-coverage` model described in the paper \[WIP\]. Used for filtering DNA/RNA sequences prior to MaP-seq experiments based on their predicted read counts, as well as for more advanced cases such as sub-pooling and barcode rebalancing. An example of using `rn-coverage` for barcode rebalancing can be found at [this link](https://drive.google.com/drive/folders/1su8oOGtnxpzIJm9vHg5tydZrnm9gQwJs?usp=drive_link).

# Installation

First, clone the repository and install the dependencies.
```
git clone https://github.com/hmblair/rn-coverage
cd rn-coverage
pip3 install -r requirements.txt
```
Don't forget to add the `./bin` directory to your path.

Then, the checkpoint must be downloaded.
```
mkdir -p checkpoints
cd checkpoints
curl -L -o rn-coverage.pt "https://www.dropbox.com/scl/fi/m539j9s7ylzdx95obkryh/RibonanzaNet-Filter.pt?rlkey=t1j2igmo2y1n3912wk7wetql4&st=tofsfmm6&dl=0"
```

# Usage

Making predictions with `rn-converage` requires tokenizing the sequences of interest and then calling the model. The former can be done with the command
```
rn-converage tokenize examples/data/test.fasta examples/data/test.nc
```
Both text and FASTA files are accepted. The latter is executed via a config file, which must be pointed to the tokenized sequences (under data.paths.predict).
```
rn-converage examples/config.yaml
```
The output `predictions/test.nc` will contain a single $`n \times 2`$ dataset `reads`, inside which are the predicted reads for 2A3 and DMS experiments. This `.nc` file can be opened with `xarray`.

See `examples` for a MWE (you will need to point the config file to the downloaded checkpoint for it to work.)

