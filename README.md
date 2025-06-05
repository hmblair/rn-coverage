# Overview

Inference for the `rn-filter` model described in the paper \[WIP\]. Used for filtering DNA/RNA sequences prior to MaP-seq experiments based on their predicted read counts, as well as for more advanced cases such as sub-pooling and barcode rebalancing.

# Installation

First, clone the repository and install the dependencies.
```
git clone https://github.com/hmblair/rn-filter
cd rn-filter
pip3 install -r requirements.txt
```
Don't forget to add the `./bin` directory to your path.

Then, the checkpoint must be downloaded.
```
mkdir -p checkpoints
cd checkpoints
curl -L -o RibonanzaNet-Filter.pt "https://www.dropbox.com/scl/fi/m539j9s7ylzdx95obkryh/RibonanzaNet-Filter.pt?rlkey=t1j2igmo2y1n3912wk7wetql4&st=tofsfmm6&dl=0"
```

# Usage

Making predictions with `rn-filter` requires tokenizing the sequences of interest and then calling the model. The former can be done with the command
```
rn-filter tokenize examples/test.fasta examples/test.nc
```
Both text and FASTA files are accepted. The latter is executed via a config file, which must be pointed to the tokenized sequences (under data.paths.predict).
```
rn-filter examples/config.yaml
```
The output `predictions/test.nc` will contain a single $`n x 2`$ dataset `reads`, inside which are the predicted reads for 2A3 and DMS experiments. This `.nc` file can be opened with `xarray`.
