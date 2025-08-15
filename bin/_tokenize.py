import xarray as xr
import numpy as np
import sys
from tqdm import tqdm

RNA_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}


def embed(sequence: str, offset: int = 0) -> np.ndarray:
    emb = np.zeros(len(sequence))
    for ix, char in enumerate(sequence):
        emb[ix] = RNA_TO_INT[char] + offset
    return emb

if len(sys.argv) not in [3, 4]:
    print("Usage: rn-coverage tokenize <input_file> <output_file> [offset]")
    sys.exit(1)

input = sys.argv[1]
output = sys.argv[2]

if (len(sys.argv) == 4):
    offset = int(sys.argv[3])
else:
    offset = 0

tokens = []
with open(input, 'r') as f:
    for line in tqdm(f):
        if ">" in line or line.strip() == "":
            continue
        tokens.append(embed(line.strip(), offset))

tokens = np.stack(tokens, axis=0).astype(np.int64)
ds = xr.Dataset({'sequence': (['batch', 'nucleotide'], tokens)})
ds.to_netcdf(output, engine='h5netcdf')
