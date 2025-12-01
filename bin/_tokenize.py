import xarray as xr
import numpy as np
import sys
from tqdm import tqdm

RNA_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}


if len(sys.argv) not in [3, 4]:
    print("Usage: rn-coverage tokenize <input_file> <output_file> [offset]")
    sys.exit(1)

input = sys.argv[1]
output = sys.argv[2]

if (len(sys.argv) == 4):
    offset = int(sys.argv[3])
else:
    offset = 0

with open(input, "r") as f:
    lines = 0
    for line in f:
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        lines += 1

with open(input, "r") as f:
    line = f.readline().strip()
    if line.startswith(">"):
        line = f.readline().strip()
    length = len(line)

tokens = np.zeros((lines, length), dtype=np.int64)

with open(input, 'r') as f:
    ix = 0
    for line in tqdm(f, total=lines):
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        for jx, char in enumerate(line):
            tokens[ix, jx] = RNA_TO_INT[char] + offset
        ix += 1

ds = xr.Dataset({'sequence': (['batch', 'nucleotide'], tokens)})
ds.to_netcdf(output, engine='h5netcdf')
