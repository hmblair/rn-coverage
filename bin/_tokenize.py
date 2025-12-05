import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
from tqdm import tqdm

from src.data.constants import RNA_TO_INT


if len(sys.argv) not in [3, 4]:
    print("Usage: rn-coverage tokenize <input_file> <output.h5> [offset]")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

if len(sys.argv) == 4:
    offset = int(sys.argv[3])
else:
    offset = 0

# First pass: count sequences and determine sequence length
num_sequences = 0
seq_length = None
with open(input_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        num_sequences += 1
        if seq_length is None:
            seq_length = len(line)

tokens = np.zeros((num_sequences, seq_length), dtype=np.int64)

# Second pass: tokenize sequences
with open(input_path, 'r') as f:
    ix = 0
    for line in tqdm(f, total=num_sequences):
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        for jx, char in enumerate(line):
            tokens[ix, jx] = RNA_TO_INT[char] + offset
        ix += 1

with h5py.File(output_path, 'w') as f:
    f.create_dataset('sequence', data=tokens)
