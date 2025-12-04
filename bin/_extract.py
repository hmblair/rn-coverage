import sys
import numpy as np
import xarray as xr

if len(sys.argv) != 3:
    print("Usage: rn-coverage extract <input.nc> <output.txt>")
    print()
    print("Extracts predicted read counts from a NetCDF prediction file.")
    print("Computes the mean across experiment types (2A3, DMS) and writes")
    print("one value per line to the output file.")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

# Load predictions
ds = xr.load_dataset(input_path)

if 'reads' not in ds:
    print(f"Error: '{input_path}' does not contain a 'reads' variable")
    sys.exit(1)

# Get reads and compute mean across experiment types
reads = ds['reads'].values  # shape: (batch, experiment_type)
mean_reads = np.mean(reads, axis=1)

# Write to text file
with open(output_path, 'w') as f:
    for val in mean_reads:
        f.write(f"{val:.6f}\n")

print(f"Extracted {len(mean_reads)} read predictions to {output_path}")
