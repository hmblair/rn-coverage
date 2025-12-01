# constants.py

NC_EXTENSION = '.nc'

# Defaults for xarray/netCDF handling
DEFAULT_BATCH_DIMENSION = 'batch'
DEFAULT_NETCDF_ENGINE = 'h5netcdf'

# RNA/DNA tokenization mapping
# Maps nucleotide characters to integer indices for model input
# T (DNA) maps to same value as U (RNA) for compatibility
RNA_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3}

# Valid phases for data pipeline
VALID_PHASES = frozenset(['train', 'validate', 'test', 'predict'])