import xarray as xr

for name in ['train', 'val']:

    file = f"{name}-onemil1.nc"
    ds = xr.load_dataset(file)

    ds['sequence'] = ds['sequence_embeddings']
    del ds['sequence_embeddings']

    ds['2A3'] = ds['reads_2A3']
    del ds['reads_2A3']

    ds['DMS'] = ds['reads_DMS']
    del ds['reads_DMS']

    ds.to_netcdf(f"{name}-onemil1-stripped.nc", engine="h5netcdf")
