import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
import matplotlib.colors as colors
from xhistogram.xarray import histogram
import seaborn as sns
import seaborn
import pandas as pd
import numpy as np
from importlib import reload
import cartopy.crs as ccrs
import cmocean.cm as cmo
import gsw
import scipy.ndimage as filter
from flox.xarray import xarray_reduce
print('Completed importing packages (1/7)')

import sys
sys.path.append('/home.ufs/amf2288/argo-intern/funcs')
import density_funcs as df
import filt_funcs as ff
print('Completed local imports (2/7)')

lon_bins = np.arange(-180,181,3)
lat_bins = np.arange(-67.5,68,3)
rho_grid = np.arange(21,29,0.0025)

def get_ds_binned(ds, lon_bins, lat_bins):
    
    ds_binned = xarray_reduce(
    ds,
    'LONGITUDE',
    'LATITUDE',
    func='mean',
    expected_groups=(
        pd.IntervalIndex.from_breaks(lon_bins),
        pd.IntervalIndex.from_breaks(lat_bins)
    ),
    fill_value=np.nan,
    skipna=True)
    
    lon_l = np.arange(lon_bins[0],lon_bins[-1],3)
    lon_r = np.arange(lon_bins[1],lon_bins[-1]+1,3)
    lon_c = (lon_l + lon_r)/2

    lat_l = np.arange(lat_bins[0],lat_bins[-1],3)
    lat_r = np.arange(lat_bins[1],lat_bins[-1]+1,3)
    lat_c = (lat_l + lat_r)/2

    ds_binned = ds_binned.assign_coords({
        "lat_c": ("LATITUDE_bins", lat_c),
        "lat_l": ("LATITUDE_bins", lat_l),
        "lat_r": ("LATITUDE_bins", lat_r),
        "lon_c": ("LONGITUDE_bins", lon_c),
        "lon_l": ("LONGITUDE_bins", lon_l),
        "lon_r": ("LONGITUDE_bins", lon_r),
    })
    
    ds_binned = ds_binned.swap_dims({"LATITUDE_bins": "lat_c", "LONGITUDE_bins": "lon_c"})
    ds_binned = ds_binned.drop_vars(["LATITUDE_bins", "LONGITUDE_bins"])
    
    return ds_binned

ds = xr.open_dataset('/swot/SUM05/amf2288/sync-boxes/ds_high_res_1.nc')
print('Completed loading ds (3/7)')

ds_rho = df.interpolate2density_prof(ds, rho_grid)
ds_rho.to_netcdf('/swot/SUM05/amf2288/sync-boxes/ds_rho_3.nc')
print('Completed interp to density (4/7)')

ds_rho_binned = get_ds_binned(ds_rho, lon_bins, lat_bins)
ds_rho_binned.to_netcdf('/swot/SUM05/amf2288/sync-boxes/ds_rho_binned_3.nc')
print('Completed binning (5/7)')

dCT_dx = ds_rho.CT.differentiate('lon_c').compute()
dSA_dx = ds_rho.SA.differentiate('lon_c').compute()
dSP_dx = ds_rho.SPICE.differentiate('lon_c').compute()
dCT_dx.to_netcdf('/swot/SUM05/amf2288/grad-boxes/dCT_dx_3.nc')
dSA_dx.to_netcdf('/swot/SUM05/amf2288/grad-boxes/dSA_dx_3.nc')
dSP_dx.to_netcdf('/swot/SUM05/amf2288/grad-boxes/dSP_dx_3.nc')
print('Completed dC_dx (6/7)')

dCT_dy = ds_rho.CT.differentiate('lat_c').compute()
dSA_dy = ds_rho.SA.differentiate('lat_c').compute()
dSP_dy = ds_rho.SPICE.differentiate('lat_c').compute()
dCT_dy.to_netcdf('/swot/SUM05/amf2288/grad-boxes/dCT_dy_3.nc')
dSA_dy.to_netcdf('/swot/SUM05/amf2288/grad-boxes/dSA_dy_3.nc')
dSP_dy.to_netcdf('/swot/SUM05/amf2288/grad-boxes/dSP_dy_3.nc')
print('Completed dC_dy (7/7)')