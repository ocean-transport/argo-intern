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
print('Completed importing packages (/)')

import os
os.chdir('/home.ufs/amf2288/argo-intern/funcs')
import density_funcs as df
import filt_funcs as ff


lon_bins = np.arange(-180,181,3)
lat_bins = np.arange(-67.5,68,3)

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

#load global dataset
ds = xr.open_dataset('/swot/SUM05/amf2288/sync-boxes/new_test.nc', chunks={'N_PROF':10000})

ds = ds.assign_attrs({"Fetched_uri":''})
unique_prof = np.arange(len(ds['N_PROF']))
ds = ds.assign_coords(N_PROF=('N_PROF', unique_prof))
print('Completed loading ds (/)')

sample_max = 2.5
ds = ds.sortby('LATITUDE').persist()
boolean_indexer = (ds.sample_rate < sample_max).compute()
ds = ds.where(boolean_indexer, drop=True)
print('Completed selecting high res profiles (/)')

#load diffusivities
ds_binned = xr.open_dataset('/swot/SUM05/amf2288/sync-boxes/globe_binned_3_z.nc', chunks={'z_c':20})
ds_diff = xr.open_dataset('/swot/SUM05/amf2288/global_diff.nc')
K_rho = ds_diff.K.interp(z_c=ds_binned.z_c)
print('Completed loading K_rho (/)')

K_rho_rho = df.density_interp_binned(K_rho, rho_grid)
print('Completed K_rho to density interp (/)')

#compute variance metrics
lfilt = 100

ct_m = ds_filt_single(ds, lfilt, 'CT')
sa_m = ds_filt_single(ds, lfilt, 'SA')
sp_m = ds_filt_single(ds, lfilt, 'SPICE')
print('Completed c_m (/)')

ct_e = ds.CT - ct_m
sa_e = ds.SA - sa_m
sp_e = ds.SP - sp_m
print('Completed c_e (/)')


ct2_var = get_ds_binned(df.interpolate2density_prof((ff.da_filt_single(ct_e.differentiate(coord='PRES_INTERPOLATED'), lfilt))**2, rho_grid), lon_bins, lat_bins)
sa2_var = get_ds_binned(df.interpolate2density_prof((ff.da_filt_single(sa_e.differentiate(coord='PRES_INTERPOLATED'), lfilt))**2, rho_grid), lon_bins, lat_bins)
sp2_var = get_ds_binned(df.interpolate2density_prof((ff.da_filt_single(sp_e.differentiate(coord='PRES_INTERPOLATED'), lfilt))**2, rho_grid), lon_bins, lat_bins)

ct2 = K_rho*ct2_var
ct2 = K_rho*ct2_var
ct2 = K_rho*ct2_var
print('Completed c2 (/)')

ct3_var = get_ds_binned(df.interpolate2density_prof(ct_m.differentiate(coord='PRES_INTERPOLATED')**2, rho_grid), lon_bins, lat_bins)
sa3_var = get_ds_binned(df.interpolate2density_prof(sa_m.differentiate(coord='PRES_INTERPOLATED')**2, rho_grid), lon_bins, lat_bins)
sp3_var = get_ds_binned(df.interpolate2density_prof(sp_m.differentiate(coord='PRES_INTERPOLATED')**2, rho_grid), lon_bins, lat_bins)

ct3 = K_rho*ct3_var
sa3 = K_rho*sa3_var
sp3 = K_rho*sp3_var
print('Completed c3 (/)')

ct_tot  = ct2 + ct3
ct2_rat = ct2/ct_tot
ct3_rat = ct3/ct_tot

sa_tot  = sa2 + sa3
sa2_rat = sa2/sa_tot
sa3_rat = sa3/sa_tot

sp_tot  = sp2 + sp3
sp2_rat = sp2/sp_tot
sp3_rat = sp3/sp_tot
print('Completed ratios (/)')

ct2.to_netcdf('/swot/SUM05/amf2288/var-boxes/ct2_3.nc')
ct3.to_netcdf('/swot/SUM05/amf2288/var-boxes/ct3_3.nc')
sa2.to_netcdf('/swot/SUM05/amf2288/var-boxes/sa2_3.nc')
sa3.to_netcdf('/swot/SUM05/amf2288/var-boxes/sa3_3.nc')
sp2.to_netcdf('/swot/SUM05/amf2288/var-boxes/sp2_3.nc')
sp3.to_netcdf('/swot/SUM05/amf2288/var-boxes/sp3_3.nc')

ct2_rat.to_netcdf('/swot/SUM05/amf2288/var-boxes/ct2_rat.nc')
ct3_rat.to_netcdf('/swot/SUM05/amf2288/var-boxes/ct3_rat.nc')
sa2_rat.to_netcdf('/swot/SUM05/amf2288/var-boxes/sa2_rat.nc')
sa3_rat.to_netcdf('/swot/SUM05/amf2288/var-boxes/sa3_rat.nc')
sp2_rat.to_netcdf('/swot/SUM05/amf2288/var-boxes/sp2_rat.nc')
sp3_rat.to_netcdf('/swot/SUM05/amf2288/var-boxes/sp3_rat.nc')

K_rho_rho.to_netcdf('/swot/SUM05/amf2288/K_rho_rho_3.nc')
print('Completed saving results (/)')