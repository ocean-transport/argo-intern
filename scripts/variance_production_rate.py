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
print('Completed importing packages (1/11)')

import sys
sys.path.append('/home.ufs/amf2288/argo-intern/funcs')
import density_funcs as df
import filt_funcs as ff
print('Completed local imports (2/11)')



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

"""
#load diffusivities
ds_binned = xr.open_dataset('/swot/SUM05/amf2288/sync-boxes/globe_binned_3_z.nc', chunks={'z_c':20})
ds_diff = xr.open_dataset('/swot/SUM05/amf2288/global_diff.nc')
diff = ds_diff.interp(z_c=ds_binned.z_c)
print('Completed loading ds_diff (3/11)')

ds_binned['K_rho'] = (['z_c', 'lon_c', 'lat_c'], diff.K.values)
ds_binned['e']     = (['z_c', 'lon_c', 'lat_c'], diff.e.values)
diff_rho = df.K_rho_interp_binned(ds_binned, rho_grid)
diff_rho.to_netcdf('/swot/SUM05/amf2288/diff_rho_3.nc')
"""
diff_rho = xr.open_dataset('/swot/SUM05/amf2288/diff_rho_3.nc')
K_rho = diff_rho.K_rho
print('Completed loading K_rho (4/11)')

"""
#load global dataset
ds = xr.open_dataset('/swot/SUM05/amf2288/sync-boxes/new_test.nc', chunks={'N_PROF':10000})
ds = ds.assign_attrs({"Fetched_uri":''})
unique_prof = np.arange(len(ds['N_PROF']))
ds = ds.assign_coords(N_PROF=('N_PROF', unique_prof))
print('Completed loading ds (5/11)')

sample_max = 2.5
ds = ds.sortby('LATITUDE').persist()
boolean_indexer = (ds.sample_rate < sample_max).compute()
ds = ds.where(boolean_indexer, drop=True)
ds.to_netcdf('/swot/SUM05/amf2288/sync-boxes/high_res.nc')
print('Completed loading ds (6/11)')
"""

ds = xr.open_dataset('/swot/SUM05/amf2288/sync-boxes/ds_high_res_1.nc')
print('Completed loading ds (5/11)')

#compute variance metrics
lfilt = 100

"""
ct_m = ff.ds_filt_single(ds, lfilt, 'CT')
sa_m = ff.ds_filt_single(ds, lfilt, 'SA')
sp_m = ff.ds_filt_single(ds, lfilt, 'SPICE')

ct_m.to_netcdf('/swot/SUM05/amf2288/var-boxes/ct_m_3.nc')
sa_m.to_netcdf('/swot/SUM05/amf2288/var-boxes/sa_m_3.nc')
sp_m.to_netcdf('/swot/SUM05/amf2288/var-boxes/sp_m_3.nc')
print('Completed c_m (7/11)')

ct_e = ds.CT - ct_m
sa_e = ds.SA - sa_m
sp_e = ds.SPICE - sp_m

ct_e.to_netcdf('/swot/SUM05/amf2288/var-boxes/ct_e_3.nc')
sa_e.to_netcdf('/swot/SUM05/amf2288/var-boxes/sa_e_3.nc')
sp_e.to_netcdf('/swot/SUM05/amf2288/var-boxes/sp_e_3.nc')
print('Completed c_e (8/11)')
"""


ct_m = xr.open_dataarray('/swot/SUM05/amf2288/var-boxes/ct_m_3.nc')
sa_m = xr.open_dataarray('/swot/SUM05/amf2288/var-boxes/sa_m_3.nc')
sp_m = xr.open_dataarray('/swot/SUM05/amf2288/var-boxes/sp_m_3.nc')

ct_e = xr.open_dataarray('/swot/SUM05/amf2288/var-boxes/ct_e_3.nc')
sa_e = xr.open_dataarray('/swot/SUM05/amf2288/var-boxes/sa_e_3.nc')
sp_e = xr.open_dataarray('/swot/SUM05/amf2288/var-boxes/sp_e_3.nc')
print('Completed loading c_m, c_e (6/11)')

"""
ct2_inside = (ff.da_filt_single(ct_e.differentiate(coord='PRES_INTERPOLATED'), lfilt))**2
sa2_inside = (ff.da_filt_single(sa_e.differentiate(coord='PRES_INTERPOLATED'), lfilt))**2
sp2_inside = (ff.da_filt_single(sp_e.differentiate(coord='PRES_INTERPOLATED'), lfilt))**2
ct3_inside = ct_m.differentiate(coord='PRES_INTERPOLATED')**2
sa3_inside = sa_m.differentiate(coord='PRES_INTERPOLATED')**2
sp3_inside = sp_m.differentiate(coord='PRES_INTERPOLATED')**2

ds['ct2_inside'] = (['N_PROF', 'PRES_INTERPOLATED'], ct2_inside.values)
ds['sa2_inside'] = (['N_PROF', 'PRES_INTERPOLATED'], sa2_inside.values)
ds['sp2_inside'] = (['N_PROF', 'PRES_INTERPOLATED'], sp2_inside.values)
ds['ct3_inside'] = (['N_PROF', 'PRES_INTERPOLATED'], ct3_inside.values)
ds['sa3_inside'] = (['N_PROF', 'PRES_INTERPOLATED'], sa3_inside.values)
ds['sp3_inside'] = (['N_PROF', 'PRES_INTERPOLATED'], sp3_inside.values)
ds.to_netcdf('/swot/SUM05/amf2288/var-boxes/var_test_3.nc')
print('Completed calculating insides (7/11)')
"""

ds = xr.open_dataset('/swot/SUM05/amf2288/var-boxes/var_test_3.nc')
print('Completed loading insides (7/11)')

ct2 = K_rho*get_ds_binned(df.interpolate2density_var(ds, rho_grid).ct2_inside, lon_bins, lat_bins)
sa2 = K_rho*get_ds_binned(df.interpolate2density_var(ds, rho_grid).sa2_inside, lon_bins, lat_bins)
sp2 = K_rho*get_ds_binned(df.interpolate2density_var(ds, rho_grid).sp2_inside, lon_bins, lat_bins)
print('Completed c2 (8/11)')

ct3 = K_rho*get_ds_binned(df.interpolate2density_prof(ds, rho_grid).ct3_inside, lon_bins, lat_bins)
sa3 = K_rho*get_ds_binned(df.interpolate2density_prof(ds, rho_grid).sa3_inside, lon_bins, lat_bins)
sp3 = K_rho*get_ds_binned(df.interpolate2density_prof(ds, rho_grid).sp3_inside, lon_bins, lat_bins)
print('Completed c3 (9/11)')

ct2.to_netcdf('/swot/SUM05/amf2288/var-boxes/ct2_3.nc')
sa2.to_netcdf('/swot/SUM05/amf2288/var-boxes/sa2_3.nc')
sp2.to_netcdf('/swot/SUM05/amf2288/var-boxes/sp2_3.nc')
ct3.to_netcdf('/swot/SUM05/amf2288/var-boxes/ct3_3.nc')
sa3.to_netcdf('/swot/SUM05/amf2288/var-boxes/sa3_3.nc')
sp3.to_netcdf('/swot/SUM05/amf2288/var-boxes/sp3_3.nc')
print('Completed saving c2, c3 (10/11)')


ct_tot  = ct2 + ct3
ct2_rat = ct2/ct_tot
ct3_rat = ct3/ct_tot

sa_tot  = sa2 + sa3
sa2_rat = sa2/sa_tot
sa3_rat = sa3/sa_tot

sp_tot  = sp2 + sp3
sp2_rat = sp2/sp_tot
sp3_rat = sp3/sp_tot

ct2_rat.to_netcdf('/swot/SUM05/amf2288/var-boxes/ct2_rat.nc')
ct3_rat.to_netcdf('/swot/SUM05/amf2288/var-boxes/ct3_rat.nc')
sa2_rat.to_netcdf('/swot/SUM05/amf2288/var-boxes/sa2_rat.nc')
sa3_rat.to_netcdf('/swot/SUM05/amf2288/var-boxes/sa3_rat.nc')
sp2_rat.to_netcdf('/swot/SUM05/amf2288/var-boxes/sp2_rat.nc')
sp3_rat.to_netcdf('/swot/SUM05/amf2288/var-boxes/sp3_rat.nc')
print('Completed saving ratios (11/11)')
