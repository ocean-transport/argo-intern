# PROCESSING FUNCTIONS

import xarray as xr
import logging
import warnings
import getpass
import numpy as np
import pandas as pd
import argopy
import gsw

from argopy import DataFetcher as ArgoDataFetcher
argo_loader = ArgoDataFetcher(src="gdac", ftp="/swot/SUM05/dbalwada/Argo_sync", progress=False)




def get_box(box, standard_grid=np.arange(0,2002,2)):
    """Takes latitude/longitude/depth data and a sample rate and returns an xarray with CT, SA, SIG0, and SPICE interpolated to a pressure grid of 2m.

    box: lat/lon in the form: box=[lon_min, lon_max, lat_min, lat_max, depth_min, depth_max]
    sample_min: minimum sample rate [m]
    """

    ds = argo_loader.region(box)
    print("loading points complete")

    ds = ds.to_xarray()
    print("to xarray complete")

    ds = ds.argo.teos10(["CT", "SA", "SIG0"])
    ds = ds.argo.point2profile()
    print("point to profile complete")

    ds_interp = get_ds_interp(ds, standard_grid)
    print("interpolation complete")

    ds_interp["SPICE"] = gsw.spiciness0(ds_interp.SA, ds_interp.CT).rename("SPICE")
    print("adding spice complete")

    #ds_interp = get_MLD(ds_interp)
    #ds_interp = add_times(ds_interp)
    #print("adding MLD complete")
    
    if 'raw_attrs' in ds_interp.attrs:
        del ds_interp.attrs['raw_attrs']

    return ds_interp




def get_ds_interp(ds, standard_grid):
    """
    NEW VERSION: NEED TO INCLUDE DOCUMENTATION!!!!!!!
    """
    
    print('NEW INTERP FUNCTION')
    profs_interp = []
    interp_step = standard_grid[1] - standard_grid[0]
    
    for n in range(0, len(ds.N_PROF)):
        prof = ds.isel(N_PROF=n).expand_dims('N_PROF')
        depth_min = int(prof.PRES.min())
        depth_min = np.ceil(depth_min / 2) * 2  # Ensure depth_min is rounded to the nearest even number
        depth_max = int(prof.PRES.max())
        depth_max = (depth_max // 2) * 2  # Ensure depth_max is rounded to the nearest even number

        # Validate standard_grid and skip the profile if invalid
        if not (np.all(np.diff(standard_grid) > 0) and np.all(standard_grid >= 0)):
            print(f"\tProfile {n} skipped due to invalid standard_grid values.")
            continue

        if depth_max > depth_min:
            dp = prof.PRES.diff('N_LEVELS')
            prof['sample_rate'] = dp
            
            try:
                prof_interp = prof.argo.interp_std_levels(np.arange(depth_min, depth_max, interp_step))
                prof_interp_reindexed = prof_interp.reindex({'PRES_INTERPOLATED': standard_grid}, method=None, fill_value=np.nan)
                profs_interp.append(prof_interp_reindexed)
            except ValueError as e:
                print(f"\tProfile {n} skipped due to interpolation error: {e}")
        
        elif depth_max > prof.PRES.max():
            print(f"\tProfile {n} has depth_max of {depth_max} but max PRES is {prof.PRES.max()}")
            
        elif depth_max <= depth_min:
            print(f"\tProfile {n} has invalid depth range: depth_min={depth_min}, depth_max={depth_max}")

    # Concatenate valid profiles
    concat_n_prof = xr.concat(profs_interp, dim='N_PROF') if profs_interp else xr.Dataset()
    
    return concat_n_prof





"""
def get_ds_interp(ds, standard_grid):
    '''
    NEW VERSION: NEED TO INCLUDE DOCUMENTATION!!!!!!!
    '''
    
    print('NEW INTERP FUNCTION')
    profs_interp = []
    interp_step = standard_grid[1]-standard_grid[0]
    
    for n in range(0, len(ds.N_PROF)):
        prof = ds.isel(N_PROF=n).expand_dims('N_PROF')
        depth_min = int(prof.PRES.min())
        depth_min = np.ceil(depth_min/2) *2 #double-check this makes sense
        depth_max = int(prof.PRES.max())
        depth_max = (depth_max // 2) * 2

        if depth_max > depth_min:
            dp = prof.PRES.diff('N_LEVELS')
            prof['sample_rate'] = dp
            
            prof_interp = prof.argo.interp_std_levels(np.arange(depth_min, depth_max, interp_step))
            prof_interp_reindexed = prof_interp.reindex({'PRES_INTERPOLATED': standard_grid}, method=None, fill_value=np.nan)
            profs_interp.append(prof_interp_reindexed)
        
        elif depth_max > prof.PRES.max():
            print(f"\tProfile {n} has depth_max of {depth_max} but max PRES is {prof.PRES.max()}")
        
        elif depth_max <= depth_min:
            print(f"\tProfile {n} has invalid depth range: depth_min={depth_min}, depth_max={depth_max}")

    concat_n_prof = xr.concat(profs_interp, dim='N_PROF')
    
    return concat_n_prof
"""




def get_float(float_ID, sample_min):
    """Takes a float ID and sample rate and returns an xarray with CT, SA, SIG0, and SPICE interpolated to a pressure grid of 2m.

    float_ID:   loads argo data from a float, based on the ID provided
    sample_min: minimum sample rate [m]
    """

    ds = argo_loader.float(float_ID)
    print("loading points complete")

    ds = ds.to_xarray()
    print("to xarray complete")

    ds = ds.argo.teos10(["CT", "SA", "SIG0"])
    ds = ds.argo.point2profile()
    print("point to profile complete")

    ds_interp = get_ds_interp(ds, 0, 2000, sample_min)
    print("interpolation complete")

    ds_interp["SPICE"] = gsw.spiciness0(ds_interp.SA, ds_interp.CT).rename("SPICE")
    print("adding spice complete")

    return ds_interp


"""
def get_ds_interp(ds, depth_min, depth_max, interp_step):
    Takes an argopy loaded xarray with sampled pressure and calculates the sampling rate, adds it as a variable, then interpolates to a standard pressure grid of size interp_step.

    ds: xarray dataset with dimensions N_LEVELS and N_PROF
    depth_min: shallowest depth for pressure grid (m)
    depth_max: deepest depth for pressure grid (m)
    interp_step: distance between pressure values for interpolated grid

    print('USING THIS DS_INTERP')
    dp = ds.PRES.diff("N_LEVELS").sortby("N_PROF")
    ds["sample_rate"] = dp
    ds_interp = ds.argo.interp_std_levels(np.arange(depth_min, depth_max, interp_step))

    number = np.arange(0, len(ds_interp.N_PROF))
    ds_interp.coords["N_PROF_NEW"] = xr.DataArray(number, dims=ds_interp.N_PROF.dims)
    
    return ds_interp
"""


"""
def get_ds_interp(ds,depth_min,depth_max,sample_rate):
    
    Takes an Argo xarray with sampled pressure and:
    1) only selects profiles that sample at a rate equal to or greater than sample_rate
    2) interpolates the pressure to a 2m grid.
    3) returns an xarray with all profiles that meet the sample rate interpolated at 2m, with a new dimension PRES_INTERPOLATED
    
    ds: xarray dataset with dimensions PRES, N_LEVELS, N_PROF; pressure PRES
    depth_min: shallowest depth selected[m]
    depth_max: deepest depth selected [m]
    sample_rate: minimum sample rate [m]
    
    median_dp=ds.PRES.where(ds.PRES<depth_max).where(ds.PRES>depth_min).diff('N_LEVELS').median('N_LEVELS')
    ind_rate=median_dp.where(median_dp<sample_rate,drop=True).N_PROF
    ds_sel=ds.sel(N_PROF=ind_rate)
    ds_interp=ds_sel.argo.interp_std_levels(np.arange(depth_min,depth_max,2))
    ds_interp=ds_interp.sortby(ds_interp.N_PROF)
    
    number=np.arange(0,len(ds_interp.N_PROF))
    ds_interp.coords['N_PROF_NEW']=xr.DataArray(number,dims=ds_interp.N_PROF.dims)
    return ds_interp
"""


def add_times(ds, variable="TIME"):
    """Takes an xarray and returns new coordinates for the whole and fractional month and year of each profile. (For example, May 10 would have month=5 and frac_month=5+(10/31). Although this function also takes into account fractional seconds, minutes, and hours in the same manor. Fractional year is calculated in the same way.)
    ds: xarray with time variable
    variable: time variable that can be used with xr.dt, default='TIME'
    """

    frac_day = (
        ds.TIME.dt.day
        + (ds.TIME.dt.hour / 24)
        + (ds.TIME.dt.minute / (24 * 60))
        + (ds.TIME.dt.minute / (24 * 60 * 60))
    )
    frac_month = ds.TIME.dt.month + (frac_day / ds.TIME.dt.days_in_month)
    frac_year = ds.TIME.dt.year + (frac_month / 12)

    month_li = []
    for i in range(0, len(ds.N_PROF)):
        month_li.append(ds.isel(N_PROF=i).TIME.dt.month)

    year_li = []
    for i in range(0, len(ds.N_PROF)):
        year_li.append(ds.isel(N_PROF=i).TIME.dt.year)

    #ds = ds.assign_coords(month=("N_PROF", month_li))
    ds = ds.assign_coords(month_frac=("N_PROF", frac_month.data))
    #ds = ds.assign_coords(year=("N_PROF", year_li))
    ds = ds.assign_coords(year_frac=("N_PROF", frac_year.data))

    return ds


def get_MLD(
    ds, threshold=0.03, variable="SIG0", dim1="N_PROF", dim2="PRES_INTERPOLATED"
):
    """Takes an xarray and returns a new coordinate "MLD" or mixed layer depth for each profile, defined using the density threshold from the surface.
    ds: xarray with profile and pressure dimensions
    threshold: density value that defines the boundary of the mixed layer, default=0.03
    variable: density coordinate, default='SIG0'
    dim1: profile dimension, default='N_PROF'
    dim2: pressure dimension, default='PRES_INTERPOLATED'
    """

    MLD_li = []

    for n in range(0, len(ds[dim1])):
        SIG0_surface = ds.isel({dim1: n})[variable].isel({dim2: 0})
        SIG0_diff = SIG0_surface + threshold
        MLD_ds = SIG0_surface.where(ds.isel({dim1: n})[variable] < SIG0_diff)
        MLD = MLD_ds.dropna(dim2).isel({dim2: -1})[dim2].values
        MLD_li.append(MLD)

    return ds.assign_coords(MLD=(dim1, MLD_li))
