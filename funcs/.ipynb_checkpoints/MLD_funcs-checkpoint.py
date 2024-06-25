# Functions related to mixed layer depth (MLD)

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argopy
import scipy.ndimage as filter
import scipy
import matplotlib
import gsw

def get_MLD(ds,threshold=0.03,variable='SIG0',dim1='N_PROF',dim2='PRES_INTERPOLATED'):
    '''Takes an xarray and returns a new coordinate "MLD" or mixed layer depth for each profile, defined using the density threshold from the surface.
    ds: xarray with profile and pressure dimensions
    threshold: density value that defines the boundary of the mixed layer, default=0.03
    variable: density coordinate, default='SIG0'
    dim1: profile dimension, default='N_PROF'
    dim2: pressure dimension, default='PRES_INTERPOLATED'
    '''
    
    MLD_li = []
    
    for n in range(0, len(ds[dim1])):
        SIG0_surface = ds.isel({dim1:n})[variable].isel({dim2:0})
        SIG0_diff    = SIG0_surface + threshold
        MLD_ds       = SIG0_surface.where(ds.isel({dim1:n})[variable] < SIG0_diff)
        MLD          = MLD_ds.dropna(dim2).isel({dim2:-1})[dim2].values
        MLD_li.append(MLD)
        
    return ds.assign_coords(MLD=(dim1,MLD_li))

def add_times(ds, variable='TIME'):
    '''Takes an xarray and returns new coordinates for the whole and fractional month and year of each profile. (For example, May 10 would have month=5 and frac_month=5+(10/31). Although this function also takes into account fractional seconds, minutes, and hours in the same manor. Fractional year is calculated in the same way.)
    ds: xarray with time variable 
    variable: time variable that can be used with xr.dt, default='TIME'
    '''
    
    frac_day = ds.TIME.dt.day + (ds.TIME.dt.hour / 24) + (ds.TIME.dt.minute / (24*60)) + (ds.TIME.dt.minute / (24*60*60))
    frac_month = ds.TIME.dt.month + (frac_day / ds.TIME.dt.days_in_month)
    frac_year = ds.TIME.dt.year + (frac_month / 12)
    
    month_li = []
    for i in range(0,len(ds.N_PROF)):
        month_li.append(ds.isel(N_PROF=i).TIME.dt.month)
        
    year_li = []
    for i in range(0,len(ds.N_PROF)):
        year_li.append(ds.isel(N_PROF=i).TIME.dt.year)
    
    
    ds = ds.assign_coords(month=('N_PROF',month_li))
    ds = ds.assign_coords(month_frac=('N_PROF',frac_month.data))
    ds = ds.assign_coords(year=('N_PROF',year_li))
    ds = ds.assign_coords(year_frac=('N_PROF',frac_year.data))
    
    return ds