import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argopy
import scipy.ndimage as filter
import scipy
import matplotlib

import filt_funcs as ff

def get_mean_variance(ds, lfilter, variable='TEMP', dim1='N_PROF', dim2='PRES_INTERPOLATED'):
    
    '''Takes an xarray and a filter scale in meters and returns an array of the filtered signal, squared.
    
    ds: xarray dataset with profiles and pressure dimensions
    lfilter: filter scale in meters
    variable: coordinate to filter, default=TEMP
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, default=PRES_INTERPOLATED'''
    
    ds_filt = ff.get_filt_single(ds, lfilter, variable=variable, dim1=dim1, dim2=dim2)
    ds_mean_variance = ds_filt**2
    
    return ds_mean_variance


def get_total_variance(ds, lfilter, variable='TEMP', dim1='N_PROF', dim2='PRES_INTERPOLATED'):
    
    '''Takes an xarray and a filter scale in meters and returns an array of the squared signal, filtered.
    
    ds: xarray dataset with profiles and pressure dimensions
    lfilter: filter scale in meters
    variable: coordinate to square, default=TEMP
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, default=PRES_INTERPOLATED'''
    
    ds_squared = ds[[variable]]**2
    ds_total_variance = ff.get_filt_single(ds_squared, lfilter, variable=variable, dim1=dim1, dim2=dim2)
    
    return ds_total_variance


def get_eddy_variance(ds, lfilter, variable='TEMP', dim1='N_PROF', dim2='PRES_INTERPOLATED'):
    
    '''Takes an xarray and a filter scale in meters and returns an array of the difference between the signal squared then filtered, and the signal filtered then squared. (This quantity we're currently defining as 'EKE')
    
    ds: xarray dataset with profiles and pressure dimensions
    lfilter: filter scale in meters
    variable: coordinate to filter, default=TEMP
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, default=PRES_INTERPOLATED'''
    
    ds_mean_variance = get_mean_variance(ds, lfilter, variable=variable, dim1=dim1, dim2=dim2)
    ds_total_variance = get_total_variance(ds, lfilter, variable=variable, dim1=dim1, dim2=dim2)
    ds_eddy_variance = ds_total_variance - ds_mean_variance
    
    return ds_eddy_variance