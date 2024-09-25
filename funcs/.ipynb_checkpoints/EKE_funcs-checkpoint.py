import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argopy
import scipy.ndimage as filter
import scipy
import matplotlib

import filt_funcs as ff

def get_filt_squared(ds, lfilter, variable='TEMP',dim1='N_PROF',dim2='PRES_INTERPOLATED'):
    
    ds_filt = ff.get_filt_single(ds, lfilter, variable=variable, dim1=dim1, dim2=dim2)
    ds_filt_squared = ds_filt**2
    
    return ds_filt_squared


def get_squared_filt(ds, lfilter, variable='TEMP',dim1='N_PROF',dim2='PRES_INTERPOLATED'):
    
    ds_squared = ds[[variable]]**2
    ds_squared_filt = ff.get_filt_single(ds_squared, lfilter, variable=variable, dim1=dim1, dim2=dim2)
    
    return ds_squared_filt


def get_EKE(ds, lfilter, variable='TEMP',dim1='N_PROF',dim2='PRES_INTERPOLATED'):
    
    ds_filt_squared = get_filt_squared(ds, lfilter, dim1=dim1, dim2=dim2, variable=variable)
    ds_squared_filt = get_squared_filt(ds, lfilter, dim1=dim2, dim2=dim2, variable=variable)
    ds_EKE = ds_squared_filt - ds_filt_squared
    
    return ds_EKE