import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argopy
import scipy.ndimage as filter
import scipy
import matplotlib

import filt_funcs as ff

def get_filt_squared(ds, lfilter, variable='TEMP'):
    
    ds_filt = ff.get_filt_single(ds, lfilter, variable=variable)
    ds_filt_squared = ds_filt**2
    
    return ds_filt_squared


def get_squared_filt(ds, lfilter, variable='TEMP'):
    
    ds_squared = ds[[variable]]**2
    ds_squared_filt = ff.get_filt_single(ds_squared, lfilter, variable=variable)
    
    return ds_squared_filt


def get_EKE(ds, lfilter, variable='TEMP'):
    
    ds_filt_squared = get_filt_squared(ds, lfilter, variable=variable)
    ds_squared_filt = get_squared_filt(ds, lfilter, variable=variable)
    ds_EKE = ds_squared_filt - ds_filt_squared
    
    return ds_EKE