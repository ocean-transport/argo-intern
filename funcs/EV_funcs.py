#EDDY VARIANCE FUNCTIONS

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argopy
import scipy.ndimage as filter
import scipy
import matplotlib

import filt_funcs as ff

def get_MV_prof(prof, lfilter, variable='CT', dim1='N_PROF', dim2='PRES_INTERPOLATED', bound=True):
    
    '''Takes a profile and a filter scale in meters and returns an array of the filtered signal, squared.
    
    ds: xarray dataset with profiles and pressure dimensions
    lfilter: filter scale in meters
    variable: coordinate to filter, default=CT
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, default=PRES_INTERPOLATED'''
    
    ds_filt = ff.get_filt_prof(prof, lfilter, variable=variable, dim1=dim1, dim2=dim2)
    ds_mean_variance = ds_filt**2
    #ds_mean_variance['TIME'] = xr.DataArray(ds.TIME)
    
    return ds_mean_variance

def get_MV(ds, lfilter, variable='CT', dim1='N_PROF', dim2='PRES_INTERPOLATED', bound=True):
    
    '''Takes an xarray and a filter scale in meters and returns an array of the filtered signal, squared.
    
    ds: xarray dataset with profiles and pressure dimensions
    lfilter: filter scale in meters
    variable: coordinate to filter, default=CT
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, default=PRES_INTERPOLATED'''
    
    ds_filt = ff.get_filt_single(ds, lfilter, variable=variable, dim1=dim1, dim2=dim2, bound=bound)
    ds_mean_variance = ds_filt**2
    ds_mean_variance['TIME'] = xr.DataArray(ds.TIME)
    
    return ds_mean_variance


def get_total_variance(ds, lfilter, variable='CT', dim1='N_PROF', dim2='PRES_INTERPOLATED', bound=True):
    
    '''Takes an xarray and a filter scale in meters and returns an array of the squared signal, filtered.
    
    ds: xarray dataset with profiles and pressure dimensions
    lfilter: filter scale in meters
    variable: coordinate to square, default=CT
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, default=PRES_INTERPOLATED'''
    
    ds_squared = ds[[variable]]**2
    ds_total_variance = ff.get_filt_single(ds_squared, lfilter, variable=variable, dim1=dim1, dim2=dim2, bound=bound)
    
    return ds_total_variance


def get_EV(ds, lfilter, variable='CT', dim1='N_PROF', dim2='PRES_INTERPOLATED', bound=True):
    
    '''Takes an xarray and a filter scale in meters and returns an array of the difference between the signal squared then filtered, and the signal filtered then squared. (This quantity we're currently defining as 'EKE')
    
    ds: xarray dataset with profiles and pressure dimensions
    lfilter: filter scale in meters
    variable: coordinate to filter, default=CT
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, default=PRES_INTERPOLATED'''
    
    ds_mean_variance = get_MV(ds, lfilter, variable=variable, dim1=dim1, dim2=dim2, bound=bound)
    ds_total_variance = get_total_variance(ds, lfilter, variable=variable, dim1=dim1, dim2=dim2, bound=bound)
    ds_eddy_variance = ds_total_variance - ds_mean_variance
    
    return ds_eddy_variance


def get_NEV(ds,ds_EV,variable,dim1='N_PROF',dim2='PRES_INTERPOLATED',coarsen_scale=40):
    
    '''Takes an xarray and its eddy variance and returns an array of the square root of the eddy variance divided by the density gradient squared. We refer to this as the 'isopycnal displacement' term.
    
    ds: xarray dataset with profiles and pressure dimensions
    ds_EV: xarray dataset of the eddy variance of ds, averaged over all profiles
    variable: coordinate to filter
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, default=PRES_INTERPOLATED
    coarsen_scale: level of smoothing that will be applied to density gradient
    '''
    
    return ((ds_EV/ get_drho_dz(ds, variable=variable, dim2=dim2, coarsen_scale=coarsen_scale).mean({dim1})**2)**(1/2))


def get_drho_dz (ds, variable, coarsen_scale, dim2='PRES_INTERPOLATED'):
    '''Takes an xarray and returns the density gradient, coarsened by the coarsen_scale.
    NOTE: I haven't figured out how to completely remove the pressure dimension from being implicit in this function. If ds has a pressure dimension different than PRES_INTERPOLATED, this function will not work properly.
    
    ds: xarray dataset with profiles and pressure dimensions
    variable: coordinate to filter
    coarsen_scale: level of smoothing to apply
    dim2: pressure dimension, default=PRES_INTERPOLATED
    '''
    
    coarsened_rho = ds[variable].coarsen(PRES_INTERPOLATED=coarsen_scale).mean()
    drho_dz_coarsened = coarsened_rho.diff(dim2)/(2*coarsen_scale)
    drho_dz = drho_dz_coarsened.interp(PRES_INTERPOLATED=ds[dim2])
    
    return drho_dz

def get_EKE_da(ds, scales, sample_max, variable, dim1='N_PROF', dim2='PRES_INTERPOLATED'):
    
    #ds = ds.where(ds.sample_rate<sample_max)
    ekes_li = []
    
    for n in range(0,len(scales)):
        ekes_li.append(get_EV(ds,scales[n],variable=variable,dim2=dim2,dim1=dim1))

    EKES_li = []
    for n in range(0,len(scales)):

        if n==0:
            EKES_li.append((ekes_li[n]).drop_vars('mask',errors='ignore').drop_vars('N_PROF_NEW',errors='ignore'))

        elif n>0:
            EKES_li.append((ekes_li[n] - ekes_li[n-1]).drop_vars('mask',errors='ignore').drop_vars('N_PROF_NEW',errors='ignore'))
    
    #eke_da = xr.concat(ekes_li, dim='eke')
    #eke_da = eke_da.assign_coords(mask=(['N_PROF','PRES_INTERPOLATED'], ekes_li[-1].mask.data))
    
    EKE_da = xr.concat(EKES_li, dim='EKE')
    EKE_da = EKE_da.assign_coords(mask=([dim1,dim2], ekes_li[-1].mask.data))
    
    return EKE_da