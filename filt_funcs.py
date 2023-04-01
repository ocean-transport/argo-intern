import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argopy
import scipy.ndimage as filter
import scipy
import matplotlib

def get_ds_interp(ds,depth_min,depth_max,sample_max):
    
    '''Takes an Argo xarray with sampled pressure and:
    1) only selects profiles that sample at a rate equal to or greater than sample_max
    2) interpolates the pressure to a 2m grid.
    3) returns an xarray with all profiles that meet the sample rate interpolated at 2m, with a new dimension PRES_INTERPOLATED
    
    ds: xarray dataset with dimensions PRES, N_LEVELS, N_PROF; pressure PRES
    depth_min: shallowest depth selected[m]
    depth_max: deepest depth selected [m]
    sample_max: minimum sample rate [m]'''
    
    median_dp=ds.PRES.where(ds.PRES<depth_max).where(ds.PRES>depth_min).diff('N_LEVELS').median('N_LEVELS')
    ind_rate=median_dp.where(median_dp<sample_max,drop=True).N_PROF
    ds_sel=ds.sel(N_PROF=ind_rate)
    ds_interp=ds_sel.argo.interp_std_levels(np.arange(depth_min,depth_max,2))
    ds_interp=ds_interp.sortby(ds_interp.N_PROF)
    
    number=np.arange(0,len(ds_interp.N_PROF))
    ds_interp.coords['N_PROF_NEW']=xr.DataArray(number,dims=ds_interp.N_PROF.dims)
    return ds_interp

def get_mask(ds, scale, dim2='PRES_INTERPOLATED', bound=False):
    
    '''Takes an xarray and returns a 1d np array with length of dim2 that contains:
    1) bound=False: ones
    2) bound=True: zeroes (the length of scale) at the top and bottom of a profile, and ones between.
    
    ds: xarray dataset with pressure dimension
    scale: int/float, used to determine the amount of pressures that will go to zero
    dim2: pressure dimension, filtering occurs along this dimension, defualt=PRES_INTERPOLATED
    bound: will boundary regions become zeros?, default=False'''
    
    if bound==False:
        mask = np.ones((len(ds[dim2])))
        
    if bound==True:
        start = ds[dim2].isel({dim2:0}).values + (scale-1)
        end = ds[dim2].isel({dim2:-1}) - (scale-1)
        mask = ds[dim2].where(ds[dim2]>start).where(ds[dim2]<end).values
        
        mask[np.greater(mask,0)] = 1
        mask[np.isnan(mask)] = 0
    
    return mask


def get_lfilters(first, last, num, log=False):
    
    '''Takes two boundaries and arrays the provided num of scales between them on either a lin or log scale. Returns a 1d np array with length num. All values are in meters.
    
    first: int/float, first scale
    last: int/float, last scale
    num: int/float, number of scales in array
    log: arrays on either a linspace (default) or logspace(==True)'''
    
    if log==False:
        lfilters = np.linspace(first, last, num)
        
    if log==True:
        first_exp = np.log10(first)
        last_exp = np.log10(last)
        lfilters = np.logspace(first_exp, last_exp, num)
        
    return lfilters

def get_nfilter(ds, lfilter, dim2='PRES_INTERPOLATED'):
    
    '''Takes an xarray (to determine dx) and a filter scale in meters. Returns the corresponding filter scale in gridpoints.
    
    ds: xarray dataset with pressure dimension
    lfilter: filter scale in meters
    dim2: pressure dimension, filtering occurs along this dimension, default=PRES_INTERPOLATED'''
    
    dx = (ds[dim2].isel({dim2:1})-ds[dim2].isel({dim2:0})).values
    sigmafilter = lfilter/np.sqrt(12)
    nfilter = sigmafilter/dx
    
    return nfilter

def get_filt_prof(prof, lfilter, variable='CT', dim1='N_PROF', dim2='PRES_INTERPOLATED'):
    
    '''Takes a profile and a filter scale in meters and returns 1d np array with the length of dim2.
    
    prof: 1d np array or single xarray profile
    lfilter: filter scale in meters
    variable: coordinate to filter, default=CT
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, filtering occurs along this dimension, default=PRES_INTERPOLATED'''
    
    nfilter = get_nfilter(prof, lfilter, dim2=dim2)
    prof_filt = filter.gaussian_filter1d(prof, sigma=nfilter, mode='wrap')
    
    return prof_filt


def get_filt_single(ds, lfilter, variable='CT', dim1='N_PROF', dim2='PRES_INTERPOLATED', bound=False):
    
    '''Takes an xarray and a filter scale in meters and returns an xarray with additional coordinates N_PRPF_NEW for a sequence that can be plotted and MASK for the boundary correction.
    
    ds: xarray dataset with pressure dimension
    lfilter: filter scale in meters
    variable: coordinate to filter, default=CT
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, filtering occurs along this dimension, default=PRES_INTERPOLATED
    bound: bound: will boundary regions become zeros?, default=False'''
    
    mask = get_mask(ds, lfilter, dim2=dim2, bound=bound)
    
    nfilter = get_nfilter(ds, lfilter, dim2=dim2)
    
    temp = np.zeros((ds[dim1].shape[0], ds[dim2].shape[0]))
    temp[:,:] = filter.gaussian_filter1d(ds[variable], sigma=nfilter, mode='wrap')
    
    ds_filt = xr.DataArray(temp, dims=['N_PROF', 'PRES_INTERPOLATED'], coords={'N_PROF':ds[dim1], 'PRES_INTERPOLATED':ds[dim2]})
    
    number=np.arange(0,len(ds_filt.N_PROF))
    ds_filt['N_PROF_NEW']=xr.DataArray(number,dims=ds[dim1].dims)
    ds_filt['MASK']=xr.DataArray(mask,dims=ds_filt[dim2].dims)
    
    return ds_filt

def get_filt_multi(ds, first, last, num, variable='CT', dim1='N_PROF', dim2='PRES_INTERPOLATED', bound=False, log=False):
    
    '''Takes an xarray and a filter scale in meters and returns an xarray with additional coordinates N_PRPF_NEW for a sequence that can be plotted, MASK for the boundary correction, and FILT_SCALE for filter scales.
    
    ds: xarray dataset with pressure dimension
    first: int/float, first scale
    last: int/float, last scale
    num: into/float, number of scales
    variable: coordinate to filter, default=CT
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, filtering occurs along this dimension, default=PRES_INTERPOLATED
    bound: bound: will boundary regions become zeros?, default=False
    log: arrays on either a linspace (default) or logspace(==True)'''
    
    lfilters = get_lfilters(first=first, last=last, num=num, log=log)
    mask = get_mask(ds, last, dim2=dim2, bound=bound)
    
    temp=np.zeros((ds[dim1].shape[0],ds[dim2].shape[0],num))
    for n in range(0,num):
        temp[:,:,n] = get_filt_single(ds=ds, lfilter=lfilters[n], variable=variable, dim1=dim1, dim2=dim2, bound=bound)
    
    ds_filt = xr.DataArray(temp, dims=['N_PROF', 'PRES_INTERPOLATED', 'FILT_SCALE'], 
                           coords={'N_PROF':ds[dim1], 'PRES_INTERPOLATED':ds[dim2], 'FILT_SCALE':lfilters})
    
    number=np.arange(0,len(ds[dim1]))
    ds_filt['N_PROF_NEW']=xr.DataArray(number,dims=ds[dim1].dims)
    ds_filt['MASK']=xr.DataArray(mask,dims=ds[dim2].dims)
    
    return ds_filt