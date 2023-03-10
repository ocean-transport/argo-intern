import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argopy
import scipy.ndimage as filter
import scipy
import matplotlib

def get_ds_interp(ds,depth_min,depth_max,sample_max):
    median_dp=ds.PRES.where(ds.PRES<depth_max).where(ds.PRES>depth_min).diff('N_LEVELS').median('N_LEVELS')
    ind_rate=median_dp.where(median_dp<sample_max,drop=True).N_PROF
    ds_sel=ds.sel(N_PROF=ind_rate)
    ds_interp=ds_sel.argo.interp_std_levels(np.arange(depth_min,depth_max,2))
    ds_interp=ds_interp.sortby(ds_interp.N_PROF)
    
    number=np.arange(0,len(ds_interp.N_PROF))
    ds_interp.coords['N_PROF_NEW']=xr.DataArray(number,dims=ds_interp.N_PROF.dims)
    return ds_interp

def get_mask(ds, scale, dim2='PRES_INTERPOLATED', bound=False):
    
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
    
    if log==False:
        lfilters = np.linspace(first, last, num)
        
    if log==True:
        first_exp = np.log10(first)
        last_exp = np.log10(last)
        lfilters = np.logspace(first_exp, last_exp, num)
        
    return lfilters

def get_nfilter(ds, lfilter, dim2='PRES_INTERPOLATED'):
    
    dx = (ds[dim2].isel({dim2:1})-ds[dim2].isel({dim2:0})).values
    sigmafilter = lfilter/np.sqrt(12)
    nfilter = sigmafilter/dx
    
    return nfilter

def get_filt_prof(prof, lfilter, variable='TEMP', dim1='N_PROF', dim2='PRES_INTERPOLATED', bound=True):
    
    mask = get_mask(prof, lfilter, dim2=dim2, bound=bound)
    
    nfilter = get_nfilter(prof, lfilter, dim2=dim2)
    
    prof_filt = filter.gaussian_filter1d(prof, sigma=nfilter, mode='wrap')
    
    return prof_filt


def get_filt_single(ds, lfilter, variable='TEMP', dim1='N_PROF', dim2='PRES_INTERPOLATED', bound=False):
    
    mask = get_mask(ds, lfilter, dim2=dim2, bound=bound)
    
    nfilter = get_nfilter(ds, lfilter, dim2=dim2)
    
    temp = np.zeros((ds[dim1].shape[0], ds[dim2].shape[0]))
    temp[:,:] = filter.gaussian_filter1d(ds[variable], sigma=nfilter, mode='wrap')
    
    ds_filt = xr.DataArray(temp, dims=['N_PROF', 'PRES_INTERPOLATED'], coords={'N_PROF':ds[dim1], 'PRES_INTERPOLATED':ds[dim2]})
    
    number=np.arange(0,len(ds_filt.N_PROF))
    ds_filt['N_PROF_NEW']=xr.DataArray(number,dims=ds[dim1].dims)
    ds_filt['MASK']=xr.DataArray(mask,dims=ds_filt[dim2].dims)
    
    return ds_filt

def get_filt_multi(ds, first, last, num, variable='TEMP', dim1='N_PROF', dim2='PRES_INTERPOLATED', bound=False, log=False):
    
    lfilters = get_lfilters(first=first, last=last, num=num, log=log)
    mask = get_mask(ds, lfilters[-1], dim2=dim2, bound=bound)
    
    temp=np.zeros((ds[dim1].shape[0],ds[dim2].shape[0],num))
    for n in range(0,num):
        temp[:,:,n] = get_filt_single(ds=ds, lfilter=lfilters[n], variable=variable, dim1=dim1, dim2=dim2, bound=bound)
    
    ds_filt = xr.DataArray(temp, dims=['N_PROF', 'PRES_INTERPOLATED', 'FILT_SCALE'], 
                           coords={'N_PROF':ds[dim1], 'PRES_INTERPOLATED':ds[dim2], 'FILT_SCALE':lfilters})
    
    number=np.arange(0,len(ds[dim1]))
    ds_filt['N_PROF_NEW']=xr.DataArray(number,dims=ds[dim1].dims)
    ds_filt['MASK']=xr.DataArray(mask,dims=ds[dim2].dims)
    
    return ds_filt