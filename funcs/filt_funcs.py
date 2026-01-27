#FILTERING FUNCTIONS

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
#import argopy
import scipy.ndimage as filter
import scipy
import matplotlib
import gsw

#import argopy
#from argopy import DataFetcher as ArgoDataFetcher
#argo_loader=ArgoDataFetcher(src='gdac',ftp="/swot/SUM05/dbalwada/Argo_sync",parallel=True,progress=True)

#import MLD_funcs as mf



def get_mask(ds, scale, variable='MLD', dim1='N_PROF', dim2='PRES_INTERPOLATED', bound=True):
    
    '''Takes an xarray and returns a dim1 length list of 1d np arrays with length of dim2 that contains:
    1) bound=False: ones
    2) bound=True: zeroes one filter scale away from the top (ML base) and bottom (profile bottom) boundaries, and ones between.
    
    ds: xarray dataset with pressure dimension
    scale: int/float, used to determine the amount of pressures that will go to zero
    variable: coordinate of mixed layer depth, default=MLD
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, filtering occurs along this dimension, defualt=PRES_INTERPOLATED
    bound: will boundary regions become zeros?, default=False'''

    mask_li = []
    
    if bound==False:
        for n in range(0,len(ds[dim1])):
            mask = np.ones((len(ds[dim2])))
            mask_li.append(mask)
        
    if bound==True:
        for n in range(0,len(ds[dim1])):
            start = ds[variable][n].values + scale
            end = ds[dim2].isel({dim2:-1}).values - scale
            mask = ds[dim2].where(ds[dim2]>start).where(ds[dim2]<end).values
            
            mask[np.greater(mask,0)] = 1
            mask[np.isnan(mask)] = 0
            mask_li.append(mask)
    
    return mask_li


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
    prof_filt = filter.gaussian_filter1d(prof, sigma=nfilter, mode='nearest')
    
    return prof_filt


def ds_filt_single(ds, lfilter, variable='CT', dim1='N_PROF', dim2='PRES_INTERPOLATED', bound=True):
    
    '''Takes an xarray and a filter scale in meters and returns an xarray with additional coordinates N_PRPF_NEW for a sequence that can be plotted and MASK for the boundary correction.
    
    ds: xarray dataset with pressure dimension
    lfilter: filter scale in meters
    variable: coordinate to filter, default=CT
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, filtering occurs along this dimension, default=PRES_INTERPOLATED
    bound: will boundary regions become zeros?, default=True'''
    
    #mask_li = get_mask(ds, lfilter, variable='MLD', dim1=dim1, dim2=dim2, bound=bound)
    
    nfilter = get_nfilter(ds, lfilter, dim2=dim2)
    
    temp = np.zeros((ds[dim1].shape[0], ds[dim2].shape[0]))
    temp[:,:] = filter.gaussian_filter1d(ds[variable], sigma=nfilter, mode='nearest')
    
    ds_filt = xr.DataArray(temp, dims=[dim1, dim2], coords={dim1:ds[dim1], dim2:ds[dim2]})
    ds_filt = ds_filt.assign_coords(lat=(dim1,ds.lat.data))
    ds_filt = ds_filt.assign_coords(lon=(dim1,ds.lon.data))
    
    number=np.arange(0,len(ds_filt[dim1]))
    ds_filt['N_PROF_NEW']=xr.DataArray(number,dims=ds[dim1].dims)
    #ds_filt=ds_filt.assign_coords(mask=((dim1,dim2),mask_li))
    
    return ds_filt

def da_filt_single(ds, lfilter, dim1='N_PROF', dim2='PRES_INTERPOLATED', bound=True, lat='lat', lon='lon'):
    
    '''Takes an xarray and a filter scale in meters and returns an xarray with additional coordinates N_PRPF_NEW for a sequence that can be plotted and MASK for the boundary correction.
    
    ds: xarray dataset with pressure dimension
    lfilter: filter scale in meters
    variable: coordinate to filter, default=CT
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, filtering occurs along this dimension, default=PRES_INTERPOLATED
    bound: will boundary regions become zeros?, default=True'''
    
    #mask_li = get_mask(ds, lfilter, variable='MLD', dim1=dim1, dim2=dim2, bound=bound)
    
    nfilter = get_nfilter(ds, lfilter, dim2=dim2)
    
    temp = np.zeros((ds[dim1].shape[0], ds[dim2].shape[0]))
    temp[:,:] = filter.gaussian_filter1d(ds, sigma=nfilter, mode='nearest')
    
    ds_filt = xr.DataArray(temp, dims=[dim1, dim2], coords={dim1:ds[dim1], dim2:ds[dim2]})
    ds_filt = ds_filt.assign_coords(lat=(dim1,ds[lat].data))
    ds_filt = ds_filt.assign_coords(lon=(dim1,ds[lon].data))
    
    number=np.arange(0,len(ds_filt[dim1]))
    ds_filt['N_PROF_NEW']=xr.DataArray(number,dims=ds[dim1].dims)
    #ds_filt=ds_filt.assign_coords(=((dim1,dim2),mask_li))
    
    return ds_filt

def get_filt_multi(ds, first, last, num, variable='CT', dim1='N_PROF', dim2='PRES_INTERPOLATED', bound=True, log=False):
    
    '''Takes an xarray and a filter scale in meters and returns an xarray with additional coordinates N_PRPF_NEW for a sequence that can be plotted, MASK for the boundary correction, and FILT_SCALE for filter scales.
    
    ds: xarray dataset with pressure dimension
    first: int/float, first scale
    last: int/float, last scale
    num: into/float, number of scales
    variable: coordinate to filter, default=CT
    dim1: profile dimension, default=N_PROF
    dim2: pressure dimension, filtering occurs along this dimension, default=PRES_INTERPOLATED
    bound: will boundary regions become zeros?, default=False
    log: arrays on either a linspace (default) or logspace(==True)'''
    
    #why is lfilter never converted to nfilter?????????????
    
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