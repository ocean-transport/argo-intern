import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argopy
import scipy.ndimage as filter
import scipy
import matplotlib
import gsw

import argopy
from argopy import DataFetcher as ArgoDataFetcher
argo_loader=ArgoDataFetcher(src='gdac',ftp="/swot/SUM05/dbalwada/Argo_sync",progress=True)

#this is my effort to allow for loading all kinds of data (argo loaded by  float ID, argo loaded by box, gliders that haven't been interpolated and ones that have) and return the same kind of data with the same dimensions and variables
def process_data(float_type, argo_ID=None, argo_box=None, float_unproc=None, float_proc=None, interp_param=None,
                 dims_li=['N_PROF','PRES_INTERPOLATED'],
                 vars_li=['SIG0']):
    '''One sentence description

    Longer description that gets into specifics

    Parameters
    ----------
    float_type: str
        One of: argo_float, argo_box, glider_unprocessed, glider_processed
    argo_ID: int, list
        Individual float ID or list of IDs
    argo_box: list, None
        Of the form: [lon_min, lon_max, lat_min, lat_max, depth_min, depth_max]
    float_unproc: xr.dataset, None
        Not-interpolated float data with dimensions: [] and variables: []
    float_proc: xr.dataset, None
        Interpolated float data with dimensions: [] and variables []
    interp_param: xr.dataset, None
        Parameters for interpolation, of the form: 
    dims_li: list
        Names of dataset's dimensions, of the form: [profiles, pressure]
    vars_li: list
        Names of dataset's variables, of the form: [density]

    Returns
    -------
    xr.dataset
        Interpolated dataset with TEOS-10 variables.
    '''

    float_types = ['argo_float', 'argo_box', 'glider_unprocessed', 'glider_processed']
    if float_type not in float_types:
        raise ValueError(f'Invalid float_type value. Please provide one of the following acceptable values {float_types}.')
        
    if float_type=='argo_float':
        if argo_ID is None or interp_param is None:
            raise TypeError(f'Must provide a float ID and interpolation parameters for float_type argo_float')
        
        ds = load_argo_float(argo_ID)
        print('loading complete')

        ds_interp = get_ds_interp(ds, interp_param)
        print('interpolation complete')

    elif float_type=='argo_box':
        if argo_box is None or interp_param is None:
            raise TypeError(f'Must provide a float box and interpolation parameters for float_type argo_box')
        
        ds = load_argo_box(argo_box)
        print('loading complete')

        ds_interp = get_ds_interp(ds, interp_param)
        print('interpolation complete')  

    elif float_type=='glider_unprocessed':
        if float_unproc is None or interp_param is None:
            raise TypeError(f'Must provide an xr ds of the unprocessed glider and interpolation parameters for float_type glider_unprocessed')

        ds = float_unproc
        print('loading complete')

        ds_interp = get_ds_interp(ds, interp_param)
        print('interpolation complete')


    elif float_type == 'glider_processed':
        if float_proc is None:
            raise TypeError(f'Must provide an xr ds of the processed glider for float_type glider_unprocessed')

        ds_interp = float_proc

    ds_interp['SA']    = gsw.SA_from_SP(ds_interp.PSAL, ds_interp.PRES, ds_interp.LONGITUDE, ds_interp.LATITUDE).rename('SA')
    ds_interp['CT']    = gsw.CT_from_t(ds_interp.SA, ds_interp.TEMP, ds_interp.PRES).rename('CT')
    ds_interp['SIG0']  = gsw.sigma0(ds_interp.SA, ds_interp.CT).rename('SIG0')
    ds_interp['SPICE'] = gsw.spiciness0(ds_interp.SA, ds_interp.CT).rename('SPICE')

    ds_interp = get_MLD(ds_interp)
    print('processing complete')

    return ds_interp

def load_argo_float(argo_ID):
    '''One sentence description

    Longer notes

    Parameters
    ----------
    argo_ID: int, list
        Description

    Returns
    -------
    xr.dataset
        Description
    '''
    ds = argo_loader.float(float_ID)
    ds = ds.to_xarray()
    ds = ds.argo.point2profile()

    return ds

def load_argo_box(argo_box):
    '''One sentence description

    Longer notes

    Parameters
    ----------
    argo_box: list
        Description

    Returns
    -------
    xr.dataset
        Description
    '''

    ds = argo_loader.region(box)
    ds = ds.to_xarray()
    ds = ds.argo.point2profile()

    return ds

#eventually will need to fix this so it works with unprocessed glider data
#wondering if Dhruv's glider paper already has a function that does this actually
def get_ds_interp(ds, interp_param):
    '''One sentence description

    Longer notes

    Parameters
    ----------
    ds: xr.dataset
        Description
    interp_param: list
        Description

    Returns
    -------
    xr.dataset
        Description
    '''
    
    dp = ds['PRES'].diff('N_LEVELS').sortby('N_PROF')
    ds["sample_rate"] = dp
    ds_interp = ds.argo.interp_std_levels(np.arange(depth_min, depth_max, interp_step))

    number = np.arange(0, len(ds_interp.N_PROF))
    ds_interp.coords["N_PROF_NEW"] = xr.DataArray(number, dims=ds_interp.N_PROF.dims)
    return ds_interp

def add_MLD():
    '''One sentence description

    Longer notes

    Parameters
    ----------

    Returns
    -------
    
    '''

    MLD_li = []
    
    for n in range(0, len(ds[dim1])):
        SIG0_surface = ds.isel({dim1:n})[variable].isel({dim2:0})
        SIG0_diff    = SIG0_surface + threshold
        MLD_ds       = SIG0_surface.where(ds.isel({dim1:n})[variable] < SIG0_diff)
        MLD          = MLD_ds.dropna(dim2).isel({dim2:-1})[dim2].values
        MLD_li.append(MLD)
        
    return ds.assign_coords(MLD=(dim1,MLD_li))
