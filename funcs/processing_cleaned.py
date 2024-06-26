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

def load_argo_box(float_type, float_ID=None, dim1='N_PROF', dim2='PRES_INTERPOLATED', var_dens='SIG0', var_time='TIME'):
    '''
    inputs:  list of latitude, longitude, depth parameters to load box
    outputs: argo_loader float object
    '''

    float_types = ['argo_float', 'argo_box', 'glider_unprocessed', 'glider_processed']
    if float_type not in float_types:
        raise ValueError(f'Invalid float_type value. Please provide one of the following acceptable values {float_types}.')
        
    if float_type=='argo_float':
        pass

    if float_type=='argo_box':
        pass

    if float_type=='glider_unprocessed':
        pass

    if float_type=='glider_processed':
        pass

    return 
    

def load_argo_float():
    '''
    inputs:  argo float ID
    outputs: argo_loader float object
    '''

def load_float_unprocessed():
    '''
    inputs:  xr dataset with dimensions of profiles and depth
    outputs: xr dataset with dimensions of profiles and ineterpolated depth
    '''

def load_float_processed():
    '''
    inputs:  xr dataset with interpolated depth
    outputs: xr dataset with all required variables
    '''

def     