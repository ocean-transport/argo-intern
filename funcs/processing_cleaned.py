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
def process_data(float_type, argo_float=None, argo_box=None, float_unproc=None, float_proc=None,
                 dims=['N_PROF','PRES_INTERPOLATED'],
                 vars=['SIG0', 'TIME']):
    '''
    inputs:  list of latitude, longitude, depth parameters to load box
    outputs: argo_loader float object
    '''

    #raising errors if anything doesn't seem right about the inputs
    float_types = ['argo_float', 'argo_box', 'glider_unprocessed', 'glider_processed']
    if float_type not in float_types:
        raise ValueError(f'Invalid float_type value. Please provide one of the following acceptable values {float_types}.')
        
    if float_type=='argo_float':
        if argo_float is None:
            raise TypeError('Must provide a float ID for float_type argo_float')

        #analysis goes here
        
        ds_process = process_argo_float()

    if float_type=='argo_box':
        if argo_float is None:
            raise TypeError('Must provide a float box for float_type argo_box')

        #analysis goes here
        
        ds_process = process_argo_box()

    if float_type=='glider_unprocessed':
        if argo_float is None:
            raise TypeError('Must provide an xr ds of the unprocessed glider for float_type glider_unprocessed')

        #analysis goes here
        
        ds_process = process_glider_unprocessed()

    if float_type=='glider_processed':
        if argo_float is None:
            raise TypeError('Must provide an xr ds of the processed glider for float_type glider_processed')

        #analysis goes here
        
        ds_process = float_proc

    print('processing complete')

    

    return 

def process_argo_float():
    '''
    '''

def process_argo_box():
    '''
    inputs:  argo float ID
    outputs: argo_loader float object
    '''

def process_float_unprocessed():
    '''
    inputs:  xr dataset with dimensions of profiles and depth
    outputs: xr dataset with dimensions of profiles and ineterpolated depth
    '''

def process_float_processed():
    '''
    inputs:  xr dataset with interpolated depth
    outputs: xr dataset with all required variables
    '''

def     