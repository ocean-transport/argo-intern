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

def load_argo_box():
    '''
    inputs:  list of latitude, longitude, depth parameters to load box
    outputs: argo_loader float object
    '''

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