# argo_box_loader

import xarray as xr
import numpy as np
import gsw
import matplotlib.pyplot as plt
import matplotlib as mpl
from importlib import reload
from cmocean import cm as cmo
import xrft
import pandas as pd
import argopy
from argopy import DataFetcher
import dask

import os
os.chdir('/home.ufs/amf2288/argo-intern/funcs')
import density_funcs as df
import EV_funcs as ef
import filt_funcs as ff
import plot_funcs as pf
import processing_funcs as prf

NW = [-180,0,0,90,0,2001]
NE = [0,180,-90,0,0,2001]
SW = [-180,0,-90,0,0,2001]
SE = [0,-180,-90,0,0,2001]

box1 = [-180,-90,  0,90,0,2001]
box2 = [- 90,  0,  0,90,0,2001]
box3 = [   0, 90,  0,90,0,2001]
box4 = [  90,180,  0,90,0,2001]
box5 = [-180,-90,-90, 0,0,2001]
box6 = [- 90,  0,-90, 0,0,2001]
box7 = [   0, 90,-90, 0,0,2001]
box8 = [  90,180,-90, 0,0,2001]

ds = prf.get_box(box1,2)