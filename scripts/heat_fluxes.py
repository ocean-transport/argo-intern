import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path
import matplotlib.colors as colors
from xhistogram.xarray import histogram
import seaborn as sns
import seaborn
import pandas as pd
import numpy as np
from importlib import reload
import cartopy.crs as ccrs
import cmocean.cm as cmo
import gsw
import scipy.ndimage as filter
from flox.xarray import xarray_reduce

print('Completed imports (1/?)')

ct_2 = xr.open_DataArray('/swot/SUM05/amf2288/var-boxes/ct_2_3.nc')
ct_3 = xr.open_DataArray('/swot/SUM05/amf2288/var-boxes/ct_2_3.nc')

ds_diff = xr.open_dataset('/swot/SUM05/amf2288/global_diff.nc')
K_rho = ds_diff.K
K_rho = K_rho.interp(z_c=ct_2.z_c)

