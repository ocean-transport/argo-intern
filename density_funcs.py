#DEPTH TO DENSITY FUNCTIONS

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argopy
import scipy.ndimage as filter
import scipy
import matplotlib
import gsw
from cmocean import cm as cmo
import scipy.interpolate as interpolate

import filt_funcs as ff
import EV_funcs as ef
import plot_funcs as pf


def func_var_int(ds, variable, rho_grid, dim1='N_PROF_NEW', dim2='PRES_INTERPOLATED', flag='group'):
    '''Takes an xarray and density grid, and returns an xarray in density space, with respect to the given variable
    
    ds: xarray in depth space
    variable: variable along which to convert to density
    rho_grid: density grid
    dim1: profiles dimension, default is N_PROF_NEW
    dim2: pressure dimension, default is PRES_INTERPOLATED
    flag: not totally sure
    '''
    
    N_PROF_num = ds[dim1].values
    
    rho = ds.SIG0
    rho_nonan = rho.where(~np.isnan(rho), drop=True)
    
    var_nonan = ds[variable].where(~np.isnan(rho), drop=True)
    
    var_nonan2 = var_nonan.where(~np.isnan(var_nonan), drop=True)
    
    if flag == 'group': # incase density is identical b/w two points (this makes things very slow)
        var_nonan = var_nonan.groupby(rho_nonan).mean()
        rho_nonan = rho_nonan.groupby(rho_nonan).mean()
    
    if (len(rho_nonan)>2) & (len(var_nonan2)>2):
        fvar = interpolate.PchipInterpolator(rho_nonan, var_nonan, extrapolate=False)
    
        var_tilde = fvar(rho_grid)
    else:
        var_tilde = np.nan*rho_grid
    
    return xr.DataArray(var_tilde.reshape((-1,1)),
                        dims = ['rho_grid', dim1],
                        coords = {'rho_grid': rho_grid, dim1: [N_PROF_num]}).rename(variable)



def interpolate2density_prof(ds_z, rho_grid, dim1='N_PROF_NEW', dim2='PRES_INTERPOLATED'):
    '''Takes an xarray in depth space and returns an xarray in density space, using the density grid provided.
    
    ds_z: xarray in depth space
    rho_grid: density grid that depth will be interpolated to
    dim1: profiles dimension, default is N_PROF_NEW to make plotting easier down the road
    dim2: pressure dimension, default is PRES_INTERPOLATED
    '''
    
    N_PROF_ind = 0
    pres_tilde_xr  = func_var_int(ds_z.isel(N_PROF=N_PROF_ind), dim2,rho_grid)
    CT_tilde_xr    = func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'CT',rho_grid)
    SA_tilde_xr    = func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'SA', rho_grid)
    SIG0_tilde_xr  = func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'SIG0', rho_grid)
    SPICE_tilde_xr = func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'SPICE', rho_grid)

    for N_PROF_ind in range(1, len(ds_z.N_PROF)):
        if np.mod(N_PROF_ind, 50)==0:
            print(N_PROF_ind)
        pres_tilde_xr  = xr.concat([pres_tilde_xr , func_var_int(ds_z.isel(N_PROF=N_PROF_ind), dim2, rho_grid)], dim=dim1)
        CT_tilde_xr    = xr.concat([CT_tilde_xr , func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'CT', rho_grid)], dim=dim1)
        SA_tilde_xr    = xr.concat([SA_tilde_xr , func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'SA', rho_grid)], dim=dim1)
        SIG0_tilde_xr  = xr.concat([SIG0_tilde_xr , func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'SIG0', rho_grid)], dim=dim1)
        SPICE_tilde_xr = xr.concat([SPICE_tilde_xr , func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'SPICE', rho_grid)], dim=dim1)
    

    ds_rho = xr.merge([pres_tilde_xr, CT_tilde_xr,
                             SA_tilde_xr, SIG0_tilde_xr, SPICE_tilde_xr])
    
    return ds_rho


def interpolate2density_dist(ds_z, rho_grid, dim1='distance', dim2='PRES_INTERPOLATED'):
    '''Takes an xarray in depth space and returns an xarray in density space, using the density grid provided.
    
    ds_z: xarray in depth space
    rho_grid: density grid that depth will be interpolated to
    dim1: distance dimension, default is distance to make plotting easier down the road
    dim2: pressure dimension, default is PRES_INTERPOLATED
    '''
    
    distance_ind = 0
    pres_tilde_xr  = func_var_int(ds_z.isel(distance=distance_ind), dim2,rho_grid)
    CT_tilde_xr    = func_var_int(ds_z.isel(distance=distance_ind), 'CT',rho_grid)
    SA_tilde_xr    = func_var_int(ds_z.isel(distance=distance_ind), 'SA', rho_grid)
    SIG0_tilde_xr  = func_var_int(ds_z.isel(distance=distance_ind), 'SIG0', rho_grid)
    SPICE_tilde_xr = func_var_int(ds_z.isel(distance=distance_ind), 'SPICE', rho_grid)

    for N_PROF_ind in range(1, len(ds_z.distance)):
        if np.mod(distance_ind, 50)==0:
            print(distance_ind)
        pres_tilde_xr  = xr.concat([pres_tilde_xr , func_var_int(ds_z.isel(distance=distance_ind), dim2, rho_grid)], dim=dim1)
        CT_tilde_xr    = xr.concat([CT_tilde_xr , func_var_int(ds_z.isel(distance=distance_ind), 'CT', rho_grid)], dim=dim1)
        SA_tilde_xr    = xr.concat([SA_tilde_xr , func_var_int(ds_z.isel(distance=distance_ind), 'SA', rho_grid)], dim=dim1)
        SIG0_tilde_xr  = xr.concat([SIG0_tilde_xr , func_var_int(ds_z.isel(distance=distance_ind), 'SIG0', rho_grid)], dim=dim1)
        SPICE_tilde_xr = xr.concat([SPICE_tilde_xr , func_var_int(ds_z.isel(distance=distance_ind), 'SPICE', rho_grid)], dim=dim1)
    

    ds_rho = xr.merge([pres_tilde_xr, CT_tilde_xr,
                             SA_tilde_xr, SIG0_tilde_xr, SPICE_tilde_xr])
    
    return ds_rho