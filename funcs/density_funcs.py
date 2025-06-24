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
#import glidertools as gt

import filt_funcs as ff
import EV_funcs as ef
import plot_funcs as pf


def var_interp_binned(ds, variable, rho_grid, dim1='z_c', dim2='lon_c', dim3='lat_c', flag='group'):
    '''
    '''
    
    lon_num = np.array(ds[dim2].values)
    lat_num = np.array(ds[dim3].values)
    
    rho = ds.SIG0
    rho_nonan = rho.where(~np.isnan(rho).compute(), drop=True).where(~np.isnan(ds[variable]).compute(), drop=True)
    var_nonan = ds[variable].where(~np.isnan(rho).compute(), drop=True).where(~np.isnan(ds[variable]).compute(), drop=True)
    var_nonan2 = var_nonan.where(~np.isnan(var_nonan).compute(), drop=True)
    
    if flag == 'group': # incase density is identical b/w two points (this makes things very slow)
        if rho_nonan.size > 2 and var_nonan.size > 2:
            var_nonan = var_nonan.groupby(rho_nonan).mean()
            rho_nonan = rho_nonan.groupby(rho_nonan).mean()
    
    if (len(rho_nonan)>2) & (len(var_nonan2)>2):
        fvar = interpolate.PchipInterpolator(rho_nonan, var_nonan, extrapolate=False)
    
        var_tilde = fvar(rho_grid)
    else:
        var_tilde = np.nan*rho_grid

    ds_rho = xr.DataArray(
        data=var_tilde.reshape((-1,1,1)), 
        dims=['rho_grid', dim2, dim3], 
        coords={
            'rho_grid': (['rho_grid'], rho_grid),
            dim2: (['lon_c'], np.array([lon_num])),  # Ensure proper shape
            dim3: (['lat_c'], np.array([lat_num]))   # Ensure proper shape
        }
    ).rename(variable)

    return ds_rho.sortby(dim2, dim3)

def K_rho_interp_binned(ds_z, rho_grid, dim1='z_c', dim2='lon_c', dim3='lat_c'):
    K_list = [var_interp_binned(ds_z.isel(lat_c=lat).isel(lon_c=lon), 'K_rho', rho_grid)
                 for lon in range(len(ds_z[dim2]))
                 for lat in range(len(ds_z[dim3]))]
    K_xr = xr.combine_by_coords(K_list)
    print('K concat complete')

    e_list = [var_interp_binned(ds_z.isel(lat_c=lat).isel(lon_c=lon), 'e', rho_grid)
                 for lon in range(len(ds_z[dim2]))
                 for lat in range(len(ds_z[dim3]))]
    e_xr = xr.combine_by_coords(e_list)
    print('e concat complete')
    
    ds_rho = xr.merge([K_xr, e_xr])
    print('ds_rho merge complete')
   
    return ds_rho

def density_interp_binned(ds_z, rho_grid, dim1='z_c', dim2='lon_c', dim3='lat_c'):
    z_c_list = [var_interp_binned(ds_z.isel(lat_c=lat).isel(lon_c=lon), 'z_c', rho_grid)
                 for lon in range(len(ds_z[dim2]))
                 for lat in range(len(ds_z[dim3]))]
    z_c_xr = xr.combine_by_coords(z_c_list)
    print('z_c concat complete')

    CT_list = [var_interp_binned(ds_z.isel(lat_c=lat).isel(lon_c=lon), 'CT', rho_grid)
                 for lon in range(len(ds_z[dim2]))
                 for lat in range(len(ds_z[dim3]))]
    CT_xr = xr.combine_by_coords(CT_list)
    print('CT concat complete')

    SA_list = [var_interp_binned(ds_z.isel(lat_c=lat).isel(lon_c=lon), 'SA', rho_grid)
                 for lon in range(len(ds_z[dim2]))
                 for lat in range(len(ds_z[dim3]))]
    SA_xr = xr.combine_by_coords(SA_list)
    print('SA concat complete')

    SIG0_list = [var_interp_binned(ds_z.isel(lat_c=lat).isel(lon_c=lon), 'SIG0', rho_grid)
                 for lon in range(len(ds_z[dim2]))
                 for lat in range(len(ds_z[dim3]))]
    SIG0_xr = xr.combine_by_coords(SIG0_list)
    print('SIG0 concat complete')

    SPICE_list = [var_interp_binned(ds_z.isel(lat_c=lat).isel(lon_c=lon), 'SPICE', rho_grid)
                 for lon in range(len(ds_z[dim2]))
                 for lat in range(len(ds_z[dim3]))]
    SPICE_xr = xr.combine_by_coords(SPICE_list)
    print('SPICE concat complete')
    
    PRES_list = [var_interp_binned(ds_z.isel(lat_c=lat).isel(lon_c=lon), 'PRES', rho_grid)
                 for lon in range(len(ds_z[dim2]))
                 for lat in range(len(ds_z[dim3]))]
    PRES_xr = xr.combine_by_coords(PRES_list)
    print('PRES concat complete')

    ds_rho = xr.merge([z_c_xr, CT_xr, SA_xr, SIG0_xr, SPICE_xr, PRES_xr])
    print('ds_rho merge complete')
   
    return ds_rho











def func_var_int(ds, variable, rho_grid, dim1='N_PROF', dim2='PRES_INTERPOLATED', flag='group'):
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
    rho_nonan = rho.where(~np.isnan(rho), drop=True).where(~np.isnan(ds[variable]), drop=True)
    
    var_nonan = ds[variable].where(~np.isnan(rho), drop=True).where(~np.isnan(ds[variable]), drop=True)
    
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



def interpolate2density_prof(ds_z, rho_grid, dim1='N_PROF', dim2='PRES_INTERPOLATED'):
    '''Takes an xarray in depth space and returns an xarray in density space, using the density grid provided.
    
    ds_z: xarray in depth space
    rho_grid: density grid that depth will be interpolated to
    dim1: profiles dimension, default is N_PROF_NEW to make plotting easier down the road
    dim2: pressure dimension, default is PRES_INTERPOLATED
    '''
    
    N_PROF_ind = 0
    PRES_INTERPOLATED_tilde_xr          = func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'PRES_INTERPOLATED',rho_grid)
    CT_tilde_xr                         = func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'CT',rho_grid)
    SA_tilde_xr                         = func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'SA', rho_grid)
    SIG0_tilde_xr                       = func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'SIG0', rho_grid)
    SPICE_tilde_xr                      = func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'SPICE', rho_grid)
    

    for N_PROF_ind in range(1, len(ds_z.N_PROF)):
        if np.mod(N_PROF_ind, 100)==0:
            print('Completed profile {}'.format(N_PROF_ind))
        PRES_INTERPOLATED_tilde_xr      = xr.concat([PRES_INTERPOLATED_tilde_xr,  func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'PRES_INTERPOLATED', rho_grid)], dim=dim1)
        CT_tilde_xr                     = xr.concat([CT_tilde_xr,    func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'CT', rho_grid)], dim=dim1)
        SA_tilde_xr                     = xr.concat([SA_tilde_xr,    func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'SA', rho_grid)], dim=dim1)
        SIG0_tilde_xr                   = xr.concat([SIG0_tilde_xr,  func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'SIG0', rho_grid)], dim=dim1)
        SPICE_tilde_xr                  = xr.concat([SPICE_tilde_xr, func_var_int(ds_z.isel(N_PROF=N_PROF_ind), 'SPICE', rho_grid)], dim=dim1)

    ds_rho = xr.merge([PRES_INTERPOLATED_tilde_xr, CT_tilde_xr,
                             SA_tilde_xr, SIG0_tilde_xr, SPICE_tilde_xr])
    
    ds_rho = ds_rho.assign_coords(TIME      =('N_PROF_NEW',ds_z.TIME.data))
    ds_rho = ds_rho.assign_coords(LATITUDE  =('N_PROF_NEW',ds_z.LATITUDE.data))
    ds_rho = ds_rho.assign_coords(LONGITUDE =('N_PROF_NEW',ds_z.LONGITUDE.data))
    
    return ds_rho

def interpolate2density_var(ds_z, rho_grid, dim1='N_PROF', dim2='PRES_INTERPOLATED'):
    '''
    '''
    
    ct2_inside_list = [func_var_int(ds_z.isel(N_PROF=N_PROF), 'ct2_inside', rho_grid)
                 for N_PROF in range(len(ds_z[dim1]))]
    ct2_inside_xr = xr.combine_by_coords(ct2_inside_list)
    print('ct2_inside concat complete')
    
    sa2_inside_list = [func_var_int(ds_z.isel(N_PROF=N_PROF), 'sa2_inside', rho_grid)
                 for N_PROF in range(len(ds_z[dim1]))]
    sa2_inside_xr = xr.combine_by_coords(sa2_inside_list)
    print('sa2_inside concat complete')
    
    sp2_inside_list = [func_var_int(ds_z.isel(N_PROF=N_PROF), 'sp2_inside', rho_grid)
                 for N_PROF in range(len(ds_z[dim1]))]
    sp2_inside_xr = xr.combine_by_coords(sp2_inside_list)
    print('sp2_inside concat complete')
    
    ct3_inside_list = [func_var_int(ds_z.isel(N_PROF=N_PROF), 'ct3_inside', rho_grid)
                 for N_PROF in range(len(ds_z[dim1]))]
    ct3_inside_xr = xr.combine_by_coords(ct3_inside_list)
    print('ct3_inside concat complete')
    
    sa3_inside_list = [func_var_int(ds_z.isel(N_PROF=N_PROF), 'sa3_inside', rho_grid)
                 for N_PROF in range(len(ds_z[dim1]))]
    sa3_inside_xr = xr.combine_by_coords(sa3_inside_list)
    print('sa3_inside concat complete')
    
    sp3_inside_list = [func_var_int(ds_z.isel(N_PROF=N_PROF), 'sp3_inside', rho_grid)
                 for N_PROF in range(len(ds_z[dim1]))]
    sp3_inside_xr = xr.combine_by_coords(sp3_inside_list)
    print('sp3_inside concat complete')
    
    ds_rho = xr.merge([ct2_inside_xr, sa2_inside_xr, sp2_inside_xr, ct3_inside_xr, sa3_inside_xr, sp3_inside_xr])
    print('ds_rho merge complete')
    
    return ds_rho







def interpolate2density_dist(ds_z, rho_grid, dim1='distance', dim2='PRES_INTERPOLATED'):
    '''Takes an xarray in depth space and returns an xarray in density space, using the density grid provided.
    
    ds_z: xarray in depth space
    rho_grid: density grid that depth will be interpolated to
    dim1: distance dimension, default is distance to make plotting easier down the road
    dim2: pressure dimension, default is PRES_INTERPOLATED
    '''
    
    distance_ind = 0
    pres_tilde_xr  = func_var_int(ds_z.isel(distance=distance_ind), dim2,    rho_grid, dim1=dim1)
    CT_tilde_xr    = func_var_int(ds_z.isel(distance=distance_ind), 'CT',    rho_grid, dim1=dim1)
    SA_tilde_xr    = func_var_int(ds_z.isel(distance=distance_ind), 'SA',    rho_grid, dim1=dim1)
    SIG0_tilde_xr  = func_var_int(ds_z.isel(distance=distance_ind), 'SIG0',  rho_grid, dim1=dim1)
    SPICE_tilde_xr = func_var_int(ds_z.isel(distance=distance_ind), 'SPICE', rho_grid, dim1=dim1)

    for distance_ind in range(1, len(ds_z.distance)):
        if np.mod(distance_ind, 50)==0:
            print(distance_ind)
        pres_tilde_xr  = xr.concat([pres_tilde_xr , func_var_int(ds_z.isel(distance=distance_ind), dim2, rho_grid, dim1=dim1)], dim=dim1)
        CT_tilde_xr    = xr.concat([CT_tilde_xr , func_var_int(ds_z.isel(distance=distance_ind), 'CT', rho_grid, dim1=dim1)], dim=dim1)
        SA_tilde_xr    = xr.concat([SA_tilde_xr , func_var_int(ds_z.isel(distance=distance_ind), 'SA', rho_grid, dim1=dim1)], dim=dim1)
        SIG0_tilde_xr  = xr.concat([SIG0_tilde_xr , func_var_int(ds_z.isel(distance=distance_ind), 'SIG0', rho_grid, dim1=dim1)], dim=dim1)
        SPICE_tilde_xr = xr.concat([SPICE_tilde_xr , func_var_int(ds_z.isel(distance=distance_ind), 'SPICE', rho_grid, dim1=dim1)], dim=dim1)
    

    ds_rho = xr.merge([pres_tilde_xr, CT_tilde_xr,
                             SA_tilde_xr, SIG0_tilde_xr, SPICE_tilde_xr])
    
    return ds_rho


def interp_distance(ds, dim1='distance', dim2='PRES_INTERPOLATED', lat='LATITUDE', lon='LONGITUDE'):
    '''Takes an xarray in terms of profiles and returns an xarray in terms of distance.
    
    ds: xarray with profile coordinate
    dim1: name of distance dimension, default='distance'
    dim2: pressure dimension, default='PRES_INTERPOLATED'
    lat: latitude variable, default='LATITUDE'
    lon: longitdue variable, default='LONGITUDE'
    '''
    
    lats=ds[lat]
    lons=ds[lon]
    
    array_distance = gt.utils.distance(lons,lats)
    array_distance = array_distance/1000
    
    cum_distance = [0]
    for i in range(1, len(array_distance)):
        dist = array_distance[i] + cum_distance[i-1]
        cum_distance.append(dist)
    
    ds_distance = xr.Dataset(data_vars=dict(
                                        CT=    ([dim1, dim2], ds.CT.data),
                                        SA=    ([dim1, dim2], ds.SA.data),
                                        SIG0=  ([dim1, dim2], ds.SIG0.data),
                                        SPICE= ([dim1, dim2], ds.SPICE.data)
                                        ),
                            coords=dict(
                                        distance=  ([dim1], cum_distance),
                                        LATITUDE=  ([dim1], lats.data),
                                        LONGITUDE= ([dim1], lons.data),
                                        TIME=      ([dim1], ds.TIME.data),
                                        PRES_INTERPOLATED= ([dim2], ds[dim2].data)
                                        ))
    
    return ds_distance





def func_var_int_pmean(ds, Pmean_smooth, Pmax, variable='SPICE', dim1='N_PROF_NEW',): 
    '''Takes a profile and mean isopycnal grid and returns a profile with the variable interpolated to that grid.
    
    ds: profile in depth space
    Pmean_smooth: smoothed mean isopycnal grid
    Pmax: maximum depth value for plotting
    variable: variable to be interpolated, default='SPICE'
    dim1: profile/distance dimension, default='N_PROF_NEW'
    '''
    
    Pmean_grid = np.linspace(0,Pmax,Pmax//2)
    
    ds_nonan = ds[variable].where(~np.isnan(ds[variable]) & ~np.isnan(Pmean_smooth), drop=True)
    
    Pmean_nonan = Pmean_smooth.where(~np.isnan(ds[variable]) & ~np.isnan(Pmean_smooth), drop=True)
    
    if len(ds_nonan) > 2:
       
        f = interpolate.PchipInterpolator(Pmean_nonan.values, ds_nonan.values , extrapolate=False)
        
        ds_on_Pmean = f(Pmean_grid)
            
        
    else:
        ds_on_Pmean = np.nan*Pmean_grid
    
    return xr.DataArray(ds_on_Pmean.reshape((-1,1)),
                        dims = ['Pmean', dim1],
                        coords = {'Pmean': Pmean_grid, dim1: [ds[dim1].values]}).rename(variable)





def ds_pmean_smooth(ds_rho, roll, dim1='N_PROF_NEW', dim2='PRES_INTERPOLATED', dim3='rho_grid'):
    '''Takes an xarray in density space and returns a smoothed isopycnal grid.
    
    ds_rho: xarray with density coordinate
    roll: smoothing factor
    dim1: profile dimension, default='N_PROF_NEW'
    dim2: pressure dimension, default='PRES_INTERPOLATED'
    dim3: density dimension, default='rho_grid'
    '''
    
    Pmean_smooth = ds_rho[dim2].mean(dim1).rolling({dim3:roll}, center=True).mean()
    
    return Pmean_smooth


def ds_pmean_var(ds_rho, Pmean_smooth, Pmax, variable3='SPICE', dim1='N_PROF_NEW'):
    '''Takes an xarray in density space and smoothed isopycnal grid and returns an xarray with the variable interpolated to that grid.
    
    ds_rho: xarray with density coordinate
    Pmean_smooth: smoothed isopycnal grid
    Pmax: maximum depth value for plotting
    variable3: variable to be interpolated, default='SPICE'
    dim1: profiles dimension, default='N_PROF_NEW'
    '''
    
    N_PROF_NEW_ind = 0

    Spice_on_Pmean = func_var_int_pmean(ds_rho.isel({dim1:N_PROF_NEW_ind}), Pmean_smooth, Pmax, variable=variable3, dim1=dim1)
    
    for N_PROF_NEW_ind in range(1, len(ds_rho[dim1])):
        Spice_on_Pmean = xr.concat([Spice_on_Pmean, func_var_int_pmean(ds_rho.isel({dim1:N_PROF_NEW_ind}), Pmean_smooth, Pmax, variable=variable3, dim1=dim1)]
                              , dim=dim1)
        
    return Spice_on_Pmean


def ds_anom(ds, dim='Pmean'):
    '''Takes an xarray and returns an xarray with the anomaly of the provided variable.
    
    ds: xarray 
    variable:
    '''
    
    n=0
    mean_prof = ds.isel({dim:n}).mean(skipna=True)
    anom_prof = ds.isel({dim:n}) - mean_prof

    for n in range(1,len(ds[dim])):
        mean_prof = ds.isel({dim:n}).mean(skipna=True)
        anom_prof_next = ds.isel({dim:n}) - mean_prof

        anom_prof = xr.concat([anom_prof, anom_prof_next], dim=dim)
        
    return anom_prof

