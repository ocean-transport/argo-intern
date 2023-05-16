#PLOTTING FUNCTIONS

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import argopy
import scipy.ndimage as filter
import scipy
import matplotlib

import filt_funcs as ff
import EV_funcs as ef


def mult_mean(ds_li, variable, dim1='N_PROF'):
    mean_li=[]
    
    for n in range(0,len(ds_li)):
        mean = ds_li[n][variable].mean({dim1})
        mean_li.append(mean)
        
    return mean_li

def mult_EV(ds_li, lfilter, variable, dim1='N_PROF', dim2='PRES_INTERPOLATED',bound=True):
    EV_li = []
    
    for n in range(0,len(ds_li)):
        EV = ef.get_EV(ds=ds_li[n],lfilter=lfilter,variable=variable,dim1=dim1,dim2=dim2,bound=bound)
        EV_li.append(EV)
        
    return EV_li

def mult_NEV(ds_li, ds_EV_li, variable, dim1='N_PROF', dim2='PRES_INTERPOLATED', coarsen_scale=40):
    NEV_li=[]
    
    for n in range(0,len(ds_li)):
        NEV=ef.get_NEV(ds=ds_li[n], ds_EV=ds_EV_li[n], variable=variable, dim1=dim1, dim2=dim2, coarsen_scale=coarsen_scale)
        NEV_li.append(NEV)
    
    return NEV_li

def plot_quad(ds_li,labels=None, bound=True):
    
    density_mean_li = mult_mean(ds_li,variable='SIG0')
    density_EV_li = mult_EV(ds_li,100,variable='SIG0',bound=bound)
    density_NEV_li = mult_NEV(ds_li,density_EV_li,variable='SIG0')

    spice_mean_li = mult_mean(ds_li,variable='SPICE')
    spice_EV_li = mult_EV(ds_li,100,variable='SPICE',bound=bound)
    
    plt.figure(figsize=(9,7))

    plt.subplot(1,4,1)
    for n in range(0,len(density_mean_li)):
        density_mean_li[n].plot(y='PRES_INTERPOLATED',label=labels[n])
    plt.ylim(-50,2050)
    plt.ylabel('Pressure(dbar)')
    plt.xlabel('Density (kg/m^3)')
    plt.grid(axis='y')
    plt.gca().invert_yaxis()
    plt.title('Mean Pot. Density')

    plt.subplot(1,4,2)
    for n in range(0,len(density_NEV_li)):
        density_NEV_li[n].where(density_NEV_li[n].MASK==1).plot(y='PRES_INTERPOLATED',label=labels[n])
    plt.ylim(-50,2050)
    plt.ylabel('')
    plt.yticks(labels=[],ticks=(np.arange(0,2001,250)))
    plt.xlabel('Isopycnal Displacement (m)')
    plt.xscale('log')
    plt.grid(axis='y')    
    plt.gca().invert_yaxis()
    plt.title('Norm. Var. of Density')

    plt.subplot(1,4,3)
    for n in range(0,len(spice_mean_li)):
        spice_mean_li[n].plot(y='PRES_INTERPOLATED',label=labels[n])
    plt.ylim(-50,2050)
    plt.ylabel('')
    plt.yticks(labels=[],ticks=(np.arange(0,2001,250)))
    plt.xlabel('Spice (kg/m^3)')
    plt.grid(axis='y')
    plt.gca().invert_yaxis()
    plt.title('Mean Spice')

    plt.subplot(1,4,4)
    for n in range(0,len(spice_EV_li)):
        spice_EV_li[n].where(spice_EV_li[n].MASK==1).plot(y='PRES_INTERPOLATED',label=labels[n])
    plt.ylim(-50,2050)
    plt.ylabel('')
    plt.yticks(labels=[],ticks=(np.arange(0,2001,250)))
    plt.xlabel('Variance of Spice (m^2)')
    plt.xscale('log')
    plt.grid(axis='y')    
    plt.gca().invert_yaxis()
    plt.title('Eddy Var. of Spice')
    
    plt.legend(bbox_to_anchor=(2.3, 0.55))
    
    plt.subplots_adjust(wspace=0.3, hspace=0.25)
    
    
def postobox(lon, lat):
    y_box = [lat[0], lat[0], lat[1], lat[1], lat[0]]
    x_box = [lon[0], lon[1], lon[1], lon[0], lon[0]]
    
    return x_box, y_box

def plot_map(box_li,stats='binned_stats.nc'):
    
    # visualization
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import MaxNLocator
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.feature as cfeature
    import cmocean
    
    ds_stats = xr.open_dataset(stats)

    x_c = ds_stats.lon
    y_c = ds_stats.lat
    # get 1st and 99th percentiles of values to plot to get a useful range for the colorscale
    EKE= ds_stats.EKE
    v1,v2 = np.nanpercentile(EKE.T,[1,99])

    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1,1,1,projection=ccrs.Robinson(central_longitude=0))
    cmap = cmocean.cm.matter_r
    pcm = ax.pcolormesh(x_c, y_c, 
                        EKE, 
                        cmap=cmap, 
                        transform=ccrs.PlateCarree(),
                        vmin=v1, vmax=v2/2)

    # gridlines and labels
    gl = ax.gridlines(color='k', linewidth=0.1, linestyle='-',
                      xlocs=np.arange(-180, 181, 60), ylocs=np.arange(-90, 91, 30),
                      draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    #plt.plot(postobox(lon, lat), color='red')
    for n in range(0,len(box_li)):
        ax.plot(postobox(box_li[n][:2],box_li[n][2:])[0],postobox(box_li[n][:2], box_li[n][2:])[1], color='black',transform=ccrs.PlateCarree(),lw=2)

    # add land and coastline
    ax.add_feature(cfeature.LAND, facecolor='grey', zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.25, zorder=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05, axes_class=plt.Axes)
    cb = fig.colorbar(pcm, cax=cax);
    cb.ax.set_ylabel(r'EKE [$m^2/s^2$]');
    
    
def TS_grids(Taxis, Saxis):
    Taxis = numpy.linspace(Taxis[0],Taxis[1],Taxis[2])
    Saxis = numpy.linspace(Saxis[0],Saxis[1],Saxis[2])
    Tgrid, Sgrid = np.meshgrid(Taxis, Saxis)
    
    return Tgrid, Sgrid


def rhospice_grids(Tgrid, Sgrid):
    rho_grid = gsw.rho(Sgrid, Tgrid, 0)
    spice_grid = gsw.spiciness0(Sgrid, Tgrid)
    
    return rho_grid, spice_grid

def plot_TS(ds_li, Taxis, Saxis, variable1='CT', variable2='SA'):
    Tgrid, Sgrid = TS_grids(Taxis=Taxis, Saxis=Saxis)
    rho_grid, spice_grid = rhospice_grids(Tgrid, Sgrid)
    
    colors=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']
    
    for n in range(0,len(ds_li)):
        plt.plot(ds_li[n][variable2],ds_li[n][variable1],'.',markersize=0.5,color=colors[n])
        
    contour_rho = plt.contour(Sgrid,Tgrid,rho_grid,colors='k')
    plt.clabel(contour_rho)
    contour_spice = plt.contour(Sgrid,Tgrid,spice_grid,colors='r')
    plt.clabel(contour_spice)
    
    plt.ylabel('Temperature (°C)')
    plt.xlabel('Salinity (g/kg)')