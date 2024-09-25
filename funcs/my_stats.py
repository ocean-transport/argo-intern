import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import xrft
import scipy
import scipy.ndimage as filter


##ARGO FUNCTIONS##
def get_ds_interp(ds,depth_min,depth_max,sample_max):
    median_dp=ds.PRES.where(ds.PRES<depth_max).where(ds.PRES>depth_min).diff('N_LEVELS').median('N_LEVELS')
    ind_rate=median_dp.where(median_dp<sample_max,drop=True).N_PROF
    ds_sel=ds.sel(N_PROF=ind_rate)
    ds_interp=ds_sel.argo.interp_std_levels(np.arange(depth_min,depth_max,2))
    ds_interp=ds_interp.sortby(ds_interp.N_PROF)
    
    number=np.arange(0,len(ds_interp.N_PROF))
    ds_interp.coords['N_PROF_NEW']=xr.DataArray(number,dims=ds_interp.N_PROF.dims)
    return ds_interp

def get_ds_filt(ds_interp,first,last,num,variable='TEMP'):
    temp_sigmas=np.logspace(first,last,num)
    sigmas=np.empty(num)
    
    for n,sig in enumerate(temp_sigmas):
        sigmas[n]=sig/4/np.sqrt(12)
    
    temp=np.zeros((ds_interp.N_PROF.shape[0],ds_interp.PRES_INTERPOLATED.shape[0],num))
    for n in range(0,num):
        temp[:,:,n]=filter.gaussian_filter1d(ds_interp[variable],sigma=sigmas[n],mode='nearest')

    ds_filt=xr.DataArray(temp,dims=['N_PROF','PRES_INTERPOLATED','FILT_SCALE'],
             coords={'N_PROF':ds_interp.N_PROF,'PRES_INTERPOLATED':ds_interp.PRES_INTERPOLATED,'FILT_SCALE':sigmas})
    
    number=np.arange(0,len(ds_filt.N_PROF))
    ds_filt['N_PROF_NEW']=xr.DataArray(number,dims=ds_filt.N_PROF.dims)
    return ds_filt

def get_var(ds_interp,ds_filt,variable='TEMP'):
    var=np.zeros(len(ds_filt.FILT_SCALE))
    for n,sig in enumerate(ds_filt.FILT_SCALE):
        prof=ds_filt.sel(FILT_SCALE=sig)
        var[n]=(prof-ds_interp[variable]).var()
    return var

##GLIDER FUNCTIONS##
def glider_ds_filt(ds_interp,first,last,num,variable='CT'):
    temp_sigmas=np.logspace(first,last,num)
    sigmas=np.empty(num)
    for n,sig in enumerate(temp_sigmas):
        sigmas[n]=sig/4/np.sqrt(12)
        
    temp_filt=np.zeros((ds_interp.ctd_pressure.shape[0],ds_interp.dives.shape[0],num))
    for n in range(0,num):
        temp_filt[:,:,n]=filter.gaussian_filter1d(ds_interp[variable],sigma=sigmas[n],mode='nearest')
    ds_filt=xr.DataArray(temp_filt,dims=['ctd_pressure','dives','filt_scale'],
        coords={'ctd_pressure':ds_interp.ctd_pressure,'dives':ds_interp.dives,'filt_scale':sigmas})
    return ds_filt

def glider_var(ds_interp,ds_filt,variable='CT'):
    var=np.zeros(len(ds_filt.filt_scale))
    for n,sig in enumerate(ds_filt.filt_scale):
        prof_filt=ds_filt.sel(filt_scale=sig)
        var[n]=(prof_filt-ds_interp.CT).var()
    return var









def spectral_funct(points, modes, slope, xmax):
    k_ar=np.logspace(0,2,modes)
    a_ar=np.sqrt(0.01*k_ar**(slope/2))
    y_ar=np.zeros((points,modes))
    signal_ar=np.zeros((points,1))
    
    x=np.linspace(0,xmax,num=points)
    
    for n in range(0,modes):
        y_ar[:,n] = np.sin(k_ar[n]*x+np.random.uniform(0,6,size=1))*a_ar[n]
        
    signal=y_ar.sum(axis=1)
    
    signal_da=xr.DataArray(signal, dims=['points'], coords={'points':x})
    signal_spec=xrft.power_spectrum(signal_da,dim='points')
    
    plt.figure(figsize=(25,3))
    plt.subplot(1,2,1)
    plt.plot(x,signal,color='black')
    plt.title("Signal (slope={})".format(slope))
    
    plt.subplot(1,2,2)
    signal_spec.plot()
    plt.plot(k_ar/(2*np.pi),(xmax/np.pi)*a_ar**2)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Power Spectrum (slope={})".format(slope))
    plt.xlabel("Wavenumber, log(1/$\lambda$)")
    plt.ylabel("Amplitude, log(a**2)")
    plt.subplots_adjust(hspace=0.2)
    
def spectral_filtered_1(points, slope, xmax,sigma):
    
    modes=200
    
    k_ar=np.logspace(0,2,modes)
    a_ar=np.sqrt(0.01*k_ar**(slope/2))
    y_ar=np.zeros((points,modes))
    signal=np.zeros((points,1))

    x=np.linspace(0,xmax,num=points)

    for n in range(0,modes):
        y_ar[:,n] = np.sin(k_ar[n]*x+np.random.uniform(0,6,size=1))*a_ar[n]

    signal=y_ar.sum(axis=1)
    signal_da=xr.DataArray(signal, dims=['points'], coords={'points':x})
    signal_spec=xrft.power_spectrum(signal_da,dim='points')
    
    filtered = scipy.ndimage.gaussian_filter1d(signal, sigma=sigma, mode='wrap')
    filtered_da=xr.DataArray(filtered, dims=['points'], coords={'points':x})
    filtered_spec=xrft.power_spectrum(filtered_da,dim='points')
    
    anom=signal-filtered
    anom_da=xr.DataArray(anom, dims=['points'], coords={'points':x})

    #Plot 1: SIGNAL + FILTER
    plt.figure(figsize=(30,10))
    plt.subplot(2,2,1)
    plt.plot(x,signal_da,color='black')
    plt.plot(x,filtered_da,color='red',linewidth=3)
    plt.title("Signal and Filtered (slope={}, $\sigma$={})".format(slope,sigma))

    #Plot 2: SIGNAL SPECTRUM
    plt.subplot(2,2,2)
    signal_spec.plot()
    plt.plot(k_ar/(2*np.pi),(xmax/np.pi)*a_ar**2)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Signal Power Spectrum (slope={})".format(slope))
    plt.xlabel("Wavenumber, log(1/$\lambda$)")
    plt.ylabel("Amplitude, log(a**2)")

    #Plot 3: SIGNAL + ANOM
    plt.subplot(2,2,3)
    plt.plot(x,anom_da, color='blue')
    plt.plot(x,signal_da, color='black')
    plt.title("Signal and Anomaly (slope={}, $\sigma$={})".format(slope,sigma))

    #Plot 4: FILTER SPECTRUM
    plt.subplot(2,2,4)
    filtered_spec.plot()
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Filtered Power Spectrum (slope={})".format(slope))
    plt.xlabel("Wavenumber, log(1/$\lambda$)")
    plt.ylabel("Amplitude, log(a**2)")
    
    plt.subplots_adjust(wspace=0.4,hspace=0.4)
    
def spectral_filtered_2(points, slope, xmax, L_filter):
    
    modes=200
    
    k_ar=np.logspace(0,2,modes)
    a_ar=np.sqrt(0.01*k_ar**(slope/2))
    y_ar=np.zeros((points,modes))
    signal=np.zeros((points,1))

    x=np.linspace(0,xmax,num=points)
    dx=x[1]-x[0]
    
    sigma=L_filter/dx/np.sqrt(12)

    for n in range(0,modes):
        y_ar[:,n] = np.sin(k_ar[n]*x+np.random.uniform(0,6,size=1))*a_ar[n]

    signal=y_ar.sum(axis=1)
    signal_da=xr.DataArray(signal, dims=['points'], coords={'points':x})
    signal_spec=xrft.power_spectrum(signal_da,dim='points')

    filtered = scipy.ndimage.gaussian_filter1d(signal, sigma=sigma, mode='wrap')
    filtered_da=xr.DataArray(filtered, dims=['points'],coords={'points':x})
    filtered_spec=xrft.power_spectrum(filtered_da,dim='points')

    #Plot 1: SIGNALS
    plt.figure(figsize=(20,5))
    plt.subplot(1,2,1)
    plt.plot(x,filtered_da,color='red',linewidth=3)
    plt.plot(x,signal_da, color='black')
    plt.title("Signal (slope={}), Gaussian Filter ($\sigma$={})".format(slope,sigma))
    
    #Plot 2: SPECTRUMS
    plt.subplot(1,2,2)
    signal_spec.plot(marker='.',color='black')
    filtered_spec.plot(marker='.',color='red')
    plt.plot(k_ar/(2*np.pi),(xmax/np.pi)*a_ar**2,color='orange')
    plt.vlines(2/xmax, 1e-8,1)
    plt.vlines(1/xmax, 1e-8,1)
    plt.vlines(1/L_filter, 1e-8,1)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Signal Spectrum (black), Filtered Spectrum (red)")
    plt.xlabel("Wavenumber, log(1/$\lambda$)")
    plt.ylabel("Amplitude, log(a**2)")
    
def filter_plot(da):
    signal = da
    plt.figure(figsize=(20,15))
    plt.subplot(4,1,1)
    plt.plot(x,signal,color='black')
    plt.ylim(-int(np.max(y_funct)+10), int(np.max(y_funct)+10))
    plt.title('Signal')

    #Gaussian window
    win_gaus = scipy.signal.windows.gaussian(points, std=4)
    filt_gaus = scipy.signal.convolve(signal, win_gaus, mode='same')
    plt.subplot(4,1,2)
    plt.plot(x,filt_gaus)
    plt.ylim(-int(np.max(y_funct)+10), int(np.max(y_funct)+10))
    plt.title("Window applied: Gaussian (\u03C3=4)")

    #Exponential window
    win_exp = scipy.signal.windows.exponential(points, tau=4)
    filt_exp = scipy.signal.convolve(signal, win_exp, mode='same')
    plt.subplot(4,1,3)
    plt.plot(x,filt_exp)
    plt.ylim(-int(np.max(y_funct)+10), int(np.max(y_funct)+10))
    plt.title("Window applied: Exponential (\u03C4=4)")
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  

    #Boxcar window
    win_av = scipy.signal.windows.boxcar(points)
    filt_av = scipy.signal.convolve(signal, win_av, mode='same')
    plt.subplot(4,1,4)
    plt.plot(x,filt_av)
    #plt.ylim(-int(np.max(y_funct)+10), int(np.max(y_funct)+10))
    plt.title("Window applied: Boxcar")
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  
    
def gaussian_plot(da):
    plt.figure(figsize=(20,30))

    signal = y_funct
    plt.subplot(10,1,1)
    plt.plot(x,signal, color='black')
    plt.ylim(-int(np.max(y_funct)+10), int(np.max(y_funct)+10))
    plt.title('Signal')

    plots=[2,3,4,5]

    for n in range(2,9,2):
        window=scipy.signal.windows.gaussian(points, std=n)
        filt_gaus = scipy.signal.convolve(signal, window, mode='same')
        plt.subplot(10,1,plots[0])
        plt.plot(x,filt_gaus)
        plt.ylim(-int(np.max(y_funct)+10), int(np.max(y_funct)+10))
        plt.title("Gaussian window with \u03C3={}".format(n))
        plots.pop(0)

    plt.subplots_adjust(wspace=0.4, hspace=0.5) 
    
def exponential_plot(da):
    plt.figure(figsize=(20,30))

    signal = y_funct
    plt.subplot(10,1,1)
    plt.plot(x,signal, color='black')
    plt.ylim(-int(np.max(y_funct)+10), int(np.max(y_funct)+10))
    plt.title('Signal')

    plots=[2,3,4,5,6,7,8,9,10]

    for n in range(5,45,10):
        window=scipy.signal.windows.exponential(points, tau=n)
        filt_gaus = scipy.signal.convolve(signal, window, mode='same')
        plt.subplot(10,1,plots[0])
        plt.plot(x,filt_gaus)
        plt.ylim(-int(np.max(y_funct)+10), int(np.max(y_funct)+10))
        plt.title("Exponential window with \u03C4={}".format(n))
        plots.pop(0)

    plt.subplots_adjust(wspace=0.4, hspace=0.5) 
    
def boxcar_plot(da):
    signal = da
    plt.figure(figsize=(20,15))
    plt.subplot(4,1,1)
    plt.plot(x,signal,color='black')
    plt.ylim(-int(np.max(y_funct)+10), int(np.max(y_funct)+10))
    plt.title('Signal')
    
    win_av = scipy.signal.windows.boxcar(points)
    filt_av = scipy.signal.convolve(signal, win_av, mode='same')
    plt.subplot(2,1,2)
    plt.plot(x,filt_av)
    #plt.ylim(-int(np.max(y_funct)+10), int(np.max(y_funct)+10))
    plt.title("Window applied: Boxcar")
    plt.subplots_adjust(wspace=0.4, hspace=0.4)  