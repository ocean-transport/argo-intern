import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import random
import xrft

def spectral_plot_example(da, points, modes, slope, a_start, k_start):
    a_li=np.zeros((modes,1))
    k_li=np.zeros((modes,1))
    y_li=np.zeros((points,modes))
    y_funct=np.zeros((points,1))

    a_li[0,0]=a_start
    k_li[0,0]=k_start

    for n in range(0,modes-1):
        this_a=a_li[n,0]
        next_a=int(this_a+slope)
        a_li[n+1,0]=next_a

        this_k=k_li[n,0]
        next_k=int(this_k+1)
        k_li[n+1,0]=next_k
        
    x = np.linspace(0, 2*np.pi, num=points)

    for n in range(0,modes):
        y_li[:,n] = np.sin(int(k_li[n])*x)*int(a_li[n])

    y_funct = y_li.sum(axis=1)

    plt.subplot(1,2,1)
    plt.plot(x,y_funct, color='black')
    plt.title("Signal")
    
    da_funct = xr.DataArray(y_funct, dims=['points'], coords={'points':x})
    da_spec = xrft.power_spectrum(da_funct, dim='points')
    plt.subplot(1,2,2)
    da_spec.plot()
    plt.xlim(7,10)
    plt.title("Sepctral Curve")
    
def spectral_plot(da, dim):
    da_spec = xrft.power_spectrum(da, dim=dim)
    da_spec.plot()
    
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