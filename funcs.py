import numpy as np
import scipy.fft as fft
from scipy.ndimage.filters import convolve1d

def random_phase_power_law_function(alpha, N, L=2*np.pi):
    '''
    Returns a periodic signal generated with a specified
    energy spectrum.
    '''
    
    #L = 2*np.pi
    #N = 128 # Change N to change resolution of the signal
    
    dx = L/N

    n = np.arange(0, N)
    x = n*dx
    k = np.arange(0, N) # all ks
    
    inc_k = slice(1, int(N/2)) # only ks for positive frequencies
    k_positive = k[inc_k] # The case for even
    k_positive_units = k_positive/ N /dx
    
    k_smallest = k_positive_units[0]
    k_break = k_smallest*5 # where slope break is

    fhat_random_phase =np.zeros_like(k).astype(np.complex128)

    Ek = np.zeros_like(k_positive_units)

    #alpha = 2. # power law slope

    C = 1/ (k_break**(-alpha)) # matching coeff.

    Ek[k_positive_units<=k_break] = 1.
    Ek[k_positive_units>k_break] = C*k_positive_units[k_positive_units>k_break]**(-alpha) # Note that the energy spectrum is square of fhat

    random_theta = np.random.uniform(low=0, high=2*np.pi, size=(len(Ek),))

    fhat_random_phase[inc_k] = Ek**0.5 * np.exp(1j*random_theta)

    # Set negative freq to be the same
    fhat_random_phase[-1: - int(N/2): -1] = np.conjugate(fhat_random_phase[inc_k])
    
    f_gen1 = fft.ifft(N*fhat_random_phase)
    
    return f_gen1.real, x, dx
    
    
def power_spectrum(y, dx): 
    '''
    Computes power spectrum of some signal,
    and the spacing dx allows one to determine the 
    corresponding frequencies in the right units [1/m].
    '''
    N = len(y)
    fft_freq = fft.fftfreq(N, d = dx) 
    
    yhat = fft.fft(y)
    
    inc_k = slice(1, int(N/2)) 
    yhat = yhat[inc_k]
    fft_freq = fft_freq[inc_k]
    
    return ((np.abs(yhat)/N )**2), fft_freq




def Ghat(k, m, sigma=1.):
     return np.exp( -1 / (2*m) * (k**(2*m)) * 
                    (2*np.pi*sigma)**(2*m) )
    
def G_m(sigma, radius, m=1.):
    """
    Computes a 1D Van der ven kernel
    
    Everything is in grid point units 
    """
    
    x = np.arange(-radius, radius+1)
    N = len(x)
    k = fft.fftfreq(N)
    
    Ghat_m = Ghat(k, m, sigma)
    
    G_m = np.roll( fft.ifft(Ghat_m), int ((N-1)/2)) 
    
    return G_m , x 

def my_filter1d_vdv(f, sigma, kernel, m=1, mode='wrap', truncate=4.0):
    """ wrapper function for convolve"""
    lw = int(truncate * float(sigma) + 0.5)
    weights,_ = kernel(sigma, lw, m)
    
    return convolve1d(f, weights, mode = mode)


def top_hat_kernel(width, radius):
    """
    Computes a 1D Gaussian convolution kernel.
    """
    x = np.arange(-radius, radius + 1) # define the range over which we need the values, +1  because of weird functionality of arange
    
    phi_x = 0.*x
    phi_x[abs(x)<=width/2] = 1./width
    
    # normalization
    phi_x /= phi_x.sum()
    
    return phi_x

def gaussian_kernel(sigma, radius):
    """
    Computes a 1D Gaussian convolution kernel.
    """
    
    p = np.polynomial.Polynomial([0, 0, -0.5 / (sigma * sigma)])
    
    x = np.arange(-radius, radius + 1)
    
    # E^(p(x))
    phi_x = np.exp(p(x), dtype=np.double)
    
    # normalization
    phi_x /= phi_x.sum()
        
    return phi_x

def my_filter1d(f, sigma, kernel, mode='wrap', truncate=4.0):
    """ wrapper function for convolve"""
    lw = int(truncate * float(sigma) + 0.5)
    weights = kernel(sigma, lw)
    
    return convolve1d(f, weights, mode = mode)