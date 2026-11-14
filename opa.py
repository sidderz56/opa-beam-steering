import numpy as np

def array_intensity(theta, N, d, wl, dphi):
    """Normalised far-field intensity of a uniform 1D emitter array.

    theta : angles in radians
    N     : number of emitters
    d     : emitter spacing (same units as wl)
    wl    : wavelength
    dphi  : phase step between neighbouring emitters (radians)
    """
    psi = 2 * np.pi / wl * d * np.sin(theta) - dphi
    af = np.exp(1j * np.outer(psi, np.arange(N))).sum(axis=1)
    return np.abs(af) ** 2 / N ** 2   # peak normalised to 1

def find_peaks(theta, I, threshold=0.0):
    """Angles (deg) and intensities of local maxima above a threshold."""
    idx = np.where((I[1:-1] > I[:-2]) & (I[1:-1] > I[2:]))[0] + 1
    idx = idx[I[idx] > threshold]
    return np.degrees(theta[idx]), I[idx]