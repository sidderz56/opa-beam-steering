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

def fwhm(theta, I):
    """Full width at half maximum (degrees) of the main beam, with linear interpolation."""
    th = np.degrees(theta)
    i0 = I.argmax()
    half = I[i0] / 2
    j = i0                                     # walk right until below half max
    while j < len(I) - 1 and I[j] > half:
        j += 1
    right = np.interp(half, [I[j], I[j - 1]], [th[j], th[j - 1]])
    j = i0                                     # walk left until below half max
    while j > 0 and I[j] > half:
        j -= 1
    left = np.interp(half, [I[j], I[j + 1]], [th[j], th[j + 1]])
    return right - left