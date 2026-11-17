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

def steering_weights(N, d, wl, theta0_deg, amp=None):
    """Complex emitter weights (amplitude × phase) that steer the beam to theta0."""
    n = np.arange(N)
    dphi = 2 * np.pi * d / wl * np.sin(np.radians(theta0_deg))
    a = np.ones(N) if amp is None else np.asarray(amp, float)
    return a * np.exp(-1j * n * dphi)

def pattern(theta, w, d, wl):
    """Far-field intensity for arbitrary complex emitter weights w.
    Normalised so the same amplitudes with perfect phases peak at 1."""
    n = np.arange(len(w))
    af = np.exp(1j * 2 * np.pi / wl * d * np.outer(np.sin(theta), n)) @ w
    return np.abs(af) ** 2 / np.sum(np.abs(w)) ** 2



def pattern_u(u, x, wl, amp=None):
    """Intensity vs u = sinθ − sinθ0 for emitters at positions x (any spacing),
    steered to θ0. Normalised so the main beam (u = 0) is 1."""
    a = np.ones(len(x)) if amp is None else np.asarray(amp, float)
    af = np.exp(1j * 2 * np.pi / wl * np.outer(u, x)) @ a
    return np.abs(af) ** 2 / np.sum(a) ** 2

def random_positions(N, d_avg, d_min, rng):
    """Emitter positions with random gaps between d_min and 2*d_avg − d_min,
    so the average gap is d_avg and no two emitters are closer than d_min."""
    gaps = rng.uniform(d_min, 2 * d_avg - d_min, N - 1)
    return np.concatenate([[0.0], np.cumsum(gaps)])

def peak_sidelobe(x, wl, fov_deg=60, n_u=8001):
    """Worst sidelobe (dB) visible at any steering angle within ±fov_deg."""
    umax = 1 + np.sin(np.radians(fov_deg))
    u = np.linspace(-umax, umax, n_u)
    I = pattern_u(u, x, wl)
    main = np.abs(u) < 1.1 * wl / (x[-1] - x[0])     # exclude the main beam
    return 10 * np.log10(I[~main].max())