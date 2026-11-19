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
    steered to θ0. amp may be complex (amplitude × phase error).
    Normalised so the error-free main beam (u = 0) is 1."""
    a = np.ones(len(x)) if amp is None else np.asarray(amp)          # no float() any more
    af = np.exp(1j * 2 * np.pi / wl * np.outer(u, x)) @ a
    return np.abs(af) ** 2 / np.sum(np.abs(a)) ** 2                  # |a| so complex weights work

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


def power_budget(x, wl, theta0_deg=0.0, amp=None, pts_per_lobe=50):
    """Split the radiated power into main lobe vs everything else.

    Integrates intensity over the visible region in u = sinθ − sinθ0
    (for isotropic emitters, power is spread evenly in sinθ).
    Main lobe = |u| < λ/L_eff, the first null of an equivalent uniform array.
    """
    x = np.asarray(x, float)
    N = len(x)
    s0 = np.sin(np.radians(theta0_deg))
    L_eff = np.ptp(x) * N / (N - 1)               # = N·d for a uniform array
    u_null = wl / L_eff
    n_u = int(2 / (u_null / pts_per_lobe)) + 1    # fine enough to resolve the main lobe
    u = np.linspace(-1 - s0, 1 - s0, n_u)         # visible region only

    I = np.empty(n_u)
    for i in range(0, n_u, 20000):                # chunked to keep memory small for large N
        I[i:i + 20000] = pattern_u(u[i:i + 20000], x, wl, amp)

    trap = lambda y, uu: np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(uu))
    main = np.abs(u) <= u_null
    total = trap(I, u)
    main_power = trap(I[main], u[main])
    return {"beam_eff": main_power / total, "main": main_power, "total": total,
            "u": u, "I": I, "u_null": u_null}


def _climb(I, i):
    """Move from index i up to the nearest local maximum."""
    while i < len(I) - 1 and I[i + 1] > I[i]:
        i += 1
    while i > 0 and I[i - 1] > I[i]:
        i -= 1
    return i

def lobe_extent(I, i):
    """Indices (l, r) of the lobe peaking at index i, out to its first minima."""
    l = r = i
    while r < len(I) - 1 and I[r + 1] < I[r]:
        r += 1
    while l > 0 and I[l - 1] < I[l]:
        l -= 1
    return l, r

def power_breakdown(pb, strong=0.5):
    """Split the power from power_budget() into: main lobe, grating lobes (other lobes
    above `strong` × peak), and sidelobes/background. Also returns the worst
    non-main level relative to the main peak, in dB."""
    u, I = pb["u"], pb["I"]
    trap = lambda y: np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(u))
    total = trap(I)

    main = np.zeros(len(I), bool)
    l0, r0 = lobe_extent(I, _climb(I, np.argmin(np.abs(u))))
    main[l0:r0 + 1] = True
    peak = I[main].max()

    peaks = list(np.where((I[1:-1] > I[:-2]) & (I[1:-1] >= I[2:]) & (I[1:-1] > strong * peak))[0] + 1)
    if I[0] > strong * peak and I[0] > I[1]:       # half-lobes sitting on the ±90° edges
        peaks.append(0)
    if I[-1] > strong * peak and I[-1] > I[-2]:
        peaks.append(len(I) - 1)

    grating = np.zeros(len(I), bool)
    for p in peaks:
        if not main[p]:
            l, r = lobe_extent(I, p)
            grating[l:r + 1] = True
    grating &= ~main

    f_main, f_grat = trap(I * main) / total, trap(I * grating) / total
    return {"main": f_main, "grating": f_grat, "other": 1 - f_main - f_grat,
            "worst_dB": 10 * np.log10(I[~main].max() / peak)}