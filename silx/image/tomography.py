# coding: utf-8
# /*##########################################################################
# Copyright (C) 2017 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ############################################################################*/
"""
This module contains utilitary functions for tomography
"""

__author__ = ["P. Paleo"]
__license__ = "MIT"
__date__ = "12/09/2017"


import numpy as np
from math import pi
from itertools import product
from bisect import bisect
from silx.math.fit import leastsq

# ------------------------------------------------------------------------------
# -------------------- Filtering-related functions -----------------------------
# ------------------------------------------------------------------------------

def compute_ramlak_filter(dwidth_padded, dtype=np.float32):
    """
    Compute the Ramachandran-Lakshminarayanan (Ram-Lak) filter, used in
    filtered backprojection.

    :param dwidth_padded: width of the 2D sinogram after padding
    :param dtype: data type
    """
    L = dwidth_padded
    h = np.zeros(L, dtype=dtype)
    L2 = L//2+1
    h[0] = 1/4.
    j = np.linspace(1, L2, L2//2, False).astype(dtype) # np < 1.9.0
    h[1:L2:2] = -1./(pi**2 * j**2)
    h[L2:] = np.copy(h[1:L2-1][::-1])
    return h


def tukey(N, alpha=0.5):
    """
    Compute the Tukey apodization window.

    :param int N: Number of points.
    :param float alpha:
    """
    apod = np.zeros(N)
    x = np.arange(N)/(N-1)
    r = alpha
    M1 = (0 <= x) * (x < r/2)
    M2 = (r/2 <= x) * (x <= 1 - r/2)
    M3 = (1 - r/2 < x) * (x <= 1)
    apod[M1] = (1 + np.cos(2*pi/r * (x[M1] - r/2)))/2.
    apod[M2] = 1.
    apod[M3] = (1 + np.cos(2*pi/r * (x[M3] - 1 + r/2)))/2.
    return apod


def lanczos(N):
    """
    Compute the Lanczos window (truncated sinc) of width N.

    :param int N: window width
    """
    x = np.arange(N)/(N-1)
    return np.sin(pi*(2*x-1))/(pi*(2*x-1))


def compute_fourier_filter(dwidth_padded, filter_name, cutoff=1.):
    """
    Compute the filter used for FBP.

    :param dwidth_padded: padded detector width. As the filtering is done by the
        Fourier convolution theorem, dwidth_padded should be at least 2*dwidth.
    :param filter_name: Name of the filter. Available filters are:
        Ram-Lak, Shepp-Logan, Cosine, Hamming, Hann, Tukey, Lanczos.
    :param cutoff: Cut-off frequency, if relevant.
    """
    Nf = dwidth_padded
    #~ filt_f = np.abs(np.fft.fftfreq(Nf))
    rl = compute_ramlak_filter(Nf, dtype=np.float64)
    filt_f = np.fft.fft(rl)

    filter_name = filter_name.lower()
    if filter_name in ["ram-lak", "ramlak"]:
        return filt_f

    w = 2 * pi * np.fft.fftfreq(dwidth_padded)
    d = cutoff
    apodization = {
        # ~OK
        "shepp-logan": np.sin(w[1:Nf]/(2*d))/(w[1:Nf]/(2*d)),
        # ~OK
        "cosine": np.cos(w[1:Nf]/(2*d)),
        # OK
        "hamming": 0.54*np.ones_like(filt_f)[1:Nf] + .46 * np.cos(w[1:Nf]/d),
        # OK
        "hann": (np.ones_like(filt_f)[1:Nf] + np.cos(w[1:Nf]/d))/2.,
        # These one is not compatible with Astra - TODO investigate why
        "tukey": np.fft.fftshift(tukey(dwidth_padded, alpha=d/2.))[1:Nf],
        "lanczos": np.fft.fftshift(lanczos(dwidth_padded))[1:Nf],
    }
    if filter_name not in apodization:
        raise ValueError("Unknown filter %s. Available filters are %s" %
                         (filter_name, str(apodization.keys())))
    filt_f[1:Nf] *= apodization[filter_name]
    return filt_f


def generate_powers():
    """
    Generate a list of powers of [2, 3, 5, 7],
    up to (2**15)*(3**9)*(5**6)*(7**5).
    """
    primes = [2, 3, 5, 7]
    maxpow = {2: 15, 3: 9, 5: 6, 7: 5}
    valuations = []
    for prime in primes:
        # disallow any odd number (for R2C transform), and any number
        # not multiple of 4 (Ram-Lak filter behaves strangely when
        # dwidth_padded/2 is not even)
        minval = 2 if prime == 2 else 0
        valuations.append(range(minval, maxpow[prime]+1))
    powers = product(*valuations)
    res = []
    for pw in powers:
        res.append(np.prod(list(map(lambda x : x[0]**x[1], zip(primes, pw)))))
    return np.unique(res)


def get_next_power(n, powers=None):
    """
    Given a number, get the closest (upper) number p such that
    p is a power of 2, 3, 5 and 7.
    """
    if powers is None:
        powers = generate_powers()
    idx = bisect(powers, n)
    if powers[idx-1] == n:
        return n
    return powers[idx]


# ------------------------------------------------------------------------------
# ------------- Functions for determining the center of rotation  --------------
# ------------------------------------------------------------------------------



def calc_center_corr(sino, fullrot=False, props=1):
    """
    Compute a guess of the Center of Rotation (CoR) of a given sinogram.
    The computation is based on the correlation between the line projections at
    angle (theta = 0) and at angle (theta = 180).

    Note that for most scans, the (theta=180) angle is not included,
    so the CoR might be underestimated.
    In a [0, 360[ scan, the projection angle at (theta=180) is exactly in the
    middle for odd number of projections.

    :param numpy.ndarray sino: Sinogram
    :param bool fullrot: optional. If False (default), the scan is assumed to
                         be [0, 180).
                         If True, the scan is assumed to be [0, 380).
    :param int props: optional. Number of propositions for the CoR
    """

    n_a, n_d = sino.shape
    first = 0
    last = -1 if not(fullrot) else n_a // 2
    proj1 = sino[first, :]
    proj2 = sino[last, :][::-1]

    # Compute the correlation in the Fourier domain
    proj1_f = np.fft.fft(proj1, 2 * n_d)
    proj2_f = np.fft.fft(proj2, 2 * n_d)
    corr = np.abs(np.fft.ifft(proj1_f * proj2_f.conj()))

    if props == 1:
        pos = np.argmax(corr)
        if pos > n_d // 2:
            pos -= n_d
        return (n_d + pos) / 2.
    else:
        corr_argsorted = np.argsort(corr)[:props]
        corr_argsorted[corr_argsorted > n_d // 2] -= n_d
        return (n_d + corr_argsorted) / 2.


def _sine_function(t, offset, amplitude, phase):
    """
    Helper function for calc_center_centroid
    """
    n_angles = t.shape[0]
    res = amplitude * np.sin(2 * pi * (1. / (2 * n_angles)) * t + phase)
    return offset + res


def _sine_function_derivative(t, params, eval_idx):
    """
    Helper function for calc_center_centroid
    """
    offset, amplitude, phase = params
    n_angles = t.shape[0]
    w = 2.0 * pi * (1. / (2.0 * n_angles)) * t + phase
    grad = (1.0, np.sin(w), amplitude*np.cos(w))
    return grad[eval_idx]


def calc_center_centroid(sino):
    """
    Compute a guess of the Center of Rotation (CoR) of a given sinogram.
    The computation is based on the computation of the centroid of each
    projection line, which should be a sine function according to the
    Helgason-Ludwig condition.
    This method is unlikely to work in local tomography.

    :param numpy.ndarray sino: Sinogram
    """

    n_a, n_d = sino.shape
    # Compute the vector of centroids of the sinogram
    i = np.arange(n_d)
    centroids = np.sum(sino*i, axis=1)/np.sum(sino, axis=1)

    # Fit with a sine function : phase, amplitude, offset
    # Using non-linear Levenberg–Marquardt  algorithm
    angles = np.linspace(0, n_a, n_a, True)
    # Initial parameter vector
    cmax, cmin = centroids.max(), centroids.min()
    offs = (cmax + cmin) / 2.
    amp = (cmax - cmin) / 2.
    phi = 1.1
    p0 = (offs, amp, phi)

    constraints = np.zeros((3, 3))

    popt, _ = leastsq(model=_sine_function,
                      xdata=angles,
                      ydata=centroids,
                      p0=p0,
                      sigma=None,
                      constraints=constraints,
                      model_deriv=None,
                      epsfcn=None,
                      deltachi=None,
                      full_output=0,
                      check_finite=True,
                      left_derivative=False,
                      max_iter=100)
    return popt[0]



# ------------------------------------------------------------------------------
# -------------------- Visualization-related functions -------------------------
# ------------------------------------------------------------------------------


def rescale_intensity(img, from_subimg=None, percentiles=None):
    """
    clamp intensity into the [2, 98] percentiles

    :param img:
    :param from_subimg:
    :param percentiles:
    :return: the rescale intensity
    """
    if percentiles is None:
        percentiles = [2, 98]
    else:
        assert type(percentiles) in (tuple, list)
        assert(len(percentiles) == 2)
    data = from_subimg if from_subimg is not None else img
    imin, imax = np.percentile(data, percentiles)
    res = np.clip(img, imin, imax)
    return res

