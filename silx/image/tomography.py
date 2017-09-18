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
from silx.math.fit import leastsq


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
