# coding: utf-8
#/*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
#############################################################################*/
"""This module provides fit functions.

List of fit functions:
-----------------------

    - :func:`sum_gauss`
    - :func:`sum_agauss`
    - :func:`sum_splitgauss`
    - :func:`sum_fastagauss`

    - :func:`sum_apvoigt`
    - :func:`sum_pvoigt`
    - :func:`sum_splitpvoigt`

    - :func:`sum_lorentz`
    - :func:`sum_alorentz`
    - :func:`sum_splitlorentz`

    - :func:`sum_downstep`
    - :func:`sum_upstep`
    - :func:`sum_slit`

    - :func:`sum_ahypermet`
    - :func:`sum_fastahypermet`

Full documentation:
-------------------

"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "17/06/2016"

import logging
import numpy

logging.basicConfig()
_logger = logging.getLogger(__name__)

cimport cython

# Rename C functions to reuse the same names for their python wrappers
from functions cimport erf_array as _erf
from functions cimport erfc_array as _erfc
from functions cimport sum_gauss as _sum_gauss
from functions cimport sum_agauss as _sum_agauss
from functions cimport sum_fastagauss as _sum_fastagauss
from functions cimport sum_splitgauss as _sum_splitgauss
from functions cimport sum_apvoigt as _sum_apvoigt
from functions cimport sum_pvoigt as _sum_pvoigt
from functions cimport sum_splitpvoigt as _sum_splitpvoigt
from functions cimport sum_lorentz as _sum_lorentz
from functions cimport sum_alorentz as _sum_alorentz
from functions cimport sum_splitlorentz as _sum_splitlorentz
from functions cimport sum_downstep as _sum_downstep
from functions cimport sum_upstep as _sum_upstep
from functions cimport sum_slit as _sum_slit
from functions cimport sum_ahypermet as _sum_ahypermet
from functions cimport sum_fastahypermet as _sum_fastahypermet


def erf(x):
    """erf(x)
    Return the gaussian error function

    :param x: Independant variable where the gaussian error function is
        calculated
    :type x: numpy.ndarray or scalar
    :return: Gaussian error function ``y=erf(x)``
    :raise: IndexError if ``x`` is an empty array
    """
    cdef:
        double[::1] x_c
        double[::1] y_c


    # force list into numpy array
    if not hasattr(x, "shape"):
        x = numpy.asarray(x)

    for len_dim in x.shape:
        if len_dim == 0:
            raise IndexError("Cannot compute erf for an empty array")

    x_c = numpy.array(x, copy=False, dtype=numpy.float64, order='C').reshape(-1)
    y_c = numpy.empty(shape=(x_c.size,), dtype=numpy.float64)

    status = _erf(&x_c[0], x_c.size, &y_c[0])

    return numpy.asarray(y_c).reshape(x.shape)


def erfc(x):
    """erfc(x) -> numpy.ndarray
    Return the gaussian complementary error function

    :param x: Independant variable where the gaussian complementary error
        function is calculated
    :type x: numpy.ndarray or scalar
    :return: Gaussian complementary error function ``y=erfc(x)``
    :type rtype: numpy.ndarray
    :raise: IndexError if ``x`` is an empty array
    """
    cdef:
        double[::1] x_c
        double[::1] y_c

    # force list into numpy array
    if not hasattr(x, "shape"):
        x = numpy.asarray(x)

    for len_dim in x.shape:
        if len_dim == 0:
            raise IndexError("Cannot compute erfc for an empty array")

    x_c = numpy.array(x, copy=False, dtype=numpy.float64, order='C').reshape(-1)
    y_c = numpy.empty(shape=(x_c.size,), dtype=numpy.float64)

    status = _erfc(&x_c[0], x_c.size, &y_c[0])

    return numpy.asarray(y_c).reshape(x.shape)


def sum_gauss(x, *params):
    """sum_gauss(x, *params) -> numpy.ndarray

    Return a sum of gaussian functions defined by *(height, centroid, fwhm)*,
    where:

        - *height* is the peak amplitude
        - *centroid* is the peak x-coordinate
        - *fwhm* is the full-width at half maximum

    :param x: Independant variable where the gaussians are calculated
    :type x: numpy.ndarray
    :param params: Array of gaussian parameters (length must be a multiple
        of 3):
        *(height1, centroid1, fwhm1, height2, centroid2, fwhm2,...)*
    :return: Array of sum of gaussian functions at each ``x`` coordinate.
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    # ensure float64 (double) type and 1D contiguous data layout in memory
    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_gauss(&x_c[0], x.size, &params_c[0], params_c.size, &y_c[0])

    if status:
        raise IndexError("Wrong number of parameters for function")

    # reshape y_c to match original, possibly unusual, data shape
    return numpy.asarray(y_c).reshape(x.shape)


def sum_agauss(x, *params):
    """sum_agauss(x, *params) -> numpy.ndarray

    Return a sum of gaussian functions defined by *(area, centroid, fwhm)*,
    where:

        - *area* is the area underneath the peak
        - *centroid* is the peak x-coordinate
        - *fwhm* is the full-width at half maximum

    :param x: Independant variable where the gaussians are calculated
    :type x: numpy.ndarray
    :param params: Array of gaussian parameters (length must be a multiple
        of 3):
        *(area1, centroid1, fwhm1, area2, centroid2, fwhm2,...)*
    :return: Array of sum of gaussian functions at each ``x`` coordinate.
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_agauss(&x_c[0], x.size, &params_c[0], params_c.size, &y_c[0])

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)


def sum_fastagauss(x, *params):
    """sum_fastagauss(x, *params) -> numpy.ndarray

    Return a sum of gaussian functions defined by *(area, centroid, fwhm)*,
    where:

        - *area* is the area underneath the peak
        - *centroid* is the peak x-coordinate
        - *fwhm* is the full-width at half maximum

    This implementation differs from :func:`sum_agauss` by the usage of a
    lookup table with precalculated exponential values. This might speed up
    the computation for large numbers of individual gaussian functions.

    :param x: Independant variable where the gaussians are calculated
    :type x: numpy.ndarray
    :param params: Array of gaussian parameters (length must be a multiple
        of 3):
        *(area1, centroid1, fwhm1, area2, centroid2, fwhm2,...)*
    :return: Array of sum of gaussian functions at each ``x`` coordinate.
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_fastagauss(&x_c[0], x.size, &params_c[0], params_c.size, &y_c[0])

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)


def sum_splitgauss(x, *params):
    """sum_splitgauss(x, *params) -> numpy.ndarray

    Return a sum of gaussian functions defined by *(area, centroid, fwhm)*,
    where:

        - *height* is the peak amplitude
        - *centroid* is the peak x-coordinate
        - *fwhm1* is the full-width at half maximum for the distribution
          when ``x < centroid``
        - *fwhm2* is the full-width at half maximum for the distribution
          when  ``x > centroid``

    :param x: Independant variable where the gaussians are calculated
    :type x: numpy.ndarray
    :param params: Array of gaussian parameters (length must be a multiple
        of 4):
        *(height1, centroid1, fwhm11, fwhm21, height2, centroid2, fwhm12, fwhm22,...)*
    :return: Array of sum of split gaussian functions at each ``x`` coordinate
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_splitgauss(&x_c[0], x.size, &params_c[0], params_c.size, &y_c[0])

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)


def sum_apvoigt(x, *params):
    """sum_apvoigt(x, *params) -> numpy.ndarray

    Return a sum of pseudo-Voigt functions, defined by *(area, centroid, fwhm,
    eta)*.

    The pseudo-Voigt profile ``PV(x)`` is an approximation of the Voigt
    profile using a linear combination of a Gaussian curve ``G(x)`` and a
    Lorentzian curve ``L(x)`` instead of their convolution.

        - *area* is the area underneath both G(x) and L(x)
        - *centroid* is the peak x-coordinate for both functions
        - *fwhm* is the full-width at half maximum of both functions
        - *eta* is the Lorentz factor: PV(x) = eta * L(x) + (1 - eta) * G(x)

    :param x: Independant variable where the gaussians are calculated
    :type x: numpy.ndarray
    :param params: Array of pseudo-Voigt parameters (length must be a multiple
        of 4):
        *(area1, centroid1, fwhm1, eta1, area2, centroid2, fwhm2, eta2,...)*
    :return: Array of sum of pseudo-Voigt functions at each ``x`` coordinate
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_apvoigt(&x_c[0], x.size, &params_c[0], params_c.size, &y_c[0])

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)


def sum_pvoigt(x, *params):
    """sum_pvoigt(x, *params) -> numpy.ndarray

    Return a sum of pseudo-Voigt functions, defined by *(height, centroid,
    fwhm, eta)*.

    The pseudo-Voigt profile ``PV(x)`` is an approximation of the Voigt
    profile using a linear combination of a Gaussian curve ``G(x)`` and a
    Lorentzian curve ``L(x)`` instead of their convolution.

        - *height* is the peak amplitude of G(x) and L(x)
        - *centroid* is the peak x-coordinate for both functions
        - *fwhm* is the full-width at half maximum of both functions
        - *eta* is the Lorentz factor: PV(x) = eta * L(x) + (1 - eta) * G(x)

    :param x: Independant variable where the gaussians are calculated
    :type x: numpy.ndarray
    :param params: Array of pseudo-Voigt parameters (length must be a multiple
        of 4):
        *(height1, centroid1, fwhm1, eta1, height2, centroid2, fwhm2, eta2,...)*
    :return: Array of sum of pseudo-Voigt functions at each ``x`` coordinate
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_pvoigt(&x_c[0], x.size, &params_c[0], params_c.size, &y_c[0])

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)


def sum_splitpvoigt(x, *params):
    """sum_splitpvoigt(x, *params) -> numpy.ndarray

    Return a sum of split pseudo-Voigt functions, defined by *(height,
    centroid, fwhm1, fwhm2, eta)*.

    The pseudo-Voigt profile ``PV(x)`` is an approximation of the Voigt
    profile using a linear combination of a Gaussian curve ``G(x)`` and a
    Lorentzian curve ``L(x)`` instead of their convolution.

        - *height* is the peak amplitudefor G(x) and L(x)
        - *centroid* is the peak x-coordinate for both functions
        - *fwhm1* is the full-width at half maximum of both functions
          when ``x < centroid``
        - *fwhm2* is the full-width at half maximum of both functions
          when ``x > centroid``
        - *eta* is the Lorentz factor: PV(x) = eta * L(x) + (1 - eta) * G(x)

    :param x: Independant variable where the gaussians are calculated
    :type x: numpy.ndarray
    :param params: Array of pseudo-Voigt parameters (length must be a multiple
        of 5):
        *(height1, centroid1, fwhm11, fwhm21, eta1,...)*
    :return: Array of sum of split pseudo-Voigt functions at each ``x``
        coordinate
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_splitpvoigt(&x_c[0], x.size, &params_c[0], params_c.size, &y_c[0])

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)


def sum_lorentz(x, *params):
    """sum_lorentz(x, *params) -> numpy.ndarray

    Return a sum of Lorentz distributions, also known as Cauchy distribution,
    defined by *(height, centroid, fwhm)*.

        - *height* is the peak amplitude
        - *centroid* is the peak x-coordinate
        - *fwhm* is the full-width at half maximum

    :param x: Independant variable where the gaussians are calculated
    :type x: numpy.ndarray
    :param params: Array of Lorentz parameters (length must be a multiple
        of 3):
        *(height1, centroid1, fwhm1,...)*
    :return: Array of sum Lorentz functions at each ``x``
        coordinate
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_lorentz(&x_c[0], x.size, &params_c[0], params_c.size, &y_c[0])

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)


def sum_alorentz(x, *params):
    """sum_alorentz(x, *params) -> numpy.ndarray

    Return a sum of Lorentz distributions, also known as Cauchy distribution,
    defined by *(area, centroid, fwhm)*.

        - *area* is the area underneath the peak
        - *centroid* is the peak x-coordinate for both functions
        - *fwhm* is the full-width at half maximum

    :param x: Independant variable where the gaussians are calculated
    :type x: numpy.ndarray
    :param params: Array of Lorentz parameters (length must be a multiple
        of 3):
        *(area1, centroid1, fwhm1,...)*
    :return: Array of sum of Lorentz functions at each ``x``
        coordinate
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_alorentz(&x_c[0],
                           x.size,
                           &params_c[0],
                           params_c.size,
                           &y_c[0])

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)


def sum_splitlorentz(x, *params):
    """sum_splitlorentz(x, *params) -> numpy.ndarray

    Return a sum of split Lorentz distributions,
    defined by *(height, centroid, fwhm1, fwhm2)*.

        - *height* is the peak amplitude
        - *centroid* is the peak x-coordinate for both functions
        - *fwhm1* is the full-width at half maximum for ``x < centroid``
        - *fwhm2* is the full-width at half maximum for ``x > centroid``

    :param x: Independant variable where the gaussians are calculated
    :type x: numpy.ndarray
    :param params: Array of Lorentz parameters (length must be a multiple
        of 4):
        *(height1, centroid1, fwhm11, fwhm21...)*
    :return: Array of sum of Lorentz functions at each ``x``
        coordinate
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_splitlorentz(&x_c[0],
                               x.size,
                               &params_c[0],
                               params_c.size,
                               &y_c[0])

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)


def sum_downstep(x, *params):
    """sum_downstep(x, *params) -> numpy.ndarray

    Return a sum of downstep functions.
    defined by *(height, centroid, fwhm)*.

        - *height* is the step's amplitude
        - *centroid* is the step's x-coordinate
        - *fwhm* is the full-width at half maximum for the derivative,
          which is a measure of the *sharpness* of the step-down's edge

    :param x: Independant variable where the gaussians are calculated
    :type x: numpy.ndarray
    :param params: Array of downstep parameters (length must be a multiple
        of 3):
        *(height1, centroid1, fwhm1,...)*
    :return: Array of sum of downstep functions at each ``x``
        coordinate
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_downstep(&x_c[0],
                           x.size,
                           &params_c[0],
                           params_c.size,
                           &y_c[0])

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)


def sum_upstep(x, *params):
    """sum_upstep(x, *params) -> numpy.ndarray

    Return a sum of upstep functions.
    defined by *(height, centroid, fwhm)*.

        - *height* is the step's amplitude
        - *centroid* is the step's x-coordinate
        - *fwhm* is the full-width at half maximum for the derivative,
          which is a measure of the *sharpness* of the step-up's edge

    :param x: Independant variable where the gaussians are calculated
    :type x: numpy.ndarray
    :param params: Array of upstep parameters (length must be a multiple
        of 3):
        *(height1, centroid1, fwhm1,...)*
    :return: Array of sum of upstep functions at each ``x``
        coordinate
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_upstep(&x_c[0],
                         x.size,
                         &params_c[0],
                         params_c.size,
                         &y_c[0])

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)


def sum_slit(x, *params):
    """sum_slit(x, *params) -> numpy.ndarray

    Return a sum of slit functions.
    defined by *(height, position, fwhm, beamfwhm)*.

        - *height* is the slit's amplitude
        - *position* is the center of the slit's x-coordinate
        - *fwhm* is the full-width at half maximum of the slit
        - *beamfwhm* is the full-width at half maximum of the
          derivative, which is a measure of the *sharpness*
          of the edges of the slit

    :param x: Independant variable where the slits are calculated
    :type x: numpy.ndarray
    :param params: Array of slit parameters (length must be a multiple
        of 4):
        *(height1, centroid1, fwhm1, beamfwhm1,...)*
    :return: Array of sum of slit functions at each ``x``
        coordinate
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_slit(&x_c[0],
                       x.size,
                       &params_c[0],
                       params_c.size,
                       &y_c[0])

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)


def sum_ahypermet(x, *params,
                  gaussian_term=True, st_term=True, lt_term=True, step_term=True):
    """sum_ahypermet(x, *params) -> numpy.ndarray

    Return a sum of ahypermet functions.
    defined by *(area, position, fwhm, st_area_r, st_slope_r, lt_area_r,
    lt_slope_r, step_height_r)*.

        - *area* is the area underneath the gaussian peak
        - *position* is the center of the various peaks and the position of
          the step down
        - *fwhm* is the full-width at half maximum of the terms
        - *st_area_r* is factor between the gaussian area and the area of the
          short tail term
        - *st_slope_r* is a ratio related to the slope of the short tail
          in the low ``x`` values (the lower, the steeper)
        - *lt_area_r* is ratio between the gaussian area and the area of the
          long tail term
        - *lt_slope_r* is a ratio related to the slope of the long tail
          in the low ``x`` values  (the lower, the steeper)
        - *step_height_r* is the ratio between the height of the step down
          and the gaussian height

    A hypermet function is a sum of four functions (terms):

        - a gaussian term
        - a long tail term
        - a short tail term
        - a step down term

    :param x: Independant variable where the hypermets are calculated
    :type x: numpy.ndarray
    :param params: Array of hypermet parameters (length must be a multiple
        of 8):
        *(area1, position1, fwhm1, st_area_r1, st_slope_r1, lt_area_r1,
        lt_slope_r1, step_height_r1...)*
    :param gaussian_term: If ``True``, enable gaussian term. Default ``True``
    :param st_term: If ``True``, enable gaussian term. Default ``True``
    :param lt_term: If ``True``, enable gaussian term. Default ``True``
    :param step_term: If ``True``, enable gaussian term. Default ``True``
    :return: Array of sum of hypermet functions at each ``x`` coordinate
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    # Sum binary flags to activate various terms of the equation
    tail_flags = 1 if gaussian_term else 0
    if st_term:
        tail_flags += 2
    if lt_term:
        tail_flags += 4
    if step_term:
        tail_flags += 8

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_ahypermet(&x_c[0],
                            x.size,
                            &params_c[0],
                            params_c.size,
                            &y_c[0],
                            tail_flags)

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)


def sum_fastahypermet(x, *params,
                      gaussian_term=True, st_term=True,
                      lt_term=True, step_term=True):
    """sum_fastahypermet(x, *params) -> numpy.ndarray

    Return a sum of hypermet functions defined by *(area, position, fwhm,
    st_area_r, st_slope_r, lt_area_r, lt_slope_r, step_height_r)*.

        - *area* is the area underneath the gaussian peak
        - *position* is the center of the various peaks and the position of
          the step down
        - *fwhm* is the full-width at half maximum of the terms
        - *st_area_r* is factor between the gaussian area and the area of the
          short tail term
        - *st_slope_r* is a parameter related to the slope of the short tail
          in the low ``x`` values (the lower, the steeper)
        - *lt_area_r* is factor between the gaussian area and the area of the
          long tail term
        - *lt_slope_r* is a parameter related to the slope of the long tail
          in the low ``x`` values  (the lower, the steeper)
        - *step_height_r* is the factor between the height of the step down
          and the gaussian height

    A hypermet function is a sum of four functions (terms):

        - a gaussian term
        - a long tail term
        - a short tail term
        - a step down term

    This function differs from :func:`sum_ahypermet` by the use of a lookup
    table for calculating exponentials. This offers better performance when
    calculating many functions for large ``x`` arrays.

    :param x: Independant variable where the hypermets are calculated
    :type x: numpy.ndarray
    :param params: Array of hypermet parameters (length must be a multiple
        of 8):
        *(area1, position1, fwhm1, st_area_r1, st_slope_r1, lt_area_r1,
        lt_slope_r1, step_height_r1...)*
    :param gaussian_term: If ``True``, enable gaussian term. Default ``True``
    :param st_term: If ``True``, enable gaussian term. Default ``True``
    :param lt_term: If ``True``, enable gaussian term. Default ``True``
    :param step_term: If ``True``, enable gaussian term. Default ``True``
    :return: Array of sum of hypermet functions at each ``x`` coordinate
    """
    cdef:
        double[::1] x_c
        double[::1] params_c
        double[::1] y_c

    # Sum binary flags to activate various terms of the equation
    tail_flags = 1 if gaussian_term else 0
    if st_term:
        tail_flags += 2
    if lt_term:
        tail_flags += 4
    if step_term:
        tail_flags += 8

    # TODO (maybe):
    # Set flags according to params, to move conditional
    # branches out of the C code.
    # E.g., set st_term = False if any of the st_slope_r params
    # (params[8*i + 4]) is 0, to prevent division by 0. Same thing for
    # lt_slope_r (params[8*i + 6]) and lt_term.

    x_c = numpy.array(x,
                      copy=False,
                      dtype=numpy.float64,
                      order='C').reshape(-1)
    params_c = numpy.array(params,
                           copy=False,
                           dtype=numpy.float64,
                           order='C').reshape(-1)
    y_c = numpy.empty(shape=(x.size,),
                      dtype=numpy.float64)

    status = _sum_fastahypermet(&x_c[0],
                               x.size,
                               &params_c[0],
                               params_c.size,
                               &y_c[0],
                               tail_flags)

    if status:
        raise IndexError("Wrong number of parameters for function")

    return numpy.asarray(y_c).reshape(x.shape)

