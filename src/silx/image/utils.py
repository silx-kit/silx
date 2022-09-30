# /*##########################################################################
# Copyright (C) 2019 European Synchrotron Radiation Facility
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

import numpy as np
from math import ceil

def gaussian_kernel(sigma, cutoff=4, force_odd_size=False):
    """
    Generates a Gaussian convolution kernel.

    :param sigma: Standard Deviation of the Gaussian curve.
    :param cutoff: Parameter tuning the truncation of the Gaussian.
        The higher cutoff, the biggest the array will be (and the closest to
        a "true" Gaussian function).
    :param force_odd_size: when set to True, the resulting array will always
        have an odd size, regardless of the values of "sigma" and "cutoff".
    :return: a numpy.ndarray containing the truncated Gaussian function.
        The array size is 2*c*s+1 where c=cutoff, s=sigma.

    Nota: due to the quick decay of the Gaussian function, small values of the
        "cutoff" parameter are usually fine. The energy difference between a
        Gaussian truncated to [-c, c] and a "true" one is
            erfc(c/(sqrt(2)*s))
        so choosing cutoff=4*sigma keeps the truncation error below 1e-4.
    """
    size = int(ceil(2 * cutoff * sigma + 1))
    if force_odd_size and size % 2 == 0:
        size += 1
    x = np.arange(size) - (size - 1.0) / 2.0
    g = np.exp(-(x / sigma) ** 2 / 2.0)
    g /= g.sum()
    return g
