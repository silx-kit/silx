#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2013-2017  European Synchrotron Radiation Facility, Grenoble, France
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.


from __future__ import division

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2013-06-13"
__status__ = "beta"

from math import ceil
import numpy
from silx.opencl.utils import get_opencl_code, calc_size


def kernel_size(sigma, odd=False, cutoff=4):
    """
    Calculate the optimal kernel size for a convolution with sigma

    :param sigma: width of the gaussian
    :param odd: enforce the kernel to be odd (more precise ?)
    """
    size = int(ceil(2 * cutoff * sigma + 1))
    if odd and size % 2 == 0:
        size += 1
    return size


def sizeof(shape, dtype="uint8"):
    """
    Calculate the number of bytes needed to allocate for a given structure

    :param shape: size or tuple of sizes
    :param dtype: data type
    """
    itemsize = numpy.dtype(dtype).itemsize
    cnt = 1
    if "__len__" in dir(shape):
        for dim in shape:
            cnt *= dim
    else:
        cnt = int(shape)
    return cnt * itemsize


def _gcd(a, b):
    """Calculate the greatest common divisor of a and b"""
    while b:
        a, b = b, a % b
    return a


def bin2RGB(img):
    """
    Perform a 2x2 binning of the image
    """
    dtype = img.dtype
    if dtype == numpy.uint8:
        out_dtype = numpy.int32
    else:
        out_dtype = dtype
    shape = img.shape
    if len(shape) == 3:
        new_shape = shape[0] // 2, shape[1] // 2, shape[2]
        new_img = img
    else:
        new_shape = shape[0] // 2, shape[1] // 2, 1
        new_img = img.reshape((shape[0], shape[1], 1))
    out = numpy.zeros(new_shape, dtype=out_dtype)
    out += new_img[::2, ::2, :]
    out += new_img[1::2, ::2, :]
    out += new_img[1::2, 1::2, :]
    out += new_img[::2, 1::2, :]
    out /= 4
    if len(shape) != 3:
        out.shape = new_shape[0], new_shape[1]
    if dtype == numpy.uint8:
        return out.astype(dtype)
    else:
        return out

