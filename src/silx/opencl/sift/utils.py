#!/usr/bin/env python
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2013-2018  European Synchrotron Radiation Facility, Grenoble, France
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

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "2018-01-09"
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


def matching_correction(matching):
    '''
    Given the matching between two list of keypoints,
    return the linear transformation to correct kp2 with respect to kp1
    '''
    N = matching.shape[0]
    # solving normals equations for least square fit
    #
    # We correct for linear transformations, mapping points (x, y)
    # to points (x', y') :
    #
    #   x' = a*x + b*y + c
    #   y' = d*x + e*y + f
    #
    # where the parameters a, ..., f  determine the linear transformation.
    # The equivalent matrix form is
    #
    #   x1  y1  1   0   0   0           a       x1'
    #   0   0   0   x1  y1  1           b       y1'
    #   x2  y2  1   0   0   0     x     c   =   x2'
    #   0   0   0   x2  y2  1           d       y2'
    #       . . . . . .                 e       .
    #                                   f       .
    X = numpy.zeros((2 * N, 6))
    X[::2, 2:] = 1, 0, 0, 0
    X[::2, 0] = matching.x[:, 0]
    X[::2, 1] = matching.y[:, 0]
    X[1::2, 0:3] = 0, 0, 0
    X[1::2, 3] = matching.x[:, 0]
    X[1::2, 4] = matching.y[:, 0]
    X[1::2, 5] = 1
    y = numpy.zeros((2 * N, 1))
    y[::2, 0] = matching.x[:, 1]
    y[1::2, 0] = matching.y[:, 1]

    # A = numpy.dot(X.transpose(), X)
    # sol = numpy.dot(numpy.linalg.inv(A), numpy.dot(X.transpose(), y))
    # sol = numpy.dot(numpy.linalg.pinv(X),y) #pseudo-inverse is slower but numerically stable
    # MSE = numpy.linalg.norm(y - numpy.dot(X,sol))**2/N #Mean Squared Error, if needed

    sol, sqmse, rnk, svals = numpy.linalg.lstsq(X, y, rcond=-1)
    return sol
