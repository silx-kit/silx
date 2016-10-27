#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Sift implementation in Python + OpenCL
#             https://github.com/silx-kit/silx
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
__date__ = "27/09/2016"
__status__ = "beta"

import numpy
from ..resources import resource_filename
from math import log, ceil

def calc_size(shape, blocksize):
    """
    Calculate the optimal size for a kernel according to the workgroup size
    """
    if "__len__" in dir(blocksize):
        return tuple((int(i) + int(j) - 1) & ~(int(j) - 1) for i, j in zip(shape, blocksize))
    else:
        return tuple((int(i) + int(blocksize) - 1) & ~(int(blocksize) - 1) for i in shape)


def nextpower(n):
    """calculate the power of two
    
    :param n: an integer, for example 100
    :return: another integer, 100-> 128
    """
    return 1<<int(ceil(log(n,2)))


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


def get_opencl_code(name):
    """Read the kernel source code  and return it.

    :param str name: Filename of the kernel source,
    :return: Corresponding surce code
    :raises: ValueError when name is not known
    """
    if not name.endswith(".cl"):
        name += ".cl"
    try:
        filename = resource_filename('opencl/sift/' + name)
    except ValueError:
        raise ValueError('Not an OpenCL kernel file: %s' % filename)
    with open(filename, "r") as fileobj:
        res = fileobj.read()
    return res.strip()


