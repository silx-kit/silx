# -*- coding: utf-8 -*-
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
Project: Sift implementation in Python + OpenCL
         https://github.com/silx-kit/silx
"""

from __future__ import division

__authors__ = ["Jérôme Kieffer", "Pierre Paleo"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/09/2017"
__status__ = "Production"

import os
import numpy
from .. import resources
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
    """Calculate the power of two

    :param n: an integer, for example 100
    :return: another integer, 100-> 128
    """
    return 1 << int(ceil(log(n, 2)))


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


def get_cl_file(resource):
    """get the full path of a openCL resource file

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx".
    See also :func:`silx.resources.register_resource_directory`.

    :param str resource: Resource name. File name contained if the `opencl`
        directory of the resources.
    :return: the full path of the openCL source file
    """
    if not resource.endswith(".cl"):
        resource += ".cl"
    return resources._resource_filename(resource,
                                        default_directory="opencl")


def read_cl_file(filename):
    """
    :param filename: read an OpenCL file and apply a preprocessor
    :return: preprocessed source code
    """
    with open(get_cl_file(filename), "r") as f:
        # Dummy preprocessor which removes the #include
        lines = [i for i in f.readlines() if not i.startswith("#include ")]
    return "".join(lines)


get_opencl_code = read_cl_file


def concatenate_cl_kernel(filenames):
    """Concatenates all the kernel from the list of files

    :param filenames: filenames containing the kernels
    :type filenames: list of str which can be filename of kernel as a string.
    :return: a string with all kernels concatenated

    this method concatenates all the kernel from the list
    """
    return os.linesep.join(read_cl_file(fn) for fn in filenames)
