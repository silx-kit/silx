# coding: utf-8
# /*##########################################################################
# Copyright (C) 2017-2018 European Synchrotron Radiation Facility
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
This module provides :func:`medfilt2d`, a 2D median filter function
with the choice between 2 implementations: 'cpp' and 'opencl'.
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "04/05/2017"


import logging

from silx.math import medianfilter as medianfilter_cpp
from silx.opencl import ocl as _ocl
if _ocl is not None:
    from silx.opencl import medfilt as medfilt_opencl
else:  # No OpenCL device or pyopencl not installed
    medfilt_opencl = None


_logger = logging.getLogger(__name__)


MEDFILT_ENGINES = ['cpp', 'opencl']


def medfilt2d(image, kernel_size=3, engine='cpp'):
    """Apply a median filter on an image.

    This median filter is using a 'nearest' padding for values
    past the array edges. If you want more padding options or
    functionalities for the median filter (conditional filter 
    for example) please have a look at
    :mod:`silx.math.medianfilter`.

    :param numpy.ndarray image: the 2D array for which we want to apply
        the median filter.
    :param kernel_size: the dimension of the kernel.
        Kernel size must be odd.
        If a scalar is given, then it is used as the size in both dimension.
        Default: (3, 3)
    :type kernel_size: A int or a list of 2 int (kernel_height, kernel_width)
    :param engine: the type of implementation to use.
        Valid values are: 'cpp' (default) and 'opencl'

    :returns: the array with the median value for each pixel.

    .. note::  if the opencl implementation is requested but
        is not present or fails, the cpp implementation is called.

    """
    if engine not in MEDFILT_ENGINES:
        err = 'silx doesn\'t have an implementation for the requested engine: '
        err += '%s' % engine
        raise ValueError(err)

    if len(image.shape) is not 2:
        raise ValueError('medfilt2d deals with arrays of dimension 2 only')

    if engine == 'cpp':
        return medianfilter_cpp.medfilt(data=image,
                                        kernel_size=kernel_size,
                                        conditional=False)
    elif engine == 'opencl':
        if medfilt_opencl is None:
            wrn = 'opencl median filter not available. '
            wrn += 'Launching cpp implementation.'
            _logger.warning(wrn)
            # instead call the cpp implementation
            return medianfilter_cpp.medfilt(data=image,
                                            kernel_size=kernel_size,
                                            conditional=False)
        else:
            try:
                medianfilter = medfilt_opencl.MedianFilter2D(image.shape,
                                                             devicetype="gpu")
                res = medianfilter.medfilt2d(image, kernel_size)
            except(RuntimeError, MemoryError, ImportError):
                wrn = 'Exception occured in opencl median filter. '
                wrn += 'To get more information see debug log.'
                wrn += 'Launching cpp implementation.'
                _logger.warning(wrn)
                _logger.debug("median filter - openCL implementation issue.",
                              exc_info=True)
                # instead call the cpp implementation
                res = medianfilter_cpp.medfilt(data=image,
                                               kernel_size=kernel_size,
                                               conditional=False)

        return res
