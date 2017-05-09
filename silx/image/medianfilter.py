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
Expose the median filter implementation (cpp, opencl ) under a common API
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "04/05/2017"

from silx.math import medianfilter as medianfilter_cpp
try:
    from silx.opencl import medfilt as medfilt_opencl
except ImportError:
    medfilt_opencl = None
import logging

_logger = logging.getLogger(__name__)

MEDFILT_ENGINES = ['cpp', 'opencl']


def medfilt(data, kernel_size=3, conditional=False, engine='cpp'):
    """Apply a 'nearest' median filter on the data.

    :param numpy.ndarray data: the array for which we want to apply
        the median filter. Should be 1D or 2D.
    :param kernel_size: the dimension of the kernel.
    :type kernel_size: For 1D should be an int for 2D should be a tuple or
        a list of (kernel_height, kernel_width)
    :param bool conditional: True if we want to apply a conditional median
        filtering.
    :param engine: the type of implementation we want to execute. Valid
        values are :attr:'MEDFILT_IMP'

    :returns: the array with the median value for each pixel.

    .. note::  if the opencl implementation is requested then it will be
        surrounded by a try-except statement and if failed
        (bad opencl installation ?) then the cpp implementation we be called.
    """
    if engine not in MEDFILT_ENGINES:
        err = 'Silx doesn\'t have an implementation for the Requested engine: '
        err += '%s' % engine
        raise ValueError(err)

    if len(data.shape) > 2:
        raise ValueError('medfilt deal with arrays of dimension <= 2')

    if engine == 'cpp':
        return medianfilter_cpp.medfilt(data=data,
                                        kernel_size=kernel_size,
                                        conditional=conditional)
    elif engine == 'opencl':
        if medfilt_opencl is None:
            wrn = 'opencl median filter module importation failed'
            wrn += 'Launching cpp implementation.'
            _logger.warning(wrn)
            # instead call the cpp implementation
            return medianfilter_cpp.medfilt(data=data,
                                            kernel_size=kernel_size,
                                            conditional=conditional)
        else:
            try:
                medianfilter = medfilt_opencl.MedianFilter2D(data.shape,
                                                             devicetype="gpu")
                print(data.shape)
                if len(data.shape) == 1:
                    res = medianfilter.medfilt1d(data, kernel_size)
                else:
                    res = medianfilter.medfilt2d(data, kernel_size)
            except(RuntimeError, MemoryError, ImportError):
                wrn = 'Exception occured opencl median filter. '
                wrn += 'To get more information see debug log.'
                wrn += 'Launching cpp implementation.'
                _logger.warning(wrn)
                _logger.debug("median filter - openCL implementation issue.",
                              exc_info=True)
                # instead call the cpp implementation
                res = medianfilter_cpp.medfilt(data=data,
                                               kernel_size=kernel_size,
                                               conditional=conditional)

        return res


def medfilt1d(data, kernel_size=3, conditional=False, engine='cpp'):
    """Apply a 'nearest' median filter on the data.

    :param numpy.ndarray data: the array for which we want to apply
        the median filter. Should be 1D.
    :param kernel_size: the dimension of the kernel.
    :type kernel_size: For 1D should be an int for 2D should be a tuple or
        a list of (kernel_height, kernel_width)
    :param bool conditional: True if we want to apply a conditional median
        filtering.
    :param engine: the type of implementation we want to execute. Valid
        values are :attr:'MEDFILT_IMP'

    :returns: the array with the median value for each pixel.

    .. note::  if the opencl implementation is requested then it will be
        surrounded by a try-except statement and if failed
        (bad opencl installation ?) then the cpp implementation we be called.
    """
    return medfilt(data, kernel_size, conditional, engine)


def medfilt2d(image, kernel_size=(3, 3), conditional=False, engine='cpp'):
    """Apply a 'nearest' median filter on the data.

    :param numpy.ndarray data: the array for which we want to apply
        the median filter. Should be 2D.
    :param kernel_size: the dimension of the kernel.
    :type kernel_size: For 1D should be an int for 2D should be a tuple or
        a list of (kernel_height, kernel_width)
    :param bool conditional: True if we want to apply a conditional median
        filtering.
    :param engine: the type of implementation we want to execute. Valid
        values are :attr:'MEDFILT_IMP'

    :returns: the array with the median value for each pixel.

    .. note::  if the opencl implementation is requested then it will be
        surrounded by a try-except statement and if failed
        (bad opencl installation ?) then the cpp implementation we be called.
    """
    return medfilt(image, kernel_size, conditional, engine)
