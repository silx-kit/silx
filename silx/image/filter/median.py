# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
# ###########################################################################*/

__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"

from . import mediantools
from numpy import asarray

def medfilt2d(input_data, kernel_size=None, conditional=0):
    """Median filter for 2-dimensional arrays.

    Apply a median filter to the input array using a local window-size
    given by kernel_size (must be odd).


    :param input_data: An 2 dimensional input array.
    :param kernel_size: A scalar or an length-2 list giving the size of the
        median filter window in each dimension.  Elements of kernel_size should
        be odd. If kernel_size is a scalar, then this scalar is used as the size
        in each dimension.
    :param conditional: If different from 0 implements a conditional median
        filter. In this case the rank of the central pixel is used to determine
        the output of the filter. If this rank is higher or equal than some
        higher threshold value or  lower or  equal  than  some  lower threshold
        value, the output of the filter would be the median value, otherwise the
        pixel remains unchanged. (for more details
          http://www.eurasip.org/Proceedings/Eusipco/Eusipco2007/Papers/c4p-h06.pdf)

    :return: An array the same size as input containing the median filtered
           result.

    """
    image = asarray(input_data)
    if kernel_size is None:
        kernel_size = [3] * 2
    kernel_size = asarray(kernel_size)
    if len(kernel_size.shape) == 0:
        kernel_size = [kernel_size.item()] * 2
    kernel_size = asarray(kernel_size)

    for size in kernel_size:
        if (size % 2) != 1:
            raise ValueError("Each element of kernel_size should be odd.")

    return mediantools._medfilt2d(image, kernel_size, conditional)

def medfilt1d(input_data, kernel_size=None, conditional=0):
    """Median filter 1-dimensional arrays.

    Apply a median filter to the input array using a local window-size
    given by kernel_size (must be odd).

  Inputs:

    :param input_data: An 1-dimensional input array.
    :param kernel_size: A scalar or an length-2 list giving the size of the
        median filter window in each dimension.  Elements of kernel_size should
        be odd. If kernel_size is a scalar, then this scalar is used as the size
        in each dimension.
    :param conditional: If different from 0 implements a conditional median
        filter.

    :return :An array the same size as input containing the median filtered
        result.

    """
    image = asarray(input_data)
    oldShape = image.shape
    image.shape = -1, 1
    if kernel_size is None:
        kernel_size = [3, 1]
    kernel_size = asarray(kernel_size)
    if len(kernel_size.shape) == 0:
        kernel_size = [kernel_size.item(), 1]
    kernel_size = asarray(kernel_size)

    for size in kernel_size:
        if (size % 2) != 1:
            image.shape = oldShape
            raise ValueError("Kernel_size should be odd.")
    output = mediantools._medfilt2d(image, kernel_size, conditional)
    output.shape = oldShape
    image.shape = oldShape
    return output
