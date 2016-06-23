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
"""This module provides a peak search function.

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
#from peaks cimport seek


def peak_search(y, fwhm, sensitivity=3.5, max_number_of_peaks=500,
                begin_index=None, end_index=None,
                debug=False, relevance_info=False):
    """Find peaks in the data.

    :param y: Data array
    :type y: numpy.ndarray
    :param fwhm: Estimated full width at half maximum of the peaks we are
        interested in
    :param sensitivity: Threshold factor used for peak search
    :param max_number_of_peaks: Maximum number of peaks in the data.
        This parameter is used to allocate memory for the output array.
        If it is too small, this function wiff fail.
    :param begin_index: Index of the first sample of the region of interest
         in the ``y`` array
    :param end_index: Index of the last sample of the region of interest in
        the ``y`` array
    :param debug: If ``True``, print debug messages. Default: ``False``
    :param relevance_info: If ``True``, add a second dimension with relevance
        information to the output array. Default: ``False``
    :return: 1D sequence with indexes of peaks in the data
        if ``relevance_info`` is ``False``.
        Else, sequence of  ``(peak_index, peak_relevance)`` tuples (one tuple
        per peak).
    :raise: ``IndexError`` if the number of peaks is too large to fit in the
        output array.
    """
    cdef:
        double[::1] y_c
        double[::1] peaks
        double[::1] relevances

    y_c = numpy.array(y,
                      copy=True,
                      dtype=numpy.float64,
                      order='C').reshape(-1)

    peaks = numpy.empty(shape=(max_number_of_peaks,),
                        dtype=numpy.float64)

    relevances = numpy.empty(shape=(max_number_of_peaks,),
                             dtype=numpy.float64)

    if debug:
        debug = 1
    else:
        debug = 0

    if begin_index is None:
        begin_index = 0
    if end_index is None:
        end_index = y_c.size - 1

    n_peaks = seek(begin_index, end_index, y_c.size,
                   fwhm, sensitivity, debug, max_number_of_peaks,
                   &y_c[0], &peaks[0], &relevances[0])

    if n_peaks < 0:
        raise IndexError("Too many peaks found for size of output array")

    if not relevance_info:
        return numpy.asarray(peaks)[0:n_peaks]
    else:
        # FIXME: maybe don't zip, return tuple (peaks, relevances)?
        return zip(numpy.asarray(peaks), numpy.asarray(relevances))[0:n_peaks]
