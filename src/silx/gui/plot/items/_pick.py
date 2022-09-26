# /*##########################################################################
#
# Copyright (c) 2019-2020 European Synchrotron Radiation Facility
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
"""This module provides classes supporting item picking."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "04/06/2019"

import numpy


class PickingResult(object):
    """Class to access picking information in a :class:`PlotWidget`"""

    def __init__(self, item, indices=None):
        """Init

        :param item: The picked item
        :param numpy.ndarray indices: Array-like of indices of picked data.
            Either 1D or 2D with dim0: data dimension and dim1: indices.
            No copy is made.
        """
        self._item = item

        if indices is None or len(indices) == 0:
            self._indices = None
        else:
            # Indices is set to None if indices array is empty
            indices = numpy.array(indices, copy=False, dtype=numpy.int64)
            self._indices = None if indices.size == 0 else indices

    def getItem(self):
        """Returns the item this results corresponds to."""
        return self._item

    def getIndices(self, copy=True):
        """Returns indices of picked data.

        If data is 1D, it returns a numpy.ndarray, otherwise
        it returns a tuple with as many numpy.ndarray as there are
        dimensions in the data.

        :param bool copy: True (default) to get a copy,
            False to return internal arrays
        :rtype: Union[None,numpy.ndarray,List[numpy.ndarray]]
        """
        if self._indices is None:
            return None
        indices = numpy.array(self._indices, copy=copy)
        return indices if indices.ndim == 1 else tuple(indices)
