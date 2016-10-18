# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
import numpy
from .. import qt


__authors__ = ["V.A. Sole"]
__license__ = "MIT"
__date__ = "17/10/2016"


# TODO: generalise perspective to n-D arrays
class ArrayTableModel(qt.QAbstractTableModel):
    """This data model provides access to 2D slices in a N-dimensional
    array.

    A slice for a 3-D array is characterized by a perspective (the number of
    the axis orthogonal to the slice) and an index at which the slice
    intersects the orthogonal axis.

    In the n-D case, only slices parallel to the last two axes are handled. A
    slice is therefore characterized by a list of indices locating the
    slice on all the :math:`n - 2` orthogonal axes.

    :param parent: Parent QObject
    :param data: Numpy array
    :param str fmt: Format string for representing numerical values.
        Default is ``"%g"``.
    :param int perspective: In the 3-D case, this parameter is the number of
        the axis orthogonal to the data slice. If the unit vectors describing
        your axes are :math:`\vec{x}, \vec{y}, \vec{z}`, a perspective of 0
        means you slices are parallel to :math:`\vec{y}\vec{z}`, 1 means they
        are parallel to :math:`\vec{x}\vec{z}` and 2 means they
        are parallel to :math:`\vec{x}\vec{y}`.
        In all other cases (1-D, 2-D, n-D), this parameter is ignored.
    """
    def __init__(self, parent=None, data=None, fmt="%g", perspective=0):
        qt.QAbstractTableModel.__init__(self, parent)

        # make sure data is an array (not a scalar, list, or list of lists)
        if data is None:
            data = numpy.array([])
        elif not isinstance(data, numpy.ndarray):
            data = numpy.array(data)
        # ensure data is at least 2-dimensional
        if len(data.shape) < 1:
            data.shape = (1, 1)
        elif len(data.shape) < 2:
            data.shape = (1, data.shape[0])

        self._array = data
        """n-dimensional numpy array"""

        self._format = fmt
        """Format string (default "%g")"""

        self._index = [0] * (len(data.shape) - 2)
        """This attribute stores the slice index. In case _array is a 3D
        array, it is an integer.  If the number of dimensions is greater
        than 3, it is a list of indices."""

        self._perspective = perspective
        """Axis number orthogonal to the 2D-slice, for a 3-D array"""

        self._assignDataFunction()

    def _assignDataFunction(self):
        shape = self._array.shape
        if len(shape) == 3:
            self._rowCount = self._rowCount3D
            self._columnCount = self._columnCount3D
            self._data = self._data3D
        else:
            # only C order array of images supported
            self._rowCount = self._rowCountND
            self._columnCount = self._columnCountND
            self._data = self._dataND

    def _rowCountND(self, parent_idx=None):
        return self._array.shape[-2]

    def _columnCountND(self, parent_idx=None):
        return self._array.shape[-1]

    def _dataND(self, index, role=qt.Qt.DisplayRole):
        if index.isValid() and role == qt.Qt.DisplayRole:
            row = index.row()
            col = index.column()
            selection = tuple(self._index + [row, col])
            return self._format % self._array[selection]
        return None

    def _rowCount3D(self, parent_idx=None):
        """Return number of rows in the slice. This is the length of the
        first axis parallel to the slice (x or y)."""
        if self._perspective == 0:
            return self._array.shape[1]
        return self._array.shape[0]

    def _columnCount3D(self, parent_idx=None):
        """Return number of columns in the slice. This is the length of the
        last axis parallel to the slice (y or z)."""
        if self._perspective == 2:
            return self._array.shape[1]
        return self._array.shape[2]

    def _data3D(self, index, role=qt.Qt.DisplayRole):
        if index.isValid() and role == qt.Qt.DisplayRole:
            row = index.row()
            col = index.column()
            if self._perspective == 0:
                selection = tuple(self._index + [row, col])
            elif self._perspective == 1:
                selection = (row, self._index[0], col)
            else:
                selection = tuple([row, col] + self._index)
            return self._format % self._array[selection]
        return None

    # Methods to be implemented to subclass QAbstractTableModel
    def rowCount(self, parent_idx=None):
        """Return the number of rows (first index) in the data slice"""
        return self._rowCount(parent_idx)

    def columnCount(self, parent_idx=None):
        """Return the number of columns (second index) in the data slice"""
        return self._columnCount(parent_idx)

    def data(self, index, role=qt.Qt.DisplayRole):
        """Return the string representation of the value at the specified
        index in the data slice.
        """
        return self._data(index, role)

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        """Return the 0-based row or column index, for display in the
        horizontal and vertical headers"""
        if role == qt.Qt.DisplayRole:
            if orientation == qt.Qt.Vertical:
                return "%d" % section
            if orientation == qt.Qt.Horizontal:
                return "%d" % section
        return None

    # Public methods
    def setArrayData(self, data, perspective=0):
        """Set the data array

        :param data: Numpy array
        :param int perspective: For a 3-D array, the array dimension acting
            as index of images, orthogonal to the image
        """
        if qt.qVersion() > "4.6":
            self.beginResetModel()
        else:
            self.reset()

        # ensure data is a numpy array
        if data is None:
            data = numpy.array([])
        elif not isinstance(data, numpy.ndarray):
            data = numpy.array(data)
        # ensure data is at least 2-dimensional
        if len(data.shape) < 1:
            data.shape = (1, 1)
        elif len(data.shape) < 2:
            data.shape = (1, data.shape[0])

        self._array = data
        self._index = [0] * (len(data.shape) - 2)
        self._perspective = perspective
        self._assignDataFunction()

        if qt.qVersion() > "4.6":
            self.endResetModel()

    def setCurrentArrayIndex(self, index):
        """Set the active slice index.

        This method is only relevant to arrays with at least 3 dimensions.

        :param sequence index: Index of the active slice in the array.
            In the general n-D case, this is a sequence of :math:`n - 2`
            indices where the slice intersects the orthogonal axes.
        :raise IndexError: If any index in the index sequence is out of bound
            on its respective axis.
        """
        shape = self._array.shape
        if len(shape) < 3:
            # index is ignored
            return
        if len(shape) == 3:
            len_ = shape[self._perspective]
            # accept integers as index in the case of 3-D arrays
            if not hasattr(index, "__len__"):
                self._index = [index]
            else:
                self._index = index
            if not 0 <= self._index[0] < len_:
                raise ValueError("Index must be a positive integer " +
                                 "lower than %d" % len_)
        else:
            # general n-D case
            for i, idx in enumerate(index):
                if not 0 <= idx < shape[i]:
                    raise IndexError("Index %d must be a positive " % idx +
                                     "integer lower than %d" % shape[i])
            self._index = index

    def setFormat(self, fmt):
        self._format = fmt

if __name__ == "__main__":
    a = qt.QApplication([])
    w = qt.QTableView()
    d = numpy.random.normal(0, 1, (5, 1000, 1000))
    for i in range(5):
        d[i, :, :] += i * 10
    # m = ArrayTableModel(fmt="%.5f")
    # m = ArrayTableModel(None, numpy.arange(100.), fmt="%.5f")
    # m = ArrayTableModel(None, numpy.ones((100,20)), fmt="%.5f")
    m = ArrayTableModel(data=d, fmt="%.5f")
    w.setModel(m)
    m.setCurrentArrayIndex(3)
    # m.setArrayData(numpy.ones((100,)))
    w.show()
    a.exec_()
