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


class NumpyArrayTableModel(qt.QAbstractTableModel):
    """This data model provides access to a given 2D slice in a N-dimensional
    array.

    :param parent: Parent QObject
    :param narray: Numpy array
    :param str fmt: Format string for representing numerical values.
        Default is ``"%g"``.
    :param int perspective:
    """
    def __init__(self, parent=None, narray=None, fmt="%g", perspective=0):
        qt.QAbstractTableModel.__init__(self, parent)
        if narray is None:
            narray = numpy.array([])
        self._array = narray
        self._format = fmt
        self._index = 0
        self.assignDataFunction(perspective)

    def rowCount(self, parent_idx=None):
        return self._rowCount(parent_idx)

    def columnCount(self, parent_idx=None):
        return self._columnCount(parent_idx)

    def data(self, index, role=qt.Qt.DisplayRole):
        return self._data(index, role)

    def _rowCount1D(self, parent_idx=None):
        return 1

    def _columnCount1D(self, parent_idx=None):
        return self._array.shape[0]

    def _data1D(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role == qt.Qt.DisplayRole:
                # row = 0
                col = index.column()
                return self._format % self._array[col]
        return None

    def _rowCount2D(self, parent_idx=None):
        return self._array.shape[0]

    def _columnCount2D(self, parent_idx=None):
        return self._array.shape[1]

    def _data2D(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role == qt.Qt.DisplayRole:
                row = index.row()
                col = index.column()
                return self._format % self._array[row, col]
        return None

    def _rowCountND(self, parent_idx=None):
        return self._array.shape[-2]

    def _columnCountND(self, parent_idx=None):
        return self._array.shape[-1]

    def _dataND(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role == qt.Qt.DisplayRole:
                row = index.row()
                col = index.column()
                actualSelection = tuple(self._index + [row, col])
                return self._format % self._array[actualSelection]
        return None

    def _rowCount3DIndex0(self, parent_idx=None):
        return self._array.shape[1]

    def _columnCount3DIndex0(self, parent_idx=None):
        return self._array.shape[2]

    def _rowCount3DIndex1(self, parent_idx=None):
        return self._array.shape[0]

    def _columnCount3DIndex1(self, parent_idx=None):
        return self._array.shape[2]

    def _rowCount3DIndex2(self, parent_idx=None):
        return self._array.shape[0]

    def _columnCount3DIndex2(self, parent_idx=None):
        return self._array.shape[1]

    def _data3DIndex0(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role == qt.Qt.DisplayRole:
                row = index.row()
                col = index.column()
                return self._format % self._array[self._index, row, col]
        return None

    def _data3DIndex1(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role == qt.Qt.DisplayRole:
                row = index.row()
                col = index.column()
                return self._format % self._array[row, self._index, col]
        return None

    def _data3DIndex2(self, index, role=qt.Qt.DisplayRole):
        if index.isValid():
            if role == qt.Qt.DisplayRole:
                row = index.row()
                col = index.column()
                return self._format % self._array[row, col, self._index]
        return None

    def setArrayData(self, data, perspective=0):
        """
        setStackData(self, data, perspective=0)
        data is a 3D array
        dimension is the array dimension acting as index of images
        """
        if qt.qVersion() > "4.6":
            self.beginResetModel()
        else:
            self.reset()
        self._array = data
        self.assignDataFunction(perspective)
        if len(data.shape) > 3:
            self._index = []
            for i in range(len(data.shape) - 2):
                self._index.append(0)
        if qt.qVersion() > "4.6":
            self.endResetModel()

    def assignDataFunction(self, dimension):
        shape = self._array.shape
        if len(shape) == 2:
            self._rowCount = self._rowCount2D
            self._columnCount = self._columnCount2D
            self._data = self._data2D
        elif len(shape) == 1:
            self._rowCount = self._rowCount1D
            self._columnCount = self._columnCount1D
            self._data = self._data1D
        elif len(shape) > 3:
            # only C order array of images supported
            self._rowCount = self._rowCountND
            self._columnCount = self._columnCountND
            self._data = self._dataND
        else:
            if dimension == 1:
                self._rowCount = self._rowCount3DIndex1
                self._columnCount = self._columnCount3DIndex1
                self._data = self._data3DIndex1
            elif dimension == 2:
                self._rowCount = self._rowCount3DIndex2
                self._columnCount = self._columnCount3DIndex2
                self._data = self._data3DIndex1
            else:
                self._rowCount = self._rowCount3DIndex0
                self._columnCount = self._columnCount3DIndex0
                self._data = self._data3DIndex0
            self._dimension = dimension

    def setCurrentArrayIndex(self, index):
        """
        This method is ignored if the current array does not
        not a 3-dimensional array.
        """
        shape = self._array.shape
        if len(shape) < 3:
            # index is ignored
            return
        if len(shape) == 3:
            shape = self._array.shape[self._dimension]
            if hasattr(index, "__len__"):
                index = index[0]
            if (index < 0) or (index >= shape):
                raise ValueError("Index must be an integer lower than %d" % shape)
            self._index = index
        else:
            # Only N-dimensional arrays of images supported
            for i in range(len(index)):
                idx = index[i]
                if (idx < 0) or (idx >= shape[i]):
                    raise ValueError("Index %d must be positive integer lower than %d" % \
                                     (idx, shape[i]))
            self._index = index

    def setFormat(self, fmt):
        self._format = fmt

if __name__ == "__main__":
    a = qt.QApplication([])
    w = qt.QTableView()
    d = numpy.random.normal(0,1, (5, 1000,1000))
    for i in range(5):
        d[i, :, :] += i
    #m = NumpyArrayTableModel(fmt="%.5f")
    #m = NumpyArrayTableModel(None, numpy.arange(100.), fmt="%.5f")
    #m = NumpyArrayTableModel(None, numpy.ones((100,20)), fmt="%.5f")
    m = NumpyArrayTableModel(None, d, fmt = "%.5f")
    w.setModel(m)
    m.setCurrentArrayIndex(4)
    #m.setArrayData(numpy.ones((100,)))
    w.show()
    a.exec_()
