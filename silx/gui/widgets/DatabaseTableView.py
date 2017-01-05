# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""
This module  define model and widget to display 1D slices from numpy
array using compound data types or hdf5 databases.
"""
from __future__ import division

from silx.gui import qt

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "05/01/2017"


class DatabaseTableModel(qt.QAbstractTableModel):
    """This data model provides access to 1D slices from numpy array using
    compound data types or hdf5 databases.

    Each entries are displayed in a single row, and each columns contain a
    specific field of the compound type.

    It also allows to display 1D arrays of simple data types.
    array.

    :param qt.QObject parent: Parent object
    :param numpy.ndarray data: A numpy array or a h5py dataset
    :param str fmt: Format string for representing numerical values.
        Default is ``"%g"``.
    """
    def __init__(self, parent=None, data=None, fmt="%g"):
        qt.QAbstractTableModel.__init__(self, parent)

        self.__format = fmt
        self.__data = None
        self.__fields = None

        # set _data
        self.setArrayData(data)

    # Methods to be implemented to subclass QAbstractTableModel
    def rowCount(self, parent_idx=None):
        """Returns number of rows to be displayed in table"""
        if self.__data is None:
            return 0
        else:
            return len(self.__data)

    def columnCount(self, parent_idx=None):
        """Returns number of columns to be displayed in table"""
        if self.__fields is None:
            return 1
        else:
            return len(self.__fields)

    def data(self, index, role=qt.Qt.DisplayRole):
        """QAbstractTableModel method to access data values
        in the format ready to be displayed"""
        if not index.isValid():
            return None

        if self.__data is None:
            return None

        if index.row() >= len(self.__data):
            return None
        data = self.__data[index.row()]

        if self.__fields is not None:
            if index.column() >= len(self.__fields):
                return None
            data = data[self.__fields[index.column()]]

        if role == qt.Qt.DisplayRole:
            try:
                return self.__format % data
            except TypeError:
                return str(data)

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        """Returns the 0-based row or column index, for display in the
        horizontal and vertical headers"""
        if role == qt.Qt.DisplayRole:
            if orientation == qt.Qt.Vertical:
                return str(section)
            if orientation == qt.Qt.Horizontal:
                if self.__fields is None:
                    if section == 0:
                        return "Data"
                    else:
                        return None
                else:
                    if section < len(self.__fields):
                        return self.__fields[section]
                    else:
                        return None
        return None

    def flags(self, index):
        """QAbstractTableModel method to inform the view whether data
        is editable or not."""
        return qt.QAbstractTableModel.flags(self, index)

    def setArrayData(self, data):
        """Set the data array and the viewing perspective.

        You can set ``copy=False`` if you need more performances, when dealing
        with a large numpy array. In this case, a simple reference to the data
        is used to access the data, rather than a copy of the array.

        .. warning::

            Any change to the data model will affect your original data
            array, when using a reference rather than a copy..

        :param data: 1D numpy array, or any object that can be
            converted to a numpy array using ``numpy.array(data)`` (e.g.
            a nested sequence).
        """
        if qt.qVersion() > "4.6":
            self.beginResetModel()

        self.__data = data
        self.__fields = None
        if data is not None:
            if data.dtype.fields is not None:
                self.__fields = list(data.dtype.fields)

        if qt.qVersion() > "4.6":
            self.endResetModel()
        else:
            self.reset()

    def arrayData(self):
        """Returns the internal data.

        :rtype: numpy.ndarray of h5py.Dataset
        """
        return self.__data

    def setFormat(self, fmt):
        """Set format string controlling how the values are represented in
        the table view.

        :param str fmt: Format string (e.g. "%.3f", "%d", "%-10.2f", "%10.3e")
            This is the C-style format string used by python when formatting
            strings with the modulus operator.
        """
        if qt.qVersion() > "4.6":
            self.beginResetModel()

        self.__format = fmt

        if qt.qVersion() > "4.6":
            self.endResetModel()
        else:
            self.reset()


class DatabaseTableView(qt.QTableView):
    """TableView using DatabaseTableModel as default model.
    """
    def __init__(self, parent=None):
        """

        :param parent: parent QWidget
        """
        qt.QWidget.__init__(self, parent)
        self.setModel(DatabaseTableModel())
