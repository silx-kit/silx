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
This module defines model and widget to display raw data using an
hexadecimal viewer.
"""
from __future__ import division

import numpy
from silx.gui import qt
import silx.io.utils
from silx.gui.widgets.TableWidget import CopySelectedCellsAction

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "27/09/2017"


class HexaTableModel(qt.QAbstractTableModel):
    """This data model provides access to a numpy void data.

    Bytes are displayed one by one as a hexadecimal viewer.

    The 16th first columns display bytes as hexadecimal, the last column
    displays the same data as ASCII.

    :param qt.QObject parent: Parent object
    :param data: A numpy array or a h5py dataset
    """
    def __init__(self, parent=None, data=None):
        qt.QAbstractTableModel.__init__(self, parent)

        self.__data = None
        self.setArrayData(data)

        if hasattr(qt.QFontDatabase, "systemFont"):
            self.__font = qt.QFontDatabase.systemFont(qt.QFontDatabase.FixedFont)
        else:
            self.__font = qt.QFont("Monospace")
            self.__font.setStyleHint(qt.QFont.TypeWriter)
        self.__palette = qt.QPalette()

    def rowCount(self, parent_idx=None):
        """Returns number of rows to be displayed in table"""
        if self.__data is None:
            return 0
        return ((len(self.__data.item()) - 1) >> 4) + 1

    def columnCount(self, parent_idx=None):
        """Returns number of columns to be displayed in table"""
        return 0x10 + 1

    def data(self, index, role=qt.Qt.DisplayRole):
        """QAbstractTableModel method to access data values
        in the format ready to be displayed"""
        if not index.isValid():
            return None

        if self.__data is None:
            return None

        row = index.row()
        column = index.column()

        if role == qt.Qt.DisplayRole:
            if column == 0x10:
                start = (row << 4)
                text = ""
                for i in range(0x10):
                    pos = start + i
                    if pos >= len(self.__data.item()):
                        break
                    data = self.__data.item()[pos]
                    value = ord(data)
                    if value > 0x20 and value < 0x7F:
                        text += chr(value)
                    else:
                        text += "."
                return text
            else:
                pos = (row << 4) + column
                if pos < len(self.__data.item()):
                    data = self.__data.item()[pos]
                    return "%02X" % ord(data)
                else:
                    return ""
        elif role == qt.Qt.FontRole:
            return self.__font

        elif role == qt.Qt.BackgroundColorRole:
            pos = (row << 4) + column
            if column != 0x10 and pos >= len(self.__data.item()):
                return self.__palette.color(qt.QPalette.Disabled, qt.QPalette.Background)
            else:
                return None

        return None

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        """Returns the 0-based row or column index, for display in the
        horizontal and vertical headers"""
        if section == -1:
            # PyQt4 send -1 when there is columns but no rows
            return None

        if role == qt.Qt.DisplayRole:
            if orientation == qt.Qt.Vertical:
                return "%02X" % (section << 4)
            if orientation == qt.Qt.Horizontal:
                if section == 0x10:
                    return "ASCII"
                else:
                    return "%02X" % section
        elif role == qt.Qt.FontRole:
            return self.__font
        elif role == qt.Qt.TextAlignmentRole:
            if orientation == qt.Qt.Vertical:
                return qt.Qt.AlignRight
            if orientation == qt.Qt.Horizontal:
                if section == 0x10:
                    return qt.Qt.AlignLeft
                else:
                    return qt.Qt.AlignCenter
        return None

    def flags(self, index):
        """QAbstractTableModel method to inform the view whether data
        is editable or not.
        """
        row = index.row()
        column = index.column()
        pos = (row << 4) + column
        if column != 0x10 and pos >= len(self.__data.item()):
            return qt.Qt.NoItemFlags
        return qt.QAbstractTableModel.flags(self, index)

    def setArrayData(self, data):
        """Set the data array.

        :param data: A numpy object or a dataset.
        """
        if qt.qVersion() > "4.6":
            self.beginResetModel()

        self.__data = data
        if self.__data is not None:
            if silx.io.utils.is_dataset(self.__data):
                self.__data = data[()]
            elif isinstance(self.__data, numpy.ndarray):
                self.__data = data[()]

        if qt.qVersion() > "4.6":
            self.endResetModel()
        else:
            self.reset()

    def arrayData(self):
        """Returns the internal data.

        :rtype: numpy.ndarray of h5py.Dataset
        """
        return self.__data


class HexaTableView(qt.QTableView):
    """TableView using HexaTableModel as default model.

    It customs the column size to provide a better layout.
    """
    def __init__(self, parent=None):
        """
        Constructor

        :param qt.QWidget parent: parent QWidget
        """
        qt.QTableView.__init__(self, parent)

        model = HexaTableModel(self)
        self.setModel(model)
        self._copyAction = CopySelectedCellsAction(self)
        self.addAction(self._copyAction)

    def copy(self):
        self._copyAction.trigger()

    def setArrayData(self, data):
        """Set the data array.

        :param data: A numpy object or a dataset.
        """
        self.model().setArrayData(data)
        self.__fixHeader()

    def __fixHeader(self):
        """Update the view according to the state of the auto-resize"""
        header = self.horizontalHeader()
        if qt.qVersion() < "5.0":
            setResizeMode = header.setResizeMode
        else:
            setResizeMode = header.setSectionResizeMode

        header.setDefaultSectionSize(30)
        header.setStretchLastSection(True)
        for i in range(0x10):
            setResizeMode(i, qt.QHeaderView.Fixed)
        setResizeMode(0x10, qt.QHeaderView.Stretch)
