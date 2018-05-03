# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2017 European Synchrotron Radiation Facility
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

"""Widget to custom NXdata groups"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "03/05/2018"

from silx.gui import qt
from silx.io import commonh5
import silx.io.nxdata


class _NxDataItem(qt.QStandardItem):

    def __init__(self):
        qt.QStandardItem.__init__(self)
        self.__error = None
        self.__title = None
        self.__signal = None
        self.__axes = []
        self.__virtual = None
        self.setEditable(False)

    def getRow(self):
        dataset = qt.QStandardItem("")
        return [self, dataset]

    def _invalidate(self):
        self.removeRows(0, self.rowCount())
        self.__virtual = None

        if self.__signal is not None:
            name = qt.QStandardItem("Signal")
            name.setEditable(False)
            value = qt.QStandardItem(self.__signal.name)
            value.setEditable(False)
            self.appendRow([name, value])
        else:
            name = qt.QStandardItem("Signal")
            name.setEditable(False)
            value = qt.QStandardItem("")
            value.setEditable(False)
            self.appendRow([name, value])

        for i in range(len(self.__axes)):
            name = qt.QStandardItem("Axis %d" % (i + 1))
            name.setEditable(False)
            axis = None
            if i < len(self.__axes):
                axis = self.__axes[i]
            if axis is not None:
                value = qt.QStandardItem(axis.name)
                value.setEditable(False)
            else:
                value = qt.QStandardItem("")
                value.setEditable(False)
            self.appendRow([name, value])

    def createVirtualGroup(self):
        name = ""
        if self.__title is not None:
            name = self.__title
        virtual = commonh5.Group(name)
        virtual.attrs["NX_class"] = "NXdata"

        if self.__title is not None:
            virtual.attrs["title"] = self.__title

        if self.__signal is not None:
            # Could be done using a link instead of a copy
            node = commonh5.Dataset("signal", self.__signal[...])
            virtual.attrs["signal"] = "signal"
            virtual.add_node(node)

        axesAttr = []
        for i in range(len(self.__axes)):
            axis = None
            if i < len(self.__axes):
                axis = self.__axes[i]
            if axis is not None:
                # Could be done using a link instead of a copy
                axis = axis[...]
            if axis is not None:
                name = "axis%d" % i
                node = commonh5.Dataset(name, axis)
                virtual.add_node(node)
            else:
                name = "."
            axesAttr.append(name)
        virtual.attrs["axes"] = axesAttr

        if not silx.io.nxdata.is_valid_nxdata(virtual):
            self.setError("This NXdata is not consistant")
        else:
            self.setError(None)
        return virtual

    def isValid(self):
        return self.__error is None

    def getVirtualGroup(self):
        if self.__virtual is None:
            self.__virtual = self.createVirtualGroup()
        return self.__virtual

    def getTitle(self):
        return self.text()

    def setTitle(self, title):
        self.setText(title)

    def setError(self, error):
        self.__error = error
        if error is None:
            self.setIcon(qt.QIcon())
        else:
            style = qt.QApplication.style()
            icon = style.standardIcon(qt.QStyle.SP_MessageBoxCritical)
            self.setIcon(icon)

    def getError(self):
        return self.__error

    def setSignal(self, dataset):
        self.__signal = dataset
        self._invalidate()

    def setAxes(self, datasets):
        self.__axes = datasets
        self._invalidate()


class CustomNxdataWidget(qt.QTreeView):

    def __init__(self, parent=None):
        qt.QTreeView.__init__(self, parent=None)
        self.__model = qt.QStandardItemModel(self)
        self.__model.setColumnCount(2)
        self.__model.setHorizontalHeaderLabels(["Name", "Dataset"])
        self.setModel(self.__model)

        header = self.header()
        if qt.qVersion() < "5.0":
            setResizeMode = header.setResizeMode
        else:
            setResizeMode = header.setSectionResizeMode
        setResizeMode(0, qt.QHeaderView.ResizeToContents)

    def getNxdataByTitle(self, title):
        for row in range(self.__model.rowCount()):
            qindex = self.__model.index(row, 0)
            item = self.model().itemFromIndex(qindex)
            if item.getTitle() == title:
                return item
        return None

    def findFreeNxdataTitle(self):
        for i in range(self.__model.rowCount() + 1):
            name = "NXData #%d" % (i + 1)
            group = self.getNxdataByTitle(name)
            if group is None:
                break
        return name

    def selectedNxdata(self):
        """Returns the list of selected NXdata"""
        result = []
        for qindex in self.selectedIndexes():
            if qindex.column() != 0:
                continue
            if not qindex.isValid():
                continue
            item = self.__model.itemFromIndex(qindex)
            if not isinstance(item, _NxDataItem):
                continue
            result.append(item.getVirtualGroup())
        return result

    def create(self, name=None):
        item = _NxDataItem()
        if name is None:
            name = self.findFreeNxdataTitle()
        item.setTitle(name)
        self.__model.addNxdata(item.getRow())

    def createFromSignal(self, dataset):
        item = _NxDataItem()
        name = self.findFreeNxdataTitle()
        item.setTitle(name)
        item.setSignal(dataset)
        item.setAxes([None] * len(dataset.shape))
        self.__model.appendRow(item.getRow())

    def createFromNxdata(self, nxdata):
        """Create a new custom NXData from an existing NXData group.

        :param h5py.Group nxdata: An h5py group following the NXData
            specification
        """
        if silx.io.nxdata.is_valid_nxdata(nxdata):
            item = _NxDataItem()
            parsedNxdata = silx.io.nxdata.NXdata(nxdata)
            title = parsedNxdata.title
            if title in [None or ""]:
                title = self.findFreeNxdataTitle()
            item.setTitle(title)
            item.setSignal(parsedNxdata.signal)
            item.setAxes(parsedNxdata.axes)
            self.model().appendRow(item.getRow())
        else:
            # FIXME: Error message, not a valid NXdata
            pass
