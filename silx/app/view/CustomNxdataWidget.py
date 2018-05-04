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
__date__ = "04/05/2018"

from silx.gui import qt
from silx.io import commonh5
import silx.io.nxdata


class _DatasetItemRow(qt.QStandardItem):

    def __init__(self, label, dataset=None):
        super(_DatasetItemRow, self).__init__(label)
        self.setEditable(False)
        self.setDropEnabled(False)
        self.setDragEnabled(False)

        self.__name = qt.QStandardItem()
        self.__name.setEditable(False)
        self.__name.setDropEnabled(True)
        self.setDataset(dataset)

    def setDataset(self, dataset):
        self.__dataset = dataset
        if self.__dataset is not None:
            name = self.__dataset.name
        else:
            name = ""
        self.__name.setText(name)
        self.__name.setDragEnabled(self.__dataset is not None)

        parent = self.parent()
        if parent is not None:
            self.parent()._datasetUpdated()

    def getDataset(self):
        return self.__dataset

    def getRow(self):
        return [self, self.__name]


class _NxDataItem(qt.QStandardItem):

    def __init__(self):
        qt.QStandardItem.__init__(self)
        self.__error = None
        self.__title = None
        self.__signal = None
        self.__axes = None
        self.__virtual = None
        self.setEditable(False)
        self.setDragEnabled(False)
        self.setDropEnabled(False)

    def getRow(self):
        dataset = qt.QStandardItem("")
        dataset.setEditable(False)
        dataset.setDragEnabled(False)
        dataset.setDropEnabled(False)
        return [self, dataset]

    def _datasetUpdated(self):
        self.__virtual = None

    def createVirtualGroup(self):
        name = ""
        if self.__title is not None:
            name = self.__title
        virtual = commonh5.Group(name)
        virtual.attrs["NX_class"] = "NXdata"

        if self.__title is not None:
            virtual.attrs["title"] = self.__title

        if self.__signal is not None:
            signal = self.__signal.getDataset()
            if signal is not None:
                # Could be done using a link instead of a copy
                node = commonh5.Dataset("signal", signal[...])
                virtual.attrs["signal"] = "signal"
                virtual.add_node(node)

        axesAttr = []
        if self.__axes is not None:
            for i, axis in enumerate(self.__axes):
                if axis is None:
                    name = "."
                else:
                    axis = axis.getDataset()
                    if axis is None:
                        name = "."
                    else:
                        name = "axis%d" % i
                        axis = axis[...]
                        node = commonh5.Dataset(name, axis)
                        virtual.add_node(node)
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
        # TODO: Do it well, bad design
        if self.__signal is None:
            item = _DatasetItemRow("Signal", dataset)
            self.appendRow(item.getRow())
            self.__signal = item
        else:
            self.__signal.setDataset(dataset)
        self._datasetUpdated()

    def setAxes(self, datasets):
        # TODO: Do it well, bad design
        if self.__axes is None:
            self.__axes = []
            for i, dataset in enumerate(datasets):
                label = "Axis %d" % (i + 1)
                item = _DatasetItemRow(label, dataset)
                self.__axes.append(item)
                self.appendRow(item.getRow())
        else:
            for i, dataset in enumerate(datasets):
                self.__axes[i].setDataset(dataset)
        self._datasetUpdated()


class Hdf5DatasetMimeData(qt.QMimeData):
    """Mimedata class to identify an internal drag and drop of a Hdf5Node."""

    MIME_TYPE = "application/x-internal-h5py-dataset"

    def __init__(self, dataset=None):
        qt.QMimeData.__init__(self)
        self.__dataset = dataset
        self.setData(self.MIME_TYPE, "".encode(encoding='utf-8'))

    def getDataset(self):
        return self.__dataset


class _Model(qt.QStandardItemModel):

    def __init__(self, parent=None):
        qt.QStandardItemModel.__init__(self, parent)
        root = self.invisibleRootItem()
        root.setDropEnabled(False)
        root.setDragEnabled(False)

    def supportedDropActions(self):
        return qt.Qt.CopyAction | qt.Qt.MoveAction

    def mimeTypes(self):
        return [Hdf5DatasetMimeData.MIME_TYPE]

    def mimeData(self, indexes):
        """
        Returns an object that contains serialized items of data corresponding
        to the list of indexes specified.

        :param List[qt.QModelIndex] indexes: List of indexes
        :rtype: qt.QMimeData
        """
        if len(indexes) > 1:
            return None
        if len(indexes) == 0:
            return None

        qindex = indexes[0]
        qindex = self.index(qindex.row(), 0, parent=qindex.parent())
        item = self.itemFromIndex(qindex)
        if isinstance(item, _DatasetItemRow):
            dataset = item.getDataset()
            if dataset is None:
                return None
            else:
                mimeData = Hdf5DatasetMimeData(dataset=item.getDataset())
        else:
            mimeData = None
        return mimeData

    def dropMimeData(self, mimedata, action, row, column, parentIndex):
        if action == qt.Qt.IgnoreAction:
            return True

        if mimedata.hasFormat(Hdf5DatasetMimeData.MIME_TYPE):
            if row != -1 or column != -1:
                # It is not a drop on a specific item
                return False
            item = self.itemFromIndex(parentIndex)
            item = item.parent().child(item.row(), 0)
            if not isinstance(item, _DatasetItemRow):
                return False

            dataset = mimedata.getDataset()
            item.setDataset(dataset)
            return True

        return False


class CustomNxdataWidget(qt.QTreeView):

    def __init__(self, parent=None):
        qt.QTreeView.__init__(self, parent=None)
        self.__model = _Model(self)
        self.__model.setColumnCount(2)
        self.__model.setHorizontalHeaderLabels(["Name", "Dataset"])
        self.setModel(self.__model)

        header = self.header()
        if qt.qVersion() < "5.0":
            setResizeMode = header.setResizeMode
        else:
            setResizeMode = header.setSectionResizeMode
        setResizeMode(0, qt.QHeaderView.ResizeToContents)

        self.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.setDropIndicatorShown(True)
        self.setDragDropOverwriteMode(True)
        self.setDragEnabled(True)
        self.viewport().setAcceptDrops(True)

        self.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.customContextMenuRequested[qt.QPoint].connect(self.__executeContextMenu)

    def defaultContextMenu(self, point):
        qindex = self.indexAt(point)
        qindex = self.__model.index(qindex.row(), 0, parent=qindex.parent())
        item = self.__model.itemFromIndex(qindex)
        if isinstance(item, _DatasetItemRow):
            if item.getDataset() is not None:
                menu = qt.QMenu()
                action = qt.QAction("Remove this dataset", menu)
                action.triggered.connect(lambda: self.__removeItemDataset(item))
                menu.addAction(action)
                return menu

        return None

    def __removeItemDataset(self, item):
        item.setDataset(None)

    def __executeContextMenu(self, point):
        menu = self.defaultContextMenu(point)
        if menu is None:
            return
        menu.exec_(qt.QCursor.pos())

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
