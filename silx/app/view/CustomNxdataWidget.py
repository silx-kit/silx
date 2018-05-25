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
__date__ = "25/05/2018"

from silx.gui import qt
from silx.io import commonh5
import silx.io.nxdata
from silx.gui.hdf5._utils import Hdf5DatasetMimeData
from silx.gui.data.TextFormatter import TextFormatter
from silx.gui.hdf5.Hdf5Formatter import Hdf5Formatter
from silx.gui import icons


_formatter = TextFormatter()
_hdf5Formatter = Hdf5Formatter(textFormatter=_formatter)


class _DatasetItemRow(qt.QStandardItem):

    def __init__(self, label, dataset=None):
        super(_DatasetItemRow, self).__init__(label)
        self.setEditable(False)
        self.setDropEnabled(False)
        self.setDragEnabled(False)

        self.__name = qt.QStandardItem()
        self.__name.setEditable(False)
        self.__name.setDropEnabled(True)

        self.__type = qt.QStandardItem()
        self.__type.setEditable(False)
        self.__type.setDropEnabled(False)
        self.__type.setDragEnabled(False)

        self.__shape = qt.QStandardItem()
        self.__shape.setEditable(False)
        self.__shape.setDropEnabled(False)
        self.__shape.setDragEnabled(False)

        self.setDataset(dataset)

    def getDefaultFormatter(self):
        return _hdf5Formatter

    def setDataset(self, dataset):
        self.__dataset = dataset
        if self.__dataset is not None:
            name = self.__dataset.name

            if silx.io.is_dataset(dataset):
                type_ = self.getDefaultFormatter().humanReadableType(dataset)
                shape = self.getDefaultFormatter().humanReadableShape(dataset)

                if dataset.shape is None:
                    icon_name = "item-none"
                elif len(dataset.shape) < 4:
                    icon_name = "item-%ddim" % len(dataset.shape)
                else:
                    icon_name = "item-ndim"
                icon = icons.getQIcon(icon_name)
            else:
                type_ = ""
                shape = ""
                icon = qt.QIcon()
        else:
            name = ""
            type_ = ""
            shape = ""
            icon = qt.QIcon()

        self.__icon = icon
        self.__name.setText(name)
        self.__name.setDragEnabled(self.__dataset is not None)
        self.__name.setIcon(self.__icon)
        self.__type.setText(type_)
        self.__shape.setText(shape)

        parent = self.parent()
        if parent is not None:
            self.parent()._datasetUpdated()

    def getDataset(self):
        return self.__dataset

    def getRow(self):
        return [self, self.__name, self.__type, self.__shape]


class _DatasetAxisItemRow(_DatasetItemRow):

    def __init__(self, dataset=None, axisId=None):
        label = "Axis %d" % (axisId + 1)
        super(_DatasetAxisItemRow, self).__init__(label=label, dataset=dataset)
        self.__axisId = axisId

    def getAxisId(self):
        return self.__axisId


class _NxDataItem(qt.QStandardItem):

    def __init__(self):
        qt.QStandardItem.__init__(self)
        self.__error = None
        self.__title = None
        self.__axes = []
        self.__virtual = None

        item = _DatasetItemRow("Signal", None)
        self.appendRow(item.getRow())
        self.__signal = item

        self.setEditable(False)
        self.setDragEnabled(False)
        self.setDropEnabled(False)
        self.setError(None)

    def getRow(self):
        row = [self]
        for _ in range(3):
            item = qt.QStandardItem("")
            item.setEditable(False)
            item.setDragEnabled(False)
            item.setDropEnabled(False)
            row.append(item)
        return row

    def _datasetUpdated(self):
        self.__virtual = None
        self.setError(None)

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

        if axesAttr != []:
            virtual.attrs["axes"] = axesAttr

        validator = silx.io.nxdata.NXdata(virtual)
        if not validator.is_valid:
            message = "<html>"
            message += "This NXdata is not consistant"
            message += "<ul>"
            for issue in validator.issues:
                message += "<li>%s</li>" % issue
            message += "</ul>"
            message += "</html>"
            self.setError(message)
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
        style = qt.QApplication.style()
        if error is None:
            message = ""
            icon = style.standardIcon(qt.QStyle.SP_DirLinkIcon)
        else:
            message = error
            icon = style.standardIcon(qt.QStyle.SP_MessageBoxCritical)
        self.setIcon(icon)
        self.setToolTip(message)

    def getError(self):
        return self.__error

    def setSignal(self, dataset):
        # TODO: Do it well, bad design
        self.__signal.setDataset(dataset)
        self._datasetUpdated()

    def setAxes(self, datasets):
        # TODO: We could avoid to remove all the items all the time
        if len(self.__axes) > 0:
            for i in reversed(range(self.rowCount())):
                item = self.child(i)
                if isinstance(item, _DatasetAxisItemRow):
                    self.removeRow(i)
        self.__axes[:] = []
        for i, dataset in enumerate(datasets):
            item = _DatasetAxisItemRow(dataset=dataset, axisId=i)
            self.__axes.append(item)
            self.appendRow(item.getRow())
        self._datasetUpdated()

    def getAxesDatasets(self):
        datasets = []
        for axis in self.__axes:
            datasets.append(axis.getDataset())
        return datasets


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

            dataset = mimedata.dataset()
            item.setDataset(dataset)
            return True

        return False


class CustomNxDataToolBar(qt.QToolBar):

    def __init__(self, parent=None):
        super(CustomNxDataToolBar, self).__init__(parent=parent)
        self.__nxdataWidget = None
        self.__createActions()
        # Initialize action state
        self.__currentSelectionChanged(qt.QModelIndex(), qt.QModelIndex())

    def __createActions(self):
        action = qt.QAction("Create a new custom NxData", self)
        action.setIcon(icons.getQIcon("nxdata-create"))
        action.triggered.connect(self.__createNewNxData)
        self.addAction(action)
        self.__addNxDataAction = action

        action = qt.QAction("Remove the selected NxData", self)
        action.setIcon(icons.getQIcon("nxdata-remove"))
        action.triggered.connect(self.__removeSelectedNxData)
        self.addAction(action)
        self.__removeNxDataAction = action

        self.addSeparator()

        action = qt.QAction("Create a new axis to the selected NxData", self)
        action.setIcon(icons.getQIcon("nxdata-axis-add"))
        action.triggered.connect(self.__appendNewAxisToSelectedNxData)
        self.addAction(action)
        self.__addNxDataAxisAction = action

        action = qt.QAction("Remove the selected NxData axis", self)
        action.setIcon(icons.getQIcon("nxdata-axis-remove"))
        action.triggered.connect(self.__removeSelectedAxis)
        self.addAction(action)
        self.__removeNxDataAxisAction = action

    def __getSelectedItem(self):
        selectionModel = self.__nxdataWidget.selectionModel()
        index = selectionModel.currentIndex()
        if not index.isValid():
            return
        model = self.__nxdataWidget.model()
        index = model.index(index.row(), 0, index.parent())
        item = model.itemFromIndex(index)
        return item

    def __createNewNxData(self):
        if self.__nxdataWidget is None:
            return
        self.__nxdataWidget.create()

    def __removeSelectedNxData(self):
        if self.__nxdataWidget is None:
            return
        item = self.__getSelectedItem()
        if isinstance(item, _NxDataItem):
            parent = item.parent()
            assert(parent is None)
            model = item.model()
            model.removeRow(item.row())

    def __appendNewAxisToSelectedNxData(self):
        if self.__nxdataWidget is None:
            return
        item = self.__getSelectedItem()
        if item is not None and not isinstance(item, _NxDataItem):
            item = item.parent()
        nxdataItem = item
        if isinstance(item, _NxDataItem):
            datasets = nxdataItem.getAxesDatasets()
            datasets.append(None)
            nxdataItem.setAxes(datasets)

    def __removeSelectedAxis(self):
        if self.__nxdataWidget is None:
            return
        item = self.__getSelectedItem()
        if isinstance(item, _DatasetAxisItemRow):
            axisId = item.getAxisId()
            nxdataItem = item.parent()
            datasets = nxdataItem.getAxesDatasets()
            del datasets[axisId]
            nxdataItem.setAxes(datasets)

    def setCustomNxDataWidget(self, widget):
        assert(isinstance(widget, CustomNxdataWidget))
        if self.__nxdataWidget is not None:
            selectionModel = self.__nxdataWidget.selectionModel()
            selectionModel.currentChanged.disconnect(self.__currentSelectionChanged)
        self.__nxdataWidget = widget
        if self.__nxdataWidget is not None:
            selectionModel = self.__nxdataWidget.selectionModel()
            selectionModel.currentChanged.connect(self.__currentSelectionChanged)

    def __currentSelectionChanged(self, current, previous):
        """Update the actions according to the table selection"""
        if not current.isValid():
            item = None
        else:
            model = self.__nxdataWidget.model()
            index = model.index(current.row(), 0, current.parent())
            item = model.itemFromIndex(index)
        self.__removeNxDataAction.setEnabled(isinstance(item, _NxDataItem))
        self.__removeNxDataAxisAction.setEnabled(isinstance(item, _DatasetAxisItemRow))
        self.__addNxDataAxisAction.setEnabled(isinstance(item, _NxDataItem) or isinstance(item, _DatasetItemRow))


class CustomNxdataWidget(qt.QTreeView):

    def __init__(self, parent=None):
        qt.QTreeView.__init__(self, parent=None)
        self.__model = _Model(self)
        self.__model.setColumnCount(2)
        self.__model.setHorizontalHeaderLabels(["Name", "Dataset", "Type", "Shape"])
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
        self.__model.appendRow(item.getRow())

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
        validator = silx.io.nxdata.NXdata(nxdata)
        if validator.is_valid:
            item = _NxDataItem()
            title = validator.title
            if title in [None or ""]:
                title = self.findFreeNxdataTitle()
            item.setTitle(title)
            item.setSignal(validator.signal)
            item.setAxes(validator.axes)
            self.model().appendRow(item.getRow())
        else:
            raise ValueError("Not a valid NXdata")
