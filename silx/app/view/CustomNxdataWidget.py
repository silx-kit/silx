# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
__date__ = "15/06/2018"

import logging
import numpy
import weakref

from silx.gui import qt
from silx.io import commonh5
import silx.io.nxdata
from silx.gui.hdf5._utils import Hdf5DatasetMimeData
from silx.gui.data.TextFormatter import TextFormatter
from silx.gui.hdf5.Hdf5Formatter import Hdf5Formatter
from silx.gui import icons


_logger = logging.getLogger(__name__)
_formatter = TextFormatter()
_hdf5Formatter = Hdf5Formatter(textFormatter=_formatter)


class _RowItems(qt.QStandardItem):
    """Define the list of items used for a specific row."""

    def type(self):
        return qt.QStandardItem.UserType + 1

    def getRowItems(self):
        """Returns the list of items used for a specific row.

        The first item should be this class.

        :rtype: List[qt.QStandardItem]
        """
        raise NotImplementedError()


class _DatasetItemRow(_RowItems):
    """Define a row which can contain a dataset."""

    def __init__(self, label="", dataset=None):
        """Constructor"""
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
        """Get the formatter used to display dataset informations.

        :rtype: Hdf5Formatter
        """
        return _hdf5Formatter

    def setDataset(self, dataset):
        """Set the dataset stored in this item.

        :param Union[numpy.ndarray,h5py.Dataset,silx.io.commonh5.Dataset] dataset:
            The dataset to store.
        """
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
        """Returns the dataset stored within the item."""
        return self.__dataset

    def getRowItems(self):
        """Returns the list of items used for a specific row.

        The first item should be this class.

        :rtype: List[qt.QStandardItem]
        """
        return [self, self.__name, self.__type, self.__shape]


class _DatasetAxisItemRow(_DatasetItemRow):
    """Define a row describing an axis."""

    def __init__(self):
        """Constructor"""
        super(_DatasetAxisItemRow, self).__init__()

    def setAxisId(self, axisId):
        """Set the id of the axis (the first axis is 0)

        :param int axisId: Identifier of this axis.
        """
        self.__axisId = axisId
        label = "Axis %d" % (axisId + 1)
        self.setText(label)

    def getAxisId(self):
        """Returns the identifier of this axis.

        :rtype: int
        """
        return self.__axisId


class _NxDataItem(qt.QStandardItem):
    """
    Define a custom NXdata.
    """

    def __init__(self):
        """Constructor"""
        qt.QStandardItem.__init__(self)
        self.__error = None
        self.__title = None
        self.__axes = []
        self.__virtual = None

        item = _DatasetItemRow("Signal", None)
        self.appendRow(item.getRowItems())
        self.__signal = item

        self.setEditable(False)
        self.setDragEnabled(False)
        self.setDropEnabled(False)
        self.__setError(None)

    def getRowItems(self):
        """Returns the list of items used for a specific row.

        The first item should be this class.

        :rtype: List[qt.QStandardItem]
        """
        row = [self]
        for _ in range(3):
            item = qt.QStandardItem("")
            item.setEditable(False)
            item.setDragEnabled(False)
            item.setDropEnabled(False)
            row.append(item)
        return row

    def _datasetUpdated(self):
        """Called when the NXdata contained of the item have changed.

        It invalidates the NXdata stored and send an event `sigNxdataUpdated`.
        """
        self.__virtual = None
        self.__setError(None)
        model = self.model()
        if model is not None:
            model.sigNxdataUpdated.emit(self.index())

    def createVirtualGroup(self):
        """Returns a new virtual Group using a NeXus NXdata structure to store
        data

        :rtype: silx.io.commonh5.Group
        """
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
                node = commonh5.DatasetProxy("signal", target=signal)
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
                    node = commonh5.DatasetProxy(name, target=axis)
                    virtual.add_node(node)
            axesAttr.append(name)

        if axesAttr != []:
            virtual.attrs["axes"] = numpy.array(axesAttr)

        validator = silx.io.nxdata.NXdata(virtual)
        if not validator.is_valid:
            message = "<html>"
            message += "This NXdata is not consistant"
            message += "<ul>"
            for issue in validator.issues:
                message += "<li>%s</li>" % issue
            message += "</ul>"
            message += "</html>"
            self.__setError(message)
        else:
            self.__setError(None)
        return virtual

    def isValid(self):
        """Returns true if the stored NXdata is valid

        :rtype: bool
        """
        return self.__error is None

    def getVirtualGroup(self):
        """Returns a cached virtual Group using a NeXus NXdata structure to
        store data.

        If the stored NXdata was invalidated, :meth:`createVirtualGroup` is
        internally called to update the cache.

        :rtype: silx.io.commonh5.Group
        """
        if self.__virtual is None:
            self.__virtual = self.createVirtualGroup()
        return self.__virtual

    def getTitle(self):
        """Returns the title of the NXdata

        :rtype: str
        """
        return self.text()

    def setTitle(self, title):
        """Set the title of the NXdata

        :param str title: The title of this NXdata
        """
        self.setText(title)

    def __setError(self, error):
        """Set the error message in case of the current state of the stored
        NXdata is not valid.

        :param str error: Message to display
        """
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
        """Returns the error message in case the NXdata is not valid.

        :rtype: str"""
        return self.__error

    def setSignalDataset(self, dataset):
        """Set the dataset to use as signal with this NXdata.

        :param Union[numpy.ndarray,h5py.Dataset,silx.io.commonh5.Dataset] dataset:
            The dataset to use as signal.
        """

        self.__signal.setDataset(dataset)
        self._datasetUpdated()

    def getSignalDataset(self):
        """Returns the dataset used as signal.

        :rtype: Union[numpy.ndarray,h5py.Dataset,silx.io.commonh5.Dataset]
        """
        return self.__signal.getDataset()

    def setAxesDatasets(self, datasets):
        """Set all the available dataset used as axes.

        Axes will be created or removed from the GUI in order to provide the
        same amount of requested axes.

        A `None` element is an axes with no dataset.

        :param List[Union[numpy.ndarray,h5py.Dataset,silx.io.commonh5.Dataset,None]] datasets:
            List of dataset to use as axes.
        """
        for i, dataset in enumerate(datasets):
            if i < len(self.__axes):
                mustAppend = False
                item = self.__axes[i]
            else:
                mustAppend = True
                item = _DatasetAxisItemRow()
            item.setAxisId(i)
            item.setDataset(dataset)
            if mustAppend:
                self.__axes.append(item)
                self.appendRow(item.getRowItems())

        # Clean up extra axis
        for i in range(len(datasets), len(self.__axes)):
            item = self.__axes.pop(len(datasets))
            self.removeRow(item.row())

        self._datasetUpdated()

    def getAxesDatasets(self):
        """Returns available axes as dataset.

        A `None` element is an axes with no dataset.

        :rtype: List[Union[numpy.ndarray,h5py.Dataset,silx.io.commonh5.Dataset,None]]
        """
        datasets = []
        for axis in self.__axes:
            datasets.append(axis.getDataset())
        return datasets


class _Model(qt.QStandardItemModel):
    """Model storing a list of custom NXdata items.

    Supports drag and drop of datasets.
    """

    sigNxdataUpdated = qt.Signal(qt.QModelIndex)
    """Emitted when stored NXdata was edited"""

    def __init__(self, parent=None):
        """Constructor"""
        qt.QStandardItemModel.__init__(self, parent)
        root = self.invisibleRootItem()
        root.setDropEnabled(True)
        root.setDragEnabled(False)

    def supportedDropActions(self):
        """Inherited method to redefine supported drop actions."""
        return qt.Qt.CopyAction | qt.Qt.MoveAction

    def mimeTypes(self):
        """Inherited method to redefine draggable mime types."""
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
        """Inherited method to handle a drop operation to this model."""
        if action == qt.Qt.IgnoreAction:
            return True

        if mimedata.hasFormat(Hdf5DatasetMimeData.MIME_TYPE):
            if row != -1 or column != -1:
                # It is not a drop on a specific item
                return False
            item = self.itemFromIndex(parentIndex)
            if item is None or item is self.invisibleRootItem():
                # Drop at the end
                dataset = mimedata.dataset()
                if silx.io.is_dataset(dataset):
                    self.createFromSignal(dataset)
                elif silx.io.is_group(dataset):
                    nxdata = dataset
                    try:
                        self.createFromNxdata(nxdata)
                    except ValueError:
                        _logger.error("Error while dropping a group as an NXdata")
                        _logger.debug("Backtrace", exc_info=True)
                        return False
                else:
                    _logger.error("Dropping a wrong object")
                    return False
            else:
                item = item.parent().child(item.row(), 0)
                if not isinstance(item, _DatasetItemRow):
                    # Dropped at a bad place
                    return False
                dataset = mimedata.dataset()
                if silx.io.is_dataset(dataset):
                    item.setDataset(dataset)
                else:
                    _logger.error("Dropping a wrong object")
                    return False
            return True

        return False

    def __getNxdataByTitle(self, title):
        """Returns an NXdata item by its title, else None.

        :rtype: Union[_NxDataItem,None]
        """
        for row in range(self.rowCount()):
            qindex = self.index(row, 0)
            item = self.itemFromIndex(qindex)
            if item.getTitle() == title:
                return item
        return None

    def findFreeNxdataTitle(self):
        """Returns an NXdata title which is not yet used.

        :rtype: str
        """
        for i in range(self.rowCount() + 1):
            name = "NXData #%d" % (i + 1)
            group = self.__getNxdataByTitle(name)
            if group is None:
                break
        return name

    def createNewNxdata(self, name=None):
        """Create a new NXdata item.

        :param Union[str,None] name: A title for the new NXdata
        """
        item = _NxDataItem()
        if name is None:
            name = self.findFreeNxdataTitle()
        item.setTitle(name)
        self.appendRow(item.getRowItems())

    def createFromSignal(self, dataset):
        """Create a new NXdata item from a signal dataset.

        This signal will also define an amount of axes according to its number
        of dimensions.

        :param Union[numpy.ndarray,h5py.Dataset,silx.io.commonh5.Dataset] dataset:
            A dataset uses as signal.
        """

        item = _NxDataItem()
        name = self.findFreeNxdataTitle()
        item.setTitle(name)
        item.setSignalDataset(dataset)
        item.setAxesDatasets([None] * len(dataset.shape))
        self.appendRow(item.getRowItems())

    def createFromNxdata(self, nxdata):
        """Create a new custom NXdata item from an existing NXdata group.

        If the NXdata is not valid, nothing is created, and an exception is
        returned.

        :param Union[h5py.Group,silx.io.commonh5.Group] nxdata: An h5py group
            following the NXData specification.
        :raise ValueError:If `nxdata` is not valid.
        """
        validator = silx.io.nxdata.NXdata(nxdata)
        if validator.is_valid:
            item = _NxDataItem()
            title = validator.title
            if title in [None or ""]:
                title = self.findFreeNxdataTitle()
            item.setTitle(title)
            item.setSignalDataset(validator.signal)
            item.setAxesDatasets(validator.axes)
            self.appendRow(item.getRowItems())
        else:
            raise ValueError("Not a valid NXdata")

    def removeNxdataItem(self, item):
        """Remove an NXdata item from this model.

        :param _NxDataItem item: An item
        """
        if isinstance(item, _NxDataItem):
            parent = item.parent()
            assert(parent is None)
            model = item.model()
            model.removeRow(item.row())
        else:
            _logger.error("Unexpected item")

    def appendAxisToNxdataItem(self, item):
        """Append a new axes to this item (or the NXdata item own by this item).

        :param Union[_NxDataItem,qt.QStandardItem] item: An item
        """
        if item is not None and not isinstance(item, _NxDataItem):
            item = item.parent()
        nxdataItem = item
        if isinstance(item, _NxDataItem):
            datasets = nxdataItem.getAxesDatasets()
            datasets.append(None)
            nxdataItem.setAxesDatasets(datasets)
        else:
            _logger.error("Unexpected item")

    def removeAxisItem(self, item):
        """Remove an axis item from this model.

        :param _DatasetAxisItemRow item: An axis item
        """
        if isinstance(item, _DatasetAxisItemRow):
            axisId = item.getAxisId()
            nxdataItem = item.parent()
            datasets = nxdataItem.getAxesDatasets()
            del datasets[axisId]
            nxdataItem.setAxesDatasets(datasets)
        else:
            _logger.error("Unexpected item")


class CustomNxDataToolBar(qt.QToolBar):
    """A specialised toolbar to manage custom NXdata model and items."""

    def __init__(self, parent=None):
        """Constructor"""
        super(CustomNxDataToolBar, self).__init__(parent=parent)
        self.__nxdataWidget = None
        self.__initContent()
        # Initialize action state
        self.__currentSelectionChanged(qt.QModelIndex(), qt.QModelIndex())

    def __initContent(self):
        """Create all expected actions and set the content of this toolbar."""
        action = qt.QAction("Create a new custom NXdata", self)
        action.setIcon(icons.getQIcon("nxdata-create"))
        action.triggered.connect(self.__createNewNxdata)
        self.addAction(action)
        self.__addNxDataAction = action

        action = qt.QAction("Remove the selected NXdata", self)
        action.setIcon(icons.getQIcon("nxdata-remove"))
        action.triggered.connect(self.__removeSelectedNxdata)
        self.addAction(action)
        self.__removeNxDataAction = action

        self.addSeparator()

        action = qt.QAction("Create a new axis to the selected NXdata", self)
        action.setIcon(icons.getQIcon("nxdata-axis-add"))
        action.triggered.connect(self.__appendNewAxisToSelectedNxdata)
        self.addAction(action)
        self.__addNxDataAxisAction = action

        action = qt.QAction("Remove the selected NXdata axis", self)
        action.setIcon(icons.getQIcon("nxdata-axis-remove"))
        action.triggered.connect(self.__removeSelectedAxis)
        self.addAction(action)
        self.__removeNxDataAxisAction = action

    def __getSelectedItem(self):
        """Get the selected item from the linked CustomNxdataWidget.

        :rtype: qt.QStandardItem
        """
        selectionModel = self.__nxdataWidget.selectionModel()
        index = selectionModel.currentIndex()
        if not index.isValid():
            return
        model = self.__nxdataWidget.model()
        index = model.index(index.row(), 0, index.parent())
        item = model.itemFromIndex(index)
        return item

    def __createNewNxdata(self):
        """Create a new NXdata item to the linked CustomNxdataWidget."""
        if self.__nxdataWidget is None:
            return
        model = self.__nxdataWidget.model()
        model.createNewNxdata()

    def __removeSelectedNxdata(self):
        """Remove the NXdata item currently selected in the linked
        CustomNxdataWidget."""
        if self.__nxdataWidget is None:
            return
        model = self.__nxdataWidget.model()
        item = self.__getSelectedItem()
        model.removeNxdataItem(item)

    def __appendNewAxisToSelectedNxdata(self):
        """Append a new axis to the NXdata item currently selected in the
        linked CustomNxdataWidget."""
        if self.__nxdataWidget is None:
            return
        model = self.__nxdataWidget.model()
        item = self.__getSelectedItem()
        model.appendAxisToNxdataItem(item)

    def __removeSelectedAxis(self):
        """Remove the axis item currently selected in the linked
        CustomNxdataWidget."""
        if self.__nxdataWidget is None:
            return
        model = self.__nxdataWidget.model()
        item = self.__getSelectedItem()
        model.removeAxisItem(item)

    def setCustomNxDataWidget(self, widget):
        """Set the linked CustomNxdataWidget to this toolbar."""
        assert(isinstance(widget, CustomNxdataWidget))
        if self.__nxdataWidget is not None:
            selectionModel = self.__nxdataWidget.selectionModel()
            selectionModel.currentChanged.disconnect(self.__currentSelectionChanged)
        self.__nxdataWidget = widget
        if self.__nxdataWidget is not None:
            selectionModel = self.__nxdataWidget.selectionModel()
            selectionModel.currentChanged.connect(self.__currentSelectionChanged)

    def __currentSelectionChanged(self, current, previous):
        """Update the actions according to the linked CustomNxdataWidget
        item selection"""
        if not current.isValid():
            item = None
        else:
            model = self.__nxdataWidget.model()
            index = model.index(current.row(), 0, current.parent())
            item = model.itemFromIndex(index)
        self.__removeNxDataAction.setEnabled(isinstance(item, _NxDataItem))
        self.__removeNxDataAxisAction.setEnabled(isinstance(item, _DatasetAxisItemRow))
        self.__addNxDataAxisAction.setEnabled(isinstance(item, _NxDataItem) or isinstance(item, _DatasetItemRow))


class _HashDropZones(qt.QStyledItemDelegate):
    """Delegate item displaying a drop zone when the item do not contains
    dataset."""

    def __init__(self, parent=None):
        """Constructor"""
        super(_HashDropZones, self).__init__(parent)
        pen = qt.QPen()
        pen.setColor(qt.QColor("#D0D0D0"))
        pen.setStyle(qt.Qt.DotLine)
        pen.setWidth(2)
        self.__dropPen = pen

    def paint(self, painter, option, index):
        """
        Paint the item

        :param qt.QPainter painter: A painter
        :param qt.QStyleOptionViewItem option: Options of the item to paint
        :param qt.QModelIndex index: Index of the item to paint
        """
        displayDropZone = False
        if index.isValid():
            model = index.model()
            rowIndex = model.index(index.row(), 0, index.parent())
            rowItem = model.itemFromIndex(rowIndex)
            if isinstance(rowItem, _DatasetItemRow):
                displayDropZone = rowItem.getDataset() is None

        if displayDropZone:
            painter.save()

            # Draw background if selected
            if option.state & qt.QStyle.State_Selected:
                colorGroup = qt.QPalette.Inactive
                if option.state & qt.QStyle.State_Active:
                    colorGroup = qt.QPalette.Active
                if not option.state & qt.QStyle.State_Enabled:
                    colorGroup = qt.QPalette.Disabled
                brush = option.palette.brush(colorGroup, qt.QPalette.Highlight)
                painter.fillRect(option.rect, brush)

            painter.setPen(self.__dropPen)
            painter.drawRect(option.rect.adjusted(3, 3, -3, -3))
            painter.restore()
        else:
            qt.QStyledItemDelegate.paint(self, painter, option, index)


class CustomNxdataWidget(qt.QTreeView):
    """Widget providing a table displaying and allowing to custom virtual
    NXdata."""

    sigNxdataItemUpdated = qt.Signal(qt.QStandardItem)
    """Emitted when the NXdata from an NXdata item was edited"""

    sigNxdataItemRemoved = qt.Signal(qt.QStandardItem)
    """Emitted when an NXdata item was removed"""

    def __init__(self, parent=None):
        """Constructor"""
        qt.QTreeView.__init__(self, parent=None)
        self.__model = _Model(self)
        self.__model.setColumnCount(4)
        self.__model.setHorizontalHeaderLabels(["Name", "Dataset", "Type", "Shape"])
        self.setModel(self.__model)

        self.setItemDelegateForColumn(1, _HashDropZones(self))

        self.__model.sigNxdataUpdated.connect(self.__nxdataUpdate)
        self.__model.rowsAboutToBeRemoved.connect(self.__rowsAboutToBeRemoved)
        self.__model.rowsAboutToBeInserted.connect(self.__rowsAboutToBeInserted)

        header = self.header()
        if qt.qVersion() < "5.0":
            setResizeMode = header.setResizeMode
        else:
            setResizeMode = header.setSectionResizeMode
        setResizeMode(0, qt.QHeaderView.ResizeToContents)
        setResizeMode(1, qt.QHeaderView.Stretch)
        setResizeMode(2, qt.QHeaderView.ResizeToContents)
        setResizeMode(3, qt.QHeaderView.ResizeToContents)

        self.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.setDropIndicatorShown(True)
        self.setDragDropOverwriteMode(True)
        self.setDragEnabled(True)
        self.viewport().setAcceptDrops(True)

        self.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.customContextMenuRequested[qt.QPoint].connect(self.__executeContextMenu)

    def __rowsAboutToBeInserted(self, parentIndex, start, end):
        if qt.qVersion()[0:2] == "5.":
            # FIXME: workaround for https://github.com/silx-kit/silx/issues/1919
            # Uses of ResizeToContents looks to break nice update of cells with Qt5
            # This patch make the view blinking
            self.repaint()

    def __rowsAboutToBeRemoved(self, parentIndex, start, end):
        """Called when an item was removed from the model."""
        items = []
        model = self.model()
        for index in range(start, end):
            qindex = model.index(index, 0, parent=parentIndex)
            item = self.__model.itemFromIndex(qindex)
            if isinstance(item, _NxDataItem):
                items.append(item)
        for item in items:
            self.sigNxdataItemRemoved.emit(item)

        if qt.qVersion()[0:2] == "5.":
            # FIXME: workaround for https://github.com/silx-kit/silx/issues/1919
            # Uses of ResizeToContents looks to break nice update of cells with Qt5
            # This patch make the view blinking
            self.repaint()

    def __nxdataUpdate(self, index):
        """Called when a virtual NXdata was updated from the model."""
        model = self.model()
        item = model.itemFromIndex(index)
        self.sigNxdataItemUpdated.emit(item)

    def createDefaultContextMenu(self, index):
        """Create a default context menu at this position.

        :param qt.QModelIndex index: Index of the item
        """
        index = self.__model.index(index.row(), 0, parent=index.parent())
        item = self.__model.itemFromIndex(index)

        menu = qt.QMenu()

        weakself = weakref.proxy(self)

        if isinstance(item, _NxDataItem):
            action = qt.QAction("Add a new axis", menu)
            action.triggered.connect(lambda: weakself.model().appendAxisToNxdataItem(item))
            action.setIcon(icons.getQIcon("nxdata-axis-add"))
            action.setIconVisibleInMenu(True)
            menu.addAction(action)
            menu.addSeparator()
            action = qt.QAction("Remove this NXdata", menu)
            action.triggered.connect(lambda: weakself.model().removeNxdataItem(item))
            action.setIcon(icons.getQIcon("remove"))
            action.setIconVisibleInMenu(True)
            menu.addAction(action)
        else:
            if isinstance(item, _DatasetItemRow):
                if item.getDataset() is not None:
                    action = qt.QAction("Remove this dataset", menu)
                    action.triggered.connect(lambda: item.setDataset(None))
                    menu.addAction(action)

            if isinstance(item, _DatasetAxisItemRow):
                menu.addSeparator()
                action = qt.QAction("Remove this axis", menu)
                action.triggered.connect(lambda: weakself.model().removeAxisItem(item))
                action.setIcon(icons.getQIcon("remove"))
                action.setIconVisibleInMenu(True)
                menu.addAction(action)

        return menu

    def __executeContextMenu(self, point):
        """Execute the context menu at this position."""
        index = self.indexAt(point)
        menu = self.createDefaultContextMenu(index)
        if menu is None or menu.isEmpty():
            return
        menu.exec_(qt.QCursor.pos())

    def removeDatasetsFrom(self, root):
        """
        Remove all datasets provided by this root

        :param root: The root file of datasets to remove
        """
        for row in range(self.__model.rowCount()):
            qindex = self.__model.index(row, 0)
            item = self.model().itemFromIndex(qindex)

            edited = False
            datasets = item.getAxesDatasets()
            for i, dataset in enumerate(datasets):
                if dataset is not None:
                    # That's an approximation, IS can't be used as h5py generates
                    # To objects for each requests to a node
                    if dataset.file.filename == root.file.filename:
                        datasets[i] = None
                        edited = True
            if edited:
                item.setAxesDatasets(datasets)

            dataset = item.getSignalDataset()
            if dataset is not None:
                # That's an approximation, IS can't be used as h5py generates
                # To objects for each requests to a node
                if dataset.file.filename == root.file.filename:
                    item.setSignalDataset(None)

    def replaceDatasetsFrom(self, removedRoot, loadedRoot):
        """
        Replace any dataset from any NXdata items using the same dataset name
        from another root.

        Usually used when a file was synchronized.

        :param removedRoot: The h5py root file which is replaced
            (which have to be removed)
        :param loadedRoot: The new h5py root file which have to be used
            instread.
        """
        for row in range(self.__model.rowCount()):
            qindex = self.__model.index(row, 0)
            item = self.model().itemFromIndex(qindex)

            edited = False
            datasets = item.getAxesDatasets()
            for i, dataset in enumerate(datasets):
                newDataset = self.__replaceDatasetRoot(dataset, removedRoot, loadedRoot)
                if dataset is not newDataset:
                    datasets[i] = newDataset
                    edited = True
            if edited:
                item.setAxesDatasets(datasets)

            dataset = item.getSignalDataset()
            newDataset = self.__replaceDatasetRoot(dataset, removedRoot, loadedRoot)
            if dataset is not newDataset:
                item.setSignalDataset(newDataset)

    def __replaceDatasetRoot(self, dataset, fromRoot, toRoot):
        """
        Replace the dataset by the same dataset name from another root.
        """
        if dataset is None:
            return None

        if dataset.file is None:
            # Not from the expected root
            return dataset

        # That's an approximation, IS can't be used as h5py generates
        # To objects for each requests to a node
        if dataset.file.filename == fromRoot.file.filename:
            # Try to find the same dataset name
            try:
                return toRoot[dataset.name]
            except Exception:
                _logger.debug("Backtrace", exc_info=True)
                return None
        else:
            # Not from the expected root
            return dataset

    def selectedItems(self):
        """Returns the list of selected items containing NXdata

        :rtype: List[qt.QStandardItem]
        """
        result = []
        for qindex in self.selectedIndexes():
            if qindex.column() != 0:
                continue
            if not qindex.isValid():
                continue
            item = self.__model.itemFromIndex(qindex)
            if not isinstance(item, _NxDataItem):
                continue
            result.append(item)
        return result

    def selectedNxdata(self):
        """Returns the list of selected NXdata

        :rtype: List[silx.io.commonh5.Group]
        """
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
