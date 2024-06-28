# /*##########################################################################
#
# Copyright (c) 2004-2023 European Synchrotron Radiation Facility
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
# ###########################################################################*/

"""Custom plot selection window for selecting 1D datasets to plot."""
from __future__ import annotations

from silx.gui import qt, plot, icons
import silx.gui
import silx.gui.plot
import silx.gui.plot.items
import silx.io
import numpy
import functools
from silx.gui.plot.LegendSelector import LegendIcon
import silx.io.url
import silx.io.utils
from silx.gui.hdf5 import _utils
import weakref


# Custom role for highlighting the drop zones
_DROP_HIGHLIGHT_ROLE = qt.Qt.UserRole + 1


class _HashDropZones(qt.QStyledItemDelegate):
    """Delegate item displaying a drop zone when the item does not contain a dataset."""

    def __init__(self, parent: qt.QWidget | None = None):
        super(_HashDropZones, self).__init__(parent)
        self.__dropPen = qt.QPen(qt.QColor("#D0D0D0"), 2, qt.Qt.DotLine)
        self.__highlightDropPen = qt.QPen(qt.QColor("#000000"), 2, qt.Qt.SolidLine)
        self.__dropTargetIndex = None

    def paint(self, painter, option, index):
        """Paint the item"""
        displayDropZone = False
        isDropHighlighted = False
        if index.isValid():
            model = index.model()
            item = model.itemFromIndex(index)
            isDropHighlighted = item.data(_DROP_HIGHLIGHT_ROLE)

            rowIndex = model.index(index.row(), 1, index.parent())
            rowItem = model.itemFromIndex(rowIndex)
            if not rowItem.data(qt.Qt.DisplayRole):
                parentIndex = model.index(
                    index.parent().row(), 1, index.parent().parent()
                )
                parentItem = model.itemFromIndex(parentIndex)
                if parentItem and parentItem.text() not in ["Y"] or index.row() == 0:
                    displayDropZone = True

        if isDropHighlighted:
            painter.save()
            painter.setPen(self.__highlightDropPen)
            painter.drawRect(option.rect.adjusted(1, 1, -1, -1))
            painter.restore()

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

            isDropTarget = self.__dropTargetIndex == index
            painter.setPen(self.__highlightDropPen if isDropTarget else self.__dropPen)
            painter.drawRect(option.rect.adjusted(3, 3, -3, -3))

            painter.drawText(
                option.rect.adjusted(3, 3, -3, -3),
                qt.Qt.AlignLeft | qt.Qt.AlignVCenter,
                " Drop a 1D dataset",
            )
            painter.restore()
        else:
            qt.QStyledItemDelegate.paint(self, painter, option, index)


class _FileListModel(qt.QStandardItemModel):
    """Model for displaying dropped file names in a TreeView widget"""

    def __init__(self, plot, parent=None):
        super().__init__(parent)
        root = self.invisibleRootItem()
        root.setDragEnabled(False)

        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["", "Dataset", ""])

        self._xParent = qt.QStandardItem("X")
        self._xParent.setEditable(False)
        root.appendRow(self._xParent)

        self._yParent = qt.QStandardItem("Y")
        self._yParent.setEditable(False)
        root.appendRow(self._yParent)

        self._plot1D = weakref.proxy(plot)
        self._index = 0

        self._xDataset = None

        self._reset()

        self.addUrl(silx.io.url.DataUrl())

        self.rowsAboutToBeRemoved.connect(self._rowsAboutToBeRemoved)

    def getXParent(self) -> qt.QStandardItem:
        """Return the parent item for the X dataset"""
        return self._xParent

    def getYParent(self) -> qt.QStandardItem:
        """Return the parent item for the Y datasets"""
        return self._yParent

    def _setXFile(self, url: silx.io.url.DataUrl | None = None):
        """Set the X dataset file in the model"""
        if url is None:
            dataName = ""
        else:
            dataName = self._getBasename(url.data_path())
            self._plot1D.setXAxisLabel(dataName)

        fileItem, iconItem, removeItem = self._createRowItems(dataName, None)
        fileItem.setData(qt.QSize(0, 30), qt.Qt.SizeHintRole)

        fileItem.setData(self._createToolTip(url), qt.Qt.ToolTipRole)

        xIndex = self._findRowContainingText("X")
        if xIndex is not None:
            self.setItem(xIndex, 1, fileItem)
            self.setItem(xIndex, 2, removeItem)

    def _addYFile(
        self,
        url: silx.io.url.DataUrl | None = None,
        curve: silx.gui.plot.items.Curve | None = None,
    ):
        """Add a Y dataset file to the model"""
        if url is None:
            dataName = ""
        else:
            dataName = self._getBasename(url.data_path())
        fileItem, iconItem, removeItem = self._createRowItems(dataName, curve)
        if not dataName:
            # provide size hint for 'empty' item
            fileItem.setData(qt.QSize(0, 30), qt.Qt.SizeHintRole)

        fileItem.setData(self._createToolTip(url), qt.Qt.ToolTipRole)

        if self.getYParent().rowCount() == 0:
            self.getYParent().appendRow([qt.QStandardItem(), fileItem])
            return

        self.getYParent().insertRow(
            self.getYParent().rowCount() - 1, [iconItem, fileItem, removeItem]
        )

    def _findRowContainingText(self, text: str) -> int | None:
        """Return the row index containing the given text, or None if not found."""
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if item and item.text() == text:
                return row
        return None

    def _createRowItems(
        self, filename: str, curve: silx.gui.plot.items.Curve | None
    ) -> tuple[qt.QStandardItem, qt.QStandardItem, qt.QStandardItem]:
        """Create the items for a row in the model"""
        fileItem = qt.QStandardItem(filename)
        fileItem.setData(curve, qt.Qt.UserRole)
        fileItem.setData(False, _DROP_HIGHLIGHT_ROLE)
        fileItem.setEditable(False)
        fileItem.setDropEnabled(True)

        iconItem = qt.QStandardItem()
        removeItem = qt.QStandardItem()
        iconItem.setEditable(False)

        return fileItem, iconItem, removeItem

    def addUrl(self, url: silx.io.url.DataUrl, node: str = "X"):
        """Add a dataset to the model"""
        if url.file_path() is not None:
            file = silx.io.open(url.file_path())
            data = file[url.data_path()]
            if silx.io.is_dataset(data) and data.ndim == 1:
                if node == "X":
                    self._xDataset = data

                    for item in self._plot1D.getItems():
                        y = item.getInfo()
                        if y is None:
                            continue
                        length = min(len(self._xDataset), len(y))
                        item.setData(self._xDataset[:length], y[:length])

                    self._setXFile(url)
                else:
                    if data is None:
                        return
                    curve = self._addPlot(self._xDataset, data)
                    self._addYFile(url, curve)

    def _addPlot(
        self, x: numpy.ndarray | None, y: numpy.ndarray
    ) -> silx.gui.plot.items.Curve:
        """Add a curve to the plot with the given x and y data."""
        if x is None:
            x = numpy.arange(len(y))

        legend = f"Curve {self._index}"
        self._index += 1

        length = min(len(x), len(y))
        curve = self._plot1D.addCurve(x=x[:length], y=y[:length], legend=legend)
        curve.setInfo(y[()])
        return curve

    def _rowsAboutToBeRemoved(self, parentIndex: qt.QModelIndex, first: int, last: int):
        """Remove the curves from the plot when a row is removed from the model."""
        parentItem = self.itemFromIndex(parentIndex)
        for row in range(first, last + 1):
            fileItem = parentItem.child(row, 1)
            curve = fileItem.data(qt.Qt.UserRole)
            if curve is not None:
                self._plot1D.removeItem(curve)

    def _updateYCurvesWithDefaultX(self):
        """Update the Y curves with the default X dataset."""
        for item in self._plot1D.getItems():
            y = item.getInfo()
            x = numpy.arange(len(y))
            item.setData(x, y)
            self._plot1D.resetZoom()

    def clearAll(self):
        """Clear all datasets from the model and the plot."""
        self._xDataset = None

        self._yParent.removeRows(0, self._yParent.rowCount())

        self._setXFile()
        self._addYFile()

        self._plot1D.clear()
        self._plot1D.setXAxisLabel("X")
        self._plot1D.resetZoom()

        self._updateYCurvesWithDefaultX()

    def _reset(self):
        """Reset the model to its initial state."""
        self._xDataset = None

        self._setXFile()
        self._addYFile()

        self._plot1D.clear()
        self._updateYCurvesWithDefaultX()

    def _createToolTip(self, url: silx.io.url.DataUrl) -> str:
        """Create the tooltip for a dataset."""
        if url is None:
            return ""
        attrs = {
            "Name": self._getBasename(url.data_path()),
            "Path": url.data_path(),
            "File name": self._getBasename(url.file_path()),
        }
        return _utils.htmlFromDict(attrs, title="HDF5 Dataset")

    def _getBasename(self, text: str) -> str:
        """Return the basename of a file path."""
        return text.split("/")[-1]


class _DropTreeView(qt.QTreeView):
    """TreeView widget for displaying dropped file names"""

    (_DESCRIPTION_COLUMN, _FILE_COLUMN, _REMOVE_COLUMN) = range(3)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setModel(model)

        self.setItemDelegateForColumn(1, _HashDropZones(self))

        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, qt.QHeaderView.Stretch)
        header.setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
        self.expandAll()

        self.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.setItemsExpandable(False)
        self.setSelectionMode(qt.QAbstractItemView.NoSelection)
        self.setFocusPolicy(qt.Qt.NoFocus)
        self.setDropIndicatorShown(True)
        self.setDragDropOverwriteMode(True)
        self.setDragEnabled(False)
        self.viewport().setAcceptDrops(True)

    def _createIconWidget(self, row: int, parentItem: qt.QStandardItem):
        """Create the icon widget for a row in the model"""
        fileItem = parentItem.child(row, 1)
        if fileItem is None:
            return
        iconItem = parentItem.child(row, 0)
        curve = fileItem.data(qt.Qt.UserRole)
        if curve is None:
            return

        legendIcon = LegendIcon(None, curve)
        widget = qt.QWidget()
        layout = qt.QHBoxLayout(widget)
        layout.addWidget(legendIcon)
        layout.setContentsMargins(4, 0, 4, 0)
        self.setIndexWidget(iconItem.index(), widget)

    def _createRemoveButton(self, row: int, parentItem: qt.QStandardItem | None):
        """Create the remove button for a row in the model"""

        # If the parentItem is None, the remove button is for the X dataset
        if parentItem is None:
            parentItem = self.model().getXParent()
            index = self.model().index(0, 2)
            button = self._getRemoveButton(None, parentItem)
            self.setIndexWidget(index, button)
            return

        # If the parentItem is not None, the remove button is for a Y dataset
        removeItem = parentItem.child(row, 2)
        if removeItem:
            button = self._getRemoveButton(removeItem, parentItem)
            self.setIndexWidget(removeItem.index(), button)

    def _getRemoveButton(
        self, removeItem: qt.QStandardItem | None, parentItem: qt.QStandardItem
    ) -> qt.QToolButton:
        """Return a remove button widget."""
        button = qt.QToolButton(self)
        button.setIcon(icons.getQIcon("remove"))
        button.setStyleSheet("QToolButton { border-radius: 0px; }")
        button.clicked.connect(
            functools.partial(self._removeFile, removeItem, parentItem)
        )
        return button

    def _removeFile(
        self, removeItem: qt.QStandardItem | None, parentItem: qt.QStandardItem
    ):
        """Remove a file from the model and the plot."""

        # If removeItem is None, the file to remove is the X dataset
        if removeItem is None:
            row = 0
            removeItem = self.model().getXParent()

        if removeItem:
            row = removeItem.row()

            # If the parentItem is None, the file to remove is the X dataset
            if parentItem is None:
                parentItem = self.model().getXParent()

            parentItem.removeRow(row)

            if parentItem == self.model().getXParent():
                self.model()._xDataset = None
                self.model()._setXFile()
                index = self.model().index(0, 2)
                self.setIndexWidget(index, None)
                self.model()._plot1D.setXAxisLabel("X")
                self.model()._updateYCurvesWithDefaultX()

    def dragEnterEvent(self, event):
        super().dragEnterEvent(event)
        self.acceptDragEvent(event)

    def dragMoveEvent(self, event):
        dropIndex = self.indexAt(event.pos())
        if (
            dropIndex.isValid()
            and self.model().itemFromIndex(dropIndex).parent() is None
        ):
            self.setDropHighlight("X")
        else:
            self.setDropHighlight("Y")
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        super().dragLeaveEvent(event)
        self.setDropHighlight(None)

    def dropEvent(self, event):
        super().dropEvent(event)
        byteString = event.mimeData().data("application/x-silx-uri")
        url = silx.io.url.DataUrl(byteString.data().decode("utf-8"))

        dropIndex = self.indexAt(event.pos())
        if (
            dropIndex.isValid()
            and self.model().itemFromIndex(dropIndex).parent() is None
        ):
            self.setX(url)
        else:
            self.addY(url)

        self.setDropHighlight(None)
        event.acceptProposedAction()

    def acceptDragEvent(self, event):
        """Accept the drag event if the mime data contains a 1D dataset."""

        if event.mimeData().hasFormat("application/x-silx-uri"):
            byteString = event.mimeData().data("application/x-silx-uri")
            url = silx.io.url.DataUrl(byteString.data().decode("utf-8"))
            with silx.io.open(url.file_path()) as file:
                data = file[url.data_path()]
                if silx.io.is_dataset(data) and data.ndim == 1:
                    event.acceptProposedAction()
        else:
            event.ignore()

    def setX(self, url: silx.io.url.DataUrl):
        """Set the X dataset in the model and the plot."""
        targetNode = "X"
        node = self.model().getXParent()
        self.model().addUrl(url, targetNode)
        self._createRemoveButton(node, None)
        self.model()._plot1D.resetZoom()

    def addY(self, url: silx.io.url.DataUrl):
        """Add a Y dataset to the model and the plot."""
        targetNode = "Y"
        node = self.model().getYParent()
        self.model().addUrl(url, targetNode)
        self._createIconWidget(node.rowCount() - 2, node)
        self._createRemoveButton(node.rowCount() - 2, node)
        self.model()._plot1D.resetZoom()

    def clear(self):
        """Clear all datasets from the model and the plot."""
        self.model().clearAll()
        index = self.model().index(0, 2)
        self.setIndexWidget(index, None)

    def setDropHighlight(self, value: str | None):
        """Set the drop highlight for the X and Y datasets."""
        xDropFileItem = self.model().itemFromIndex(
            self.model().sibling(
                0, 1, self.model().indexFromItem(self.model().getXParent())
            )
        )
        xDropFileItem.setData(value == "X", _DROP_HIGHLIGHT_ROLE)
        yDropFileItem = (
            self.model().getYParent().child(self.model().getYParent().rowCount() - 1, 1)
        )
        yDropFileItem.setData(value == "Y", _DROP_HIGHLIGHT_ROLE)


class _DropPlot1D(plot.Plot1D):
    """Plot1D widget for displaying 1D datasets that can accept drops."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._treeView = None
        self.dropOverlay = DropOverlay(self)

    def setTreeView(self, treeView: qt.QTreeView):
        """Set the TreeView widget for the plot."""
        self._treeView = treeView

    def dragEnterEvent(self, event):
        super().dragEnterEvent(event)
        self._treeView.acceptDragEvent(event)
        if event.isAccepted():
            self._showDropOverlay(event)
            self.dropOverlay.show()

    def dragMoveEvent(self, event):
        super().dragMoveEvent(event)
        self._showDropOverlay(event)

    def dragLeaveEvent(self, event):
        super().dragLeaveEvent(event)
        self.dropOverlay.hide()

    def dropEvent(self, event):
        super().dropEvent(event)
        self.dropOverlay.hide()
        byteString = event.mimeData().data("application/x-silx-uri")
        url = silx.io.url.DataUrl(byteString.data().decode("utf-8"))

        plotArea = self.getWidgetHandle()
        dropPosition = plotArea.mapFrom(self, event.pos())

        plotBounds = self.getPlotBoundsInPixels()
        left, top, width, height = plotBounds
        yAreaTop = top + height

        if dropPosition.y() > yAreaTop:
            self._treeView.setX(url)
        else:
            self._treeView.addY(url)

        self.resetZoom()
        event.acceptProposedAction()

    def setXAxisLabel(self, label: str):
        """Set the label for the X axis."""
        xAxis = self.getXAxis()
        xAxis.setLabel(label)

    def _showDropOverlay(self, event):
        """Show the drop overlay at the drop position."""
        plotArea = self.getWidgetHandle()
        dropPosition = plotArea.mapFrom(self, event.pos())
        offset = plotArea.mapTo(self, qt.QPoint(0, 0))

        plotBounds = self.getPlotBoundsInPixels()
        left, top, width, height = plotBounds
        yAreaTop = top + height

        if dropPosition.y() > yAreaTop:
            rect = qt.QRect(left + offset.x(), yAreaTop + offset.y(), width, 50)
        else:
            rect = qt.QRect(left + offset.x(), top + offset.y(), width, height)

        self.dropOverlay.setGeometry(rect)


class _PlotToolBar(qt.QToolBar):
    """Toolbar widget for the plot."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def addClearAction(self, treeView: qt.QTreeView):
        """Add the clear action to the toolbar."""
        icon = self.style().standardIcon(qt.QStyle.SP_TrashIcon)
        clearAction = qt.QAction(icon, "Clear All", self)
        clearAction.triggered.connect(treeView.clear)
        self.addAction(clearAction)


class DropOverlay(qt.QWidget):
    """Overlay widget for displaying drop zones on the plot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(qt.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(qt.Qt.WA_NoSystemBackground)
        self.hide()

    def paintEvent(self, event):
        """Paint the overlay."""
        painter = qt.QPainter(self)
        painter.setRenderHint(qt.QPainter.Antialiasing)
        brush_color = qt.QColor(0, 0, 0, 50)
        painter.setBrush(qt.QBrush(brush_color))
        painter.drawRect(self.rect())


class CustomPlotSelectionWindow(qt.QMainWindow):
    """A customized plot selection window allowing the user to select and display 1D data sets."""

    sigVisibilityChanged = qt.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot selection")

        self._plot1D = _DropPlot1D()
        model = _FileListModel(self._plot1D)

        self._treeView = _DropTreeView(model, self)
        self._plot1D.setTreeView(self._treeView)

        centralWidget = qt.QSplitter()

        centralWidget.addWidget(self._treeView)
        centralWidget.addWidget(self._plot1D)

        centralWidget.setCollapsible(0, False)
        centralWidget.setCollapsible(1, False)
        centralWidget.setStretchFactor(1, 1)

        self.setCentralWidget(centralWidget)

        toolbar = _PlotToolBar(self)
        toolbar.addClearAction(self._treeView)
        self.addToolBar(qt.Qt.TopToolBarArea, toolbar)

    def getPlot1D(self) -> _DropPlot1D:
        """Return the plot widget."""
        return self._plot1D

    def getTreeView(self) -> _DropTreeView:
        """Return the tree view widget."""
        return self._treeView

    def showEvent(self, event):
        super().showEvent(event)
        self.sigVisibilityChanged.emit(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.sigVisibilityChanged.emit(False)

    def setX(self, dataUrl: silx.io.url.DataUrl):
        """Set the X dataset in the model and the plot."""
        self._treeView.setX(dataUrl)

    def addY(self, dataUrl: silx.io.url.DataUrl):
        """Add a Y dataset to the model and the plot."""
        self._treeView.addY(dataUrl)
