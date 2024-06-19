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

""""""

from silx.gui import qt, plot, icons
import silx.io
import numpy
import functools
from silx.gui.plot.LegendSelector import LegendIcon
import silx.io.utils
from silx.gui.hdf5 import _utils


_DROP_HIGHLIGHT_ROLE = qt.Qt.UserRole + 1


class _HashDropZones(qt.QStyledItemDelegate):
    """Delegate item displaying a drop zone when the item does not contain a dataset."""

    def __init__(self, parent=None):
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
                "Drop a 1D dataset",
            )
            painter.restore()
        else:
            qt.QStyledItemDelegate.paint(self, painter, option, index)


class _FileListModel(qt.QStandardItemModel):
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

        self._plot1D = plot
        self._index = 0

        self._xDataset = None

        self._reset()

        self.addUrl(silx.io.url.DataUrl())

        self.rowsAboutToBeRemoved.connect(self._rowAboutToBeRemoved)

    def getXParent(self):
        return self._xParent

    def getYParent(self):
        return self._yParent

    def _setXFile(self, url=None, curve=None):
        if url is None:
            dataName = ""
        else:
            dataName = self._getBasename(url.data_path())
            self._plot1D.setXAxisLabel(dataName)

        fileItem, iconItem, removeItem = self._createRowItems(dataName, curve)
        fileItem.setData(qt.QSize(0, 30), qt.Qt.SizeHintRole)

        fileItem.setData(self._createToolTip(url), qt.Qt.ToolTipRole)

        xIndex = self._findRowContainingText("X")
        if xIndex is not None:
            self.setItem(xIndex, 1, fileItem)
            self.setItem(xIndex, 2, removeItem)

    def _addYFile(self, url=None, curve=None, node="X"):
        if url is None:
            dataName = ""
        else:
            dataName = self._getBasename(url.data_path())
        fileItem, iconItem, removeItem = self._createRowItems(dataName, curve)
        if not dataName:
            fileItem.setData(qt.QSize(0, 30), qt.Qt.SizeHintRole)

        fileItem.setData(self._createToolTip(url), qt.Qt.ToolTipRole)

        if self.getYParent().rowCount() == 0:
            self.getYParent().appendRow([qt.QStandardItem(), fileItem])
            return

        self.getYParent().insertRow(
            self.getYParent().rowCount() - 1, [iconItem, fileItem, removeItem]
        )

    def _findRowContainingText(self, text):
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if item and item.text() == text:
                return row
        return None

    def _createRowItems(self, filename, curve):
        fileItem = qt.QStandardItem(filename)
        fileItem.setData(curve, qt.Qt.UserRole)
        fileItem.setData(False, _DROP_HIGHLIGHT_ROLE)
        fileItem.setEditable(False)
        fileItem.setDropEnabled(True)

        iconItem = qt.QStandardItem()
        removeItem = qt.QStandardItem()
        iconItem.setEditable(False)

        return fileItem, iconItem, removeItem

    def fileItemExists(self, filename):
        for row in range(self.getYParent().rowCount()):
            item = self.item(row, 1)
            if item and item.text() == filename:
                return True
        return False

    def addUrl(self, url, node="X"):
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

    def _addPlot(self, x, y):
        if x is None:
            x = numpy.arange(len(y))

        legend = f"Curve {self._index}"
        self._index += 1

        length = min(len(x), len(y))
        curve = self._plot1D.addCurve(x=x[:length], y=y[:length], legend=legend)
        curve.setInfo(y[()])
        return curve

    def _rowAboutToBeRemoved(self, parentIndex, first, last):
        parentItem = self.itemFromIndex(parentIndex)
        for row in range(first, last + 1):
            fileItem = parentItem.child(row, 1)
            curve = fileItem.data(qt.Qt.UserRole)
            if curve is not None:
                self._plot1D.removeItem(curve)

    def _updateYCurvesWithDefaultX(self):
        for item in self._plot1D.getItems():
            y = item.getInfo()
            x = numpy.arange(len(y))
            item.setData(x, y)
            self._plot1D.resetZoom()

    def clearAll(self):
        self._xDataset = None

        self._yParent.removeRows(0, self._yParent.rowCount())

        self._setXFile()
        self._addYFile()

        self._plot1D.clear()
        self._plot1D.setXAxisLabel("X")
        self._plot1D.resetZoom()

        self._updateYCurvesWithDefaultX()

    def _reset(self):
        self._xDataset = None

        self._setXFile()
        self._addYFile()

        self._plot1D.clear()
        self._updateYCurvesWithDefaultX()

    def _createToolTip(self, url):
        if url is None:
            return ""
        attrs = {
            "Name": self._getBasename(url.data_path()),
            "Path": url.data_path(),
            "File name": self._getBasename(url.file_path()),
        }
        return _utils.htmlFromDict(attrs, title="HDF5 Dataset")

    def _getBasename(self, text):
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

    def _createIconWidget(self, row, parentItem):
        fileItem = parentItem.child(row, 1)
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

    def _createRemoveButton(self, row, parentItem):
        if parentItem is None:
            parentItem = self.model().getXParent()
            index = self.model().index(0, 2)
            button = qt.QToolButton(self)
            button.setIcon(icons.getQIcon("remove"))
            button.clicked.connect(
                functools.partial(self._removeFile, None, parentItem)
            )
            self.setIndexWidget(index, button)
            return

        removeItem = parentItem.child(row, 2)
        if removeItem:
            button = qt.QToolButton(self)
            button.setIcon(icons.getQIcon("remove"))
            button.clicked.connect(
                functools.partial(self._removeFile, removeItem, parentItem)
            )
            self.setIndexWidget(removeItem.index(), button)

    def _removeFile(self, removeItem, parentItem):
        if removeItem is None:
            row = 0
            removeItem = self.model().getXParent()

        if removeItem:
            row = removeItem.row()

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
        if event.mimeData().hasFormat("application/x-silx-uri"):
            byteString = event.mimeData().data("application/x-silx-uri")
            url = silx.io.url.DataUrl(byteString.data().decode("utf-8"))
            with silx.io.open(url.file_path()) as file:
                data = file[url.data_path()]
                if (
                    silx.io.is_dataset(data)
                    and data.ndim == 1
                    and not self.model().fileItemExists(url.data_path())
                ):
                    event.acceptProposedAction()
        else:
            event.ignore()

    def setX(self, url):
        targetNode = "X"
        node = self.model().getXParent()
        self.model().addUrl(url, targetNode)
        self._createRemoveButton(node, None)
        self.model()._plot1D.resetZoom()

    def addY(self, url):
        targetNode = "Y"
        node = self.model().getYParent()
        self.model().addUrl(url, targetNode)
        self._createIconWidget(node.rowCount() - 2, node)
        self._createRemoveButton(node.rowCount() - 2, node)
        self.model()._plot1D.resetZoom()

    def clear(self):
        self.model().clearAll()
        index = self.model().index(0, 2)
        self.setIndexWidget(index, None)

    def setDropHighlight(self, value):
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._treeView = None

    def setTreeView(self, treeView):
        self._treeView = treeView

    def dragEnterEvent(self, event):
        super().dragEnterEvent(event)
        self._treeView.acceptDragEvent(event)

    def dropEvent(self, event):
        super().dropEvent(event)
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

    def setXAxisLabel(self, label):
        xAxis = self.getXAxis()
        xAxis.setLabel(label)


class _PlotToolBar(qt.QToolBar):
    def __init__(self, parent=None):
        super().__init__(parent)

    def addClearAction(self, treeView):
        icon = self.style().standardIcon(qt.QStyle.SP_TrashIcon)
        clearAction = qt.QAction(icon, "Clear All", self)
        clearAction.triggered.connect(treeView.clear)
        self.addAction(clearAction)


class CustomPlotSelectionWindow(qt.QMainWindow):
    sigVisibilityChanged = qt.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot selection")

        self.plot1D = _DropPlot1D()
        model = _FileListModel(self.plot1D)

        self.treeView = _DropTreeView(model, self)
        self.plot1D.setTreeView(self.treeView)

        centralWidget = qt.QSplitter()

        centralWidget.addWidget(self.treeView)
        centralWidget.addWidget(self.plot1D)

        centralWidget.setCollapsible(0, False)
        centralWidget.setCollapsible(1, False)
        centralWidget.setStretchFactor(1, 1)

        self.setCentralWidget(centralWidget)

        self.toolbar = _PlotToolBar(self)
        self.toolbar.addClearAction(self.treeView)
        self.addToolBar(qt.Qt.TopToolBarArea, self.toolbar)

    def showEvent(self, event):
        super().showEvent(event)
        self.sigVisibilityChanged.emit(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.sigVisibilityChanged.emit(False)
