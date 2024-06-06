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
from silx.gui.hdf5._utils import Hdf5DatasetMimeData
from silx.gui.plot.LegendSelector import LegendIcon


class _HashDropZones(qt.QStyledItemDelegate):
    """Delegate item displaying a drop zone when the item does not contain a dataset."""

    def __init__(self, parent=None):
        """Constructor"""
        super(_HashDropZones, self).__init__(parent)
        self.__dropPen = qt.QPen(qt.QColor("#D0D0D0"), 2, qt.Qt.DotLine)
        self.__highlightDropPen = qt.QPen(qt.QColor("#000000"), 2, qt.Qt.SolidLine)
        self.__dropTargetIndex = None 

    def paint(self, painter, option, index):
        """Paint the item"""
        displayDropZone = False
        if index.isValid():
            model = index.model()
            rowIndex = model.index(index.row(), 1, index.parent())
            rowItem = model.itemFromIndex(rowIndex)
            if not rowItem.data(qt.Qt.DisplayRole):
                parentIndex = model.index(index.parent().row(), 1, index.parent().parent())
                parentItem = model.itemFromIndex(parentIndex)
                if parentItem and parentItem.text() not in ["Y"] or index.row() == 0:
                    displayDropZone = True
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
            painter.restore()
        else:
            qt.QStyledItemDelegate.paint(self, painter, option, index)

    def setDropTarget(self, index):
        """Set the current drop target index"""
        self.__dropTargetIndex = index

    def clearDropTarget(self):
        """Clear the current drop target index"""
        self.__dropTargetIndex = None

        
class _FileListModel(qt.QStandardItemModel):
    def __init__(self, plot, parent=None):
        """Constructor"""
        super().__init__(parent)
        root = self.invisibleRootItem()
        root.setDropEnabled(True)
        root.setDragEnabled(False)

        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["", "File Name", ""])

        self._xParent = qt.QStandardItem("X")
        self._xParent.setEditable(False)
        root.appendRow(self._xParent)

        self._yParent = qt.QStandardItem("Y")
        self._yParent.setEditable(False)
        root.appendRow(self._yParent)

        self._plot1D = plot
        self._index = 0

        self._xDataset = None

        self._addXFile("")
        self._addYFile("")

        self.addUrl(silx.io.url.DataUrl())

        self.rowsAboutToBeRemoved.connect(self._rowAboutToBeRemoved)

    def supportedDropActions(self):
        """Inherited method to redefine supported drop actions."""
        return qt.Qt.CopyAction | qt.Qt.MoveAction

    def mimeTypes(self):
        """Inherited method to redefine draggable mime types."""
        return [Hdf5DatasetMimeData.MIME_TYPE]

    def getXParent(self):
        return self._xParent

    def getYParent(self):
        return self._yParent

    def _addXFile(self, filename, curve=None):
        fileItem, iconItem, removeItem = self._createRowItems(filename, curve)

        xIndex = self._findRowContainingText("X")
        if xIndex is not None:
            self.setItem(xIndex, 1, fileItem)
            self.setItem(xIndex, 2, removeItem)

    def _addYFile(self, filename, curve=None, node="X"):
        fileItem, iconItem, removeItem = self._createRowItems(filename, curve)

        if self.getYParent().rowCount() == 0:
            self.getYParent().appendRow([None, fileItem])
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
        fileItem.setEditable(False)

        iconItem = qt.QStandardItem()
        removeItem = qt.QStandardItem()
        iconItem.setEditable(False)

        return fileItem, iconItem, removeItem

    def fileExists(self, filename):
        for row in range(self.getXParent().rowCount()):
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
                        y = item.getData()[1]
                        item.setData(self._xDataset, y)

                    self._addXFile(url.data_path())
                else:
                    if data is None:
                        return
                    curve = self._addPlot(self._xDataset, data)
                    self._addYFile(url.data_path(), curve)

    def _addPlot(self, x, y):
        if x is None:
            x = numpy.arange(len(y))

        if len(x) != len(y):
            length = min(len(x), len(y))
            x = x[:length]
            y = y[:length]

        legend_name = f"Curve {self._index}"
        self._index += 1

        curve = self._plot1D.addCurve(x=x, y=y, legend=legend_name)
        return curve

    def _rowAboutToBeRemoved(self, parentIndex, first, last):
        parentItem = self.itemFromIndex(parentIndex)
        for row in range(first, last + 1):
            fileItem = parentItem.child(row, 1)
            curve = fileItem.data(qt.Qt.UserRole)
            if curve is None:
                return
            self._plot1D.removeItem(curve)

    def _updateYCurvesWithDefaultX(self):
        for item in self._plot1D.getItems():
            y = item.getData()[1]
            x = numpy.arange(len(y))
            item.setData(x, y)
            self._plot1D.resetZoom()


class _DropTreeView(qt.QTreeView):
    """TreeView widget for displaying dropped file names"""

    (_DESCRIPTION_COLUMN, _FILE_COLUMN, _REMOVE_COLUMN) = range(3)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self.setModel(self._model)

        self._delegate = _HashDropZones(self)
        self.setItemDelegateForColumn(1, self._delegate)

        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, qt.QHeaderView.Stretch)
        header.setSectionResizeMode(2, qt.QHeaderView.ResizeToContents)
        self.expandAll()

        self.setSelectionMode(qt.QAbstractItemView.SingleSelection)
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
                self.model()._addXFile("")
                index = self.model().index(0, 2)
                self.setIndexWidget(index, None)
                self.model()._updateYCurvesWithDefaultX()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-silx-uri"):
            byteString = event.mimeData().data("application/x-silx-uri")
            url = silx.io.url.DataUrl(byteString.data().decode("utf-8"))
            with silx.io.open(url.file_path()) as file:
                data = file[url.data_path()]
                if (
                    silx.io.is_dataset(data)
                    and data.ndim == 1
                    and not self.model().fileExists(url.data_path())
                ):
                    event.acceptProposedAction()
                    self._delegate.setDropTarget(None)
                    self.viewport().update()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        dropPosition = self.indexAt(event.pos())
        if dropPosition.isValid():
            self._delegate.setDropTarget(dropPosition)
            self.viewport().update()
        event.acceptProposedAction()
    
    def dragLeaveEvent(self, event):
        self._delegate.clearDropTarget()
        self.viewport().update()        

    def dropEvent(self, event):
        byteString = event.mimeData().data("application/x-silx-uri")
        url = silx.io.url.DataUrl(byteString.data().decode("utf-8"))

        dropPosition = self.indexAt(event.pos())
        if not dropPosition.isValid():
            self._model.addUrl(url, node="Y")
            parentNode = self._model.getYParent()
            self._createIconWidget(parentNode.rowCount() - 2, parentNode)
            self._createRemoveButton(parentNode.rowCount() - 2, parentNode)
        else:
            dropItem = self.model().itemFromIndex(dropPosition)
            parentItem = dropItem.parent()

            if parentItem is None:
                targetNode = "X"
                node = self.model().getXParent()
                self.model().addUrl(url, node=targetNode)
                self._createRemoveButton(node, None)
                self.model()._plot1D.resetZoom()
            else:
                targetNode = "Y"
                parentNode = self.model().getYParent()
                self._model.addUrl(url, node=targetNode)
                self._createIconWidget(parentNode.rowCount() - 2, parentItem)
                self._createRemoveButton(parentNode.rowCount() - 2, parentItem)
                self._model._plot1D.resetZoom()

        self._delegate.clearDropTarget()
        self.viewport().update()
        event.acceptProposedAction()


class DropPlot1D(plot.Plot1D):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        self._model = None
        self._treeView = None

    def setModel(self, model):
        self._model = model

    def setTreeView(self, treeView):
        self._treeView = treeView

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-silx-uri"):
            byteString = event.mimeData().data("application/x-silx-uri")
            url = silx.io.url.DataUrl(byteString.data().decode("utf-8"))
            with silx.io.open(url.file_path()) as file:
                data = file[url.data_path()]
                if (
                    silx.io.is_dataset(data)
                    and data.ndim == 1
                    and not self._model.fileExists(url.data_path())
                ):
                    event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        byteString = event.mimeData().data("application/x-silx-uri")
        url = silx.io.url.DataUrl(byteString.data().decode("utf-8"))

        dropPosition = event.pos()

        plotBounds = self.getPlotBoundsInPixels()
        left, top, width, height = plotBounds

        xAreaHeight = int(height * 0.1)
        yAreaTop = top + height - xAreaHeight

        if dropPosition.y() > yAreaTop:
            targetNode = "X"
            parentItem = self._model.getXParent()
        else:
            targetNode = "Y"
            parentItem = self._model.getYParent()

        with silx.io.open(url.file_path()) as file:
            data = file[url.data_path()]
            if silx.io.is_dataset(data) and data.ndim == 1:
                if targetNode == "X" and self._model._xDataset is None:
                    node = self._model.getXParent()
                    self._model.addUrl(url, node=targetNode)
                    self._treeView._createRemoveButton(node, None)
                    self._model._updateYCurvesWithDefaultX()
                else:
                    parentNode = self._model.getYParent()
                    self._model.addUrl(url, node=targetNode)
                    self._treeView._createIconWidget(
                        parentNode.rowCount() - 2, parentItem
                    )
                    self._treeView._createRemoveButton(
                        parentNode.rowCount() - 2, parentItem
                    )

        self.resetZoom()
        event.acceptProposedAction()


class CustomPlotSelectionWindow(qt.QMainWindow):
    sigVisibilityChanged = qt.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot selection Window")

        self.plot1D = DropPlot1D()
        model = _FileListModel(self.plot1D)
        self.plot1D.setModel(model)

        self.treeView = _DropTreeView(model, self)
        self.plot1D.setTreeView(self.treeView)

        centralWidget = qt.QSplitter()

        centralWidget.addWidget(self.treeView)
        centralWidget.addWidget(self.plot1D)

        centralWidget.setCollapsible(0, False)
        centralWidget.setCollapsible(1, False)
        centralWidget.setStretchFactor(1, 1)

        self.setCentralWidget(centralWidget)

    def showEvent(self, event):
        super().showEvent(event)
        self.sigVisibilityChanged.emit(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.sigVisibilityChanged.emit(False)


if __name__ == "__main__":
    app = qt.QApplication([])

    window = CustomPlotSelectionWindow()
    window.show()

    app.exec_()
