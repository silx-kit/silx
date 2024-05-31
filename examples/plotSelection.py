from silx.gui import qt, plot, icons
import silx.io
import numpy
import functools
from silx.gui.hdf5._utils import Hdf5DatasetMimeData
from silx.gui.plot.LegendSelector import LegendIcon

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

            painter.setPen(self.__dropPen)
            painter.drawRect(option.rect.adjusted(3, 3, -3, -3))
            painter.restore()
        else:
            qt.QStyledItemDelegate.paint(self, painter, option, index)


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
        root.appendRow(self._xParent)

        self._yParent = qt.QStandardItem("Y")
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

    def _addXFile(self, file_name, curve=None):
        fileItem, iconItem, removeItem = self._createRowItems(file_name, curve)

        xIndex = self._findRowContainingText("X")
        print(f"xIndex : {xIndex}")
        if xIndex is not None:
            self.setItem(xIndex, 1, fileItem) 
            self.setItem(xIndex, 2, removeItem)
        
        print(f"Added X file: {fileItem.data(qt.Qt.DisplayRole)}")

    def _addYFile(self, file_name, curve=None, node="X"):
        fileItem, iconItem, removeItem = self._createRowItems(file_name, curve)

        if self.getYParent().rowCount() == 0:
            self.getYParent().appendRow([None, fileItem])
            return
    
        self.getYParent().insertRow(self.getYParent().rowCount()-1, [iconItem, fileItem, removeItem])
        print(f"Added Y file: {fileItem.data(qt.Qt.DisplayRole)}")

    def _findRowContainingText(self, text):
        for row in range(self.rowCount()):
            item = self.item(row, 0)
            if item and item.text() == text:
                return row
        return None

    def _createRowItems(self, file_name, curve):
        fileItem = qt.QStandardItem(file_name)
        fileItem.setData(curve, qt.Qt.UserRole)

        iconItem = qt.QStandardItem()
        removeItem = qt.QStandardItem()
        iconItem.setEditable(False)

        return fileItem, iconItem, removeItem

    def fileExists(self, file_name):
        for row in range(self.getXParent().rowCount()):
            item = self.item(row, 1) 
            print(f"File exist : item {item}")
            if item and item.text() == file_name:
                print("File exist : True")
                return True
        print("File exist : False")
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
                            print("No data")
                            return
                        curve = self._addPlot(self._xDataset, data)
                        self._addYFile(url.data_path(), curve)
            else:
                print(f"Data is not a valid dataset or not 1D : {data}")

    def _addPlot(self, x, y):
        if x is None:
            print("X dataset is not set")
            x = numpy.arange(len(y))
        
        if len(x) != len(y):
            print("x and y have different length")
            length = min(len(x), len(y))
            x = x[:length]
            y = y[:length]

        legend_name = "Curve {}".format(self._index)
        self._index += 1

        curve = self._plot1D.addCurve(x=x, y=y, legend=legend_name)
        return curve 
    
    def _rowAboutToBeRemoved(self, parentIndex, first, last):
        print("rowAboutToBeRemoved")
        parentItem = self.itemFromIndex(parentIndex)
        for row in range(first, last+1):
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

    (
        _DESCRIPTION_COLUMN,
        _FILE_COLUMN,
        _REMOVE_COLUMN
    ) = range(3)

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self.setModel(self._model)

        self.setItemDelegateForColumn(1, _HashDropZones(self))
        
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
            parentItem = self._model.getXParent()
            index = self.model().index(0, 2)  
            button = qt.QToolButton(self)
            button.setIcon(icons.getQIcon("remove"))
            button.clicked.connect(functools.partial(self._removeFile, None, parentItem))
            self.setIndexWidget(index, button)
            return

        removeItem = parentItem.child(row, 2)
        print(f"Trying to create remove button at row {row}")
        print(f"Remove item: {removeItem}, row: {row}")
        if removeItem:
            button = qt.QToolButton(self)
            button.setIcon(icons.getQIcon("remove"))
            button.clicked.connect(functools.partial(self._removeFile, removeItem, parentItem))
            self.setIndexWidget(removeItem.index(), button)
        else:
            print(f"Remove item is None at row {row}")

    def _removeFile(self, removeItem, parentItem):
        if removeItem is None:
            row = 0
            removeItem = self._model.getXParent()

        if removeItem:
            row = removeItem.row()
            print(f"removeFile : {row}")

            if parentItem is None:
                parentItem = self._model.getXParent()
    
            parentItem.removeRow(row)

            if parentItem == self._model.getXParent():
                self._model._xDataset = None
                self._model._addXFile("")
                index = self.model().index(0, 2)  
                self.setIndexWidget(index, None)
                self._model._updateYCurvesWithDefaultX()

    def dragEnterEvent(self, event):
        print("dragEnterEvent called")
        if event.mimeData().hasFormat("application/x-silx-uri"):
            byteString = event.mimeData().data("application/x-silx-uri")
            url = silx.io.url.DataUrl(byteString.data().decode("utf-8"))
            with silx.io.open(url.file_path()) as file:
                data = file[url.data_path()]
                if silx.io.is_dataset(data) and data.ndim == 1 and not self._model.fileExists(url.data_path()):
                    event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        print("dropEvent called")
        byteString = event.mimeData().data("application/x-silx-uri")
        url = silx.io.url.DataUrl(byteString.data().decode("utf-8"))

        dropPosition = self.indexAt(event.pos())
        if not dropPosition.isValid():
            return

        dropItem = self._model.itemFromIndex(dropPosition)
        parentItem = dropItem.parent()

        # X condition
        if parentItem is None:
            targetNode = "X"
            node = self._model.getXParent()
            self._model.addUrl(url, node=targetNode)
            self._createRemoveButton(node, parentItem)
            self._model._plot1D.resetZoom()
            return

        targetNode = "Y"
        parentNode = self._model.getYParent()   
        self._model.addUrl(url, node=targetNode)
        self._createIconWidget(parentNode.rowCount() - 2, parentItem)
        self._createRemoveButton(parentNode.rowCount() - 2, parentItem)
        self._model._plot1D.resetZoom()
        event.acceptProposedAction()
        
class MainWindow(qt.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Drag And Drop Project")

        self.plot1D = plot.Plot1D()
        model = _FileListModel(self.plot1D)
        self.treeView = _DropTreeView(model, self)
        
        centralWidget = qt.QSplitter()

        centralWidget.addWidget(self.treeView)
        centralWidget.addWidget(self.plot1D)

        centralWidget.setCollapsible(0, False)
        centralWidget.setCollapsible(1, False)
        centralWidget.setStretchFactor(1, 1)
        
        self.setCentralWidget(centralWidget)

if __name__ == "__main__":
    app = qt.QApplication([])

    window = MainWindow()
    window.show()

    app.exec_()
