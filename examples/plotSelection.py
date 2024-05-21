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
            rowIndex = model.index(index.row(), 0, index.parent())
            rowItem = model.itemFromIndex(rowIndex)
            if not rowItem.data(qt.Qt.DisplayRole):
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

class _IconDropZones(_HashDropZones):
    """Delegate item displaying a LegendIcon when the item contains one."""

    def paint(self, painter, option, index):
        """
        Paint the item

        :param qt.QPainter painter: A painter
        :param qt.QStyleOptionViewItem option: Options of the item to paint
        :param qt.QModelIndex index: Index of the item to paint
        """
        if index.isValid():
            legendIcon = index.data(qt.Qt.UserRole)
            if isinstance(legendIcon, LegendIcon):
                legendIcon.paint(painter, option.rect, option.palette)
                return
        super(_IconDropZones, self).paint(painter, option, index)


class _FileListModel(qt.QStandardItemModel):
    def __init__(self, parent=None):
        """Constructor"""
        super().__init__(parent)
        root = self.invisibleRootItem()
        root.setDropEnabled(True)
        root.setDragEnabled(False)

        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["File Name", "Icon", ""])

    def supportedDropActions(self):
        """Inherited method to redefine supported drop actions."""
        return qt.Qt.CopyAction | qt.Qt.MoveAction
    
    def mimeTypes(self):
        """Inherited method to redefine draggable mime types."""
        return [Hdf5DatasetMimeData.MIME_TYPE]
    
    def addFile(self, file_name, legendIcon):
        file_item = qt.QStandardItem(file_name)
        icon_item = qt.QStandardItem()
        icon_item.setData(legendIcon, qt.Qt.UserRole)
        remove_item = qt.QStandardItem()
        

        
        icon_item.setEditable(False)

        if self.rowCount() == 0:
            self.appendRow([file_item])
            return
        self.insertRow(self.rowCount()-1, [file_item, icon_item, remove_item])

        icon_index = icon_item.index()
        remove_index = icon_index.siblingAtColumn(2)
        
        print(f"Added file: {file_item.data(qt.Qt.DisplayRole)}")
        return remove_index

    def fileExists(self, file_name):
        for row in range(self.rowCount()):
            item = self.item(row, 0)  
            if item and item.text() == file_name:
                return True
        return False
    
class _DropTreeView(qt.QTreeView):
    """TreeView widget for displaying dropped file names"""

    def __init__(self, plot, parent=None):
        super().__init__(parent)
        self.__model = _FileListModel()
        self.setModel(self.__model)

        self.setItemDelegateForColumn(0, _HashDropZones(self))
        self.setItemDelegateForColumn(1, _IconDropZones(self))
        self.__model.addFile("", None)

        header = self.header()
        header.setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, qt.QHeaderView.Stretch)

        self.setUrl(silx.io.url.DataUrl())
        self.plot = plot
        self.curves = []

        self.setDropIndicatorShown(True)
        self.setDragDropOverwriteMode(True)
        self.setDragEnabled(True)
        self.viewport().setAcceptDrops(True)

    def setPlot(self, dataset):
        x_values = numpy.arange(0, len(dataset))
        legend_name = "Curve {}".format(len(self.curves) + 1)
        curve = self.plot.addCurve(x=x_values, y=dataset, legend=legend_name)
        legendIcon = LegendIcon(self, curve)
        self.curves.append(curve)
        return legendIcon

    def setUrl(self, url):
        if url.file_path() is not None:
            file = silx.io.open(url.file_path())
            data = file[url.data_path()]
            if silx.io.is_dataset(data):
                if data.ndim == 1:
                    legendIcon = self.setPlot(data)
                    self.__model.addFile(url.data_path(), legendIcon)
                    self.createRemoveButton(self.__model.rowCount() - 2)
                
    def createRemoveButton(self, row):
        remove_item = self.__model.item(row, 2)
        print(f"Remove item : {remove_item}, row : {row}")       
        if remove_item:
            print("Tesyt")
            button = qt.QPushButton(self)
            button.setIcon(icons.getQIcon("remove"))
            button.clicked.connect(functools.partial(self.removeFile, remove_item))
            self.setIndexWidget(remove_item.index(), button)

    def removeFile(self, remove_item):
        if remove_item:
                row = remove_item.row()
                print(f"Row :  {row}")
                legend = self.curves.pop(row)
                self.plot.removeCurve(legend)
                
                self.model().removeRow(row)
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-silx-uri"):
            byteString = event.mimeData().data("application/x-silx-uri")
            url = silx.io.url.DataUrl(byteString.data().decode("utf-8"))
            with silx.io.open(url.file_path()) as file:
                data = file[url.data_path()]
                if silx.io.is_dataset(data) and data.ndim == 1 and not self.__model.fileExists(url.data_path()):
                    event.acceptProposedAction()
        print("dragEnterEvent called")
    
    def dropEvent(self, event):
        byteString = event.mimeData().data("application/x-silx-uri")
        url = silx.io.url.DataUrl(byteString.data().decode("utf-8"))
        self.setUrl(url)
        event.acceptProposedAction()
        print("dropEvent called")

class ToolBar(qt.QToolBar):
    def __init__(self, tree_view, plot, parent=None):
        super().__init__(parent)
        self.tree_view = tree_view
        self.plot = plot

        removeAction = self.addAction(icons.getQIcon("remove"), "Remove File")
        removeAction.triggered.connect(self.removeFile)

    def removeFile(self):
        selected_indexes = self.tree_view.selectedIndexes()
        for index in selected_indexes:
            if index.column() == 0:
                row = index.row()
                self.plot.removeCurve(self.tree_view.curves[row])
                self.tree_view.model().removeRow(row)
    
class MainWindow(qt.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Drag And Drop Project")

        self.plot1D = plot.Plot1D()
        self.treeView = _DropTreeView(self.plot1D, self)
        self.toolBar = ToolBar(self.treeView, self.plot1D)

        centralWidget = qt.QWidget()
        mainLayout = qt.QHBoxLayout(centralWidget)

        verticalLayout = qt.QVBoxLayout()
        verticalLayout.addWidget(self.toolBar)
        verticalLayout.addWidget(self.treeView)
        

        mainLayout.addLayout(verticalLayout)
        mainLayout.addWidget(self.plot1D)

        self.setCentralWidget(centralWidget)

if __name__ == "__main__":
    app = qt.QApplication([])

    window = MainWindow()
    window.show()

    app.exec_()
