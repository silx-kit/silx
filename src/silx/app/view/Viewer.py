# /*##########################################################################
# Copyright (C) 2016-2023 European Synchrotron Radiation Facility
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
"""Browse a data file with a GUI"""

from __future__ import annotations

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "15/01/2019"


import os
import logging
import functools
import traceback
from types import TracebackType
from typing import Optional

import silx.io.nxdata
from silx.gui import qt
from silx.gui import icons
import silx.gui.hdf5
from .ApplicationContext import ApplicationContext
from .CustomNxdataWidget import CustomNxdataWidget
from .CustomNxdataWidget import CustomNxDataToolBar
from ..utils import parseutils
from silx.gui.utils import projecturl
from .DataPanel import DataPanel


_logger = logging.getLogger(__name__)


class Viewer(qt.QMainWindow):
    """
    This window allows to browse a data file like images or HDF5 and it's
    content.
    """

    def __init__(self, parent=None, settings=None):
        """
        Constructor
        """

        qt.QMainWindow.__init__(self, parent)
        self.setWindowTitle("Silx viewer")

        silxIcon = icons.getQIcon("silx")
        self.setWindowIcon(silxIcon)

        self.__error = ""

        self.__context = self.createApplicationContext(settings)
        self.__context.restoreLibrarySettings()

        self.__dialogState = None
        self.__customNxDataItem = None
        self.__treeview = silx.gui.hdf5.Hdf5TreeView(self)
        self.__treeview.setExpandsOnDoubleClick(False)
        """Silx HDF5 TreeView"""

        rightPanel = qt.QSplitter(self)
        rightPanel.setOrientation(qt.Qt.Vertical)
        self.__splitter2 = rightPanel

        self.__displayIt = None
        self.__treeWindow = self.__createTreeWindow(self.__treeview)

        # Custom the model to be able to manage the life cycle of the files
        treeModel = silx.gui.hdf5.Hdf5TreeModel(self.__treeview, ownFiles=False)
        treeModel.sigH5pyObjectLoaded.connect(self.__h5FileLoaded)
        treeModel.sigH5pyObjectRemoved.connect(self.__h5FileRemoved)
        treeModel.sigH5pyObjectSynchronized.connect(self.__h5FileSynchonized)
        treeModel.setDatasetDragEnabled(True)
        self.__treeModelSorted = silx.gui.hdf5.NexusSortFilterProxyModel(
            self.__treeview
        )
        self.__treeModelSorted.setSourceModel(treeModel)
        self.__treeModelSorted.sort(0, qt.Qt.AscendingOrder)
        self.__treeModelSorted.setSortCaseSensitivity(qt.Qt.CaseInsensitive)

        self.__treeview.setModel(self.__treeModelSorted)
        rightPanel.addWidget(self.__treeWindow)

        self.__customNxdata = CustomNxdataWidget(self)
        self.__customNxdata.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        # optimise the rendering
        self.__customNxdata.setUniformRowHeights(True)
        self.__customNxdata.setIconSize(qt.QSize(16, 16))
        self.__customNxdata.setExpandsOnDoubleClick(False)

        self.__customNxdataWindow = self.__createCustomNxdataWindow(self.__customNxdata)
        self.__customNxdataWindow.setVisible(False)
        rightPanel.addWidget(self.__customNxdataWindow)

        rightPanel.setStretchFactor(1, 1)
        rightPanel.setCollapsible(0, False)
        rightPanel.setCollapsible(1, False)

        self.__dataPanel = DataPanel(self, self.__context)

        spliter = qt.QSplitter(self)
        spliter.addWidget(rightPanel)
        spliter.addWidget(self.__dataPanel)
        spliter.setStretchFactor(1, 1)
        spliter.setCollapsible(0, False)
        spliter.setCollapsible(1, False)
        self.__splitter = spliter

        main_panel = qt.QWidget(self)
        layout = qt.QVBoxLayout()
        layout.addWidget(spliter)
        layout.setStretchFactor(spliter, 1)
        main_panel.setLayout(layout)

        self.setCentralWidget(main_panel)

        self.__treeview.activated.connect(self.displaySelectedData)
        self.__customNxdata.activated.connect(self.displaySelectedCustomData)
        self.__customNxdata.sigNxdataItemRemoved.connect(self.__customNxdataRemoved)
        self.__customNxdata.sigNxdataItemUpdated.connect(self.__customNxdataUpdated)
        self.__treeview.addContextMenuCallback(self.customContextMenu)

        treeModel = self.__treeview.findHdf5TreeModel()
        columns = list(treeModel.COLUMN_IDS)
        columns.remove(treeModel.VALUE_COLUMN)
        columns.remove(treeModel.NODE_COLUMN)
        columns.remove(treeModel.DESCRIPTION_COLUMN)
        columns.insert(1, treeModel.DESCRIPTION_COLUMN)
        self.__treeview.header().setSections(columns)

        self._iconUpward = icons.getQIcon("plot-yup")
        self._iconDownward = icons.getQIcon("plot-ydown")

        self.createActions()
        self.createMenus()
        self.__context.restoreSettings()

    def createApplicationContext(self, settings):
        return ApplicationContext(self, settings)

    def __createTreeWindow(self, treeView):
        toolbar = qt.QToolBar(self)
        toolbar.setIconSize(qt.QSize(16, 16))
        toolbar.setStyleSheet("QToolBar { border: 0px }")

        action = qt.QAction(toolbar)
        action.setIcon(icons.getQIcon("view-refresh"))
        action.setText("Refresh")
        action.setToolTip("Refresh all selected items")
        action.triggered.connect(self.__refreshSelected)
        action.setShortcuts(
            [
                qt.QKeySequence(qt.Qt.Key_F5),
                qt.QKeySequence(qt.Qt.CTRL | qt.Qt.Key_R),
            ]
        )
        toolbar.addAction(action)
        treeView.addAction(action)
        self.__refreshAction = action

        action = qt.QAction(toolbar)
        # action.setIcon(icons.getQIcon("view-refresh"))
        action.setText("Close")
        action.setToolTip("Close selected item")
        action.triggered.connect(self.__removeSelected)
        action.setShortcut(qt.QKeySequence.Delete)
        treeView.addAction(action)
        self.__closeAction = action

        toolbar.addSeparator()

        action = qt.QAction(toolbar)
        action.setIcon(icons.getQIcon("tree-expand-all"))
        action.setText("Expand all")
        action.setToolTip("Expand all selected items")
        action.triggered.connect(self.__expandAllSelected)
        action.setShortcut(qt.QKeySequence(qt.Qt.CTRL | qt.Qt.Key_Plus))
        toolbar.addAction(action)
        treeView.addAction(action)
        self.__expandAllAction = action

        action = qt.QAction(toolbar)
        action.setIcon(icons.getQIcon("tree-collapse-all"))
        action.setText("Collapse all")
        action.setToolTip("Collapse all selected items")
        action.triggered.connect(self.__collapseAllSelected)
        action.setShortcut(qt.QKeySequence(qt.Qt.CTRL | qt.Qt.Key_Minus))
        toolbar.addAction(action)
        treeView.addAction(action)
        self.__collapseAllAction = action

        action = qt.QAction("&Sort file content", toolbar)
        action.setIcon(icons.getQIcon("tree-sort"))
        action.setToolTip("Toggle sorting of file content")
        action.setCheckable(True)
        action.setChecked(True)
        action.triggered.connect(self.setContentSorted)
        toolbar.addAction(action)
        treeView.addAction(action)
        self._sortContentAction = action

        widget = qt.QWidget(self)
        layout = qt.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(treeView)
        return widget

    def __removeSelected(self):
        """Close selected items"""
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

        selection = self.__treeview.selectionModel()
        indexes = selection.selectedIndexes()
        selectedItems = []
        model = self.__treeview.model()
        h5files = set([])
        while len(indexes) > 0:
            index = indexes.pop(0)
            if index.column() != 0:
                continue
            h5 = model.data(index, role=silx.gui.hdf5.Hdf5TreeModel.H5PY_OBJECT_ROLE)
            rootIndex = index
            # Reach the root of the tree
            while rootIndex.parent().isValid():
                rootIndex = rootIndex.parent()
            rootRow = rootIndex.row()
            relativePath = self.__getRelativePath(model, rootIndex, index)
            selectedItems.append((rootRow, relativePath))
            h5files.add(h5.file)

        if len(h5files) != 0:
            model = self.__treeview.findHdf5TreeModel()
            for h5 in h5files:
                row = model.h5pyObjectRow(h5)
                model.removeH5pyObject(h5)

        qt.QApplication.restoreOverrideCursor()

    def __refreshSelected(self):
        """Refresh all selected items"""
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

        selection = self.__treeview.selectionModel()
        indexes = selection.selectedIndexes()
        selectedItems = []
        model = self.__treeview.model()
        h5files = []
        while len(indexes) > 0:
            index = indexes.pop(0)
            if index.column() != 0:
                continue
            rootIndex = index
            # Reach the root of the tree
            while rootIndex.parent().isValid():
                rootIndex = rootIndex.parent()
            rootRow = rootIndex.row()
            relativePath = self.__getRelativePath(model, rootIndex, index)
            selectedItems.append((rootRow, relativePath))
            h5 = model.data(
                rootIndex, role=silx.gui.hdf5.Hdf5TreeModel.H5PY_OBJECT_ROLE
            )
            item = model.data(
                rootIndex, role=silx.gui.hdf5.Hdf5TreeModel.H5PY_ITEM_ROLE
            )
            h5files.append((h5, item._openedPath))

        if len(h5files) == 0:
            qt.QApplication.restoreOverrideCursor()
            return

        model = self.__treeview.findHdf5TreeModel()
        for h5, filename in h5files:
            self.__synchronizeH5pyObject(h5, filename)

        model = self.__treeview.model()
        itemSelection = qt.QItemSelection()
        for rootRow, relativePath in selectedItems:
            rootIndex = model.index(rootRow, 0, qt.QModelIndex())
            index = self.__indexFromPath(model, rootIndex, relativePath)
            if index is None:
                continue
            indexEnd = model.index(index.row(), model.columnCount() - 1, index.parent())
            itemSelection.select(index, indexEnd)
        selection.select(itemSelection, qt.QItemSelectionModel.ClearAndSelect)

        qt.QApplication.restoreOverrideCursor()

    def __synchronizeH5pyObject(self, h5, filename: Optional[str] = None):
        model = self.__treeview.findHdf5TreeModel()
        # This is buggy right now while h5py do not allow to close a file
        # while references are still used.
        # FIXME: The architecture have to be reworked to support this feature.
        # model.synchronizeH5pyObject(h5)

        if filename is None:
            filename = f"{h5.file.filename}::{h5.name}"
        row = model.h5pyObjectRow(h5)
        index = self.__treeview.model().index(row, 0, qt.QModelIndex())
        paths = self.__getPathFromExpandedNodes(self.__treeview, index)
        model.removeH5pyObject(h5)
        model.insertFile(filename, row)
        index = self.__treeview.model().index(row, 0, qt.QModelIndex())
        self.__expandNodesFromPaths(self.__treeview, index, paths)

    def __getRelativePath(self, model, rootIndex, index):
        """Returns a relative path from an index to his rootIndex.

        If the path is empty the index is also the rootIndex.
        """
        path = ""
        while index.isValid():
            if index == rootIndex:
                return path
            name = model.data(index)
            if path == "":
                path = name
            else:
                path = name + "/" + path
            index = index.parent()

        # index is not a children of rootIndex
        raise ValueError("index is not a children of the rootIndex")

    def __getPathFromExpandedNodes(self, view, rootIndex):
        """Return relative path from the root index of the extended nodes"""
        model = view.model()
        rootPath = None
        paths = []
        indexes = [rootIndex]
        while len(indexes):
            index = indexes.pop(0)
            if not view.isExpanded(index):
                continue

            node = model.data(index, role=silx.gui.hdf5.Hdf5TreeModel.H5PY_ITEM_ROLE)
            path = node._getCanonicalName()
            if rootPath is None:
                rootPath = path
            path = path[len(rootPath) :]
            paths.append(path)

            for child in range(model.rowCount(index)):
                childIndex = model.index(child, 0, index)
                indexes.append(childIndex)
        return paths

    def __indexFromPath(self, model, rootIndex, path):
        elements = path.split("/")
        if elements[0] == "":
            elements.pop(0)
        index = rootIndex
        while len(elements) != 0:
            element = elements.pop(0)
            found = False
            for child in range(model.rowCount(index)):
                childIndex = model.index(child, 0, index)
                name = model.data(childIndex)
                if element == name:
                    index = childIndex
                    found = True
                    break
            if not found:
                return None
        return index

    def __expandNodesFromPaths(self, view, rootIndex, paths):
        model = view.model()
        for path in paths:
            index = self.__indexFromPath(model, rootIndex, path)
            if index is not None:
                view.setExpanded(index, True)

    def __expandAllSelected(self):
        """Expand all selected items of the tree.

        The depth is fixed to avoid infinite loop with recurssive links.
        """
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

        selection = self.__treeview.selectionModel()
        indexes = selection.selectedIndexes()
        model = self.__treeview.model()
        while len(indexes) > 0:
            index = indexes.pop(0)
            if isinstance(index, tuple):
                index, depth = index
            else:
                depth = 0
            if index.column() != 0:
                continue

            if depth > 10:
                # Avoid infinite loop with recursive links
                break

            if model.hasChildren(index):
                self.__treeview.setExpanded(index, True)
                for row in range(model.rowCount(index)):
                    childIndex = model.index(row, 0, index)
                    indexes.append((childIndex, depth + 1))
        qt.QApplication.restoreOverrideCursor()

    def __collapseAllSelected(self):
        """Collapse all selected items of the tree.

        The depth is fixed to avoid infinite loop with recurssive links.
        """
        selection = self.__treeview.selectionModel()
        indexes = selection.selectedIndexes()
        model = self.__treeview.model()
        while len(indexes) > 0:
            index = indexes.pop(0)
            if isinstance(index, tuple):
                index, depth = index
            else:
                depth = 0
            if index.column() != 0:
                continue

            if depth > 10:
                # Avoid infinite loop with recursive links
                break

            if model.hasChildren(index):
                self.__treeview.setExpanded(index, False)
                for row in range(model.rowCount(index)):
                    childIndex = model.index(row, 0, index)
                    indexes.append((childIndex, depth + 1))

    def __createCustomNxdataWindow(self, customNxdataWidget):
        toolbar = CustomNxDataToolBar(self)
        toolbar.setCustomNxDataWidget(customNxdataWidget)
        toolbar.setIconSize(qt.QSize(16, 16))
        toolbar.setStyleSheet("QToolBar { border: 0px }")

        widget = qt.QWidget(self)
        layout = qt.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(customNxdataWidget)
        return widget

    def __h5FileLoaded(self, loadedH5, filename):
        self.__context.pushRecentFile(loadedH5.file.filename)
        if filename == self.__displayIt:
            self.__displayIt = None
            self.displayData(loadedH5)

    def __h5FileRemoved(self, removedH5):
        self.__dataPanel.removeDatasetsFrom(removedH5)
        self.__customNxdata.removeDatasetsFrom(removedH5)
        removedH5.close()

    def __h5FileSynchonized(self, removedH5, loadedH5):
        self.__dataPanel.replaceDatasetsFrom(removedH5, loadedH5)
        self.__customNxdata.replaceDatasetsFrom(removedH5, loadedH5)
        removedH5.close()

    def closeEvent(self, event):
        self.__context.saveSettings()

        # Clean up as much as possible Python objects
        self.displayData(None)
        customModel = self.__customNxdata.model()
        customModel.clear()
        hdf5Model = self.__treeview.findHdf5TreeModel()
        hdf5Model.clear()

    def saveSettings(self, settings):
        """Save the window settings to this settings object

        :param qt.QSettings settings: Initialized settings
        """
        isFullScreen = bool(self.windowState() & qt.Qt.WindowFullScreen)
        if isFullScreen:
            # show in normal to catch the normal geometry
            self.showNormal()

        settings.beginGroup("mainwindow")
        settings.setValue("size", self.size())
        settings.setValue("pos", self.pos())
        settings.setValue("full-screen", isFullScreen)
        settings.endGroup()

        settings.beginGroup("mainlayout")
        settings.setValue("spliter", self.__splitter.sizes())
        settings.setValue("spliter2", self.__splitter2.sizes())
        isVisible = self.__customNxdataWindow.isVisible()
        settings.setValue("custom-nxdata-window-visible", isVisible)
        settings.endGroup()

        settings.beginGroup("content")
        isSorted = self._sortContentAction.isChecked()
        settings.setValue("is-sorted", isSorted)
        settings.endGroup()

        if isFullScreen:
            self.showFullScreen()

    def restoreSettings(self, settings):
        """Restore the window settings using this settings object

        :param qt.QSettings settings: Initialized settings
        """
        settings.beginGroup("mainwindow")
        size = settings.value("size", qt.QSize(640, 480))
        pos = settings.value("pos", qt.QPoint())
        isFullScreen = settings.value("full-screen", False)
        isFullScreen = parseutils.to_bool(isFullScreen, False)
        settings.endGroup()

        settings.beginGroup("mainlayout")
        try:
            data = settings.value("spliter")
            data = [int(d) for d in data]
            self.__splitter.setSizes(data)
        except Exception:
            _logger.debug("Backtrace", exc_info=True)
        try:
            data = settings.value("spliter2")
            data = [int(d) for d in data]
            self.__splitter2.setSizes(data)
        except Exception:
            _logger.debug("Backtrace", exc_info=True)
        isVisible = settings.value("custom-nxdata-window-visible", False)
        isVisible = parseutils.to_bool(isVisible, False)
        self.__customNxdataWindow.setVisible(isVisible)
        self._displayCustomNxdataWindow.setChecked(isVisible)
        settings.endGroup()

        settings.beginGroup("content")
        isSorted = settings.value("is-sorted", True)
        isSorted = parseutils.to_bool(isSorted, True)
        self.setContentSorted(isSorted)
        settings.endGroup()

        if not pos.isNull():
            self.move(pos)
        if not size.isNull():
            self.resize(size)
        if isFullScreen:
            self.showFullScreen()

    def createActions(self):
        action = qt.QAction("E&xit", self)
        action.setShortcut(qt.QKeySequence.Quit)
        action.setStatusTip("Exit the application")
        action.triggered.connect(self.close)
        self._exitAction = action

        action = qt.QAction("&Open...", self)
        action.setShortcut(qt.QKeySequence.Open)
        action.setStatusTip("Open a file")
        action.triggered.connect(self.open)
        self._openAction = action

        menu = qt.QMenu("Open Recent", self)
        menu.setStatusTip("Open a recently opened file")
        self._openRecentMenu = menu

        action = qt.QAction("Close All", self)
        action.setShortcut(qt.QKeySequence.Close)
        action.setStatusTip("Close all opened files")
        action.triggered.connect(self.closeAll)
        self._closeAllAction = action

        action = qt.QAction("&About", self)
        action.setStatusTip("Show the application's About box")
        action.triggered.connect(self.about)
        self._aboutAction = action

        action = qt.QAction("&Documentation", self)
        action.setStatusTip("Show the Silx library's documentation")
        action.triggered.connect(self.showDocumentation)
        self._documentationAction = action

        # Plot backend

        self._plotBackendMenu = qt.QMenu("Plot rendering backend", self)
        self._plotBackendMenu.setStatusTip("Select plot rendering backend")

        group = qt.QActionGroup(self)
        group.setExclusive(True)

        action = qt.QAction("matplotlib", self)
        action.setStatusTip("Plot will be rendered using matplotlib")
        action.setCheckable(True)
        action.triggered.connect(self.__forceMatplotlibBackend)
        group.addAction(action)
        self._plotBackendMenu.addAction(action)
        self._usePlotWithMatplotlib = action

        action = qt.QAction("OpenGL", self)
        action.setStatusTip("Plot will be rendered using OpenGL")
        action.setCheckable(True)
        action.triggered.connect(self.__forceOpenglBackend)
        group.addAction(action)
        self._plotBackendMenu.addAction(action)
        self._usePlotWithOpengl = action

        # Plot image orientation

        self._plotImageOrientationMenu = qt.QMenu(
            "Default plot image y-axis orientation", self
        )
        self._plotImageOrientationMenu.setStatusTip(
            "Select the default y-axis orientation used by plot displaying images"
        )

        group = qt.QActionGroup(self)
        group.setExclusive(True)

        action = qt.QAction("Downward, origin on top", self)
        action.setIcon(self._iconDownward)
        action.setStatusTip("Plot images will use a downward Y-axis orientation")
        action.setCheckable(True)
        action.triggered.connect(self.__forcePlotImageDownward)
        group.addAction(action)
        self._plotImageOrientationMenu.addAction(action)
        self._useYAxisOrientationDownward = action

        action = qt.QAction("Upward, origin on bottom", self)
        action.setIcon(self._iconUpward)
        action.setStatusTip("Plot images will use a upward Y-axis orientation")
        action.setCheckable(True)
        action.triggered.connect(self.__forcePlotImageUpward)
        group.addAction(action)
        self._plotImageOrientationMenu.addAction(action)
        self._useYAxisOrientationUpward = action

        # mpl layout

        action = qt.QAction("Use MPL tight layout", self)
        action.setCheckable(True)
        action.triggered.connect(self.__forceMplTightLayout)
        self._useMplTightLayout = action

        # Windows

        action = qt.QAction("Show custom NXdata selector", self)
        action.setStatusTip(
            "Show a widget which allow to create plot by selecting data and axes"
        )
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_F6))
        action.toggled.connect(self.__toggleCustomNxdataWindow)
        self._displayCustomNxdataWindow = action

    def __toggleCustomNxdataWindow(self):
        isVisible = self._displayCustomNxdataWindow.isChecked()
        self.__customNxdataWindow.setVisible(isVisible)

    def __updateFileMenu(self):
        files = self.__context.getRecentFiles()
        self._openRecentMenu.clear()
        self._openRecentMenu.setEnabled(len(files) != 0)
        if len(files) != 0:
            for filePath in files:
                baseName = os.path.basename(filePath)
                action = qt.QAction(baseName, self)
                action.setToolTip(filePath)
                action.triggered.connect(
                    functools.partial(self.__openRecentFile, filePath)
                )
                self._openRecentMenu.addAction(action)
            self._openRecentMenu.addSeparator()
            baseName = os.path.basename(filePath)
            action = qt.QAction("Clear history", self)
            action.setToolTip("Clear the history of the recent files")
            action.triggered.connect(self.__clearRecentFile)
            self._openRecentMenu.addAction(action)

    def __clearRecentFile(self):
        self.__context.clearRencentFiles()

    def __openRecentFile(self, filePath):
        self.appendFile(filePath)

    def __updateOptionMenu(self):
        """Update the state of the checked options as it is based on global
        environment values."""

        # plot backend

        title = self._plotBackendMenu.title().split(": ", 1)[0]
        backend = self.__context.getDefaultPlotBackend()
        self._plotBackendMenu.setTitle(f"{title}: {backend}")

        action = self._usePlotWithMatplotlib
        action.setChecked(backend in ["matplotlib", "mpl"])
        title = action.text().split(" (", 1)[0]
        if not action.isChecked():
            title += " (applied after application restart)"
        action.setText(title)

        action = self._usePlotWithOpengl
        action.setChecked(backend in ["opengl", "gl"])
        title = action.text().split(" (", 1)[0]
        if not action.isChecked():
            title += " (applied after application restart)"
        action.setText(title)

        # plot orientation

        menu = self._plotImageOrientationMenu
        if silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION == "downward":
            menu.setIcon(self._iconDownward)
        else:
            menu.setIcon(self._iconUpward)

        action = self._useYAxisOrientationDownward
        action.setChecked(
            silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION == "downward"
        )
        title = action.text().split(" (", 1)[0]
        if not action.isChecked():
            title += " (applied after application restart)"
        action.setText(title)

        action = self._useYAxisOrientationUpward
        action.setChecked(
            silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION != "downward"
        )
        title = action.text().split(" (", 1)[0]
        if not action.isChecked():
            title += " (applied after application restart)"
        action.setText(title)

        # mpl

        action = self._useMplTightLayout
        action.setChecked(silx.config._MPL_TIGHT_LAYOUT)

    def createMenus(self):
        fileMenu = self.menuBar().addMenu("&File")
        fileMenu.addAction(self._openAction)
        fileMenu.addMenu(self._openRecentMenu)
        fileMenu.addAction(self._closeAllAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self._exitAction)
        fileMenu.aboutToShow.connect(self.__updateFileMenu)

        optionMenu = self.menuBar().addMenu("&Options")
        optionMenu.addMenu(self._plotImageOrientationMenu)
        optionMenu.addMenu(self._plotBackendMenu)
        optionMenu.addAction(self._useMplTightLayout)
        optionMenu.aboutToShow.connect(self.__updateOptionMenu)

        viewMenu = self.menuBar().addMenu("&Views")
        viewMenu.addAction(self._displayCustomNxdataWindow)

        helpMenu = self.menuBar().addMenu("&Help")
        helpMenu.addAction(self._aboutAction)
        helpMenu.addAction(self._documentationAction)

        self.__errorButton = qt.QToolButton(self)
        self.__errorButton.setIcon(
            self.style().standardIcon(qt.QStyle.SP_MessageBoxWarning)
        )
        self.__errorButton.setToolTip(
            "An error occured!\nClick to display last error\nor check messages in the console"
        )
        self.__errorButton.setVisible(False)
        self.__errorButton.clicked.connect(self.__errorButtonClicked)
        self.menuBar().setCornerWidget(self.__errorButton)

    def open(self):
        dialog = self.createFileDialog()
        if self.__dialogState is None:
            currentDirectory = os.getcwd()
            dialog.setDirectory(currentDirectory)
        else:
            dialog.restoreState(self.__dialogState)

        result = dialog.exec()
        if not result:
            return

        self.__dialogState = dialog.saveState()

        filenames = dialog.selectedFiles()
        for filename in filenames:
            self.appendFile(filename)

    def closeAll(self):
        """Close all currently opened files"""
        model = self.__treeview.findHdf5TreeModel()
        model.clear()

    def createFileDialog(self):
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle("Open")
        dialog.setModal(True)

        # NOTE: hdf5plugin have to be loaded before
        extensions = {}
        for description, ext in silx.io.supported_extensions().items():
            extensions[description] = " ".join(sorted(list(ext)))

        # Add extensions supported by fabio
        extensions["NeXus layout from EDF files"] = "*.edf"
        extensions["NeXus layout from TIFF image files"] = "*.tif *.tiff"
        extensions["NeXus layout from CBF files"] = "*.cbf"
        extensions["NeXus layout from MarCCD image files"] = "*.mccd"

        all_supported_extensions = set()
        for name, exts in extensions.items():
            exts = exts.split(" ")
            all_supported_extensions.update(exts)
        all_supported_extensions = sorted(list(all_supported_extensions))

        filters = []
        filters.append("All supported files (%s)" % " ".join(all_supported_extensions))
        for name, extension in extensions.items():
            filters.append("%s (%s)" % (name, extension))
        filters.append("All files (*)")

        dialog.setNameFilters(filters)
        dialog.setFileMode(qt.QFileDialog.ExistingFiles)
        return dialog

    def about(self):
        from .About import About

        About.about(self, "Silx viewer")

    def showDocumentation(self):
        subpath = "index.html"
        url = projecturl.getDocumentationUrl(subpath)
        qt.QDesktopServices.openUrl(qt.QUrl(url))

    def setContentSorted(self, sort):
        """Set whether file content should be sorted or not.

        :param bool sort:
        """
        sort = bool(sort)
        if sort != self.isContentSorted():
            # save expanded nodes
            pathss = []
            root = qt.QModelIndex()
            model = self.__treeview.model()
            for i in range(model.rowCount(root)):
                index = model.index(i, 0, root)
                paths = self.__getPathFromExpandedNodes(self.__treeview, index)
                pathss.append(paths)

            self.__treeview.setModel(
                self.__treeModelSorted if sort else self.__treeModelSorted.sourceModel()
            )
            self._sortContentAction.setChecked(self.isContentSorted())

            # restore expanded nodes
            model = self.__treeview.model()
            for i in range(model.rowCount(root)):
                index = model.index(i, 0, root)
                paths = pathss.pop(0)
                self.__expandNodesFromPaths(self.__treeview, index, paths)

    def isContentSorted(self):
        """Returns whether the file content is sorted or not.

        :rtype: bool
        """
        return self.__treeview.model() is self.__treeModelSorted

    def __forcePlotImageDownward(self):
        silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION = "downward"

    def __forcePlotImageUpward(self):
        silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION = "upward"

    def __forceMatplotlibBackend(self):
        silx.config.DEFAULT_PLOT_BACKEND = "matplotlib"

    def __forceOpenglBackend(self):
        silx.config.DEFAULT_PLOT_BACKEND = "opengl", "matplotlib"

    def __forceMplTightLayout(self):
        silx.config._MPL_TIGHT_LAYOUT = self._useMplTightLayout.isChecked()

    def appendFile(self, filename):
        if self.__displayIt is None:
            # Store the file to display it (loading could be async)
            self.__displayIt = filename
        self.__treeview.findHdf5TreeModel().appendFile(filename)

    def displaySelectedData(self):
        """Called to update the dataviewer with the selected data."""
        selected = list(self.__treeview.selectedH5Nodes(ignoreBrokenLinks=False))
        if len(selected) == 1:
            # Update the viewer for a single selection
            data = selected[0]
            self.__dataPanel.setData(data)
        else:
            _logger.debug("Too many data selected")

    def displayData(self, data):
        """Called to update the dataviewer with a secific data."""
        self.__dataPanel.setData(data)

    def displaySelectedCustomData(self):
        selected = list(self.__customNxdata.selectedItems())
        if len(selected) == 1:
            # Update the viewer for a single selection
            item = selected[0]
            self.__dataPanel.setCustomDataItem(item)
        else:
            _logger.debug("Too many items selected")

    def __customNxdataRemoved(self, item):
        if self.__dataPanel.getCustomNxdataItem() is item:
            self.__dataPanel.setCustomDataItem(None)

    def __customNxdataUpdated(self, item):
        if self.__dataPanel.getCustomNxdataItem() is item:
            self.__dataPanel.setCustomDataItem(item)

    def __makeSureCustomNxDataWindowIsVisible(self):
        if not self.__customNxdataWindow.isVisible():
            self.__customNxdataWindow.setVisible(True)
            self._displayCustomNxdataWindow.setChecked(True)

    def useAsNewCustomSignal(self, h5dataset):
        self.__makeSureCustomNxDataWindowIsVisible()
        model = self.__customNxdata.model()
        model.createFromSignal(h5dataset)

    def useAsNewCustomNxdata(self, h5nxdata):
        self.__makeSureCustomNxDataWindowIsVisible()
        model = self.__customNxdata.model()
        model.createFromNxdata(h5nxdata)

    def customContextMenu(self, event):
        """Called to populate the context menu

        :param silx.gui.hdf5.Hdf5ContextMenuEvent event: Event
            containing expected information to populate the context menu
        """
        selectedObjects = event.source().selectedH5Nodes(ignoreBrokenLinks=False)
        menu = event.menu()

        if not menu.isEmpty():
            menu.addSeparator()

        for obj in selectedObjects:
            h5 = obj.h5py_object

            name = obj.name
            if name.startswith("/"):
                name = name[1:]
            if name == "":
                name = "the root"

            action = qt.QAction("Show %s" % name, event.source())
            action.triggered.connect(lambda: self.displayData(h5))
            menu.addAction(action)

            if silx.io.is_dataset(h5):
                action = qt.QAction("Use as a new custom signal", event.source())
                action.triggered.connect(lambda: self.useAsNewCustomSignal(h5))
                menu.addAction(action)

            if silx.io.is_group(h5) and silx.io.nxdata.is_valid_nxdata(h5):
                action = qt.QAction("Use as a new custom NXdata", event.source())
                action.triggered.connect(lambda: self.useAsNewCustomNxdata(h5))
                menu.addAction(action)

            if silx.io.is_file(h5):
                action = qt.QAction("Close %s" % obj.local_filename, event.source())
                action.triggered.connect(
                    lambda: self.__treeview.findHdf5TreeModel().removeH5pyObject(h5)
                )
                menu.addAction(action)
                action = qt.QAction(
                    "Synchronize %s" % obj.local_filename, event.source()
                )
                action.triggered.connect(lambda: self.__synchronizeH5pyObject(h5))
                menu.addAction(action)

    def __errorButtonClicked(self):
        button = qt.QMessageBox.warning(
            self,
            "Error",
            self.getError(),
            qt.QMessageBox.Reset | qt.QMessageBox.Close,
            qt.QMessageBox.Close,
        )
        if button == qt.QMessageBox.Reset:
            self.setError("")

    def getError(self) -> str:
        """Returns error information string"""
        return self.__error

    def setError(self, error: str):
        """Set error information string"""
        if error == self.__error:
            return

        self.__error = error
        self.__errorButton.setVisible(error != "")

    def setErrorFromException(
        self, type_: type[BaseException], value: BaseException, trace: TracebackType
    ):
        """Set information about the last exception that occured"""
        formattedTrace = "\n".join(traceback.format_tb(trace))
        self.setError(f"{type_.__name__}:\n{value}\n\n{formattedTrace}")
