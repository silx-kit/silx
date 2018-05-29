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
"""Browse a data file with a GUI"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "29/05/2018"


import os
import collections
import logging
import functools

import silx.io.nxdata
from silx.gui import qt
from silx.gui import icons
from .ApplicationContext import ApplicationContext
from .CustomNxdataWidget import CustomNxdataWidget
from .CustomNxdataWidget import CustomNxDataToolBar
from . import utils


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
        # Import it here to be sure to use the right logging level
        import silx.gui.hdf5
        from silx.gui.data.DataViewerFrame import DataViewerFrame

        qt.QMainWindow.__init__(self, parent)
        self.setWindowTitle("Silx viewer")

        self.__context = ApplicationContext(self, settings)
        self.__context.restoreLibrarySettings()

        self.__dialogState = None
        self.__customNxDataItem = None
        self.__treeview = silx.gui.hdf5.Hdf5TreeView(self)
        self.__treeview.setExpandsOnDoubleClick(False)
        """Silx HDF5 TreeView"""

        rightPanel = qt.QSplitter(self)
        rightPanel.setOrientation(qt.Qt.Vertical)
        self.__splitter2 = rightPanel

        # Custom the model to be able to manage the life cycle of the files
        treeModel = silx.gui.hdf5.Hdf5TreeModel(self.__treeview, ownFiles=False)
        treeModel.sigH5pyObjectLoaded.connect(self.__h5FileLoaded)
        treeModel.sigH5pyObjectRemoved.connect(self.__h5FileRemoved)
        treeModel.sigH5pyObjectSynchronized.connect(self.__h5FileSynchonized)
        treeModel.setDatasetDragEnabled(True)
        treeModel2 = silx.gui.hdf5.NexusSortFilterProxyModel(self.__treeview)
        treeModel2.setSourceModel(treeModel)
        self.__treeview.setModel(treeModel2)
        rightPanel.addWidget(self.__treeview)

        self.__customNxdata = CustomNxdataWidget(self)
        self.__customNxdata.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        # optimise the rendering
        self.__customNxdata.setUniformRowHeights(True)
        self.__customNxdata.setIconSize(qt.QSize(16, 16))
        self.__customNxdata.setExpandsOnDoubleClick(False)

        self.__customNxdataWindow = self.__createCustomNxdataWindow(self.__customNxdata)
        self.__customNxdataWindow.setVisible(False)
        rightPanel.addWidget(self.__customNxdataWindow)

        self.__dataViewer = DataViewerFrame(self)
        self.__dataViewer.setGlobalHooks(self.__context)

        rightPanel.setStretchFactor(1, 1)
        rightPanel.setCollapsible(0, False)
        rightPanel.setCollapsible(1, False)

        spliter = qt.QSplitter(self)
        spliter.addWidget(rightPanel)
        spliter.addWidget(self.__dataViewer)
        spliter.setStretchFactor(1, 1)
        self.__splitter = spliter

        main_panel = qt.QWidget(self)
        layout = qt.QVBoxLayout()
        layout.addWidget(spliter)
        layout.setStretchFactor(spliter, 1)
        main_panel.setLayout(layout)

        self.setCentralWidget(main_panel)

        self.__treeview.activated.connect(self.displaySelectedData)
        self.__customNxdata.activated.connect(self.displayCustomData)
        self.__customNxdata.sigNxdataItemRemoved.connect(self.__customNxdataRemoved)
        self.__customNxdata.sigNxdataItemUpdated.connect(self.__customNxdataUpdated)
        self.__treeview.addContextMenuCallback(self.customContextMenu)

        treeModel = self.__treeview.findHdf5TreeModel()
        columns = list(treeModel.COLUMN_IDS)
        columns.remove(treeModel.DESCRIPTION_COLUMN)
        columns.remove(treeModel.NODE_COLUMN)
        self.__treeview.header().setSections(columns)

        self._iconUpward = icons.getQIcon('plot-yup')
        self._iconDownward = icons.getQIcon('plot-ydown')

        self.createActions()
        self.createMenus()
        self.__context.restoreSettings()

    def __createCustomNxdataWindow(self, customNxdataWidget):
        toolbar = CustomNxDataToolBar(self)
        toolbar.setCustomNxDataWidget(customNxdataWidget)
        toolbar.setIconSize(qt.QSize(16, 16))
        toolbar.setStyleSheet("QToolBar { border: 0px }")

        widget = qt.QWidget()
        layout = qt.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(toolbar)
        layout.addWidget(customNxdataWidget)
        return widget

    def __h5FileLoaded(self, loadedH5):
        self.__context.pushRecentFile(loadedH5.file.filename)

    def __h5FileRemoved(self, removedH5):
        data = self.__dataViewer.data()
        if data is not None:
            if data.file is not None:
                # That's an approximation, IS can't be used as h5py generates
                # To objects for each requests to a node
                if data.file.filename == removedH5.file.filename:
                    self.__dataViewer.setData(None)
        self.__customNxdata.removeDatasetsFrom(removedH5)
        removedH5.close()

    def __h5FileSynchonized(self, removedH5, loadedH5):
        data = self.__dataViewer.data()
        if data is not None:
            if data.file is not None:
                if data.file.filename == removedH5.file.filename:
                    # Try to synchonize the viewed data
                    try:
                        # TODO: It have to update the data without changing the
                        # view which is not so easy
                        newData = loadedH5[data.name]
                        self.__dataViewer.setData(newData)
                    except Exception:
                        _logger.debug("Backtrace", exc_info=True)
        self.__customNxdata.replaceDatasetsFrom(removedH5, loadedH5)
        removedH5.close()

    def closeEvent(self, event):
        self.__context.saveSettings()

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
        try:
            if not isinstance(isFullScreen, bool):
                isFullScreen = utils.stringToBool(isFullScreen)
        except ValueError:
            isFullScreen = False
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
        try:
            if not isinstance(isVisible, bool):
                isVisible = utils.stringToBool(isVisible)
        except ValueError:
            isVisible = False
        self.__customNxdataWindow.setVisible(isVisible)
        self._displayCustomNxdataWindow.setChecked(isVisible)

        settings.endGroup()

        if not pos.isNull():
            self.move(pos)
        if not size.isNull():
            self.resize(size)
        if isFullScreen:
            self.showFullScreen()

    def createActions(self):
        action = qt.QAction("E&xit", self)
        action.setShortcuts(qt.QKeySequence.Quit)
        action.setStatusTip("Exit the application")
        action.triggered.connect(self.close)
        self._exitAction = action

        action = qt.QAction("&Open...", self)
        action.setStatusTip("Open a file")
        action.triggered.connect(self.open)
        self._openAction = action

        action = qt.QAction("Open Recent", self)
        action.setStatusTip("Open a recently openned file")
        action.triggered.connect(self.open)
        self._openRecentAction = action

        action = qt.QAction("&About", self)
        action.setStatusTip("Show the application's About box")
        action.triggered.connect(self.about)
        self._aboutAction = action

        # Plot backend

        action = qt.QAction("Plot rendering backend", self)
        action.setStatusTip("Select plot rendering backend")
        self._plotBackendSelection = action

        menu = qt.QMenu()
        action.setMenu(menu)
        group = qt.QActionGroup(self)
        group.setExclusive(True)

        action = qt.QAction("matplotlib", self)
        action.setStatusTip("Plot will be rendered using matplotlib")
        action.setCheckable(True)
        action.triggered.connect(self.__forceMatplotlibBackend)
        group.addAction(action)
        menu.addAction(action)
        self._usePlotWithMatplotlib = action

        action = qt.QAction("OpenGL", self)
        action.setStatusTip("Plot will be rendered using OpenGL")
        action.setCheckable(True)
        action.triggered.connect(self.__forceOpenglBackend)
        group.addAction(action)
        menu.addAction(action)
        self._usePlotWithOpengl = action

        # Plot image orientation

        action = qt.QAction("Default plot image y-axis orientation", self)
        action.setStatusTip("Select the default y-axis orientation used by plot displaying images")
        self._plotImageOrientation = action

        menu = qt.QMenu()
        action.setMenu(menu)
        group = qt.QActionGroup(self)
        group.setExclusive(True)

        action = qt.QAction("Downward, origin on top", self)
        action.setIcon(self._iconDownward)
        action.setStatusTip("Plot images will use a downward Y-axis orientation")
        action.setCheckable(True)
        action.triggered.connect(self.__forcePlotImageDownward)
        group.addAction(action)
        menu.addAction(action)
        self._useYAxisOrientationDownward = action

        action = qt.QAction("Upward, origin on bottom", self)
        action.setIcon(self._iconUpward)
        action.setStatusTip("Plot images will use a upward Y-axis orientation")
        action.setCheckable(True)
        action.triggered.connect(self.__forcePlotImageUpward)
        group.addAction(action)
        menu.addAction(action)
        self._useYAxisOrientationUpward = action

        # Windows

        action = qt.QAction("Show custom NXdata selector", self)
        action.setStatusTip("Show a widget which allow to create plot by selecting data and axes")
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_F5))
        action.toggled.connect(self.__toggleCustomNxdataWindow)
        self._displayCustomNxdataWindow = action

    def __toggleCustomNxdataWindow(self):
        isVisible = self._displayCustomNxdataWindow.isChecked()
        self.__customNxdataWindow.setVisible(isVisible)

    def __updateFileMenu(self):
        files = self.__context.getRecentFiles()
        self._openRecentAction.setEnabled(len(files) != 0)
        menu = None
        if len(files) != 0:
            menu = qt.QMenu()
            for filePath in files:
                baseName = os.path.basename(filePath)
                action = qt.QAction(baseName, self)
                action.setToolTip(filePath)
                action.triggered.connect(functools.partial(self.__openRecentFile, filePath))
                menu.addAction(action)
            menu.addSeparator()
            baseName = os.path.basename(filePath)
            action = qt.QAction("Clear history", self)
            action.setToolTip("Clear the history of the recent files")
            action.triggered.connect(self.__clearRecentFile)
            menu.addAction(action)
        self._openRecentAction.setMenu(menu)

    def __clearRecentFile(self):
        self.__context.clearRencentFiles()

    def __openRecentFile(self, filePath):
        self.appendFile(filePath)

    def __updateOptionMenu(self):
        """Update the state of the checked options as it is based on global
        environment values."""

        # plot backend

        action = self._plotBackendSelection
        title = action.text().split(": ", 1)[0]
        action.setText("%s: %s" % (title, silx.config.DEFAULT_PLOT_BACKEND))

        action = self._usePlotWithMatplotlib
        action.setChecked(silx.config.DEFAULT_PLOT_BACKEND in ["matplotlib", "mpl"])
        title = action.text().split(" (", 1)[0]
        if not action.isChecked():
            title += " (applied after application restart)"
        action.setText(title)

        action = self._usePlotWithOpengl
        action.setChecked(silx.config.DEFAULT_PLOT_BACKEND in ["opengl", "gl"])
        title = action.text().split(" (", 1)[0]
        if not action.isChecked():
            title += " (applied after application restart)"
        action.setText(title)

        # plot orientation

        action = self._plotImageOrientation
        if silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION == "downward":
            action.setIcon(self._iconDownward)
        else:
            action.setIcon(self._iconUpward)
        action.setIconVisibleInMenu(True)

        action = self._useYAxisOrientationDownward
        action.setChecked(silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION == "downward")
        title = action.text().split(" (", 1)[0]
        if not action.isChecked():
            title += " (applied after application restart)"
        action.setText(title)

        action = self._useYAxisOrientationUpward
        action.setChecked(silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION != "downward")
        title = action.text().split(" (", 1)[0]
        if not action.isChecked():
            title += " (applied after application restart)"
        action.setText(title)

    def createMenus(self):
        fileMenu = self.menuBar().addMenu("&File")
        fileMenu.addAction(self._openAction)
        fileMenu.addAction(self._openRecentAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self._exitAction)
        fileMenu.aboutToShow.connect(self.__updateFileMenu)

        optionMenu = self.menuBar().addMenu("&Options")
        optionMenu.addAction(self._plotImageOrientation)
        optionMenu.addAction(self._plotBackendSelection)
        optionMenu.aboutToShow.connect(self.__updateOptionMenu)

        windowMenu = self.menuBar().addMenu("&Windows")
        windowMenu.addAction(self._displayCustomNxdataWindow)

        helpMenu = self.menuBar().addMenu("&Help")
        helpMenu.addAction(self._aboutAction)

    def open(self):
        dialog = self.createFileDialog()
        if self.__dialogState is None:
            currentDirectory = os.getcwd()
            dialog.setDirectory(currentDirectory)
        else:
            dialog.restoreState(self.__dialogState)

        result = dialog.exec_()
        if not result:
            return

        self.__dialogState = dialog.saveState()

        filenames = dialog.selectedFiles()
        for filename in filenames:
            self.appendFile(filename)

    def createFileDialog(self):
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle("Open")
        dialog.setModal(True)

        # NOTE: hdf5plugin have to be loaded before
        extensions = collections.OrderedDict()
        for description, ext in silx.io.supported_extensions().items():
            extensions[description] = " ".join(sorted(list(ext)))

        try:
            # NOTE: hdf5plugin have to be loaded before
            import fabio
        except Exception:
            _logger.debug("Backtrace while loading fabio", exc_info=True)
            fabio = None

        if fabio is not None:
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

    def __forcePlotImageDownward(self):
        silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION = "downward"

    def __forcePlotImageUpward(self):
        silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION = "upward"

    def __forceMatplotlibBackend(self):
        silx.config.DEFAULT_PLOT_BACKEND = "matplotlib"

    def __forceOpenglBackend(self):
        silx.config.DEFAULT_PLOT_BACKEND = "opengl"

    def appendFile(self, filename):
        self.__treeview.findHdf5TreeModel().appendFile(filename)

    def displaySelectedData(self):
        """Called to update the dataviewer with the selected data.
        """
        selected = list(self.__treeview.selectedH5Nodes(ignoreBrokenLinks=False))
        if len(selected) == 1:
            # Update the viewer for a single selection
            data = selected[0]
            self.displayData(data)
        else:
            _logger.debug("Too much data selected")

    def displayData(self, data):
        """Called to update the dataviewer with the data.
        """
        self.__customNxDataItem = None
        self.__dataViewer.setData(data)

    def displayCustomData(self):
        selected = list(self.__customNxdata.selectedItems())
        if len(selected) == 1:
            # Update the viewer for a single selection
            item = selected[0]
            self.__customNxDataItem = item
            data = item.getVirtualGroup()
            self.__dataViewer.setData(data)

    def __customNxdataRemoved(self, item):
        if self.__customNxDataItem is item:
            self.__customNxDataId = None
            self.__dataViewer.setData(None)

    def __customNxdataUpdated(self, item):
        if self.__customNxDataItem is item:
            data = item.getVirtualGroup()
            self.__dataViewer.setData(data)

    def __makeSureCustomNxDataWindowIsVisible(self):
        if not self.__customNxdataWindow.isVisible():
            self.__customNxdataWindow.setVisible(True)
            self._displayCustomNxdataWindow.setChecked(True)

    def useAsNewCustomSignal(self, h5dataset):
        self.__makeSureCustomNxDataWindowIsVisible()
        self.__customNxdata.createFromSignal(h5dataset)

    def useAsNewCustomNxdata(self, h5nxdata):
        self.__makeSureCustomNxDataWindowIsVisible()
        self.__customNxdata.createFromNxdata(h5nxdata)

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

            action = qt.QAction("Show %s" % obj.name, event.source())
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
                action = qt.QAction("Remove %s" % obj.local_filename, event.source())
                action.triggered.connect(lambda: self.__treeview.findHdf5TreeModel().removeH5pyObject(h5))
                menu.addAction(action)
                action = qt.QAction("Synchronize %s" % obj.local_filename, event.source())
                action.triggered.connect(lambda: self.__treeview.findHdf5TreeModel().synchronizeH5pyObject(h5))
                menu.addAction(action)
