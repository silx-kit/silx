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
__date__ = "25/04/2018"


import os
import collections

from silx.gui import qt
from .ApplicationContext import ApplicationContext


class Viewer(qt.QMainWindow):
    """
    This window allows to browse a data file like images or HDF5 and it's
    content.
    """

    def __init__(self):
        """
        :param files_: List of HDF5 or Spec files (pathes or
            :class:`silx.io.spech5.SpecH5` or :class:`h5py.File`
            instances)
        """
        # Import it here to be sure to use the right logging level
        import silx.gui.hdf5
        from silx.gui.data.DataViewerFrame import DataViewerFrame

        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Silx viewer")

        self.__asyncload = False
        self.__dialogState = None
        self.__treeview = silx.gui.hdf5.Hdf5TreeView(self)
        """Silx HDF5 TreeView"""

        self.__dataViewer = DataViewerFrame(self)
        context = ApplicationContext(self)
        self.__dataViewer.setGlobalHooks(context)
        vSpliter = qt.QSplitter(qt.Qt.Vertical)
        vSpliter.addWidget(self.__dataViewer)
        vSpliter.setSizes([10, 0])

        spliter = qt.QSplitter(self)
        spliter.addWidget(self.__treeview)
        spliter.addWidget(vSpliter)
        spliter.setStretchFactor(1, 1)

        main_panel = qt.QWidget(self)
        layout = qt.QVBoxLayout()
        layout.addWidget(spliter)
        layout.setStretchFactor(spliter, 1)
        main_panel.setLayout(layout)

        self.setCentralWidget(main_panel)

        model = self.__treeview.selectionModel()
        model.selectionChanged.connect(self.displayData)
        self.__treeview.addContextMenuCallback(self.closeAndSyncCustomContextMenu)

        treeModel = self.__treeview.findHdf5TreeModel()
        columns = list(treeModel.COLUMN_IDS)
        columns.remove(treeModel.DESCRIPTION_COLUMN)
        columns.remove(treeModel.NODE_COLUMN)
        self.__treeview.header().setSections(columns)

        self.createActions()
        self.createMenus()

    def createActions(self):
        action = qt.QAction("E&xit", self)
        action.setShortcuts(qt.QKeySequence.Quit)
        action.setStatusTip("Exit the application")
        action.triggered.connect(self.close)
        self._exitAction = action

        action = qt.QAction("&Open", self)
        action.setStatusTip("Open a file")
        action.triggered.connect(self.open)
        self._openAction = action

        action = qt.QAction("&About", self)
        action.setStatusTip("Show the application's About box")
        action.triggered.connect(self.about)
        self._aboutAction = action

    def createMenus(self):
        fileMenu = self.menuBar().addMenu("&File")
        fileMenu.addAction(self._openAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self._exitAction)
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
        import silx.io
        extensions = collections.OrderedDict()
        for description, ext in silx.io.supported_extensions().items():
            extensions[description] = " ".join(sorted(list(ext)))

        # NOTE: hdf5plugin have to be loaded before
        import fabio
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

    def appendFile(self, filename):
        self.__treeview.findHdf5TreeModel().appendFile(filename)

    def displayData(self):
        """Called to update the dataviewer with the selected data.
        """
        selected = list(self.__treeview.selectedH5Nodes(ignoreBrokenLinks=False))
        if len(selected) == 1:
            # Update the viewer for a single selection
            data = selected[0]
            self.__dataViewer.setData(data)

    def useAsyncLoad(self, useAsync):
        self.__asyncload = useAsync

    def closeAndSyncCustomContextMenu(self, event):
        """Called to populate the context menu

        :param silx.gui.hdf5.Hdf5ContextMenuEvent event: Event
            containing expected information to populate the context menu
        """
        selectedObjects = event.source().selectedH5Nodes(ignoreBrokenLinks=False)
        menu = event.menu()

        if not menu.isEmpty():
            menu.addSeparator()

        # Import it here to be sure to use the right logging level
        import h5py
        for obj in selectedObjects:
            if obj.ntype is h5py.File:
                action = qt.QAction("Remove %s" % obj.local_filename, event.source())
                action.triggered.connect(lambda: self.__treeview.findHdf5TreeModel().removeH5pyObject(obj.h5py_object))
                menu.addAction(action)
                action = qt.QAction("Synchronize %s" % obj.local_filename, event.source())
                action.triggered.connect(lambda: self.__treeview.findHdf5TreeModel().synchronizeH5pyObject(obj.h5py_object))
                menu.addAction(action)
