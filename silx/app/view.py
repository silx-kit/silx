# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
__date__ = "12/04/2017"

import sys
import os
import argparse
import logging
import collections


logging.basicConfig()
_logger = logging.getLogger(__name__)
"""Module logger"""

try:
    # it should be loaded before h5py
    import hdf5plugin  # noqa
except ImportError:
    hdf5plugin = None

try:
    import h5py
    import silx.gui.hdf5
except ImportError:
    h5py = None

try:
    import fabio
except ImportError:
    fabio = None

from silx.gui import qt
from silx.gui.data.DataViewerFrame import DataViewerFrame


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
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Silx viewer")

        self.__asyncload = False
        self.__dialogState = None
        self.__treeview = silx.gui.hdf5.Hdf5TreeView(self)
        """Silx HDF5 TreeView"""

        self.__dataViewer = DataViewerFrame(self)
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

        self.__treeview.selectionModel().selectionChanged.connect(self.displayData)

        self.__treeview.addContextMenuCallback(self.customContextMenu)
        # lambda function will never be called cause we store it as weakref
        self.__treeview.addContextMenuCallback(lambda event: None)
        # you have to store it first
        self.__store_lambda = lambda event: self.closeAndSyncCustomContextMenu(event)
        self.__treeview.addContextMenuCallback(self.__store_lambda)

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

        extensions = collections.OrderedDict()
        # expect h5py
        extensions["HDF5 files"] = "*.h5"
        # no dependancy
        extensions["Spec files"] = "*.dat *.spec *.mca"
        # expect fabio
        extensions["EDF files"] = "*.edf"
        extensions["TIFF image files"] = "*.tif *.tiff"
        extensions["NumPy binary files"] = "*.npy"
        extensions["CBF files"] = "*.cbf"
        extensions["MarCCD image files"] = "*.mccd"

        filters = []
        filters.append("All supported files (%s)" % " ".join(extensions.values()))
        for name, extension in extensions.items():
            filters.append("%s (%s)" % (name, extension))
        filters.append("All files (*)")

        dialog.setNameFilters(filters)
        dialog.setFileMode(qt.QFileDialog.ExistingFiles)
        return dialog

    def about(self):
        import silx._version
        message = """<p align="center"><b>Silx viewer</b>
        <br />
        <br />{silx_version}
        <br />
        <br /><a href="{project_url}">Upstream project on GitHub</a>
        </p>
        <p align="left">
        <dl>
        <dt><b>Silx version</b></dt><dd>{silx_version}</dd>
        <dt><b>Qt version</b></dt><dd>{qt_version}</dd>
        <dt><b>Qt binding</b></dt><dd>{qt_binding}</dd>
        <dt><b>Python version</b></dt><dd>{python_version}</dd>
        <dt><b>Optional libraries</b></dt><dd>{optional_lib}</dd>
        </dl>
        </p>
        <p>
        Copyright (C) <a href="{esrf_url}">European Synchrotron Radiation Facility</a>
        </p>
        """
        def format_optional_lib(name, isAvailable):
            if isAvailable:
                template = '<b>%s</b> is <font color="green">installed</font>'
            else:
                template = '<b>%s</b> is <font color="red">not installed</font>'
            return template % name

        optional_lib = []
        optional_lib.append(format_optional_lib("FabIO", fabio is not None))
        optional_lib.append(format_optional_lib("H5py", h5py is not None))
        optional_lib.append(format_optional_lib("hdf5plugin", hdf5plugin is not None))

        info = dict(
            esrf_url="http://www.esrf.eu",
            project_url="https://github.com/silx-kit/silx",
            silx_version=silx._version.version,
            qt_binding=qt.BINDING,
            qt_version=qt.qVersion(),
            python_version=sys.version.replace("\n", "<br />"),
            optional_lib="<br />".join(optional_lib)
        )
        qt.QMessageBox.about(self, "About Menu", message.format(**info))

    def appendFile(self, filename):
        self.__treeview.findHdf5TreeModel().appendFile(filename)

    def displayData(self):
        """Called to update the dataviewer with the selected data.
        """
        selected = list(self.__treeview.selectedH5Nodes())
        if len(selected) == 1:
            # Update the viewer for a single selection
            data = selected[0]
            self.__dataViewer.setData(data)

    def useAsyncLoad(self, useAsync):
        self.__asyncload = useAsync

    def customContextMenu(self, event):
        """Called to populate the context menu

        :param silx.gui.hdf5.Hdf5ContextMenuEvent event: Event
            containing expected information to populate the context menu
        """
        selectedObjects = event.source().selectedH5Nodes()
        menu = event.menu()

        hasDataset = False
        for obj in selectedObjects:
            if obj.ntype is h5py.Dataset:
                hasDataset = True
                break

        if len(menu.children()):
            menu.addSeparator()

        if hasDataset:
            action = qt.QAction("Do something on the datasets", event.source())
            menu.addAction(action)

    def closeAndSyncCustomContextMenu(self, event):
        """Called to populate the context menu

        :param silx.gui.hdf5.Hdf5ContextMenuEvent event: Event
            containing expected information to populate the context menu
        """
        selectedObjects = event.source().selectedH5Nodes()
        menu = event.menu()

        if len(menu.children()):
            menu.addSeparator()

        for obj in selectedObjects:
            if obj.ntype is h5py.File:
                action = qt.QAction("Remove %s" % obj.local_filename, event.source())
                action.triggered.connect(lambda: self.__treeview.findHdf5TreeModel().removeH5pyObject(obj.h5py_object))
                menu.addAction(action)
                action = qt.QAction("Synchronize %s" % obj.local_filename, event.source())
                action.triggered.connect(lambda: self.__treeview.findHdf5TreeModel().synchronizeH5pyObject(obj.h5py_object))
                menu.addAction(action)


def main(argv):
    """
    Main function to launch the viewer as an application

    :param argv: Command line arguments
    :returns: exit status
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'files',
        type=argparse.FileType('rb'),
        nargs=argparse.ZERO_OR_MORE,
        help='Data file to show (h5 file, edf files, spec files)')

    options = parser.parse_args(argv[1:])

    if h5py is None:
        message = "Module 'h5py' is not installed but is mandatory."\
            + " You can install it using \"pip install h5py\"."
        _logger.error(message)
        return -1

    if hdf5plugin is None:
        message = "Module 'hdf5plugin' is not installed. It supports some hdf5"\
            + " compressions. You can install it using \"pip install hdf5plugin\"."
        _logger.warning(message)

    app = qt.QApplication([])
    sys.excepthook = qt.exceptionHandler
    window = Viewer()
    window.resize(qt.QSize(640, 480))

    for f in options.files:
        filename = f.name
        f.close()
        window.appendFile(filename)

    window.show()
    result = app.exec_()
    # remove ending warnings relative to QTimer
    app.deleteLater()
    return result
