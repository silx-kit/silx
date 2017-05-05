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
import argparse
import logging


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
        self.setWindowTitle("Silx HDF5 widget example")

        self.__asyncload = False
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

        self.__treeview.activated.connect(self.displayData)

        self.__treeview.addContextMenuCallback(self.customContextMenu)
        # lambda function will never be called cause we store it as weakref
        self.__treeview.addContextMenuCallback(lambda event: None)
        # you have to store it first
        self.__store_lambda = lambda event: self.closeAndSyncCustomContextMenu(event)
        self.__treeview.addContextMenuCallback(self.__store_lambda)

    def appendFile(self, filename):
        self.__treeview.findHdf5TreeModel().appendFile(filename)

    def displayData(self):
        """Called to update the dataviewer with the selected data.
        """
        selected = list(self.__treeview.selectedH5Nodes())
        if len(selected) == 1:
            # Update the viewer for a single selection
            data = selected[0]
            # data is a hdf5.H5Node object
            # data.h5py_object is a Group/Dataset object (from h5py, spech5, fabioh5)
            # The dataviewer can display both
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
