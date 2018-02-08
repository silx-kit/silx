# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
# ###########################################################################*/
"""
This module contains an :class:`ImageFileDialog`.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "08/02/2018"

import logging
from silx.gui import qt
from silx.gui.hdf5.Hdf5Formatter import Hdf5Formatter
import silx.io
from .AbstractDataFileDialog import AbstractDataFileDialog
try:
    import fabio
except ImportError:
    fabio = None


_logger = logging.getLogger(__name__)


class _DataPreview(qt.QWidget):
    """Provide a preview of the selected image"""

    def __init__(self, parent=None):
        super(_DataPreview, self).__init__(parent)

        self.__formatter = Hdf5Formatter(self)
        self.__data = None
        self.__info = qt.QTableView(self)
        self.__model = qt.QStandardItemModel(self)
        self.__info.setModel(self.__model)
        self.__info.horizontalHeader().hide()
        self.__info.horizontalHeader().setStretchLastSection(True)
        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.__info)
        self.setLayout(layout)

    def colormap(self):
        return None

    def setColormap(self, colormap):
        # Ignored
        pass

    def sizeHint(self):
        return qt.QSize(200, 200)

    def setData(self, data, fromDataSelector=False):
        self.__info.setEnabled(data is not None)
        if data is None:
            self.__model.clear()
        else:
            self.__model.clear()

            if silx.io.is_dataset(data):
                kind = "Dataset"
            elif silx.io.is_group(data):
                kind = "Group"
            elif silx.io.is_file(data):
                kind = "File"
            else:
                kind = "Unknown"

            headers = []

            basename = data.name.split("/")[-1]
            if basename == "":
                basename = "/"
            headers.append("Basename")
            self.__model.appendRow([qt.QStandardItem(basename)])
            headers.append("Kind")
            self.__model.appendRow([qt.QStandardItem(kind)])
            if hasattr(data, "dtype"):
                headers.append("Type")
                text = self.__formatter.humanReadableType(data)
                self.__model.appendRow([qt.QStandardItem(text)])
            if hasattr(data, "shape"):
                headers.append("Shape")
                text = self.__formatter.humanReadableShape(data)
                self.__model.appendRow([qt.QStandardItem(text)])
            self.__model.setVerticalHeaderLabels(headers)
        self.__data = data

    def __imageItem(self):
        image = self.__plot.getImage("data")
        return image

    def data(self):
        if self.__data is not None:
            if hasattr(self.__data, "name"):
                # in case of HDF5
                if self.__data.name is None:
                    # The dataset was closed
                    self.__data = None
        return self.__data

    def clear(self):
        self.__data = None
        self.__info.setText("")


class DataFileDialog(AbstractDataFileDialog):
    """The DataFileDialog class provides a dialog that allow users to select
    any datasets or groups from an HDF5-like file.

    The DataFileDialog class enables a user to traverse the file system in
    order to select one file. Then to traverse the file to select an HDF5 node.

    The selected data is any kind of group or dataset.

    Using an DataFileDialog can be done like that.

    .. code-block:: python

        dialog = DataFileDialog()
        result = dialog.exec_()
        if result:
            print("Selection:")
            print(dialog.selectedFile())
            print(dialog.selectedData())
            print(dialog.selectedPath())
        else:
            print("Nothing selected")

        # Make sure loaded files are closed properly
        dialog.clear()

    If you want to manage the life cycle of the object on your own, it is
    better to avoid to use `selectedData` but to load again the data with the
    our `io` API.

    .. code-block:: python

        dialog = DataFileDialog()
        result = dialog.exec_()
        if not result:
            return
        url = dialog.selectedPath()
        dialog.clear()

        # here you can manage the way you want
        with silx.io.open(url):
            pass
    """

    def _createPreviewWidget(self, parent):
        previewWidget = _DataPreview(parent)
        previewWidget.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        return previewWidget

    def _createSelectorWidget(self, parent):
        # There is no selector
        return None

    def _createPreviewToolbar(self, parent, dataPreviewWidget, dataSelectorWidget):
        # There is no toolbar
        return None

    def _isDataSupportable(self, data):
        """Check if the selected data can be supported at one point.

        If true, the data selector will be checked and it will update the data
        preview. Else the selecting is disabled.

        :rtype: bool
        """
        # Everything is supported
        return True

    def _isFabioFilesSupported(self):
        # Everything is supported
        return False

    def _isDataSupported(self, data):
        """Check if the data can be returned by the dialog.

        If true, this data can be returned by the dialog and the open button
        will be enabled. If false the button will be disabled.

        :rtype: bool
        """
        return True

    def _displayedDataInfo(self, dataBeforeSelection, dataAfterSelection):
        """Returns the text displayed under the data preview.

        This zone is used to display error in case or problem of data selection
        or problems with IO.

        :param numpy.ndarray dataAfterSelection: Data as it is after the
            selection widget (basically the data from the preview widget)
        :param numpy.ndarray dataAfterSelection: Data as it is before the
            selection widget (basically the data from the browsing widget)
        :rtype: bool
        """
        return u""
