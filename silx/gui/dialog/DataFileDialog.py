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
This module contains an :class:`DataFileDialog`.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "14/02/2018"

import logging
from silx.gui import qt
from silx.gui.hdf5.Hdf5Formatter import Hdf5Formatter
import silx.io
from .AbstractDataFileDialog import AbstractDataFileDialog
from silx.third_party import enum
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
            if hasattr(data, "attrs") and "NX_class" in data.attrs:
                headers.append("NX_class")
                value = data.attrs["NX_class"]
                formatter = self.__formatter.textFormatter()
                old = formatter.useQuoteForText()
                formatter.setUseQuoteForText(False)
                text = self.__formatter.textFormatter().toString(value)
                formatter.setUseQuoteForText(old)
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
    """The `DataFileDialog` class provides a dialog that allow users to select
    any datasets or groups from an HDF5-like file.

    The `DataFileDialog` class enables a user to traverse the file system in
    order to select an HDF5-like file. Then to traverse the file to select an
    HDF5 node.

    .. image:: img/datafiledialog.png

    The selected data is any kind of group or dataset. It can be restricted
    to only existing datasets or only existing groups using
    :meth:`setFilterMode`. A callback can be defining using
    :meth:`setFilterCallback` to filter even more data which can be returned.

    Filtering data which can be returned by a `DataFileDialog` can be done like
    that:

    .. code-block:: python

        # Force to return only a dataset
        dialog = DataFileDialog()
        dialog.setFilterMode(DataFileDialog.FilterMode.ExistingDataset)

    .. code-block:: python

        def customFilter(obj):
            if "NX_class" in obj.attrs:
                return obj.attrs["NX_class"] in [b"NXentry", u"NXentry"]
            return False

        # Force to return an NX entry
        dialog = DataFileDialog()
        # 1st, filter out everything which is not a group
        dialog.setFilterMode(DataFileDialog.FilterMode.ExistingGroup)
        # 2nd, check what NX_class is an NXentry
        dialog.setFilterCallback(customFilter)

    Executing a `DataFileDialog` can be done like that:

    .. code-block:: python

        dialog = DataFileDialog()
        result = dialog.exec_()
        if result:
            print("Selection:")
            print(dialog.selectedFile())
            print(dialog.selectedUrl())
        else:
            print("Nothing selected")

    If the selection is a dataset you can access to the data using
    :meth:`selectedData`.

    If the selection is a group or if you want to read the selected object on
    your own you can use the `silx.io` API.

    .. code-block:: python

        url = dialog.selectedUrl()
        with silx.io.open(url) as data:
            pass

    Or by loading the file first

    .. code-block:: python

        url = dialog.selectedDataUrl()
        with silx.io.open(url.file_path()) as h5:
            data = h5[url.data_path()]

    Or by using `h5py` library

    .. code-block:: python

        url = dialog.selectedDataUrl()
        with h5py.File(url.file_path()) as h5:
            data = h5[url.data_path()]
    """

    class FilterMode(enum.Enum):
        """This enum is used to indicate what the user may select in the
        dialog; i.e. what the dialog will return if the user clicks OK."""

        AnyNode = 0
        """Any existing node from an HDF5-like file."""
        ExistingDataset = 1
        """An existing HDF5-like dataset."""
        ExistingGroup = 2
        """An existing HDF5-like group. A file root is a group."""

    def __init__(self, parent=None):
        AbstractDataFileDialog.__init__(self, parent=parent)
        self.__filter = DataFileDialog.FilterMode.AnyNode
        self.__filterCallback = None

    def selectedData(self):
        """Returns the selected data by using the :meth:`silx.io.get_data`
        API with the selected URL provided by the dialog.

        If the URL identify a group of a file it will raise an exception. For
        group or file you have to use on your own the API :meth:`silx.io.open`.

        :rtype: numpy.ndarray
        :raise ValueError: If the URL do not link to a dataset
        """
        url = self.selectedUrl()
        return silx.io.get_data(url)

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
        if self.__filter == DataFileDialog.FilterMode.AnyNode:
            accepted = True
        elif self.__filter == DataFileDialog.FilterMode.ExistingDataset:
            accepted = silx.io.is_dataset(data)
        elif self.__filter == DataFileDialog.FilterMode.ExistingGroup:
            accepted = silx.io.is_group(data)
        else:
            raise ValueError("Filter %s is not supported" % self.__filter)
        if not accepted:
            return False
        if self.__filterCallback is not None:
            try:
                return self.__filterCallback(data)
            except Exception:
                _logger.error("Error while executing custom callback", exc_info=True)
                return False
        return True

    def setFilterCallback(self, callback):
        """Set the filter callback. This filter is applied only if the filter
        mode (:meth:`filterMode`) first accepts the selected data.

        It is not supposed to be set while the dialog is being used.

        :param callable callback: Define a custom function returning a boolean
            and taking as argument an h5-like node. If the function returns true
            the dialog can return the associated URL.
        """
        self.__filterCallback = callback

    def setFilterMode(self, mode):
        """Set the filter mode.

        It is not supposed to be set while the dialog is being used.

        :param DataFileDialog.FilterMode mode: The new filter.
        """
        self.__filter = mode

    def fileMode(self):
        """Returns the filter mode.

        :rtype: DataFileDialog.FilterMode
        """
        return self.__filter

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
