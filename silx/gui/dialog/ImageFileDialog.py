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
__date__ = "12/10/2017"

import os
import logging
import silx.io
from silx.gui.plot import actions
from silx.gui import qt
from silx.gui.plot.PlotWidget import PlotWidget
from silx.gui.hdf5.Hdf5TreeModel import Hdf5TreeModel

_logger = logging.getLogger(__name__)


class _ImagePreview(qt.QWidget):

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.__plot = PlotWidget(self)
        self.__plot.setAxesDisplayed(False)

        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.__plot)
        self.setLayout(layout)

        tools = qt.QToolBar(self)
        tools.addAction(actions.mode.ZoomModeAction(self.__plot, self))
        tools.addAction(actions.mode.PanModeAction(self.__plot, self))

        tools2 = qt.QToolBar(self)
        tools2.addAction(actions.control.ColormapAction(self.__plot, self))

        self.__size = qt.QLabel()
        status = qt.QStatusBar(self)
        status.addPermanentWidget(self.__size)

        self.__plot.addToolBar(tools)
        self.__plot.addToolBar(tools2)
        self.__plot.setStatusBar(status)

    def setImage(self, image):
        if image is None:
            self.clear()
            return

        self.__plot.addImage(legend="data", data=image)
        self.__plot.resetZoom()
        axis = self.__plot.getXAxis()
        axis.setLimitsConstraints(0, image.shape[1])
        axis = self.__plot.getYAxis()
        axis.setLimitsConstraints(0, image.shape[0])

        shape = [str(i) for i in image.shape]
        text = u" \u00D7 ".join(shape)
        self.__size.setText(text)

    def clear(self):
        self.__size.setText("")
        image = self.__plot.getImage("data")
        if image is not None:
            self.__plot.removeImage(legend="data")


class ImageFileDialog(qt.QDialog):
    """The ImageFileDialog class provides a dialog that allow users to select
    an image from a file.

    The ImageFileDialog class enables a user to traverse the file system in
    order to select one file. Then to traverse the file to select a frame or
    a slice of a dataset.

    The selected data is an image in 2 dimension.

    Using an ImageFileDialog can be done like that.

    .. code-block:: python

        dialog = ImageFileDialog()
        result = dialog.exec_()
        if result:
            print("Selection:")
            print(dialog.selectedFile())
            print(dialog.selectedImage())
            print(dialog.selectedImagePath())
        else:
            print("Nothing selected")
    """

    def __init__(self, parent=None):
        super(ImageFileDialog, self).__init__(parent)

        self.__selectedFile = None
        self.__selectedImage = None
        self.__selectedImagePath = None

        self.__h5 = None
        self.__fileModel = qt.QFileSystemModel(self)
        self.__dataModel = Hdf5TreeModel(self)
        self.__initLayout()

        path = os.getcwd()
        self.__fileModel.setRootPath(path)

    def __initLayout(self):
        self.__fileLocationView = qt.QColumnView(self)
        self.__fileLocationView.setModel(self.__fileModel)
        self.__fileLocationView.selectionModel().selectionChanged.connect(self.__fileSelected)
        self.__fileLocationView.setSelectionMode(qt.QAbstractItemView.SingleSelection)

        self.__dataLocationView = qt.QColumnView(self)
        self.__dataLocationView.setModel(self.__dataModel)
        self.__dataLocationView.selectionModel().selectionChanged.connect(self.__dataSelected)
        self.__dataLocationView.setSelectionMode(qt.QAbstractItemView.SingleSelection)

        self.__data = _ImagePreview(self)
        self.__data.setMinimumSize(200, 200)
        self.__data.setMaximumSize(400, 16777215)
        self.__data.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)

        self.__buttons = qt.QDialogButtonBox(self)
        types = qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel
        self.__buttons.setStandardButtons(types)
        self.__buttons.accepted.connect(self.accept)
        self.__buttons.rejected.connect(self.reject)

        layout = qt.QVBoxLayout(self)

        layoutLeft = qt.QVBoxLayout()
        layoutLeft.addWidget(self.__fileLocationView)
        layoutLeft.addWidget(self.__dataLocationView)

        layoutCentral = qt.QHBoxLayout()
        layoutCentral.addLayout(layoutLeft)
        layoutCentral.addWidget(self.__data)

        layout.addLayout(layoutCentral)
        layout.addWidget(self.__buttons)

        self.setLayout(layout)

    def __fileSelected(self, selected, deselected):
        self.__selectedFile = None
        for i in selected.indexes():
            path = self.__fileModel.filePath(i)
            if os.path.isfile(path):
                if self.__h5 is not None:
                    self.__dataModel.removeH5pyObject(self.__h5)
                    self.__h5 = None
                try:
                    self.__h5 = silx.io.open(path)
                    self.__selectedFile = path
                except IOError as e:
                    print(e)
                else:
                    self.__dataModel.insertH5pyObject(self.__h5)

    def __dataSelected(self):
        indexes = self.__dataLocationView.selectionModel().selectedIndexes()
        self.__selectedImage = None

        if len(indexes) == 1:
            index = indexes[0]
            model = self.__dataModel
            obj = model.data(index, model.H5PY_OBJECT_ROLE)
            if silx.io.is_dataset(obj):
                if len(obj.shape) == 2:
                    self.__selectedImage = obj

        self.__data.setImage(self.__selectedImage)

    # Selected file

    def setDirectory(self, path):
        """Sets the image dialog's current directory."""
        self.__fileModel.reset()
        self.__fileModel.setRootPath(path)

    def selectedFile(self):
        """Returns the file path containing the selected data.

        :rtype: str
        """
        return self.__selectedFile

    def selectFile(self, path):
        """Sets the image dialog's current file."""
        raise NotImplementedError()

    # Selected image

    def selectImagePath(self, path):
        """Sets the image dialog's current image path."""
        raise NotImplementedError()

    def selectedImagePath(self):
        """Returns the URI from the file path to the image.

        :rtype: str
        """
        return self.__selectedImagePath

    def selectedImage(self):
        """Returns the numpy array selected.

        :rtype: numpy.ndarray
        """
        return self.__selectedImage

    # Filters

    def nameFilters(self):
        """Returns the file type filters that are in operation on this file
        dialog."""
        raise NotImplementedError()

    def selectNameFilter(self, filter):
        """Sets the current file type filter. Multiple filters can be passed
        in filter by separating them with semicolons or spaces.
        """
        raise NotImplementedError()

    def selectedNameFilter(self):
        """Returns the filter that the user selected in the file dialog."""
        raise NotImplementedError()

    # State

    def restoreState(self, state):
        """Restores the dialogs's layout, history and current directory to the
        state specified."""
        raise NotImplementedError()

    def saveState(self):
        """Saves the state of the dialog's layout, history and current
        directory."""
        raise NotImplementedError()
