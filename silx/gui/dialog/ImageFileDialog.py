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


def _indexFromH5Object(model, h5Object):
    """This code should be inside silx"""
    if h5Object is None:
        return qt.QModelIndex()

    filename = h5Object.file.filename

    # Seach for the right roots
    rootIndices = []
    for index in range(model.rowCount(qt.QModelIndex())):
        index = model.index(index, 0, qt.QModelIndex())
        obj = model.data(index, Hdf5TreeModel.H5PY_OBJECT_ROLE)
        if obj.file.filename == filename:
            # We can have many roots with different subtree of the same
            # root
            rootIndices.append(index)

    if len(rootIndices) == 0:
        # No root found
        return qt.QModelIndex()

    path = h5Object.name + "/"
    path = path.replace("//", "/")

    # Search for the right node
    found = False
    foundIndices = []
    for _ in range(1000 * len(rootIndices)):
        # Avoid too much iterations, in case of recurssive links
        if len(foundIndices) == 0:
            if len(rootIndices) == 0:
                # Nothing found
                break
            # Start fron a new root
            foundIndices.append(rootIndices.pop(0))

            obj = model.data(index, Hdf5TreeModel.H5PY_OBJECT_ROLE)
            p = obj.name + "/"
            p = p.replace("//", "/")
            if path == p:
                found = True
                break

        parentIndex = foundIndices[-1]
        for index in range(model.rowCount(parentIndex)):
            index = model.index(index, 0, parentIndex)
            obj = model.data(index, Hdf5TreeModel.H5PY_OBJECT_ROLE)

            p = obj.name + "/"
            p = p.replace("//", "/")
            if path == p:
                foundIndices.append(index)
                found = True
                break
            elif path.startswith(p):
                foundIndices.append(index)
                break
        else:
            # Nothing found, start again with another root
            foundIndices = []

        if found:
            break

    if found:
        return foundIndices[-1]
    return qt.QModelIndex()


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


class _SideBar(qt.QListView):
    """Sidebar containing shortcuts for common directories"""

    def __init__(self, parent=None):
        super(_SideBar, self).__init__(parent)
        model = self._createModel()
        self.setModel(model)
        self.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)

    def _createModel(self):

        # Get default shortcut
        # There is no other way
        d = qt.QFileDialog()
        urls = d.sidebarUrls()
        d = None

        model = qt.QStandardItemModel(self)

        names = {}
        names[""] = "Computer"
        names[qt.QDir.rootPath()] = "Computer"
        names[qt.QDir.homePath()] = "Home"

        style = qt.QApplication.style()

        for url in urls:
            path = url.toLocalFile()
            if path in names:
                name = names[path]
            else:
                name = path.rsplit("/", 1)[-1]

            if name == "Computer":
                icon = style.standardIcon(qt.QStyle.SP_ComputerIcon)
            elif not os.path.exists(path):
                icon = style.standardIcon(qt.QStyle.SP_MessageBoxCritical)
            elif os.path.isdir(path):
                icon = style.standardIcon(qt.QStyle.SP_DirIcon)
            elif os.path.isfile(path):
                icon = style.standardIcon(qt.QStyle.SP_FileIcon)
            else:
                icon = None

            item = qt.QStandardItem(name)
            item.setIcon(icon)
            item.setData(url, role=qt.Qt.UserRole)
            model.appendRow(item)

        return model

    def sizeHint(self):
        index = self.model().index(0, 0)
        return self.sizeHintForIndex(index) + qt.QSize(2 * self.frameWidth(), 2 * self.frameWidth())


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
        self.__fileModel.directoryLoaded.connect(self.__directoryLoaded)

        self.__dataModel = Hdf5TreeModel(self)
        self.__initLayout()

        path = os.getcwd()
        self.__fileModel.setRootPath(path)

    def _createSideBar(self):
        return _SideBar(self)

    def __initLayout(self):
        self.__sidebar = self._createSideBar()
        self.__sidebar.clicked.connect(self.__shortcutClicked)
        self.__sidebar.setSelectionMode(qt.QAbstractItemView.SingleSelection)

        self.__fileLocationView = qt.QListView(self)
        self.__fileLocationView.setModel(self.__fileModel)
        self.__fileLocationView.selectionModel().selectionChanged.connect(self.__fileSelected)
        self.__fileLocationView.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.__fileLocationView.activated.connect(self.__fileActivated)

        self.__dataLocationView = qt.QListView(self)
        self.__dataLocationView.setModel(self.__dataModel)
        self.__dataLocationView.selectionModel().selectionChanged.connect(self.__dataSelected)
        self.__dataLocationView.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.__dataLocationView.activated.connect(self.__dataActivated)

        self.__data = _ImagePreview(self)
        self.__data.setMinimumSize(200, 200)
        self.__data.setMaximumSize(400, 16777215)
        self.__data.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)

        self.__buttons = qt.QDialogButtonBox(self)
        types = qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel
        self.__buttons.setStandardButtons(types)
        self.__buttons.accepted.connect(self.accept)
        self.__buttons.rejected.connect(self.reject)

        datasetSelection = qt.QWidget(self)
        layoutLeft = qt.QVBoxLayout()
        layoutLeft.setContentsMargins(0, 0, 0, 0)
        layoutLeft.addWidget(self.__fileLocationView)
        layoutLeft.addWidget(self.__dataLocationView)
        datasetSelection.setLayout(layoutLeft)
        datasetSelection.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)

        imageSelection = qt.QWidget(self)
        imageLayout = qt.QVBoxLayout()
        imageLayout.addWidget(self.__data)
        imageSelection.setLayout(imageLayout)

        self.__splitter = qt.QSplitter(self)
        self.__splitter.setContentsMargins(0, 0, 0, 0)
        self.__splitter.addWidget(self.__sidebar)
        self.__splitter.addWidget(datasetSelection)
        self.__splitter.addWidget(imageSelection)
        self.__splitter.setStretchFactor(1, 10)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.__splitter)
        layout.addWidget(self.__buttons)

        self.setLayout(layout)

    def __shortcutClicked(self):
        indexes = self.__sidebar.selectionModel().selectedIndexes()
        if len(indexes) == 1:
            index = indexes[0]
            url = self.__sidebar.model().data(index, role=qt.Qt.UserRole)
            path = url.toLocalFile()
            if path == "":
                path = qt.QDir.rootPath()
            self.__fileModel.setRootPath(path)

    def __fileActivated(self, index):
        if self.__fileModel.isDir(index):
            view = self.__fileLocationView
            view.setRootIndex(index)

    def __directoryLoaded(self, path):
        index = self.__fileModel.index(path)
        view = self.__fileLocationView
        view.setRootIndex(index)

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
                    _logger.error("Error while loading file %s: %s", path, e.args[0])
                    _logger.debug("Backtrace", exc_info=True)
                else:
                    self.__dataModel.insertH5pyObject(self.__h5)
                    index = _indexFromH5Object(self.__dataModel, self.__h5)
                    view = self.__dataLocationView
                    view.setRootIndex(index)

    def __dataActivated(self, index):
        obj = index.data(role=Hdf5TreeModel.H5PY_OBJECT_ROLE)
        if silx.io.is_group(obj):
            view = self.__dataLocationView
            view.setRootIndex(index)

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
