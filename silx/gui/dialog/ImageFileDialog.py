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
__date__ = "13/10/2017"

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


class _IconProvider(object):

    FileDialogToParentDir = qt.QStyle.SP_CustomBase + 1

    FileDialogToParentFile = qt.QStyle.SP_CustomBase + 2

    def __init__(self):
        self.__iconFileDialogToParentDir = None
        self.__iconFileDialogToParentFile = None

    def _createIconToParent(self, standardPixmap):
        """

        FIXME: It have to be tested for some OS (arrow icon do not have always
        the same direction)
        """
        style = qt.QApplication.style()
        baseIcon = style.standardIcon(qt.QStyle.SP_FileDialogToParent)
        backgroundIcon = style.standardIcon(standardPixmap)
        icon = qt.QIcon(self)

        sizes = baseIcon.availableSizes()
        sizes = sorted(sizes, key=lambda s: s.height())
        sizes = filter(lambda s: s.height() < 100, sizes)
        if len(sizes) > 0:
            baseSize = sizes[-1]
        else:
            baseIcon.availableSizes()[0]
        size = qt.QSize(baseSize.width(), baseSize.height() * 3 / 2)

        modes = [qt.QIcon.Normal, qt.QIcon.Disabled]
        for mode in modes:
            pixmap = qt.QPixmap(size)
            pixmap.fill(qt.Qt.transparent)
            painter = qt.QPainter(pixmap)
            painter.drawPixmap(0, 0, backgroundIcon.pixmap(baseSize, mode=mode))
            painter.drawPixmap(0, size.height() / 3, baseIcon.pixmap(baseSize, mode=mode))
            painter.end()
            icon.addPixmap(pixmap, mode=mode)

        return icon

    def getFileDialogToParentDir(self):
        if self.__iconFileDialogToParentDir is None:
            self.__iconFileDialogToParentDir = self._createIconToParent(qt.QStyle.SP_DirIcon)
        return self.__iconFileDialogToParentDir

    def getFileDialogToParentFile(self):
        if self.__iconFileDialogToParentFile is None:
            self.__iconFileDialogToParentFile = self._createIconToParent(qt.QStyle.SP_FileIcon)
        return self.__iconFileDialogToParentFile

    def icon(self, kind):
        if kind == self.FileDialogToParentDir:
            return self.getFileDialogToParentDir()
        elif kind == self.FileDialogToParentFile:
            return self.getFileDialogToParentFile()
        else:
            style = qt.QApplication.style()
            icon = style.standardIcon(kind)
            return icon


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
        self.setUniformItemSizes(True)
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

    _defaultIconProvider = None
    """Lazy loaded default icon provider"""

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
        self.__showAsListView()

        path = os.getcwd()
        self.__fileModel.setRootPath(path)

    def _createSideBar(self):
        return _SideBar(self)

    def iconProvider(self):
        iconProvider = self.__class__._defaultIconProvider
        if iconProvider is None:
            iconProvider = _IconProvider()
            self.__class__._defaultIconProvider = iconProvider
        return iconProvider

    def _createBrowseToolBar(self):
        toolbar = qt.QToolBar(self)
        iconProvider = self.iconProvider()

        back = qt.QAction(toolbar)
        back.setText("Back")
        back.setIcon(iconProvider.icon(qt.QStyle.SP_ArrowBack))
        back.triggered.connect(self.__navigateBack)

        forward = qt.QAction(toolbar)
        forward.setText("Forward")
        forward.setIcon(iconProvider.icon(qt.QStyle.SP_ArrowForward))
        forward.triggered.connect(self.__navigateForward)

        parentDirectory = qt.QAction(toolbar)
        parentDirectory.setText("Parent directory")
        parentDirectory.setIcon(iconProvider.icon(qt.QStyle.SP_FileDialogToParent))
        parentDirectory.triggered.connect(self.__navigateToParent)

        fileDirectory = qt.QAction(toolbar)
        fileDirectory.setText("Root of the file")
        fileDirectory.setIcon(iconProvider.icon(iconProvider.FileDialogToParentFile))
        fileDirectory.triggered.connect(self.__navigateToParentFile)

        parentFileDirectory = qt.QAction(toolbar)
        parentFileDirectory.setText("Parent directory of the file")
        parentFileDirectory.setIcon(iconProvider.icon(iconProvider.FileDialogToParentDir))
        parentFileDirectory.triggered.connect(self.__navigateToParentDir)

        listView = qt.QAction(toolbar)
        listView.setText("List view")
        listView.setIcon(iconProvider.icon(qt.QStyle.SP_FileDialogListView))
        listView.triggered.connect(self.__showAsListView)
        listView.setCheckable(True)

        detailView = qt.QAction(toolbar)
        detailView.setText("List view")
        detailView.setIcon(iconProvider.icon(qt.QStyle.SP_FileDialogDetailedView))
        detailView.triggered.connect(self.__showAsDetailedView)
        detailView.setCheckable(True)

        self.__listViewAction = listView
        self.__detailViewAction = detailView

        toolbar.addAction(back)
        toolbar.addAction(forward)
        toolbar.addSeparator()
        toolbar.addAction(parentDirectory)
        toolbar.addAction(fileDirectory)
        toolbar.addAction(parentFileDirectory)
        toolbar.addSeparator()
        toolbar.addAction(listView)
        toolbar.addAction(detailView)

        toolbar.setStyleSheet("QToolBar { border: 0px }")

        return toolbar

    def __initLayout(self):
        self.__sidebar = self._createSideBar()
        self.__sidebar.clicked.connect(self.__shortcutClicked)
        self.__sidebar.setSelectionMode(qt.QAbstractItemView.SingleSelection)

        self.__browserView = qt.QListView(self)
        self.__browserView.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.__browserView.activated.connect(self.__browsedItemActivated)
        self.__browserView.setUniformItemSizes(True)
        self.__browserView.setResizeMode(qt.QListView.Adjust)
        self.__updateBrowserModel(self.__fileModel)

        self.__data = _ImagePreview(self)
        self.__data.setMinimumSize(200, 200)
        self.__data.setMaximumSize(400, 16777215)
        self.__data.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)

        self.__buttons = qt.QDialogButtonBox(self)
        types = qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel
        self.__buttons.setStandardButtons(types)
        self.__buttons.accepted.connect(self.accept)
        self.__buttons.rejected.connect(self.reject)

        self.__browseToolBar = self._createBrowseToolBar()

        datasetSelection = qt.QWidget(self)
        layoutLeft = qt.QVBoxLayout()
        layoutLeft.setContentsMargins(0, 0, 0, 0)
        layoutLeft.addWidget(self.__browseToolBar)
        layoutLeft.addWidget(self.__browserView)
        datasetSelection.setLayout(layoutLeft)
        datasetSelection.setSizePolicy(qt.QSizePolicy.MinimumExpanding, qt.QSizePolicy.Expanding)

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

    def __navigateBack(self):
        raise NotImplementedError()

    def __navigateForward(self):
        raise NotImplementedError()

    def __navigateToParent(self):
        index = self.__browserView.rootIndex()
        if index.model() is self.__fileModel:
            # browse throw the file system
            index = index.parent()
            if index.isValid():
                self.__browserView.setRootIndex(index)
        elif index.model() is self.__dataModel:
            index = index.parent()
            if index.isValid():
                # browse throw the hdf5
                self.__browserView.setRootIndex(index)
            else:
                # go back to the file system
                self.__navigateToParentDir()

    def __navigateToParentFile(self):
        index = self.__browserView.rootIndex()
        if index.model() is self.__dataModel:
            index = _indexFromH5Object(self.__dataModel, self.__h5)
            self.__browserView.setRootIndex(index)

    def __updateBrowserModel(self, model):
        if self.__browserView.selectionModel() is not None:
            self.__browserView.selectionModel().selectionChanged.disconnect()
        self.__browserView.setModel(model)
        self.__browserView.selectionModel().selectionChanged.connect(self.__browsedItemSelected)

    def __navigateToParentDir(self):
        index = self.__browserView.rootIndex()
        if index.model() is self.__dataModel:
            self.__updateBrowserModel(self.__fileModel)
            index = self.__fileModel.index(self.__h5.file.filename)
            index = index.parent()
            self.__browserView.setRootIndex(index)
            self.__dataModel.removeH5pyObject(self.__h5)
            self.__h5 = None

    def __showAsListView(self):
        mode = qt.QListView.IconMode
        self.__browserView.setViewMode(mode)
        self.__listViewAction.setChecked(True)
        self.__detailViewAction.setChecked(False)

    def __showAsDetailedView(self):
        mode = qt.QListView.ListMode
        self.__browserView.setViewMode(mode)
        self.__listViewAction.setChecked(False)
        self.__detailViewAction.setChecked(True)

    def __shortcutClicked(self):
        indexes = self.__sidebar.selectionModel().selectedIndexes()
        if len(indexes) == 1:
            index = indexes[0]
            url = self.__sidebar.model().data(index, role=qt.Qt.UserRole)
            path = url.toLocalFile()
            if path == "":
                path = qt.QDir.rootPath()
            self.__fileModel.setRootPath(path)

    def __browsedItemActivated(self, index):
        if index.model() is self.__fileModel:
            if self.__fileModel.isDir(index):
                self.__browserView.setRootIndex(index)
            path = self.__fileModel.filePath(index)
            if os.path.isfile(path):
                self.__fileActivated(index)
        elif index.model() is self.__dataModel:
            obj = index.data(role=Hdf5TreeModel.H5PY_OBJECT_ROLE)
            if silx.io.is_group(obj):
                self.__browserView.setRootIndex(index)

    def __browsedItemSelected(self, selected, deselected):
        indexes = self.__browserView.selectionModel().selectedIndexes()
        if len(indexes) == 1:
            index = indexes[0]
            if index.model() is self.__fileModel:
                self.__dataSelected(None)
            elif index.model() is self.__dataModel:
                self.__dataSelected(index)

    def __directoryLoaded(self, path):
        index = self.__fileModel.index(path)
        self.__browserView.setRootIndex(index)

    def __fileActivated(self, index):
        self.__selectedFile = None
        path = self.__fileModel.filePath(index)
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
                self.__updateBrowserModel(self.__dataModel)
                self.__browserView.setRootIndex(index)

    def __dataSelected(self, index):
        self.__selectedImage = None
        if index is not None and index.model() is self.__dataModel:
            obj = index.data(self.__dataModel.H5PY_OBJECT_ROLE)
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
