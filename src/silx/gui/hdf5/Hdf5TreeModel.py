# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "12/03/2019"


import os
import logging
from typing import Optional
import functools
from .. import qt
from .. import icons
from .Hdf5Node import Hdf5Node
from .Hdf5Item import Hdf5Item
from .Hdf5LoadingItem import Hdf5LoadingItem
from . import _utils
from ... import io as silx_io

import h5py


_logger = logging.getLogger(__name__)


def _createRootLabel(h5obj):
    """
    Create label for the very first npde of the tree.

    :param h5obj: The h5py object to display in the GUI
    :type h5obj: h5py-like object
    :rtpye: str
    """
    if silx_io.is_file(h5obj):
        label = os.path.basename(h5obj.filename)
    else:
        filename = os.path.basename(h5obj.file.filename)
        path = h5obj.name
        if path.startswith("/"):
            path = path[1:]
        label = "%s::%s" % (filename, path)
    return label


class LoadingItemRunnable(qt.QRunnable):
    """Runner to process item loading from a file"""

    class __Signals(qt.QObject):
        """Signal holder"""
        itemReady = qt.Signal(object, object, object)
        runnerFinished = qt.Signal(object)

    def __init__(self, filename, item):
        """Constructor

        :param LoadingItemWorker worker: Object holding data and signals
        """
        super(LoadingItemRunnable, self).__init__()
        self.filename = filename
        self.oldItem = item
        self.signals = self.__Signals()

    def setFile(self, filename, item):
        self.filenames.append((filename, item))

    @property
    def itemReady(self):
        return self.signals.itemReady

    @property
    def runnerFinished(self):
        return self.signals.runnerFinished

    def __loadItemTree(self, oldItem, h5obj):
        """Create an item tree used by the GUI from an h5py object.

        :param Hdf5Node oldItem: The current item displayed the GUI
        :param h5py.File h5obj: The h5py object to display in the GUI
        :rtpye: Hdf5Node
        """
        text = _createRootLabel(h5obj)
        item = Hdf5Item(
            text=text,
            obj=h5obj,
            parent=oldItem.parent,
            populateAll=True,
            openedPath=oldItem._openedPath,
        )
        return item

    def run(self):
        """Process the file loading. The worker is used as holder
        of the data and the signal. The result is sent as a signal.
        """
        h5file = None
        try:
            h5file = silx_io.open(self.filename)
            newItem = self.__loadItemTree(self.oldItem, h5file)
            error = None
        except IOError as e:
            # Should be logged
            error = e
            newItem = None
            if h5file is not None:
                h5file.close()

        self.itemReady.emit(self.oldItem, newItem, error)
        self.runnerFinished.emit(self)

    def autoDelete(self):
        return True


class Hdf5TreeModel(qt.QAbstractItemModel):
    """Tree model storing a list of :class:`h5py.File` like objects.

    The main column display the :class:`h5py.File` list and there hierarchy.
    Other columns display information on node hierarchy.
    """

    H5PY_ITEM_ROLE = qt.Qt.UserRole
    """Role to reach h5py item from an item index"""

    H5PY_OBJECT_ROLE = qt.Qt.UserRole + 1
    """Role to reach h5py object from an item index"""

    USER_ROLE = qt.Qt.UserRole + 2
    """Start of range of available user role for derivative models"""

    NAME_COLUMN = 0
    """Column id containing HDF5 node names"""

    TYPE_COLUMN = 1
    """Column id containing HDF5 dataset types"""

    SHAPE_COLUMN = 2
    """Column id containing HDF5 dataset shapes"""

    VALUE_COLUMN = 3
    """Column id containing HDF5 dataset values"""

    DESCRIPTION_COLUMN = 4
    """Column id containing HDF5 node description/title/message"""

    NODE_COLUMN = 5
    """Column id containing HDF5 node type"""

    LINK_COLUMN = 6
    """Column id containing HDF5 link type"""

    COLUMN_IDS = [
        NAME_COLUMN,
        TYPE_COLUMN,
        SHAPE_COLUMN,
        VALUE_COLUMN,
        DESCRIPTION_COLUMN,
        NODE_COLUMN,
        LINK_COLUMN,
    ]
    """List of logical columns available"""

    sigH5pyObjectLoaded = qt.Signal(object)
    """Emitted when a new root item was loaded and inserted to the model."""

    sigH5pyObjectRemoved = qt.Signal(object)
    """Emitted when a root item is removed from the model."""

    sigH5pyObjectSynchronized = qt.Signal(object, object)
    """Emitted when an item was synchronized."""

    def __init__(self, parent=None, ownFiles=True):
        """
        Constructor

        :param qt.QWidget parent: Parent widget
        :param bool ownFiles: If true (default) the model will manage the files
            life cycle when they was added using path (like DnD).
        """
        super(Hdf5TreeModel, self).__init__(parent)

        self.header_labels = [None] * len(self.COLUMN_IDS)
        self.header_labels[self.NAME_COLUMN] = 'Name'
        self.header_labels[self.TYPE_COLUMN] = 'Type'
        self.header_labels[self.SHAPE_COLUMN] = 'Shape'
        self.header_labels[self.VALUE_COLUMN] = 'Value'
        self.header_labels[self.DESCRIPTION_COLUMN] = 'Description'
        self.header_labels[self.NODE_COLUMN] = 'Node'
        self.header_labels[self.LINK_COLUMN] = 'Link'

        # Create items
        self.__root = Hdf5Node()
        self.__fileDropEnabled = True
        self.__fileMoveEnabled = True
        self.__datasetDragEnabled = False

        self.__animatedIcon = icons.getWaitIcon()
        self.__animatedIcon.iconChanged.connect(self.__updateLoadingItems)
        self.__runnerSet = set([])

        # store used icons to avoid the cache to release it
        self.__icons = []
        self.__icons.append(icons.getQIcon("item-none"))
        self.__icons.append(icons.getQIcon("item-0dim"))
        self.__icons.append(icons.getQIcon("item-1dim"))
        self.__icons.append(icons.getQIcon("item-2dim"))
        self.__icons.append(icons.getQIcon("item-3dim"))
        self.__icons.append(icons.getQIcon("item-ndim"))

        self.__ownFiles = ownFiles
        self.__openedFiles = []
        """Store the list of files opened by the model itself."""
        # FIXME: It should be managed one by one by Hdf5Item itself

        # It is not possible to override the QObject destructor nor
        # to access to the content of the Python object with the `destroyed`
        # signal cause the Python method was already removed with the QWidget,
        # while the QObject still exists.
        # We use a static method plus explicit references to objects to
        # release. The callback do not use any ref to self.
        onDestroy = functools.partial(self._closeFileList, self.__openedFiles)
        self.destroyed.connect(onDestroy)

    @staticmethod
    def _closeFileList(fileList):
        """Static method to close explicit references to internal objects."""
        _logger.debug("Clear Hdf5TreeModel")
        for obj in fileList:
            _logger.debug("Close file %s", obj.filename)
            obj.close()
        fileList[:] = []

    def _closeOpened(self):
        """Close files which was opened by this model.

        File are opened by the model when it was inserted using
        `insertFileAsync`, `insertFile`, `appendFile`."""
        self._closeFileList(self.__openedFiles)

    def __updateLoadingItems(self, icon):
        for i in range(self.__root.childCount()):
            item = self.__root.child(i)
            if isinstance(item, Hdf5LoadingItem):
                index1 = self.index(i, 0, qt.QModelIndex())
                index2 = self.index(i, self.columnCount() - 1, qt.QModelIndex())
                self.dataChanged.emit(index1, index2)

    def __itemReady(self, oldItem, newItem, error):
        """Called at the end of a concurent file loading, when the loading
        item is ready. AN error is defined if an exception occured when
        loading the newItem .

        :param Hdf5Node oldItem: current displayed item
        :param Hdf5Node newItem: item loaded, or None if error is defined
        :param Exception error: An exception, or None if newItem is defined
        """
        row = self.__root.indexOfChild(oldItem)

        rootIndex = qt.QModelIndex()
        self.beginRemoveRows(rootIndex, row, row)
        self.__root.removeChildAtIndex(row)
        self.endRemoveRows()

        if newItem is not None:
            rootIndex = qt.QModelIndex()
            if self.__ownFiles:
                self.__openedFiles.append(newItem.obj)
            self.beginInsertRows(rootIndex, row, row)
            self.__root.insertChild(row, newItem)
            self.endInsertRows()

            if isinstance(oldItem, Hdf5LoadingItem):
                self.sigH5pyObjectLoaded.emit(newItem.obj)
            else:
                self.sigH5pyObjectSynchronized.emit(oldItem.obj, newItem.obj)

        # FIXME the error must be displayed

    def isFileDropEnabled(self):
        return self.__fileDropEnabled

    def setFileDropEnabled(self, enabled):
        self.__fileDropEnabled = enabled

    fileDropEnabled = qt.Property(bool, isFileDropEnabled, setFileDropEnabled)
    """Property to enable/disable file dropping in the model."""

    def isDatasetDragEnabled(self):
        return self.__datasetDragEnabled

    def setDatasetDragEnabled(self, enabled):
        self.__datasetDragEnabled = enabled

    datasetDragEnabled = qt.Property(bool, isDatasetDragEnabled, setDatasetDragEnabled)
    """Property to enable/disable drag of datasets."""

    def isFileMoveEnabled(self):
        return self.__fileMoveEnabled

    def setFileMoveEnabled(self, enabled):
        self.__fileMoveEnabled = enabled

    fileMoveEnabled = qt.Property(bool, isFileMoveEnabled, setFileMoveEnabled)
    """Property to enable/disable drag-and-drop of files to
    change the ordering in the model."""

    def supportedDropActions(self):
        if self.__fileMoveEnabled or self.__fileDropEnabled:
            return qt.Qt.CopyAction | qt.Qt.MoveAction
        else:
            return 0

    def mimeTypes(self):
        types = []
        if self.__fileMoveEnabled or self.__datasetDragEnabled:
            types.append(_utils.Hdf5DatasetMimeData.MIME_TYPE)
        return types

    def mimeData(self, indexes):
        """
        Returns an object that contains serialized items of data corresponding
        to the list of indexes specified.

        :param List[qt.QModelIndex] indexes: List of indexes
        :rtype: qt.QMimeData
        """
        if len(indexes) == 0:
            return None

        indexes = [i for i in indexes if i.column() == 0]
        if len(indexes) > 1:
            raise NotImplementedError("Drag of multi rows is not implemented")
        if len(indexes) == 0:
            raise NotImplementedError("Drag of cell is not implemented")

        node = self.nodeFromIndex(indexes[0])

        if self.__fileMoveEnabled and node.parent is self.__root:
            mimeData = _utils.Hdf5DatasetMimeData(node=node, isRoot=True)
        elif self.__datasetDragEnabled:
            mimeData = _utils.Hdf5DatasetMimeData(node=node)
        else:
            mimeData = None
        return mimeData

    def flags(self, index):
        defaultFlags = qt.QAbstractItemModel.flags(self, index)

        if index.isValid():
            node = self.nodeFromIndex(index)
            if self.__fileMoveEnabled and node.parent is self.__root:
                # that's a root
                return qt.Qt.ItemIsDragEnabled | defaultFlags
            elif self.__datasetDragEnabled:
                return qt.Qt.ItemIsDragEnabled | defaultFlags
            return defaultFlags
        elif self.__fileDropEnabled or self.__fileMoveEnabled:
            return qt.Qt.ItemIsDropEnabled | defaultFlags
        else:
            return defaultFlags

    def dropMimeData(self, mimedata, action, row, column, parentIndex):
        if action == qt.Qt.IgnoreAction:
            return True

        if self.__fileMoveEnabled and mimedata.hasFormat(_utils.Hdf5DatasetMimeData.MIME_TYPE):
            if mimedata.isRoot():
                dragNode = mimedata.node()
                parentNode = self.nodeFromIndex(parentIndex)
                if parentNode is not dragNode.parent:
                    return False

                if row == -1:
                    # append to the parent
                    row = parentNode.childCount()
                else:
                    # insert at row
                    pass

                dragNodeParent = dragNode.parent
                sourceRow = dragNodeParent.indexOfChild(dragNode)
                self.moveRow(parentIndex, sourceRow, parentIndex, row)
                return True

        if self.__fileDropEnabled and mimedata.hasFormat("text/uri-list"):

            parentNode = self.nodeFromIndex(parentIndex)
            if parentNode is not self.__root:
                while(parentNode is not self.__root):
                    node = parentNode
                    parentNode = node.parent
                row = parentNode.indexOfChild(node)
            else:
                if row == -1:
                    row = self.__root.childCount()

            messages = []
            for url in mimedata.urls():
                try:
                    self.insertFileAsync(url.toLocalFile(), row)
                    row += 1
                except IOError as e:
                    messages.append(e.args[0])
            if len(messages) > 0:
                title = "Error occurred when loading files"
                message = "<html>%s:<ul><li>%s</li><ul></html>" % (title, "</li><li>".join(messages))
                qt.QMessageBox.critical(None, title, message)
            return True

        return False

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        if orientation == qt.Qt.Horizontal:
            if role in [qt.Qt.DisplayRole, qt.Qt.EditRole]:
                return self.header_labels[section]
        return None

    def insertNode(self, row, node):
        if row == -1:
            row = self.__root.childCount()
        self.beginInsertRows(qt.QModelIndex(), row, row)
        self.__root.insertChild(row, node)
        self.endInsertRows()

    def moveRow(self, sourceParentIndex, sourceRow, destinationParentIndex, destinationRow):
        if sourceRow == destinationRow or sourceRow == destinationRow - 1:
            # abort move, same place
            return
        return self.moveRows(sourceParentIndex, sourceRow, 1, destinationParentIndex, destinationRow)

    def moveRows(self, sourceParentIndex, sourceRow, count, destinationParentIndex, destinationRow):
        self.beginMoveRows(sourceParentIndex, sourceRow, sourceRow, destinationParentIndex, destinationRow)
        sourceNode = self.nodeFromIndex(sourceParentIndex)
        destinationNode = self.nodeFromIndex(destinationParentIndex)

        if sourceNode is destinationNode and sourceRow < destinationRow:
            item = sourceNode.child(sourceRow)
            destinationNode.insertChild(destinationRow, item)
            sourceNode.removeChildAtIndex(sourceRow)
        else:
            item = sourceNode.removeChildAtIndex(sourceRow)
            destinationNode.insertChild(destinationRow, item)

        self.endMoveRows()
        return True

    def index(self, row, column, parent=qt.QModelIndex()):
        try:
            node = self.nodeFromIndex(parent)
            return self.createIndex(row, column, node.child(row))
        except IndexError:
            return qt.QModelIndex()

    def data(self, index, role=qt.Qt.DisplayRole):
        node = self.nodeFromIndex(index)

        if role == self.H5PY_ITEM_ROLE:
            return node

        if role == self.H5PY_OBJECT_ROLE:
            return node.obj

        if index.column() == self.NAME_COLUMN:
            return node.dataName(role)
        elif index.column() == self.TYPE_COLUMN:
            return node.dataType(role)
        elif index.column() == self.SHAPE_COLUMN:
            return node.dataShape(role)
        elif index.column() == self.VALUE_COLUMN:
            return node.dataValue(role)
        elif index.column() == self.DESCRIPTION_COLUMN:
            return node.dataDescription(role)
        elif index.column() == self.NODE_COLUMN:
            return node.dataNode(role)
        elif index.column() == self.LINK_COLUMN:
            return node.dataLink(role)
        else:
            return None

    def columnCount(self, parent=qt.QModelIndex()):
        return len(self.COLUMN_IDS)

    def hasChildren(self, parent=qt.QModelIndex()):
        node = self.nodeFromIndex(parent)
        if node is None:
            return 0
        return node.hasChildren()

    def rowCount(self, parent=qt.QModelIndex()):
        node = self.nodeFromIndex(parent)
        if node is None:
            return 0
        return node.childCount()

    def parent(self, child):
        if not child.isValid():
            return qt.QModelIndex()

        node = self.nodeFromIndex(child)

        if node is None:
            return qt.QModelIndex()

        parent = node.parent

        if parent is None:
            return qt.QModelIndex()

        grandparent = parent.parent
        if grandparent is None:
            return qt.QModelIndex()
        row = grandparent.indexOfChild(parent)

        assert row != - 1
        return self.createIndex(row, 0, parent)

    def nodeFromIndex(self, index):
        return index.internalPointer() if index.isValid() else self.__root

    def _closeFileIfOwned(self, node):
        """"Close the file if it was loaded from a filename or a
        drag-and-drop"""
        obj = node.obj
        for f in self.__openedFiles:
            if f is obj:
                _logger.debug("Close file %s", obj.filename)
                obj.close()
                self.__openedFiles.remove(obj)

    def synchronizeIndex(self, index):
        """
        Synchronize a file a given its index.

        Basically close it and load it again.

        :param qt.QModelIndex index: Index of the item to update
        """
        node = self.nodeFromIndex(index)
        if node.parent is not self.__root:
            return

        filename = node.obj.filename
        self.insertFileAsync(filename, index.row(), synchronizingNode=node)

    @staticmethod
    def __areH5pyObjectEqual(obj1, obj2):
        """Compare commonh5/h5py object without comparing data"""
        if isinstance(obj1, h5py.HLObject):  # Priority to h5py __eq__
            return obj1 == obj2

        # else compare commonh5 objects
        if not isinstance(obj2, type(obj1)):
            return False
        def key(item):
            if item.file is None:
                return item.name
            return item.file.filename, item.file.mode, item.name
        return key(obj1) == key(obj2)

    def h5pyObjectRow(self, h5pyObject):
        for row in range(self.__root.childCount()):
            item = self.__root.child(row)
            if self.__areH5pyObjectEqual(item.obj, h5pyObject):
                return row
        return -1

    def synchronizeH5pyObject(self, h5pyObject):
        """
        Synchronize a h5py object in all the tree.

        Basically close it and load it again.

        :param h5py.File h5pyObject: A :class:`h5py.File` object.
        """
        index = 0
        while index < self.__root.childCount():
            item = self.__root.child(index)
            if self.__areH5pyObjectEqual(item.obj, h5pyObject):
                qindex = self.index(index, 0, qt.QModelIndex())
                self.synchronizeIndex(qindex)
            index += 1

    def removeIndex(self, index):
        """
        Remove an item from the model using its index.

        :param qt.QModelIndex index: Index of the item to remove
        """
        node = self.nodeFromIndex(index)
        if node.parent != self.__root:
            return
        self._closeFileIfOwned(node)
        self.beginRemoveRows(qt.QModelIndex(), index.row(), index.row())
        self.__root.removeChildAtIndex(index.row())
        self.endRemoveRows()
        self.sigH5pyObjectRemoved.emit(node.obj)

    def removeH5pyObject(self, h5pyObject):
        """
        Remove an item from the model using the holding h5py object.
        It can remove more than one item.

        :param h5py.File h5pyObject: A :class:`h5py.File` object.
        """
        index = 0
        while index < self.__root.childCount():
            item = self.__root.child(index)
            if self.__areH5pyObjectEqual(item.obj, h5pyObject):
                qindex = self.index(index, 0, qt.QModelIndex())
                self.removeIndex(qindex)
            else:
                index += 1

    def insertH5pyObject(
        self,
        h5pyObject,
        text: Optional[str] = None,
        row: int = -1,
        filename: Optional[str] = None,
    ):
        """Append an HDF5 object from h5py to the tree.

        :param h5pyObject: File handle/descriptor for a :class:`h5py.File`
            or any other class of h5py file structure.
        """
        if text is None:
            text = _createRootLabel(h5pyObject)
        if row == -1:
            row = self.__root.childCount()
        self.insertNode(
            row,
            Hdf5Item(
                text=text,
                obj=h5pyObject,
                parent=self.__root,
                openedPath=filename,
            )
        )

    def hasPendingOperations(self):
        return len(self.__runnerSet) > 0

    def insertFileAsync(self, filename, row=-1, synchronizingNode=None):
        if not os.path.isfile(filename):
            raise IOError("Filename '%s' must be a file path" % filename)

        # create temporary item
        if synchronizingNode is None:
            text = os.path.basename(filename)
            item = Hdf5LoadingItem(
                text=text,
                parent=self.__root,
                animatedIcon=self.__animatedIcon,
                openedPath=filename,
            )
            self.insertNode(row, item)
        else:
            item = synchronizingNode

        # start loading the real one
        runnable = LoadingItemRunnable(filename, item)
        runnable.itemReady.connect(self.__itemReady)
        runnable.runnerFinished.connect(self.__releaseRunner)
        self.__runnerSet.add(runnable)
        qt.silxGlobalThreadPool().start(runnable)

    def __releaseRunner(self, runner):
        self.__runnerSet.remove(runner)

    def insertFile(self, filename, row=-1):
        """Load a HDF5 file into the data model.

        :param filename: file path.
        """
        try:
            h5file = silx_io.open(filename)
            if self.__ownFiles:
                self.__openedFiles.append(h5file)
            self.sigH5pyObjectLoaded.emit(h5file)
            self.insertH5pyObject(h5file, row=row, filename=filename)
        except IOError:
            _logger.debug("File '%s' can't be read.", filename, exc_info=True)
            raise

    def clear(self):
        """Remove all the content of the model"""
        for _ in range(self.rowCount()):
            qindex = self.index(0, 0, qt.QModelIndex())
            self.removeIndex(qindex)

    def appendFile(self, filename):
        self.insertFile(filename, -1)

    def indexFromH5Object(self, h5Object):
        """Returns a model index from an h5py-like object.

        :param object h5Object: An h5py-like object
        :rtype: qt.QModelIndex
        """
        if h5Object is None:
            return qt.QModelIndex()

        filename = h5Object.file.filename

        # Seach for the right roots
        rootIndices = []
        for index in range(self.rowCount(qt.QModelIndex())):
            index = self.index(index, 0, qt.QModelIndex())
            obj = self.data(index, Hdf5TreeModel.H5PY_OBJECT_ROLE)
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

                obj = self.data(index, Hdf5TreeModel.H5PY_OBJECT_ROLE)
                p = obj.name + "/"
                p = p.replace("//", "/")
                if path == p:
                    found = True
                    break

            parentIndex = foundIndices[-1]
            for index in range(self.rowCount(parentIndex)):
                index = self.index(index, 0, parentIndex)
                obj = self.data(index, Hdf5TreeModel.H5PY_OBJECT_ROLE)

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
