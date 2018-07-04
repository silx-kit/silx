# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
__date__ = "30/04/2018"


import logging
from .. import qt
from ...utils import weakref as silxweakref
from .Hdf5TreeModel import Hdf5TreeModel
from .Hdf5HeaderView import Hdf5HeaderView
from .NexusSortFilterProxyModel import NexusSortFilterProxyModel
from .Hdf5Item import Hdf5Item
from . import _utils

_logger = logging.getLogger(__name__)


class Hdf5TreeView(qt.QTreeView):
    """TreeView which allow to browse HDF5 file structure.

    .. image:: img/Hdf5TreeView.png

    It provides columns width auto-resizing and additional
    signals.

    The default model is a :class:`NexusSortFilterProxyModel` sourcing
    a :class:`Hdf5TreeModel`. The :class:`Hdf5TreeModel` is reachable using
    :meth:`findHdf5TreeModel`. The default header is :class:`Hdf5HeaderView`.

    Context menu is managed by the :meth:`setContextMenuPolicy` with the value
    Qt.CustomContextMenu. This policy must not be changed, otherwise context
    menus will not work anymore. You can use :meth:`addContextMenuCallback` and
    :meth:`removeContextMenuCallback` to add your custum actions according
    to the selected objects.
    """
    def __init__(self, parent=None):
        """
        Constructor

        :param parent qt.QWidget: The parent widget
        """
        qt.QTreeView.__init__(self, parent)

        model = self.createDefaultModel()
        self.setModel(model)

        self.setHeader(Hdf5HeaderView(qt.Qt.Horizontal, self))
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.sortByColumn(0, qt.Qt.AscendingOrder)
        # optimise the rendering
        self.setUniformRowHeights(True)

        self.setIconSize(qt.QSize(16, 16))
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(qt.QAbstractItemView.DragDrop)
        self.showDropIndicator()

        self.__context_menu_callbacks = silxweakref.WeakList()
        self.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._createContextMenu)

    def createDefaultModel(self):
        """Creates and returns the default model.

        Inherite to custom the default model"""
        model = Hdf5TreeModel(self)
        proxy_model = NexusSortFilterProxyModel(self)
        proxy_model.setSourceModel(model)
        return proxy_model

    def __removeContextMenuProxies(self, ref):
        """Callback to remove dead proxy from the list"""
        self.__context_menu_callbacks.remove(ref)

    def _createContextMenu(self, pos):
        """
        Create context menu.

        :param pos qt.QPoint: Position of the context menu
        """
        actions = []

        menu = qt.QMenu(self)

        hovered_index = self.indexAt(pos)
        hovered_node = self.model().data(hovered_index, Hdf5TreeModel.H5PY_ITEM_ROLE)
        if hovered_node is None or not isinstance(hovered_node, Hdf5Item):
            return

        hovered_object = _utils.H5Node(hovered_node)
        event = _utils.Hdf5ContextMenuEvent(self, menu, hovered_object)

        for callback in self.__context_menu_callbacks:
            try:
                callback(event)
            except KeyboardInterrupt:
                raise
            except Exception:
                # make sure no user callback crash the application
                _logger.error("Error while calling callback", exc_info=True)
                pass

        if not menu.isEmpty():
            for action in actions:
                menu.addAction(action)
            menu.popup(self.viewport().mapToGlobal(pos))

    def addContextMenuCallback(self, callback):
        """Register a context menu callback.

        The callback will be called when a context menu is requested with the
        treeview and the list of selected h5py objects in parameters. The
        callback must return a list of :class:`qt.QAction` object.

        Callbacks are stored as saferef. The object must store a reference by
        itself.
        """
        self.__context_menu_callbacks.append(callback)

    def removeContextMenuCallback(self, callback):
        """Unregister a context menu callback"""
        self.__context_menu_callbacks.remove(callback)

    def findHdf5TreeModel(self):
        """Find the Hdf5TreeModel from the stack of model filters.

        :returns: A Hdf5TreeModel, else None
        :rtype: Hdf5TreeModel
        """
        model = self.model()
        while model is not None:
            if isinstance(model, qt.QAbstractProxyModel):
                model = model.sourceModel()
            else:
                break
        if model is None:
            return None
        if isinstance(model, Hdf5TreeModel):
            return model
        else:
            return None

    def dragEnterEvent(self, event):
        model = self.findHdf5TreeModel()
        if model is not None and model.isFileDropEnabled() and event.mimeData().hasFormat("text/uri-list"):
            self.setState(qt.QAbstractItemView.DraggingState)
            event.accept()
        else:
            qt.QTreeView.dragEnterEvent(self, event)

    def dragMoveEvent(self, event):
        model = self.findHdf5TreeModel()
        if model is not None and model.isFileDropEnabled() and event.mimeData().hasFormat("text/uri-list"):
            event.setDropAction(qt.Qt.CopyAction)
            event.accept()
        else:
            qt.QTreeView.dragMoveEvent(self, event)

    def selectedH5Nodes(self, ignoreBrokenLinks=True):
        """Returns selected h5py objects like :class:`h5py.File`,
        :class:`h5py.Group`, :class:`h5py.Dataset` or mimicked objects.

        :param ignoreBrokenLinks bool: Returns objects which are not not
            broken links.
        :rtype: iterator(:class:`_utils.H5Node`)
        """
        for index in self.selectedIndexes():
            if index.column() != 0:
                continue
            item = self.model().data(index, Hdf5TreeModel.H5PY_ITEM_ROLE)
            if item is None:
                continue
            if isinstance(item, Hdf5Item):
                if ignoreBrokenLinks and item.isBrokenObj():
                    continue
                yield _utils.H5Node(item)

    def __intermediateModels(self, index):
        """Returns intermediate models from the view model to the
        model of the index."""
        models = []
        targetModel = index.model()
        model = self.model()
        while model is not None:
            if model is targetModel:
                # found
                return models
            models.append(model)
            if isinstance(model, qt.QAbstractProxyModel):
                model = model.sourceModel()
            else:
                break
        raise RuntimeError("Model from the requested index is not reachable from this view")

    def mapToModel(self, index):
        """Map an index from any model reachable by the view to an index from
        the very first model connected to the view.

        :param qt.QModelIndex index: Index from the Hdf5Tree model
        :rtype: qt.QModelIndex
        :return: Index from the model connected to the view
        """
        if not index.isValid():
            return index
        models = self.__intermediateModels(index)
        for model in reversed(models):
            index = model.mapFromSource(index)
        return index

    def setSelectedH5Node(self, h5Object):
        """
        Select the specified node of the tree using an h5py node.

        - If the item is found, parent items are expended, and then the item
          is selected.
        - If the item is not found, the selection do not change.
        - A none argument allow to deselect everything

        :param h5py.Node h5Object: The node to select
        """
        if h5Object is None:
            self.setCurrentIndex(qt.QModelIndex())
            return

        model = self.findHdf5TreeModel()
        index = model.indexFromH5Object(h5Object)
        index = self.mapToModel(index)
        if index.isValid():
            # Update the GUI
            i = index
            while i.isValid():
                self.expand(i)
                i = i.parent()
            self.setCurrentIndex(index)

    def mousePressEvent(self, event):
        """Override mousePressEvent to provide a consistante compatible API
        between Qt4 and Qt5
        """
        super(Hdf5TreeView, self).mousePressEvent(event)
        if event.button() != qt.Qt.LeftButton:
            # Qt5 only sends itemClicked on left button mouse click
            if qt.qVersion() > "5":
                qindex = self.indexAt(event.pos())
                self.clicked.emit(qindex)
