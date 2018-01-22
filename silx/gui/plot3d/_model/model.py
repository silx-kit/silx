# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
This module provides the :class:`SceneWidget` content and parameters model.
"""

from __future__ import absolute_import, division

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "11/01/2018"


import weakref

from ... import qt

from .core import BaseRow
from .items import Settings, nodeFromItem


class Root(BaseRow):
    """Root node of :class:`SceneWidget` parameters.

    It has two children:
    - Settings
    - Scene group
    """

    def __init__(self, model, sceneWidget):
        super(Root, self).__init__()
        self._sceneWidget = weakref.ref(sceneWidget)
        self.setParent(model)  # Needed for Root

    def children(self):
        sceneWidget = self._sceneWidget()
        if sceneWidget is None:
            return ()
        else:
            return super(Root, self).children()


class SceneModel(qt.QAbstractItemModel):
    """Model of a :class:`SceneWidget`.

    :param SceneWidget parent: The SceneWidget this model represents.
    """

    def __init__(self, parent):
        self._sceneWidget = weakref.ref(parent)

        super(SceneModel, self).__init__(parent)
        self._root = Root(self, parent)
        self._root.addRow(Settings(parent))
        self._root.addRow(nodeFromItem(parent.getSceneGroup()))

    def sceneWidget(self):
        """Returns the :class:`SceneWidget` this model represents.

        In case the widget has already been deleted, it returns None

        :rtype: SceneWidget
        """
        return self._sceneWidget()

    def _itemFromIndex(self, index):
        """Returns the corresponding :class:`Node` or :class:`Item3D`.

        :param QModelIndex index:
        :rtype: Node or Item3D
        """
        return index.internalPointer() if index.isValid() else self._root

    def index(self, row, column, parent=qt.QModelIndex()):
        """See :meth:`QAbstractItemModel.index`"""
        if column >= self.columnCount(parent) or row >= self.rowCount(parent):
            return qt.QModelIndex()

        item = self._itemFromIndex(parent)
        return self.createIndex(row, column, item.children()[row])

    def parent(self, index):
        """See :meth:`QAbstractItemModel.parent`"""
        if not index.isValid():
            return qt.QModelIndex()

        item = self._itemFromIndex(index)
        parent = item.parent()

        ancestor = parent.parent()

        if ancestor is not self:  # root node
            children = ancestor.children()
            row = children.index(parent)
            return self.createIndex(row, 0, parent)

        return qt.QModelIndex()

    def rowCount(self, parent=qt.QModelIndex()):
        """See :meth:`QAbstractItemModel.rowCount`"""
        item = self._itemFromIndex(parent)
        return item.rowCount()

    def columnCount(self, parent=qt.QModelIndex()):
        """See :meth:`QAbstractItemModel.columnCount`"""
        item = self._itemFromIndex(parent)
        return item.columnCount()

    def data(self, index, role=qt.Qt.DisplayRole):
        """See :meth:`QAbstractItemModel.data`"""
        item = self._itemFromIndex(index)
        column = index.column()
        return item.data(column, role)

    def setData(self, index, value, role=qt.Qt.EditRole):
        """See :meth:`QAbstractItemModel.setData`"""
        item = self._itemFromIndex(index)
        column = index.column()
        if item.setData(column, value, role):
            self.dataChanged.emit(index, index)
            return True
        return False

    def flags(self, index):
        """See :meth:`QAbstractItemModel.flags`"""
        item = self._itemFromIndex(index)
        column = index.column()
        return item.flags(column)

    def headerData(self, section, orientation, role=qt.Qt.DisplayRole):
        """See :meth:`QAbstractItemModel.headerData`"""
        if orientation == qt.Qt.Horizontal and role == qt.Qt.DisplayRole:
            return 'Item' if section == 0 else 'Value'
        else:
            return None
