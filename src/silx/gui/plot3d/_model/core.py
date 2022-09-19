# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
This module provides base classes to implement models for 3D scene content.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "11/01/2018"


import collections
import weakref

from ....utils.weakref import WeakMethodProxy
from ... import qt


class BaseRow(qt.QObject):
    """Base class for rows of the tree model.

    The root node parent MUST be set to the QAbstractItemModel it belongs to.
    By default item is enabled.

    :param children: Iterable of BaseRow to start with (not signaled)
    """

    def __init__(self, children=()):
        self.__modelRef = None
        self.__parentRef = None
        super(BaseRow, self).__init__()
        self.__children = []
        for row in children:
            assert isinstance(row, BaseRow)
            row.setParent(self)
            self.__children.append(row)
        self.__flags = collections.defaultdict(lambda: qt.Qt.ItemIsEnabled)
        self.__tooltip = None

    def setParent(self, parent):
        """Override :meth:`QObject.setParent` to cache model and parent"""
        self.__parentRef = None if parent is None else weakref.ref(parent)

        if isinstance(parent, qt.QAbstractItemModel):
            model = parent
        elif isinstance(parent, BaseRow):
            model = parent.model()
        else:
            model = None

        self._updateModel(model)

        super(BaseRow, self).setParent(parent)

    def parent(self):
        """Override :meth:`QObject.setParent` to use cached parent

        :rtype: Union[QObject, None]"""
        return self.__parentRef() if self.__parentRef is not None else None

    def _updateModel(self, model):
        """Update the model this row belongs to"""
        if model != self.model():
            self.__modelRef = weakref.ref(model) if model is not None else None
            for child in self.children():
                child._updateModel(model)

    def model(self):
        """Return the model this node belongs to or None if not in a model.

        :rtype: Union[QAbstractItemModel, None]
        """
        return self.__modelRef() if self.__modelRef is not None else None

    def index(self, column=0):
        """Return corresponding index in the model or None if not in a model.

        :param int column: The column to make the index for
        :rtype: Union[QModelIndex, None]
        """
        parent = self.parent()
        model = self.model()

        if model is None:  # Not in a model
            return None
        elif parent is model:  # Root node
            return qt.QModelIndex()
        else:
            index = parent.index()
            row = parent.children().index(self)
            return model.index(row, column, index)

    def columnCount(self):
        """Returns number of columns (default: 2)

        :rtype: int
        """
        return 2

    def children(self):
        """Returns the list of children nodes

        :rtype: tuple of Node
        """
        return tuple(self.__children)

    def rowCount(self):
        """Returns number of rows

        :rtype: int
        """
        return len(self.__children)

    def addRow(self, row, index=None):
        """Add a node to the children

        :param BaseRow row: The node to add
        :param int index: The index at which to insert it or
                          None to append
        """
        if index is None:
            index = self.rowCount()
        assert index <= self.rowCount()

        model = self.model()

        if model is not None:
            parent = self.index()
            model.beginInsertRows(parent, index, index)

        self.__children.insert(index, row)
        row.setParent(self)

        if model is not None:
            model.endInsertRows()

    def removeRow(self, row):
        """Remove a row from the children list.

        It removes either a node or a row index.

        :param row: BaseRow object or index of row to remove
        :type row: Union[BaseRow, int]
        """
        if isinstance(row, BaseRow):
            row = self.__children.index(row)
        else:
            row = int(row)
        assert row < self.rowCount()

        model = self.model()

        if model is not None:
            index = self.index()
            model.beginRemoveRows(index, row, row)

        node = self.__children.pop(row)
        node.setParent(None)

        if model is not None:
            model.endRemoveRows()

    def data(self, column, role):
        """Returns data for given column and role

        :param int column: Column index for this row
        :param int role: The role to get
        :return: Corresponding data (Default: None)
        """
        if role == qt.Qt.ToolTipRole and self.__tooltip is not None:
            return self.__tooltip
        else:
            return None

    def setData(self, column, value, role):
        """Set data for given column and role

        :param int column: Column index for this row
        :param value: The data to set
        :param int role: The role to set
        :return: True on success, False on failure
        :rtype: bool
        """
        return False

    def setToolTip(self, tooltip):
        """Set the tooltip of the whole row.

        If None there is no tooltip.

        :param Union[str, None] tooltip:
        """
        self.__tooltip = tooltip

    def setFlags(self, flags, column=None):
        """Set the static flags to return.

        Default is ItemIsEnabled for all columns.

        :param int column: The column for which to set the flags
        :param flags: Item flags
        """
        if column is None:
            self.__flags = collections.defaultdict(lambda: flags)
        else:
            self.__flags[column] = flags

    def flags(self, column):
        """Returns flags for given column

        :rtype: int
        """
        return self.__flags[column]


class StaticRow(BaseRow):
    """Row with static data.

    :param tuple display: List of data for DisplayRole for each column
    :param dict roles: Optional mapping of roles to list of data.
    :param children: Iterable of BaseRow to start with (not signaled)
    """

    def __init__(self, display=('', None), roles=None, children=()):
        super(StaticRow, self).__init__(children)
        self._dataByRoles = {} if roles is None else roles
        self._dataByRoles[qt.Qt.DisplayRole] = display

    def data(self, column, role):
        if role in self._dataByRoles:
            data = self._dataByRoles[role]
            if column < len(data):
                return data[column]
        return super(StaticRow, self).data(column, role)

    def columnCount(self):
        return len(self._dataByRoles[qt.Qt.DisplayRole])


class ProxyRow(BaseRow):
    """Provides a node to proxy a data accessible through functions.

    Warning: Only weak reference are kept on fget and fset.

    :param str name: The name of this node
    :param callable fget: A callable returning the data
    :param callable fset:
        An optional callable setting the data with data as a single argument.
    :param notify:
        An optional signal emitted when data has changed.
    :param callable toModelData:
        An optional callable to convert from fget
        callable to data returned by the model.
    :param callable fromModelData:
        An optional callable converting data provided to the model to
        data for fset.
    :param editorHint: Data to provide as UserRole for editor selection/setup
    """

    def __init__(self,
                 name='',
                 fget=None,
                 fset=None,
                 notify=None,
                 toModelData=None,
                 fromModelData=None,
                 editorHint=None):

        super(ProxyRow, self).__init__()
        self.__name = name
        self.__editorHint = editorHint

        assert fget is not None
        self._fget = WeakMethodProxy(fget)
        self._fset = WeakMethodProxy(fset) if fset is not None else None
        if fset is not None:
            self.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsEditable, 1)
        self._toModelData = toModelData
        self._fromModelData = fromModelData

        if notify is not None:
            notify.connect(self._notified)  # TODO support sigItemChanged flags

    def _notified(self, *args, **kwargs):
        """Send update to the model upon signal notifications"""
        index = self.index(column=1)
        model = self.model()
        if model is not None:
            model.dataChanged.emit(index, index)

    def data(self, column, role):
        if column == 0:
            if role == qt.Qt.DisplayRole:
                return self.__name

        elif column == 1:
            if role == qt.Qt.UserRole:  # EditorHint
                return self.__editorHint
            elif role == qt.Qt.DisplayRole or (role == qt.Qt.EditRole and
                                               self._fset is not None):
                data = self._fget()
                if self._toModelData is not None:
                    data = self._toModelData(data)
                return data

        return super(ProxyRow, self).data(column, role)

    def setData(self, column, value, role):
        if role == qt.Qt.EditRole and self._fset is not None:
            if self._fromModelData is not None:
                value = self._fromModelData(value)
            self._fset(value)
            return True

        return super(ProxyRow, self).setData(column, value, role)


class ColorProxyRow(ProxyRow):
    """Provides a proxy to a QColor property.

    The color is returned through the decorative role.

    See :class:`ProxyRow`
    """

    def data(self, column, role):
        if column == 1:  # Show color as decoration, not text
            if role == qt.Qt.DisplayRole:
                return None
            if role == qt.Qt.DecorationRole:
                role = qt.Qt.DisplayRole
        return super(ColorProxyRow, self).data(column, role)


class AngleDegreeRow(ProxyRow):
    """ProxyRow patching display of column 1 to add degree symbol

    See :class:`ProxyRow`
    """

    def __init__(self, *args, **kwargs):
        super(AngleDegreeRow, self).__init__(*args, **kwargs)

    def data(self, column, role):
        if column == 1 and role == qt.Qt.DisplayRole:
            return u'%g°' % super(AngleDegreeRow, self).data(column, role)
        else:
            return super(AngleDegreeRow, self).data(column, role)
