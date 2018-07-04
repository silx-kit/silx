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
Python implementation of classes which are not provided by default by PySide.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "17/01/2017"


from PySide.QtGui import QAbstractProxyModel
from PySide.QtCore import QModelIndex
from PySide.QtCore import Qt
from PySide.QtGui import QItemSelection
from PySide.QtGui import QItemSelectionRange


class QIdentityProxyModel(QAbstractProxyModel):
    """Python translation of the source code of Qt c++ file"""

    def __init__(self, parent=None):
        super(QIdentityProxyModel, self).__init__(parent)
        self.__ignoreNextLayoutAboutToBeChanged = False
        self.__ignoreNextLayoutChanged = False
        self.__persistentIndexes = []

    def columnCount(self, parent):
        parent = self.mapToSource(parent)
        return self.sourceModel().columnCount(parent)

    def dropMimeData(self, data, action, row, column, parent):
        parent = self.mapToSource(parent)
        return self.sourceModel().dropMimeData(data, action, row, column, parent)

    def index(self, row, column, parent=QModelIndex()):
        parent = self.mapToSource(parent)
        i = self.sourceModel().index(row, column, parent)
        return self.mapFromSource(i)

    def insertColumns(self, column, count, parent=QModelIndex()):
        parent = self.mapToSource(parent)
        return self.sourceModel().insertColumns(column, count, parent)

    def insertRows(self, row, count, parent=QModelIndex()):
        parent = self.mapToSource(parent)
        return self.sourceModel().insertRows(row, count, parent)

    def mapFromSource(self, sourceIndex):
        if self.sourceModel() is None or not sourceIndex.isValid():
            return QModelIndex()
        index = self.createIndex(sourceIndex.row(), sourceIndex.column(), sourceIndex.internalPointer())
        return index

    def mapSelectionFromSource(self, sourceSelection):
        proxySelection = QItemSelection()
        if self.sourceModel() is None:
            return proxySelection

        cursor = sourceSelection.constBegin()
        end = sourceSelection.constEnd()
        while cursor != end:
            topLeft = self.mapFromSource(cursor.topLeft())
            bottomRight = self.mapFromSource(cursor.bottomRight())
            proxyRange = QItemSelectionRange(topLeft, bottomRight)
            proxySelection.append(proxyRange)
            cursor += 1
        return proxySelection

    def mapSelectionToSource(self, proxySelection):
        sourceSelection = QItemSelection()
        if self.sourceModel() is None:
            return sourceSelection

        cursor = proxySelection.constBegin()
        end = proxySelection.constEnd()
        while cursor != end:
            topLeft = self.mapToSource(cursor.topLeft())
            bottomRight = self.mapToSource(cursor.bottomRight())
            sourceRange = QItemSelectionRange(topLeft, bottomRight)
            sourceSelection.append(sourceRange)
            cursor += 1
        return sourceSelection

    def mapToSource(self, proxyIndex):
        if self.sourceModel() is None or not proxyIndex.isValid():
            return QModelIndex()
        return self.sourceModel().createIndex(proxyIndex.row(), proxyIndex.column(), proxyIndex.internalPointer())

    def match(self, start, role, value, hits=1, flags=Qt.MatchFlags(Qt.MatchStartsWith | Qt.MatchWrap)):
        if self.sourceModel() is None:
            return []

        start = self.mapToSource(start)
        sourceList = self.sourceModel().match(start, role, value, hits, flags)
        proxyList = []
        for cursor in sourceList:
            proxyList.append(self.mapFromSource(cursor))
        return proxyList

    def parent(self, child):
        sourceIndex = self.mapToSource(child)
        sourceParent = sourceIndex.parent()
        index = self.mapFromSource(sourceParent)
        return index

    def removeColumns(self, column, count, parent=QModelIndex()):
        parent = self.mapToSource(parent)
        return self.sourceModel().removeColumns(column, count, parent)

    def removeRows(self, row, count, parent=QModelIndex()):
        parent = self.mapToSource(parent)
        return self.sourceModel().removeRows(row, count, parent)

    def rowCount(self, parent=QModelIndex()):
        parent = self.mapToSource(parent)
        return self.sourceModel().rowCount(parent)

    def setSourceModel(self, newSourceModel):
        """Bind and unbind the source model events"""
        self.beginResetModel()

        sourceModel = self.sourceModel()
        if sourceModel is not None:
            sourceModel.rowsAboutToBeInserted.disconnect(self.__rowsAboutToBeInserted)
            sourceModel.rowsInserted.disconnect(self.__rowsInserted)
            sourceModel.rowsAboutToBeRemoved.disconnect(self.__rowsAboutToBeRemoved)
            sourceModel.rowsRemoved.disconnect(self.__rowsRemoved)
            sourceModel.rowsAboutToBeMoved.disconnect(self.__rowsAboutToBeMoved)
            sourceModel.rowsMoved.disconnect(self.__rowsMoved)
            sourceModel.columnsAboutToBeInserted.disconnect(self.__columnsAboutToBeInserted)
            sourceModel.columnsInserted.disconnect(self.__columnsInserted)
            sourceModel.columnsAboutToBeRemoved.disconnect(self.__columnsAboutToBeRemoved)
            sourceModel.columnsRemoved.disconnect(self.__columnsRemoved)
            sourceModel.columnsAboutToBeMoved.disconnect(self.__columnsAboutToBeMoved)
            sourceModel.columnsMoved.disconnect(self.__columnsMoved)
            sourceModel.modelAboutToBeReset.disconnect(self.__modelAboutToBeReset)
            sourceModel.modelReset.disconnect(self.__modelReset)
            sourceModel.dataChanged.disconnect(self.__dataChanged)
            sourceModel.headerDataChanged.disconnect(self.__headerDataChanged)
            sourceModel.layoutAboutToBeChanged.disconnect(self.__layoutAboutToBeChanged)
            sourceModel.layoutChanged.disconnect(self.__layoutChanged)

        super(QIdentityProxyModel, self).setSourceModel(newSourceModel)

        sourceModel = self.sourceModel()
        if sourceModel is not None:
            sourceModel.rowsAboutToBeInserted.connect(self.__rowsAboutToBeInserted)
            sourceModel.rowsInserted.connect(self.__rowsInserted)
            sourceModel.rowsAboutToBeRemoved.connect(self.__rowsAboutToBeRemoved)
            sourceModel.rowsRemoved.connect(self.__rowsRemoved)
            sourceModel.rowsAboutToBeMoved.connect(self.__rowsAboutToBeMoved)
            sourceModel.rowsMoved.connect(self.__rowsMoved)
            sourceModel.columnsAboutToBeInserted.connect(self.__columnsAboutToBeInserted)
            sourceModel.columnsInserted.connect(self.__columnsInserted)
            sourceModel.columnsAboutToBeRemoved.connect(self.__columnsAboutToBeRemoved)
            sourceModel.columnsRemoved.connect(self.__columnsRemoved)
            sourceModel.columnsAboutToBeMoved.connect(self.__columnsAboutToBeMoved)
            sourceModel.columnsMoved.connect(self.__columnsMoved)
            sourceModel.modelAboutToBeReset.connect(self.__modelAboutToBeReset)
            sourceModel.modelReset.connect(self.__modelReset)
            sourceModel.dataChanged.connect(self.__dataChanged)
            sourceModel.headerDataChanged.connect(self.__headerDataChanged)
            sourceModel.layoutAboutToBeChanged.connect(self.__layoutAboutToBeChanged)
            sourceModel.layoutChanged.connect(self.__layoutChanged)

        self.endResetModel()

    def __columnsAboutToBeInserted(self, parent, start, end):
        parent = self.mapFromSource(parent)
        self.beginInsertColumns(parent, start, end)

    def __columnsAboutToBeMoved(self, sourceParent, sourceStart, sourceEnd, destParent, dest):
        sourceParent = self.mapFromSource(sourceParent)
        destParent = self.mapFromSource(destParent)
        self.beginMoveColumns(sourceParent, sourceStart, sourceEnd, destParent, dest)

    def __columnsAboutToBeRemoved(self, parent, start, end):
        parent = self.mapFromSource(parent)
        self.beginRemoveColumns(parent, start, end)

    def __columnsInserted(self, parent, start, end):
        self.endInsertColumns()

    def __columnsMoved(self, sourceParent, sourceStart, sourceEnd, destParent, dest):
        self.endMoveColumns()

    def __columnsRemoved(self, parent, start, end):
        self.endRemoveColumns()

    def __dataChanged(self, topLeft, bottomRight):
        topLeft = self.mapFromSource(topLeft)
        bottomRight = self.mapFromSource(bottomRight)
        self.dataChanged(topLeft, bottomRight)

    def __headerDataChanged(self, orientation, first, last):
        self.headerDataChanged(orientation, first, last)

    def __layoutAboutToBeChanged(self):
        """Store persistent indexes"""
        if self.__ignoreNextLayoutAboutToBeChanged:
            return

        for proxyPersistentIndex in self.persistentIndexList():
            self.__proxyIndexes.append()
            sourcePersistentIndex = self.mapToSource(proxyPersistentIndex)
            mapping = proxyPersistentIndex, sourcePersistentIndex
            self.__persistentIndexes.append(mapping)

        self.layoutAboutToBeChanged()

    def __layoutChanged(self):
        """Restore persistent indexes"""
        if self.__ignoreNextLayoutChanged:
            return

        for mapping in self.__persistentIndexes:
            proxyIndex, sourcePersistentIndex = mapping
            sourcePersistentIndex = self.mapFromSource(sourcePersistentIndex)
            self.changePersistentIndex(proxyIndex, sourcePersistentIndex)

        self.__persistentIndexes = []

        self.layoutChanged()

    def __modelAboutToBeReset(self):
        self.beginResetModel()

    def __modelReset(self):
        self.endResetModel()

    def __rowsAboutToBeInserted(self, parent, start, end):
        parent = self.mapFromSource(parent)
        self.beginInsertRows(parent, start, end)

    def __rowsAboutToBeMoved(self, sourceParent, sourceStart, sourceEnd, destParent, dest):
        sourceParent = self.mapFromSource(sourceParent)
        destParent = self.mapFromSource(destParent)
        self.beginMoveRows(sourceParent, sourceStart, sourceEnd, destParent, dest)

    def __rowsAboutToBeRemoved(self, parent, start, end):
        parent = self.mapFromSource(parent)
        self.beginRemoveRows(parent, start, end)

    def __rowsInserted(self, parent, start, end):
        self.endInsertRows()

    def __rowsMoved(self, sourceParent, sourceStart, sourceEnd, destParent, dest):
        self.endMoveRows()

    def __rowsRemoved(self, parent, start, end):
        self.endRemoveRows()
