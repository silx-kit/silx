# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2017 European Synchrotron Radiation Facility
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
This module provides a tree widget to set/view parameters of a ScalarFieldView.
"""

from __future__ import absolute_import

__authors__ = ["D. N."]
__license__ = "MIT"
__date__ = "02/10/2017"


from silx.gui import qt

from .SubjectItem import SubjectItem


class TreeViewModelBase(qt.QStandardItemModel):
    """Default model for parameter TreeView"""

    def __init__(self, parent=None):
        super(TreeViewModelBase, self).__init__(parent)
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(['Name', 'Value'])


class ItemDelegate(qt.QStyledItemDelegate):
    """
    Delegate for the QTreeView filled with SubjectItems.
    """

    sigDelegateEvent = qt.Signal(str)

    def __init__(self, parent=None):
        super(ItemDelegate, self).__init__(parent)

    def createEditor(self, parent, option, index):
        item = index.model().itemFromIndex(index)
        if item:
            if isinstance(item, SubjectItem):
                editor = item.getEditor(parent, option, index)
                if editor:
                    editor.setAutoFillBackground(True)
                    if hasattr(editor, 'sigViewTask'):
                        editor.sigViewTask.connect(self.__viewTask)
                    return editor

        editor = super(ItemDelegate, self).createEditor(parent,
                                                        option,
                                                        index)
        return editor

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def setEditorData(self, editor, index):
        item = index.model().itemFromIndex(index)
        if item:
            if isinstance(item, SubjectItem) and item.setEditorData(editor):
                return
        super(ItemDelegate, self).setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        item = index.model().itemFromIndex(index)
        if isinstance(item, SubjectItem) and item._setModelData(editor):
            return
        super(ItemDelegate, self).setModelData(editor, model, index)

    def __viewTask(self, task):
        self.sigDelegateEvent.emit(task)


class TreeView(qt.QTreeView):
    """
    TreeView displaying SubjectItems.
    """

    def __init__(self, parent=None):
        super(TreeView, self).__init__(parent)
        self.__openedIndex = None

        self.setIconSize(qt.QSize(16, 16))

        header = self.header()
        if hasattr(header, 'setSectionResizeMode'):  # Qt5
            header.setSectionResizeMode(qt.QHeaderView.ResizeToContents)
        else:  # Qt4
            header.setResizeMode(qt.QHeaderView.ResizeToContents)

        delegate = ItemDelegate()
        self.setItemDelegate(delegate)
        self.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.setSelectionMode(qt.QAbstractItemView.SingleSelection)

        self.clicked.connect(self.__clicked)

    def setModel(self, model):
        """
        Reimplementation of the QTreeView.setModel method. It connects the
        rowsRemoved signal and opens the persistent editors.

        :param qt.QStandardItemModel model: the model
        """

        prevModel = self.model()
        if prevModel:
            self.__openPersistentEditors(qt.QModelIndex(), False)
            try:
                prevModel.rowsRemoved.disconnect(self.rowsRemoved)
            except TypeError:
                pass

        super(TreeView, self).setModel(model)
        model.rowsRemoved.connect(self.rowsRemoved)
        self.__openPersistentEditors(qt.QModelIndex())

    def __openPersistentEditors(self, parent=None, openEditor=True):
        """
        Opens or closes the items persistent editors.

        :param qt.QModelIndex parent: starting index, or None if the whole tree
            is to be considered.
        :param bool openEditor: True to open the editors, False to close them.
        """
        model = self.model()

        if not model:
            return

        if not parent or not parent.isValid():
            parent = self.model().invisibleRootItem().index()

        if openEditor:
            meth = self.openPersistentEditor
        else:
            meth = self.closePersistentEditor

        curParent = parent
        children = [model.index(row, 0, curParent)
                    for row in range(model.rowCount(curParent))]

        columnCount = model.columnCount()

        while len(children) > 0:
            curParent = children.pop(-1)

            children.extend([model.index(row, 0, curParent)
                             for row in range(model.rowCount(curParent))])

            for colIdx in range(columnCount):
                sibling = model.sibling(curParent.row(),
                                        colIdx,
                                        curParent)
                item = model.itemFromIndex(sibling)
                if isinstance(item, SubjectItem) and item.persistent:
                    meth(sibling)

    def rowsAboutToBeRemoved(self, parent, start, end):
        """
        Reimplementation of the QTreeView.rowsAboutToBeRemoved. Closes all
        persistent editors under parent.

        :param qt.QModelIndex parent: Parent index
        :param int start: Start index from parent index (inclusive)
        :param int end: End index from parent index (inclusive)
        """
        self.__openPersistentEditors(parent, False)
        super(TreeView, self).rowsAboutToBeRemoved(parent, start, end)

    def rowsRemoved(self, parent, start, end):
        """
        Called when QTreeView.rowsRemoved is emitted. Opens all persistent
        editors under parent.

        :param qt.QModelIndex parent: Parent index
        :param int start: Start index from parent index (inclusive)
        :param int end: End index from parent index (inclusive)
        """
        super(TreeView, self).rowsRemoved(parent, start, end)
        self.__openPersistentEditors(parent, True)

    def rowsInserted(self, parent, start, end):
        """
        Reimplementation of the QTreeView.rowsInserted. Opens all persistent
        editors under parent.

        :param qt.QModelIndex parent: Parent index
        :param int start: Start index from parent index
        :param int end: End index from parent index
        """
        self.__openPersistentEditors(parent, False)
        super(TreeView, self).rowsInserted(parent, start, end)
        self.__openPersistentEditors(parent)

    def keyReleaseEvent(self, event):
        """
        Reimplementation of the QTreeView.keyReleaseEvent.
        At the moment only Key_Delete is handled. It calls the selected item's
        queryRemove method, and deleted the item if needed.

        :param qt.QKeyEvent event: A key event
        """

        # TODO : better filtering
        key = event.key()
        modifiers = event.modifiers()

        if key == qt.Qt.Key_Delete and modifiers == qt.Qt.NoModifier:
            self.__removeIsosurfaces()

        super(TreeView, self).keyReleaseEvent(event)

    def __clicked(self, index):
        """
        Called when the QTreeView.clicked signal is emitted. Calls the item's
        leftClick method.

        :param qt.QIndex index: An index
        """
        item = self.model().itemFromIndex(index)
        if isinstance(item, SubjectItem):
            item.leftClicked()
