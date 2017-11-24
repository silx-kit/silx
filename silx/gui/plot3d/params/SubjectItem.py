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
This module provides the base class for parameter tree items.
"""

from __future__ import absolute_import

__authors__ = ["D. N."]
__license__ = "MIT"
__date__ = "02/10/2017"


import weakref

from ... import qt


class SubjectItem(qt.QStandardItem):
    """
    Base class for observers items.

    Subclassing:
    ------------
    The following method can/should be reimplemented:
    - _init
    - _pullData
    - _pushData
    - _setModelData
    - _subjectChanged
    - getEditor
    - getSignals
    - leftClicked
    - queryRemove
    - setEditorData

    Also the following attributes are available:
    - editable
    - persistent

    :param subject: object that this item will be observing.
    """

    editable = False
    """ boolean: set to True to make the item editable. """

    persistent = False
    """
    boolean: set to True to make the editor persistent.
        See : Qt.QAbstractItemView.openPersistentEditor
    """

    def __init__(self, subject, *args):

        super(SubjectItem, self).__init__(*args)

        self.setEditable(self.editable)

        self.__subject = None
        self.setSubject(subject)

    def setData(self, value, role=qt.Qt.UserRole, pushData=True):
        """
        Overloaded method from QStandardItem. The pushData keyword tells
        the item to push data to the subject if the role is equal to EditRole.
        This is useful to let this method know if the setData method was called
        internally or from the view.

        :param value: the value ti set to data
        :param role: role in the item
        :param pushData: if True push value in the existing data.
        """
        if role == qt.Qt.EditRole and pushData:
            setValue = self._pushData(value, role)
            if setValue != value:
                value = setValue
        super(SubjectItem, self).setData(value, role)

    def getSubject(self):
        """The subject this item is observing"""
        return None if self.__subject is None else self.__subject()

    def setSubject(self, subject):
        if self.__subject is not None:
            raise ValueError('Subject already set '
                             ' (subject change not supported).')
        if subject is None:
            self.__subject = None
        else:
            self.__subject = weakref.ref(subject)
        if subject is not None:
            self._init()
            self._connectSignals()

    def _connectSignals(self):
        """
        Connects the signals. Called when the subject is set.
        """

        def gen_slot(_sigIdx):
            def slotfn(*args, **kwargs):
                self._subjectChanged(signalIdx=_sigIdx,
                                     args=args,
                                     kwargs=kwargs)
            return slotfn

        if self.getSubject() is not None:
            self.__slots = slots = []

            signals = self.getSignals()

            if signals:
                if not isinstance(signals, (list, tuple)):
                    signals = [signals]
                for sigIdx, signal in enumerate(signals):
                    slot = gen_slot(sigIdx)
                    signal.connect(slot)
                    slots.append((signal, slot))

    def _disconnectSignals(self):
        """
        Disconnects all subject's signal
        """
        if self.__slots:
            for signal, slot in self.__slots:
                try:
                    signal.disconnect(slot)
                except TypeError:
                    pass

    def _enableRow(self, enable):
        """
        Set the enabled state for this cell, or for the whole row
        if this item has a parent.

        :param bool enable: True if we wan't to enable the cell
        """
        parent = self.parent()
        model = self.model()
        if model is None or parent is None:
            # no parent -> no siblings
            self.setEnabled(enable)
            return

        for col in range(model.columnCount()):
            sibling = parent.child(self.row(), col)
            sibling.setEnabled(enable)

    #################################################################
    # Methods to overload
    #################################################################

    def getSignals(self):
        """
        Returns the list of this items subject's signals that
        this item will be listening to.

        :return: list.
        """
        return None

    def _subjectChanged(self, signalIdx=None, args=None, kwargs=None):
        """
        Called when one of the signals is triggered. Default implementation
        just calls _pullData, compares the result to the current value stored
        as Qt.EditRole, and stores the new value if it is different. It also
        stores its str representation as Qt.DisplayRole

        :param signalIdx: index of the triggered signal. The value passed
            is the same as the signal position in the list returned by
            SubjectItem.getSignals.
        :param args: arguments received from the signal
        :param kwargs: keyword arguments received from the signal
        """
        data = self._pullData()
        if data == self.data(qt.Qt.EditRole):
            return
        self.setData(data, role=qt.Qt.DisplayRole, pushData=False)
        self.setData(data, role=qt.Qt.EditRole, pushData=False)

    def _pullData(self):
        """
        Pulls data from the subject.

        :return: subject data
        """
        return None

    def _pushData(self, value, role=qt.Qt.UserRole):
        """
        Pushes data to the subject and returns the actual value that was stored

        :return: the value that was stored
        """
        return value

    def _init(self):
        """
        Called when the subject is set.
        :return:
        """
        self._subjectChanged()

    def getEditor(self, parent, option, index):
        """
        Returns the editor widget used to edit this item's data. The arguments
        are the one passed to the QStyledItemDelegate.createEditor method.

        :param parent: the Qt parent of the editor
        :param option:
        :param index:
        :return:
        """
        return None

    def setEditorData(self, editor):
        """
        This is called by the View's delegate just before the editor is shown,
        its purpose it to setup the editors contents. Return False to use
        the delegate's default behaviour.

        :param editor:
        :return:
        """
        return True

    def _setModelData(self, editor):
        """
        This is called by the View's delegate just before the editor is closed,
        its allows this item to update itself with data from the editor.

        :param editor:
        :return:
        """
        return False

    def queryRemove(self, view=None):
        """
        This is called by the view to ask this items if it (the view) can
        remove it. Return True to let the view know that the item can be
        removed.

        :param view:
        :return:
        """
        return False

    def leftClicked(self):
        """
        This method is called by the view when the item's cell if left clicked.

        :return:
        """
        pass
