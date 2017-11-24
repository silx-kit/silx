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
This module provides a parameter tree color item and associated color editor.
"""

from __future__ import absolute_import

__authors__ = ["D. N."]
__license__ = "MIT"
__date__ = "02/10/2017"

import sys

from ... import qt
from .SubjectItem import SubjectItem


class ColorEditor(qt.QWidget):
    """
    QColor editor.
    """

    sigColorChanged = qt.Signal(object)
    """Signal emitted when the color changed in the dialog."""

    def getColor(self):
        """Returns current color of the editor

        :rtype: QColor
        """
        return qt.QColor(self.__color)

    def setColor(self, color):
        """Set the editor current color

        :param color: The current color
        """
        self._setColor(color)
        self.__previousColor = color

    color = property(lambda self: qt.QColor(self.__color))

    def __init__(self, *args, **kwargs):
        super(ColorEditor, self).__init__(*args, **kwargs)
        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        button = qt.QToolButton()
        icon = qt.QIcon(qt.QPixmap(32, 32))
        button.setIcon(icon)
        layout.addWidget(button)
        button.clicked.connect(self.__showColorDialog)
        layout.addStretch(1)

        self.__color = None
        self.__previousColor = None

    def sizeHint(self):
        return qt.QSize(0, 0)

    def _setColor(self, qColor):
        button = self.findChild(qt.QToolButton)
        pixmap = qt.QPixmap(32, 32)
        pixmap.fill(qColor)
        button.setIcon(qt.QIcon(pixmap))
        self.__color = qColor

    def __showColorDialog(self):
        dialog = qt.QColorDialog(parent=self)
        if sys.platform == 'darwin':
            # Use of native color dialog on macos might cause problems
            dialog.setOption(qt.QColorDialog.DontUseNativeDialog, True)

        self.__previousColor = self.__color
        dialog.setAttribute(qt.Qt.WA_DeleteOnClose)
        dialog.setModal(True)
        dialog.currentColorChanged.connect(self.__colorChanged)
        dialog.finished.connect(self.__dialogClosed)
        dialog.show()

    def __colorChanged(self, color):
        self.__color = color
        self._setColor(color)
        self.sigColorChanged.emit(color)

    def __dialogClosed(self, result):
        if result == qt.QDialog.Rejected:
            self.__colorChanged(self.__previousColor)
        self.__previousColor = None


class ColorItem(SubjectItem):
    """Parameter tree editable color item."""

    editable = True
    persistent = True

    def getEditor(self, parent, option, index):
        editor = ColorEditor(parent)
        editor.setColor(self.getColor())

        # Wrapping call in lambda is a workaround for PySide with Python 3
        editor.sigColorChanged.connect(
            lambda color: self._editorSlot(color))
        return editor

    def _editorSlot(self, color):
        self.setData(color, qt.Qt.EditRole)

    def _pushData(self, value, role=qt.Qt.UserRole):
        self.setColor(value)
        return self.getColor()

    def _pullData(self):
        self.getColor()

    def setColor(self, color):
        """Set the current color

        Override to implement actual color setter

        :param QColor color: The current color
        """
        pass

    def getColor(self):
        """Return the current color

        Override to implement color getter

        :rtype: QColor
        """
        raise NotImplementedError(
            "This method must be implemented in subclass")
