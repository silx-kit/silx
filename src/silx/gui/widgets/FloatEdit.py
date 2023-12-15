# /*##########################################################################
#
# Copyright (c) 2004-2023 European Synchrotron Radiation Facility
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
"""Module contains a float editor
"""
from __future__ import annotations


__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "02/10/2017"

from .. import qt


class FloatEdit(qt.QLineEdit):
    """Field to edit a float value.

    The value can be modified with :meth:`value` and :meth:`setValue`.

    The property :meth:`widgetResizable` allow to change the default
    behaviour in order to automatically resize the widget to the displayed value.
    Use :meth:`setMinimumWidth` to enforce the minimum width.

    :param parent: Parent of the widget
    :param value: The value to set the QLineEdit to.
    """

    _QLineEditPrivateHorizontalMargin = 2
    """Constant from Qt source code"""

    def __init__(self, parent: qt.QWidget | None = None, value: float | None = None):
        qt.QLineEdit.__init__(self, parent)
        validator = qt.QDoubleValidator(self)
        self.__widgetResizable: bool = False
        self.__minimumWidth = 30
        """Store the minimum width requested by the user, the real one is
        dynamic"""
        self.setValidator(validator)
        self.setAlignment(qt.Qt.AlignRight)
        self.textChanged.connect(self.__textChanged)
        if value is not None:
            self.setValue(value)

    def value(self) -> float:
        """Return the QLineEdit current value as a float."""
        text = self.text()
        value, validated = self.validator().locale().toDouble(text)
        if not validated:
            self.setValue(value)
        return value

    def setValue(self, value: float):
        """Set the current value of the LineEdit

        :param value: The value to set the QLineEdit to.
        """
        locale = self.validator().locale()
        if qt.BINDING == "PySide6":
            # Fix for PySide6 not selecting the right method
            text = locale.toString(float(value), "g")
        else:
            text = locale.toString(float(value))

        self.setText(text)
        if self.__widgetResizable:
            self.__forceMinimumWidthFromContent()

    def __textChanged(self, text: str):
        if self.__widgetResizable:
            self.__forceMinimumWidthFromContent()

    def widgetResizable(self) -> bool:
        """
        Returns whether or not the widget auto resizes itself based on it's content
        """
        return self.__widgetResizable

    def setWidgetResizable(self, resizable: bool):
        """
        If true, the widget will automatically resize itself to its displayed content.

        This avoids to have to scroll to see the widget's content, and allow to take
        advantage of extra space.
        """
        if self.__widgetResizable == resizable:
            return
        self.__widgetResizable = resizable
        self.updateGeometry()
        if resizable:
            self.__forceMinimumWidthFromContent()
        else:
            qt.QLineEdit.setMinimumWidth(self, self.__minimumWidth)

    def __minimumWidthFromContent(self) -> int:
        """Minimum size for the widget to properly read the actual number"""
        text = self.text()
        font = self.font()
        metrics = qt.QFontMetrics(font)
        margins = self.textMargins()
        width = (
            metrics.horizontalAdvance(text)
            + self._QLineEditPrivateHorizontalMargin * 2
            + margins.left()
            + margins.right()
        )
        width = max(self.__minimumWidth, width)
        opt = qt.QStyleOptionFrame()
        self.initStyleOption(opt)
        s = self.style().sizeFromContents(
            qt.QStyle.CT_LineEdit, opt, qt.QSize(width, self.height())
        )
        return s.width()

    def sizeHint(self) -> qt.QSize:
        sizeHint = qt.QLineEdit.sizeHint(self)
        if not self.__widgetResizable:
            return sizeHint
        width = self.__minimumWidthFromContent()
        return qt.QSize(width, sizeHint.height())

    def __forceMinimumWidthFromContent(self):
        width = self.__minimumWidthFromContent()
        qt.QLineEdit.setMinimumWidth(self, width)
        self.updateGeometry()

    def setMinimumWidth(self, width: int):
        self.__minimumWidth = width
        qt.QLineEdit.setMinimumWidth(self, width)
        self.updateGeometry()

    def minimumWidth(self) -> int:
        """Returns the user defined minimum width."""
        return self.__minimumWidth
