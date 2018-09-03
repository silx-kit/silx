# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""This module provides a flow layout for QWidget: :class:`FlowLayout`.
"""

from __future__ import division

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "20/07/2018"


from .. import qt


class FlowLayout(qt.QLayout):
    """Layout widgets on (possibly) multiple lines in the available width.

    See Qt :class:`QLayout` for API documentation.

    Adapted from C++ `Qt FlowLayout example
    <http://doc.qt.io/qt-5/qtwidgets-layouts-flowlayout-example.html>`_

    :param QWidget parent: See :class:`QLayout`
    """

    def __init__(self, parent=None):
        super(FlowLayout, self).__init__(parent)
        self._items = []
        self._horizontalSpacing = -1
        self._verticalSpacing = -1

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        else:
            return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        else:
            return None

    def expandingDirections(self):
        return 0

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(qt.QRect(0, 0, width, 0), test=True)

    def setGeometry(self, rect):
        super(FlowLayout, self).setGeometry(rect)
        self._layout(rect)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = qt.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())

        left, top, right, bottom = self.getContentsMargins()
        size += qt.QSize(left + right, top + bottom)
        return size

    def _layout(self, rect, test=False):
        left, top, right, bottom = self.getContentsMargins()
        effectiveRect = rect.adjusted(left, top, -right, -bottom)
        x, y = effectiveRect.x(), effectiveRect.y()
        lineHeight = 0

        for item in self._items:
            widget = item.widget()
            spaceX = self.horizontalSpacing()
            if spaceX == -1:
                spaceX = widget.style().layoutSpacing(
                    qt.QSizePolicy.PushButton,
                    qt.QSizePolicy.PushButton,
                    qt.Qt.Horizontal)
            spaceY = self.verticalSpacing()
            if spaceY == -1:
                spaceY = widget.style().layoutSpacing(
                    qt.QSizePolicy.PushButton,
                    qt.QSizePolicy.PushButton,
                    qt.Qt.Vertical)

            nextX = x + item.sizeHint().width() + spaceX
            if (nextX - spaceX) > effectiveRect.right() and lineHeight > 0:
                x = effectiveRect.x()
                y += lineHeight + spaceY
                nextX = x + item.sizeHint().width() + spaceX
                lineHeight = 0

            if not test:
                item.setGeometry(qt.QRect(qt.QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y() + bottom

    def setHorizontalSpacing(self, spacing):
        """Set the horizontal spacing between widgets laid out side by side

        :param int spacing:
        """
        self._horizontalSpacing = spacing
        self.update()

    def horizontalSpacing(self):
        """Returns the horizontal spacing between widgets laid out side by side

        :rtype: int
        """
        if self._horizontalSpacing >= 0:
            return self._horizontalSpacing
        else:
            return self._smartSpacing(qt.QStyle.PM_LayoutHorizontalSpacing)

    def setVerticalSpacing(self, spacing):
        """Set the vertical spacing between lines

        :param int spacing:
        """
        self._verticalSpacing = spacing
        self.update()

    def verticalSpacing(self):
        """Returns the vertical spacing between lines

        :rtype: int
        """
        if self._verticalSpacing >= 0:
            return self._verticalSpacing
        else:
            return self._smartSpacing(qt.QStyle.PM_LayoutVerticalSpacing)

    def _smartSpacing(self, pm):
        parent = self.parent()
        if parent is None:
            return -1
        if parent.isWidgetType():
            return parent.style().pixelMetric(pm, None, parent)
        else:
            return parent.spacing()
