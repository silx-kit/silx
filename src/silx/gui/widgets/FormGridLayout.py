# /*##########################################################################
#
# Copyright (c) 2022 European Synchrotron Radiation Facility
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
"""This module provides a form layout for QWidget: :class:`FormGridLayout`.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "29/09/2022"


import typing
from .. import qt


class FormGridLayout(qt.QGridLayout):
    """A layout with the API of :class:`qt.QFormLayout` based on a :class:`qt.QGridLayout`.

    This allow a bit more flexibility, like allow vertical expanding
    of the rows.
    """
    def __init__(self, parent):
        super(FormGridLayout, self).__init__(parent)
        self.__cursor = 0

    def _addCell(self, something, row, column, rowSpan=1, columnSpan=1):
        if isinstance(something, qt.QLayout):
            self.addLayout(something, row, column, rowSpan, columnSpan)
        else:
            if isinstance(something, str):
                something = qt.QLabel(something)
            self.addWidget(something, row, column, rowSpan, columnSpan)

    def addRow(self, label: typing.Union[str, qt.QWidget, qt.QLayout], field: typing.Union[None, qt.QWidget, qt.QLayout] = None):
        """
        Adds a new row to the bottom of this form layout.

        If field is defined, the given label and field are added.

        Else, the label is a widget and spans both columns.
        """
        if field is None:
            self._addCell(label, self.__cursor, 0, 1, 2)
        else:
            self._addCell(label, self.__cursor, 0)
            self._addCell(field, self.__cursor, 1)
        self.__cursor += 1

    def addItem(self, item: qt.QLayoutItem):
        """
        Adds a new layout item to the bottom of this form layout.
        """
        super(FormGridLayout, self).addItem(item)
        self.__cursor += 1
