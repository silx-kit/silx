# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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

from __future__ import division

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "02/10/2017"

from .. import qt


class FloatEdit(qt.QLineEdit):
    """Field to edit a float value.

    :param parent: See :class:`QLineEdit`
    :param float value: The value to set the QLineEdit to.
    """
    def __init__(self, parent=None, value=None):
        qt.QLineEdit.__init__(self, parent)
        validator = qt.QDoubleValidator(self)
        self.setValidator(validator)
        self.setAlignment(qt.Qt.AlignRight)
        if value is not None:
            self.setValue(value)

    def value(self):
        """Return the QLineEdit current value as a float."""
        text = self.text()
        value, validated = self.validator().locale().toDouble(text)
        if not validated:
            self.setValue(value)
        return value

    def setValue(self, value):
        """Set the current value of the LineEdit

        :param float value: The value to set the QLineEdit to.
        """
        text = self.validator().locale().toString(float(value))
        self.setText(text)
