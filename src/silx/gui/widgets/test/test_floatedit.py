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
"""Tests for FloatEdit"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "09/05/2023"

import pytest
import weakref
from silx.gui import qt
from silx.gui.widgets.FloatEdit import FloatEdit
from silx.gui.utils import validators


@pytest.fixture
def floatEdit(qapp, qapp_utils):
    widget = FloatEdit()
    widget.setAttribute(qt.Qt.WA_DeleteOnClose)
    yield widget
    widget.close()
    ref = weakref.ref(widget)
    widget = None
    qapp_utils.qWaitForDestroy(ref)


def test_show(qapp_utils, floatEdit):
    qapp_utils.qWaitForWindowExposed(floatEdit)


def test_value(floatEdit):
    floatEdit.setValue(1.5)
    assert floatEdit.value() == 1.5


def test_none_value(floatEdit):
    v = validators.AdvancedDoubleValidator()
    v.setAllowEmpty(True)
    floatEdit.setValidator(v)
    floatEdit.setValue(None)
    assert floatEdit.value() is None
    floatEdit.setText("")
    assert floatEdit.value() is None


def test_original_value(floatEdit):
    """
    Check that we can retrieve the original value while it was not edited by the user.
    """
    floatEdit.setValue(0.123456789)
    assert floatEdit.value() == 0.123456789
    floatEdit.setCursorPosition(2)
    floatEdit.insert("1")
    assert floatEdit.value() == pytest.approx(0.1123456)


def test_quantity_value(floatEdit):
    """
    Check that the widget supports quantity validator.
    """
    v = validators.DoublePintValidator()
    floatEdit.setValidator(v)

    floatEdit.setValue((0.12, "mm"))
    assert floatEdit.value() == (0.12, "mm")
    floatEdit.setCursorPosition(3)
    floatEdit.insert("1")
    assert floatEdit.value() == (0.112, "mm")
