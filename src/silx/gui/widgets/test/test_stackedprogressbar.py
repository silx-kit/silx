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
"""Tests for StackedProgressBar"""

__license__ = "MIT"

import pytest
from silx.gui import qt
from silx.gui.widgets.StackedProgressBar import StackedProgressBar


@pytest.fixture
def stackedProgressBar(qWidgetFactory):
    yield qWidgetFactory(StackedProgressBar)


def test_show(qapp_utils, stackedProgressBar: StackedProgressBar):
    pass


def test_value(qapp_utils, stackedProgressBar: StackedProgressBar):
    stackedProgressBar.setRange(0, 100)
    stackedProgressBar.setProgressItem("foo", value=0)
    stackedProgressBar.setProgressItem("foo", value=50)
    stackedProgressBar.setProgressItem("foo", value=100)


def test_animation(qapp_utils, stackedProgressBar: StackedProgressBar):
    stackedProgressBar.setRange(0, 100)
    stackedProgressBar.setProgressItem("foo", value=0, striped=True, animated=True)
    stackedProgressBar.setProgressItem("foo", value=50)
    stackedProgressBar.setProgressItem("foo", value=100)


def test_stack(qapp_utils, stackedProgressBar: StackedProgressBar):
    stackedProgressBar.setRange(0, 100)
    stackedProgressBar.setProgressItem("foo1", value=10, color=qt.QColor("#FF0000"))
    stackedProgressBar.setProgressItem("foo2", value=50, color=qt.QColor("#00FF00"))
    stackedProgressBar.setProgressItem("foo3", value=20, color=qt.QColor("#0000FF"))
