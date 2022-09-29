# /*##########################################################################
#
# Copyright (c) 2020 European Synchrotron Radiation Facility
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
"""Tests for LegendIconWidget"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "23/10/2020"

import unittest

from silx.gui import qt
from silx.gui.widgets.LegendIconWidget import LegendIconWidget
from silx.gui.utils.testutils import TestCaseQt
from silx.utils.testutils import ParametricTestCase


class TestLegendIconWidget(TestCaseQt, ParametricTestCase):
    """Tests for TestRangeSlider"""

    def setUp(self):
        self.widget = LegendIconWidget()
        self.widget.show()
        self.qWaitForWindowExposed(self.widget)

    def tearDown(self):
        self.widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.widget.close()
        del self.widget
        self.qapp.processEvents()

    def testCreate(self):
        self.qapp.processEvents()

    def testColormap(self):
        self.widget.setColormap("viridis")
        self.qapp.processEvents()

    def testSymbol(self):
        self.widget.setSymbol("o")
        self.widget.setSymbolColormap("viridis")
        self.qapp.processEvents()
