# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Test TableWidget"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "05/12/2016"


import unittest
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.widgets.TableWidget import TableWidget


class TestTableWidget(TestCaseQt):
    def setUp(self):
        super(TestTableWidget, self).setUp()
        self._result = []

    def testShow(self):
        table = TableWidget()
        table.setColumnCount(10)
        table.setRowCount(7)
        table.enableCut()
        table.enablePaste()
        table.show()
        table.hide()
        self.qapp.processEvents()
