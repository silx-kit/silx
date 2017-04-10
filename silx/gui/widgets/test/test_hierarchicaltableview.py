# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "07/04/2017"

import unittest

from .. import HierarchicalTableView
from ...test.utils import TestCaseQt
from silx.gui import qt


class TableModel(HierarchicalTableView.HierarchicalTableModel):

    def __init__(self, parent):
        HierarchicalTableView.HierarchicalTableModel.__init__(self, parent)
        self.__content = {}

    def rowCount(self, parent=qt.QModelIndex()):
        return 3

    def columnCount(self, parent=qt.QModelIndex()):
        return 3

    def setData1(self):
        if qt.qVersion() > "4.6":
            self.beginResetModel()
        else:
            self.reset()

        content = {}
        content[0, 0] = ("title", True, (1, 3))
        content[0, 1] = ("a", True, (2, 1))
        content[1, 1] = ("b", False, (1, 2))
        content[1, 2] = ("c", False, (1, 1))
        content[2, 2] = ("d", False, (1, 1))
        self.__content = content
        if qt.qVersion() > "4.6":
            self.endResetModel()

    def data(self, index, role=qt.Qt.DisplayRole):
        if not index.isValid():
            return None
        cell = self.__content.get((index.column(), index.row()), None)
        if cell is None:
            return None

        if role == self.SpanRole:
            return cell[2]
        elif role == self.IsHeaderRole:
            return cell[1]
        elif role == qt.Qt.DisplayRole:
            return cell[0]
        return None


class TestHierarchicalTableView(TestCaseQt):
    """Test for HierarchicalTableView"""

    def testEmpty(self):
        widget = HierarchicalTableView.HierarchicalTableView()
        widget.show()
        self.qWaitForWindowExposed(widget)

    def testModel(self):
        widget = HierarchicalTableView.HierarchicalTableView()
        model = TableModel(widget)
        # set the data before using the model into the widget
        model.setData1()
        widget.setModel(model)
        span = widget.rowSpan(0, 0), widget.columnSpan(0, 0)
        self.assertEqual(span, (1, 3))
        widget.show()
        self.qWaitForWindowExposed(widget)

    def testModelUpdate(self):
        widget = HierarchicalTableView.HierarchicalTableView()
        model = TableModel(widget)
        widget.setModel(model)
        # set the data after using the model into the widget
        model.setData1()
        span = widget.rowSpan(0, 0), widget.columnSpan(0, 0)
        self.assertEqual(span, (1, 3))


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loader(TestHierarchicalTableView))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
