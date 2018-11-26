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
__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "05/12/2016"

import unittest

from .. import PeriodicTable
from silx.gui.utils.testutils import TestCaseQt
from silx.gui import qt


class TestPeriodicTable(TestCaseQt):
    """Basic test for ArrayTableWidget with a numpy array"""

    def testShow(self):
        """basic test (instantiation done in setUp)"""
        pt = PeriodicTable.PeriodicTable()
        pt.show()
        self.qWaitForWindowExposed(pt)

    def testSelectable(self):
        """basic test (instantiation done in setUp)"""
        pt = PeriodicTable.PeriodicTable(selectable=True)
        self.assertTrue(pt.selectable)

    def testCustomElements(self):
        PTI = PeriodicTable.ColoredPeriodicTableItem
        my_items = [
            PTI("Xx", 42, 43, 44, "xaxatorium", 1002.2,
                bgcolor="#FF0000"),
            PTI("Yy", 25, 22, 44, "yoyotrium", 8.8)
        ]

        pt = PeriodicTable.PeriodicTable(elements=my_items)

        pt.setSelection(["He", "Xx"])
        selection = pt.getSelection()
        self.assertEqual(len(selection), 1)  # "He" not found
        self.assertEqual(selection[0].symbol, "Xx")
        self.assertEqual(selection[0].Z, 42)
        self.assertEqual(selection[0].col, 43)
        self.assertAlmostEqual(selection[0].mass, 1002.2)
        self.assertEqual(qt.QColor(selection[0].bgcolor),
                         qt.QColor(qt.Qt.red))

        self.assertTrue(pt.isElementSelected("Xx"))
        self.assertFalse(pt.isElementSelected("Yy"))
        self.assertRaises(KeyError, pt.isElementSelected, "Yx")

    def testVeryCustomElements(self):
        class MyPTI(PeriodicTable.PeriodicTableItem):
            def __init__(self, *args):
                PeriodicTable.PeriodicTableItem.__init__(self, *args[:6])
                self.my_feature = args[6]

        my_items = [
            MyPTI("Xx", 42, 43, 44, "xaxatorium", 1002.2, "spam"),
            MyPTI("Yy", 25, 22, 44, "yoyotrium", 8.8, "eggs")
        ]

        pt = PeriodicTable.PeriodicTable(elements=my_items)

        pt.setSelection(["Xx", "Yy"])
        selection = pt.getSelection()
        self.assertEqual(len(selection), 2)
        self.assertEqual(selection[1].symbol, "Yy")
        self.assertEqual(selection[1].Z, 25)
        self.assertEqual(selection[1].col, 22)
        self.assertEqual(selection[1].row, 44)
        self.assertAlmostEqual(selection[0].mass, 1002.2)
        self.assertAlmostEqual(selection[0].my_feature, "spam")


class TestPeriodicCombo(TestCaseQt):
    """Basic test for ArrayTableWidget with a numpy array"""
    def setUp(self):
        super(TestPeriodicCombo, self).setUp()
        self.pc = PeriodicTable.PeriodicCombo()

    def tearDown(self):
        del self.pc
        super(TestPeriodicCombo, self).tearDown()

    def testShow(self):
        """basic test (instantiation done in setUp)"""
        self.pc.show()
        self.qWaitForWindowExposed(self.pc)

    def testSelect(self):
        self.pc.setSelection("Sb")
        selection = self.pc.getSelection()
        self.assertIsInstance(selection,
                              PeriodicTable.PeriodicTableItem)
        self.assertEqual(selection.symbol, "Sb")
        self.assertEqual(selection.Z, 51)
        self.assertEqual(selection.name, "antimony")


class TestPeriodicList(TestCaseQt):
    """Basic test for ArrayTableWidget with a numpy array"""
    def setUp(self):
        super(TestPeriodicList, self).setUp()
        self.pl = PeriodicTable.PeriodicList()

    def tearDown(self):
        del self.pl
        super(TestPeriodicList, self).tearDown()

    def testShow(self):
        """basic test (instantiation done in setUp)"""
        self.pl.show()
        self.qWaitForWindowExposed(self.pl)

    def testSelect(self):
        self.pl.setSelectedElements(["Li", "He", "Au"])
        sel_elmts = self.pl.getSelection()

        self.assertEqual(len(sel_elmts), 3,
                         "Wrong number of elements selected")
        for e in sel_elmts:
            self.assertIsInstance(e, PeriodicTable.PeriodicTableItem)
            self.assertIn(e.symbol, ["Li", "He", "Au"])
            self.assertIn(e.Z, [2, 3, 79])
            self.assertIn(e.name, ["lithium", "helium", "gold"])


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPeriodicTable))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPeriodicList))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestPeriodicCombo))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
