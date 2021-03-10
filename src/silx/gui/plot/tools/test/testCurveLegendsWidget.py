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
__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/08/2018"


import unittest

from silx.gui import qt
from silx.utils.testutils import ParametricTestCase
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.plot import PlotWindow
from silx.gui.plot.tools import CurveLegendsWidget


class TestCurveLegendsWidget(TestCaseQt, ParametricTestCase):
    """Tests for CurveLegendsWidget class"""

    def setUp(self):
        super(TestCurveLegendsWidget, self).setUp()
        self.plot = PlotWindow()

        self.legends = CurveLegendsWidget.CurveLegendsWidget()
        self.legends.setPlotWidget(self.plot)

        dock = qt.QDockWidget()
        dock.setWindowTitle('Curve Legends')
        dock.setWidget(self.legends)
        self.plot.addTabbedDockWidget(dock)

        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

    def tearDown(self):
        del self.legends
        self.qapp.processEvents()
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        super(TestCurveLegendsWidget, self).tearDown()

    def _assertNbLegends(self, count):
        """Check the number of legends in the CurveLegendsWidget"""
        children = self.legends.findChildren(CurveLegendsWidget._LegendWidget)
        self.assertEqual(len(children), count)

    def testAddRemoveCurves(self):
        """Test CurveLegendsWidget while adding/removing curves"""
        self.plot.addCurve((0, 1), (1, 2), legend='a')
        self._assertNbLegends(1)
        self.plot.addCurve((0, 1), (2, 3), legend='b')
        self._assertNbLegends(2)

        # Detached/attach
        self.legends.setPlotWidget(None)
        self._assertNbLegends(0)

        self.legends.setPlotWidget(self.plot)
        self._assertNbLegends(2)

        self.plot.clear()
        self._assertNbLegends(0)

    def testUpdateCurves(self):
        """Test CurveLegendsWidget while updating curves """
        self.plot.addCurve((0, 1), (1, 2), legend='a')
        self._assertNbLegends(1)
        self.plot.addCurve((0, 1), (2, 3), legend='b')
        self._assertNbLegends(2)

        # Activate curve
        self.plot.setActiveCurve('a')
        self.qapp.processEvents()
        self.plot.setActiveCurve('b')
        self.qapp.processEvents()

        # Change curve style
        curve = self.plot.getCurve('a')
        curve.setLineWidth(2)
        for linestyle in (':', '', '--', '-'):
            with self.subTest(linestyle=linestyle):
                curve.setLineStyle(linestyle)
                self.qapp.processEvents()
                self.qWait(1000)

        for symbol in ('o', 'd', '', 's'):
            with self.subTest(symbol=symbol):
                curve.setSymbol(symbol)
                self.qapp.processEvents()
                self.qWait(1000)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            TestCurveLegendsWidget))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
