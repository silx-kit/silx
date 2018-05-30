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
__date__ = "26/03/2018"


import unittest

import numpy

from silx.gui import qt
from silx.utils.testutils import ParametricTestCase
from silx.gui.test.utils import TestCaseQt, SignalListener
from silx.gui.plot import PlotWindow
from silx.gui.plot.tools import roi


class TestSelectionManager(TestCaseQt, ParametricTestCase):
    """Tests for RegionOfInterestManager class"""

    def setUp(self):
        super(TestSelectionManager, self).setUp()
        self.plot = PlotWindow()

        self.roiTableWidget = roi.RegionOfInterestTableWidget()
        dock = qt.QDockWidget()
        dock.setWidget(self.roiTableWidget)
        self.plot.addDockWidget(qt.Qt.BottomDockWidgetArea, dock)

        self.plot.show()
        self.qWaitForWindowExposed(self.plot)

    def tearDown(self):
        del self.roiTableWidget
        self.qapp.processEvents()
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        super(TestSelectionManager, self).tearDown()

    def testSelection(self):
        """Test selection of different shapes"""
        tests = (  # shape, points=[list of (x, y), list of (x, y)]
            ('point', numpy.array(([(10., 15.)], [(20., 25.)]))),
            ('rectangle', numpy.array((((1., 10.), (11., 20.)),
                                       ((2., 3.), (12., 13.))))),
            ('polygon', numpy.array((((0., 1.), (0., 10.), (10., 0.)),
                                     ((5., 6.), (5., 16.), (15., 6.))))),
            ('line', numpy.array((((10., 20.), (10., 30.)),
                                  ((30., 40.), (30., 50.))))),
            ('hline', numpy.array((((10., 20.), (10., 30.)),
                                   ((30., 40.), (30., 50.))))),
            ('vline', numpy.array((((10., 20.), (10., 30.)),
                                   ((30., 40.), (30., 50.))))),
        )

        for kind, points in tests:
            with self.subTest(kind=kind):
                selector = roi.RegionOfInterestManager(self.plot)
                self.roiTableWidget.setRegionOfInterestManager(selector)
                selector.start(kind)

                self.assertEqual(selector.getSelections(), ())

                finishListener = SignalListener()
                selector.sigSelectionFinished.connect(finishListener)

                changedListener = SignalListener()
                selector.sigSelectionChanged.connect(changedListener)

                # Add a point
                selector.addSelection(kind, points[0])
                self.qapp.processEvents()
                self.assertTrue(numpy.all(numpy.equal(
                    selector.getSelectionPoints(), (points[0],))))
                self.assertEqual(changedListener.callCount(), 1)

                # Remove it
                selector.removeSelection(selector.getSelections()[0])
                self.assertEqual(selector.getSelections(), ())
                self.assertEqual(changedListener.callCount(), 2)

                # Add two point
                selector.addSelection(kind, points[0])
                self.qapp.processEvents()
                selector.addSelection(kind, points[1])
                self.qapp.processEvents()
                self.assertTrue(numpy.all(numpy.equal(
                    selector.getSelectionPoints(),
                    (points[0], points[1]))))
                self.assertEqual(changedListener.callCount(), 4)

                # Reset it
                result = selector.clearSelections()
                self.assertTrue(result)
                self.assertEqual(selector.getSelections(), ())
                self.assertEqual(changedListener.callCount(), 5)

                changedListener.clear()

                # Add two point
                selector.addSelection(kind, points[0])
                self.qapp.processEvents()
                selector.addSelection(kind, points[1])
                self.qapp.processEvents()
                self.assertTrue(numpy.all(numpy.equal(
                    selector.getSelectionPoints(),
                    (points[0], points[1]))))
                self.assertEqual(changedListener.callCount(), 2)

                # stop
                result = selector.stop()
                self.assertTrue(result)
                self.assertTrue(numpy.all(numpy.equal(
                    selector.getSelectionPoints(),
                    (points[0], points[1]))))

                self.qapp.processEvents()
                self.assertEqual(finishListener.callCount(), 1)

                # Try to set max selection to 1 while 2 selections
                with self.assertRaises(ValueError):
                    selector.setMaxSelections(1)

                # restart
                changedListener.clear()
                selector.clearSelections()
                self.assertEqual(selector.getSelections(), ())
                self.assertEqual(changedListener.callCount(), 1)
                selector.start(kind)

                # Set max limit to 1
                selector.setMaxSelections(1)

                # Add a point
                selector.addSelection(kind, points[0])
                self.qapp.processEvents()
                self.assertTrue(numpy.all(numpy.equal(
                    selector.getSelectionPoints(), (points[0],))))
                self.assertEqual(changedListener.callCount(), 2)

                # Try to add a 2nd point while max selection is 1
                with self.assertRaises(RuntimeError):
                    selector.addSelection(kind, points[1])

                # stop
                selector.stop()
                selector.clearSelections()
                self.qapp.processEvents()
                self.assertEqual(finishListener.callCount(), 2)
                self.assertEqual(changedListener.callCount(), 3)

    def testChangeInteractionMode(self):
        """Test change of interaction mode"""
        selector = roi.RegionOfInterestManager(self.plot)
        self.roiTableWidget.setRegionOfInterestManager(selector)
        selector.start('point')

        interactiveModeToolBar = self.plot.getInteractiveModeToolBar()
        panAction = interactiveModeToolBar.getPanModeAction()

        for kind in selector.getSupportedSelectionKinds():
            with self.subTest(kind=kind):
                # Change to pan mode
                panAction.trigger()

                # Change to selection mode
                selectionAction = selector.getDrawSelectionModeAction(kind)
                selectionAction.trigger()

                self.assertEqual(kind, selector.getSelectionKind())

        selector.clearSelections()


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            TestSelectionManager))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
