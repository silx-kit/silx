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


class TestRegionOfInterestManager(TestCaseQt, ParametricTestCase):
    """Tests for RegionOfInterestManager class"""

    def setUp(self):
        super(TestRegionOfInterestManager, self).setUp()
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
        super(TestRegionOfInterestManager, self).tearDown()

    def test(self):
        """Test ROI of different shapes"""
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
                manager = roi.RegionOfInterestManager(self.plot)
                self.roiTableWidget.setRegionOfInterestManager(manager)
                manager.start(kind)

                self.assertEqual(manager.getRegionOfInterests(), ())

                finishListener = SignalListener()
                manager.sigInteractiveModeFinished.connect(finishListener)

                changedListener = SignalListener()
                manager.sigRegionOfInterestChanged.connect(changedListener)

                # Add a point
                manager.createRegionOfInterest(kind, points[0])
                self.qapp.processEvents()
                self.assertTrue(numpy.all(numpy.equal(
                    manager.getRegionOfInterestPoints(), (points[0],))))
                self.assertEqual(changedListener.callCount(), 1)

                # Remove it
                manager.removeRegionOfInterest(manager.getRegionOfInterests()[0])
                self.assertEqual(manager.getRegionOfInterests(), ())
                self.assertEqual(changedListener.callCount(), 2)

                # Add two point
                manager.createRegionOfInterest(kind, points[0])
                self.qapp.processEvents()
                manager.createRegionOfInterest(kind, points[1])
                self.qapp.processEvents()
                self.assertTrue(numpy.all(numpy.equal(
                    manager.getRegionOfInterestPoints(),
                    (points[0], points[1]))))
                self.assertEqual(changedListener.callCount(), 4)

                # Reset it
                result = manager.clearRegionOfInterests()
                self.assertTrue(result)
                self.assertEqual(manager.getRegionOfInterests(), ())
                self.assertEqual(changedListener.callCount(), 5)

                changedListener.clear()

                # Add two point
                manager.createRegionOfInterest(kind, points[0])
                self.qapp.processEvents()
                manager.createRegionOfInterest(kind, points[1])
                self.qapp.processEvents()
                self.assertTrue(numpy.all(numpy.equal(
                    manager.getRegionOfInterestPoints(),
                    (points[0], points[1]))))
                self.assertEqual(changedListener.callCount(), 2)

                # stop
                result = manager.stop()
                self.assertTrue(result)
                self.assertTrue(numpy.all(numpy.equal(
                    manager.getRegionOfInterestPoints(),
                    (points[0], points[1]))))

                self.qapp.processEvents()
                self.assertEqual(finishListener.callCount(), 1)

                manager.clearRegionOfInterests()

    def testMaxROI(self):
        """Test Max ROI"""
        kind = 'rectangle'
        points = numpy.array((((1., 10.), (11., 20.)),
                              ((2., 3.), (12., 13.))))

        manager = roi.InteractiveRegionOfInterestManager(self.plot)
        self.roiTableWidget.setRegionOfInterestManager(manager)
        self.assertEqual(manager.getRegionOfInterests(), ())

        changedListener = SignalListener()
        manager.sigRegionOfInterestChanged.connect(changedListener)

        # Add two point
        manager.createRegionOfInterest(kind, points[0])
        manager.createRegionOfInterest(kind, points[1])
        self.qapp.processEvents()
        self.assertTrue(numpy.all(numpy.equal(
            manager.getRegionOfInterestPoints(),
            (points[0], points[1]))))
        self.assertEqual(changedListener.callCount(), 2)

        # Try to set max ROI to 1 while there is 2 ROIs
        with self.assertRaises(ValueError):
            manager.setMaxRegionOfInterests(1)

        manager.clearRegionOfInterests()
        self.assertEqual(manager.getRegionOfInterests(), ())
        self.assertEqual(changedListener.callCount(), 3)

        # Set max limit to 1
        manager.setMaxRegionOfInterests(1)

        # Add a point
        manager.createRegionOfInterest(kind, points[0])
        self.qapp.processEvents()
        self.assertTrue(numpy.all(numpy.equal(
            manager.getRegionOfInterestPoints(), (points[0],))))
        self.assertEqual(changedListener.callCount(), 4)

        # Add a 2nd point while max ROI is 1
        manager.createRegionOfInterest(kind, points[1])
        self.qapp.processEvents()
        self.assertTrue(numpy.all(numpy.equal(
            manager.getRegionOfInterestPoints(), (points[1],))))
        self.assertEqual(changedListener.callCount(), 6)

    def testChangeInteractionMode(self):
        """Test change of interaction mode"""
        manager = roi.RegionOfInterestManager(self.plot)
        self.roiTableWidget.setRegionOfInterestManager(manager)
        manager.start('point')

        interactiveModeToolBar = self.plot.getInteractiveModeToolBar()
        panAction = interactiveModeToolBar.getPanModeAction()

        for kind in manager.getSupportedRegionOfInterestKinds():
            with self.subTest(kind=kind):
                # Change to pan mode
                panAction.trigger()

                # Change to interactive ROI mode
                action = manager.getInteractionModeAction(kind)
                action.trigger()

                self.assertEqual(kind, manager.getRegionOfInterestKind())

        manager.clearRegionOfInterests()


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(
            TestRegionOfInterestManager))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
