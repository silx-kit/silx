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
"""Basic tests for PlotWidget"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "20/11/2018"


import unittest
from silx.gui.plot import PlotWidget
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.plot.utils.axis import SyncAxes


class TestAxisSync(TestCaseQt):
    """Tests AxisSync class"""

    def setUp(self):
        TestCaseQt.setUp(self)
        self.plot1 = PlotWidget()
        self.plot2 = PlotWidget()
        self.plot3 = PlotWidget()

    def tearDown(self):
        self.plot1 = None
        self.plot2 = None
        self.plot3 = None
        TestCaseQt.tearDown(self)

    def testMoveFirstAxis(self):
        """Test synchronization after construction"""
        _sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis(), self.plot3.getXAxis()])

        self.plot1.getXAxis().setLimits(10, 500)
        self.assertEqual(self.plot1.getXAxis().getLimits(), (10, 500))
        self.assertEqual(self.plot2.getXAxis().getLimits(), (10, 500))
        self.assertEqual(self.plot3.getXAxis().getLimits(), (10, 500))

    def testMoveSecondAxis(self):
        """Test synchronization after construction"""
        _sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis(), self.plot3.getXAxis()])

        self.plot2.getXAxis().setLimits(10, 500)
        self.assertEqual(self.plot1.getXAxis().getLimits(), (10, 500))
        self.assertEqual(self.plot2.getXAxis().getLimits(), (10, 500))
        self.assertEqual(self.plot3.getXAxis().getLimits(), (10, 500))

    def testMoveTwoAxes(self):
        """Test synchronization after construction"""
        _sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis(), self.plot3.getXAxis()])

        self.plot1.getXAxis().setLimits(1, 50)
        self.plot2.getXAxis().setLimits(10, 500)
        self.assertEqual(self.plot1.getXAxis().getLimits(), (10, 500))
        self.assertEqual(self.plot2.getXAxis().getLimits(), (10, 500))
        self.assertEqual(self.plot3.getXAxis().getLimits(), (10, 500))

    def testDestruction(self):
        """Test synchronization when sync object is destroyed"""
        sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis(), self.plot3.getXAxis()])
        del sync

        self.plot1.getXAxis().setLimits(10, 500)
        self.assertEqual(self.plot1.getXAxis().getLimits(), (10, 500))
        self.assertNotEqual(self.plot2.getXAxis().getLimits(), (10, 500))
        self.assertNotEqual(self.plot3.getXAxis().getLimits(), (10, 500))

    def testAxisDestruction(self):
        """Test synchronization when an axis disappear"""
        _sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis(), self.plot3.getXAxis()])

        # Destroy the plot is possible
        import weakref
        plot = weakref.ref(self.plot2)
        self.plot2 = None
        result = self.qWaitForDestroy(plot)
        if not result:
            # We can't test
            self.skipTest("Object not destroyed")

        self.plot1.getXAxis().setLimits(10, 500)
        self.assertEqual(self.plot3.getXAxis().getLimits(), (10, 500))

    def testStop(self):
        """Test synchronization after calling stop"""
        sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis(), self.plot3.getXAxis()])
        sync.stop()

        self.plot1.getXAxis().setLimits(10, 500)
        self.assertEqual(self.plot1.getXAxis().getLimits(), (10, 500))
        self.assertNotEqual(self.plot2.getXAxis().getLimits(), (10, 500))
        self.assertNotEqual(self.plot3.getXAxis().getLimits(), (10, 500))

    def testStopMovingStart(self):
        """Test synchronization after calling stop, moving an axis, then start again"""
        sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis(), self.plot3.getXAxis()])
        sync.stop()
        self.plot1.getXAxis().setLimits(10, 500)
        self.plot2.getXAxis().setLimits(1, 50)
        self.assertEqual(self.plot1.getXAxis().getLimits(), (10, 500))
        sync.start()

        # The first axis is the reference
        self.assertEqual(self.plot1.getXAxis().getLimits(), (10, 500))
        self.assertEqual(self.plot2.getXAxis().getLimits(), (10, 500))
        self.assertEqual(self.plot3.getXAxis().getLimits(), (10, 500))

    def testDoubleStop(self):
        """Test double stop"""
        sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis(), self.plot3.getXAxis()])
        sync.stop()
        self.assertRaises(RuntimeError, sync.stop)

    def testDoubleStart(self):
        """Test double stop"""
        sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis(), self.plot3.getXAxis()])
        self.assertRaises(RuntimeError, sync.start)

    def testScale(self):
        """Test scale change"""
        _sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis(), self.plot3.getXAxis()])
        self.plot1.getXAxis().setScale(self.plot1.getXAxis().LOGARITHMIC)
        self.assertEqual(self.plot1.getXAxis().getScale(), self.plot1.getXAxis().LOGARITHMIC)
        self.assertEqual(self.plot2.getXAxis().getScale(), self.plot1.getXAxis().LOGARITHMIC)
        self.assertEqual(self.plot3.getXAxis().getScale(), self.plot1.getXAxis().LOGARITHMIC)

    def testDirection(self):
        """Test direction change"""
        _sync = SyncAxes([self.plot1.getYAxis(), self.plot2.getYAxis(), self.plot3.getYAxis()])
        self.plot1.getYAxis().setInverted(True)
        self.assertEqual(self.plot1.getYAxis().isInverted(), True)
        self.assertEqual(self.plot2.getYAxis().isInverted(), True)
        self.assertEqual(self.plot3.getYAxis().isInverted(), True)

    def testSyncCenter(self):
        """Test direction change"""
        # Not the same scale
        self.plot1.getXAxis().setLimits(0, 200)
        self.plot2.getXAxis().setLimits(0, 20)
        self.plot3.getXAxis().setLimits(0, 2)
        _sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis(), self.plot3.getXAxis()],
                         syncLimits=False, syncCenter=True)

        self.assertEqual(self.plot1.getXAxis().getLimits(), (0, 200))
        self.assertEqual(self.plot2.getXAxis().getLimits(), (100 - 10, 100 + 10))
        self.assertEqual(self.plot3.getXAxis().getLimits(), (100 - 1, 100 + 1))

    def testSyncCenterAndZoom(self):
        """Test direction change"""
        # Not the same scale
        self.plot1.getXAxis().setLimits(0, 200)
        self.plot2.getXAxis().setLimits(0, 20)
        self.plot3.getXAxis().setLimits(0, 2)
        _sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis(), self.plot3.getXAxis()],
                         syncLimits=False, syncCenter=True, syncZoom=True)

        # Supposing all the plots use the same size
        self.assertEqual(self.plot1.getXAxis().getLimits(), (0, 200))
        self.assertEqual(self.plot2.getXAxis().getLimits(), (0, 200))
        self.assertEqual(self.plot3.getXAxis().getLimits(), (0, 200))

    def testAddAxis(self):
        """Test synchronization after construction"""
        sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis()])
        sync.addAxis(self.plot3.getXAxis())

        self.plot1.getXAxis().setLimits(10, 500)
        self.assertEqual(self.plot1.getXAxis().getLimits(), (10, 500))
        self.assertEqual(self.plot2.getXAxis().getLimits(), (10, 500))
        self.assertEqual(self.plot3.getXAxis().getLimits(), (10, 500))

    def testRemoveAxis(self):
        """Test synchronization after construction"""
        sync = SyncAxes([self.plot1.getXAxis(), self.plot2.getXAxis(), self.plot3.getXAxis()])
        sync.removeAxis(self.plot3.getXAxis())

        self.plot1.getXAxis().setLimits(10, 500)
        self.assertEqual(self.plot1.getXAxis().getLimits(), (10, 500))
        self.assertEqual(self.plot2.getXAxis().getLimits(), (10, 500))
        self.assertNotEqual(self.plot3.getXAxis().getLimits(), (10, 500))
