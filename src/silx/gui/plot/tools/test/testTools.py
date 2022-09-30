# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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
"""Basic tests for silx.gui.plot.tools package"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/03/2018"


import functools
import unittest
import numpy

from silx.utils.testutils import LoggingValidator
from silx.gui.utils.testutils import qWaitForWindowExposedAndActivate
from silx.gui import qt
from silx.gui.plot import PlotWindow
from silx.gui.plot import tools
from silx.gui.plot.test.utils import PlotWidgetTestCase


class TestPositionInfo(PlotWidgetTestCase):
    """Tests for PositionInfo widget."""

    def _createPlot(self):
        return PlotWindow()

    def setUp(self):
        super(TestPositionInfo, self).setUp()
        self.mouseMove(self.plot, pos=(0, 0))
        self.qapp.processEvents()
        self.qWait(100)

    def tearDown(self):
        super(TestPositionInfo, self).tearDown()

    def _test(self, positionWidget, converterNames, **kwargs):
        """General test of PositionInfo.

        - Add it to a toolbar and
        - Move mouse around the center of the PlotWindow.
        """
        toolBar = qt.QToolBar()
        self.plot.addToolBar(qt.Qt.BottomToolBarArea, toolBar)

        toolBar.addWidget(positionWidget)

        converters = positionWidget.getConverters()
        self.assertEqual(len(converters), len(converterNames))
        for index, name in enumerate(converterNames):
            self.assertEqual(converters[index][0], name)

        self.qapp.processEvents()
        with LoggingValidator(tools.__name__, **kwargs):
            # Move mouse to center
            center = self.plot.size() / 2
            self.mouseMove(self.plot, pos=(center.width(), center.height()))
            # Move out
            self.mouseMove(self.plot, pos=(1, 1))

    def testDefaultConverters(self):
        """Test PositionInfo with default converters"""
        positionWidget = tools.PositionInfo(plot=self.plot)
        self._test(positionWidget, ('X', 'Y'))

    def testCustomConverters(self):
        """Test PositionInfo with custom converters"""
        converters = [
            ('Coords', lambda x, y: (int(x), int(y))),
            ('Radius', lambda x, y: numpy.sqrt(x * x + y * y)),
            ('Angle', lambda x, y: numpy.degrees(numpy.arctan2(y, x)))
        ]
        positionWidget = tools.PositionInfo(plot=self.plot,
                                            converters=converters)
        self._test(positionWidget, ('Coords', 'Radius', 'Angle'))

    def testFailingConverters(self):
        """Test PositionInfo with failing custom converters"""
        def raiseException(x, y):
            raise RuntimeError()

        positionWidget = tools.PositionInfo(
            plot=self.plot,
            converters=[('Exception', raiseException)])
        self._test(positionWidget, ['Exception'], error=2)

    def testUpdate(self):
        """Test :meth:`PositionInfo.updateInfo`"""
        calls = []

        def update(calls, x, y):  # Get number of calls
            calls.append((x, y))
            return len(calls)

        positionWidget = tools.PositionInfo(
            plot=self.plot,
            converters=[('Call count', functools.partial(update, calls))])

        positionWidget.updateInfo()
        self.assertEqual(len(calls), 1)


class TestPlotToolsToolbars(PlotWidgetTestCase):
    """Tests toolbars from silx.gui.plot.tools"""

    def test(self):
        """"Add all toolbars"""
        for tbClass in (tools.InteractiveModeToolBar,
                        tools.ImageToolBar,
                        tools.CurveToolBar,
                        tools.OutputToolBar):
            tb = tbClass(parent=self.plot, plot=self.plot)
        self.plot.addToolBar(tb)
