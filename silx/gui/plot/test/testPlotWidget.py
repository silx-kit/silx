# coding: utf-8
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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "01/09/2017"


import unittest
import logging
import contextlib
import numpy

from silx.test.utils import ParametricTestCase
from silx.gui.test.utils import SignalListener
from silx.gui.test.utils import TestCaseQt
from silx.test import utils
from silx.utils import deprecation

from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.Colormap import Colormap
from silx.gui.plot.backends.BackendMatplotlib import BackendMatplotlibQt


SIZE = 1024
"""Size of the test image"""

DATA_2D = numpy.arange(SIZE ** 2).reshape(SIZE, SIZE)
"""Image data set"""


logger = logging.getLogger(__name__)


class _PlotWidgetTest(TestCaseQt):
    """Base class for tests of PlotWidget, not a TestCase in itself.

    plot attribute is the PlotWidget created for the test.
    """

    def __init__(self, methodName='runTest'):
        TestCaseQt.__init__(self, methodName=methodName)
        self.__mousePos = None

    def _createPlot(self):
        return PlotWidget()

    def setUp(self):
        super(_PlotWidgetTest, self).setUp()
        self.plot = self._createPlot()
        self.plot.show()
        self.plotAlive = True
        self.qWaitForWindowExposed(self.plot)
        TestCaseQt.mouseClick(self, self.plot, button=qt.Qt.LeftButton, pos=(0, 0))

    def __onPlotDestroyed(self):
        self.plotAlive = False

    def _waitForPlotClosed(self):
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.destroyed.connect(self.__onPlotDestroyed)
        self.plot.close()
        del self.plot
        for _ in range(100):
            if not self.plotAlive:
                break
            self.qWait(10)
        else:
            logger.error("Plot is still alive")

    def tearDown(self):
        self.qapp.processEvents()
        self._waitForPlotClosed()
        super(_PlotWidgetTest, self).tearDown()

    def _logMplEvents(self, event):
        self.__mplEvents.append(event)

    @contextlib.contextmanager
    def _waitForMplEvent(self, plot, mplEventType):
        """Check if an event was received by the MPL backend.

        :param PlotWidget plot: A plot widget or a MPL plot backend
        :param str mplEventType: MPL event type
        :raises RuntimeError: When the event did not happen
        """
        self.__mplEvents = []
        if isinstance(plot, BackendMatplotlibQt):
            backend = plot
        else:
            backend = plot._backend

        callbackId = backend.mpl_connect(mplEventType, self._logMplEvents)
        received = False
        yield
        for _ in range(100):
            if len(self.__mplEvents) > 0:
                received = True
                break
            self.qWait(10)
        backend.mpl_disconnect(callbackId)
        del self.__mplEvents
        if not received:
            self.logScreenShot()
            raise RuntimeError("MPL event %s expected but nothing received" % mplEventType)

    def __plotHandleEvent(self, event, *args, **kwargs):
        self.__plotEvents.add(event)
        self.__patchedPlot._eventHandler._old_handleEvent(event, *args, **kwargs)

    @contextlib.contextmanager
    def _waitForPlotEvent(self, plot, plotEventType):
        """Check if an event was received by the Silx Plot.

        :param PlotWidget plot: A plot widget or a MPL plot backend
        :param str plotEventType: Silx plot event type
        :raises RuntimeError: When the event did not happen
        """
        if isinstance(plot, BackendMatplotlibQt):
            backend = plot
        else:
            backend = plot._backend
        plot = backend._plot

        self.__plotEvents = set([])
        plot._eventHandler._old_handleEvent = plot._eventHandler.handleEvent
        plot._eventHandler.handleEvent = self.__plotHandleEvent
        self.__patchedPlot = plot

        received = False
        yield
        for _ in range(100):
            if plotEventType in self.__plotEvents:
                received = True
                break
            self.qWait(10)

        plot._eventHandler.handleEvent = plot._eventHandler._old_handleEvent
        del plot._eventHandler._old_handleEvent
        del self.__patchedPlot

        if not received:
            self.logScreenShot()
            raise RuntimeError("Backend function _%s expected but nothing received" % plotEventType)
        del self.__plotEvents

    def _haveMplEvent(self, widget, pos):
        """Check if the widget at this position is a matplotlib widget."""
        if isinstance(pos, qt.QPoint):
            pass
        else:
            pos = qt.QPoint(pos[0], pos[1])
        pos = widget.mapTo(widget.window(), pos)
        target = widget.window().childAt(pos)

        # Check if the target is a MPL container
        backend = target
        if hasattr(target, "_backend"):
            backend = target._backend
        haveEvent = isinstance(backend, BackendMatplotlibQt)
        return haveEvent

    def _patchPos(self, widget, pos):
        """Return a real position relative to the widget.

        If pos is None, the returned value is the center of the widget,
        as the default behaviour of functions like QTest.mouseMove.
        Else the position is returned as it is.
        """
        if pos is None:
            pos = widget.size() / 2
            pos = pos.width(), pos.height()
        return pos

    def _checkMouseMove(self, widget, pos):
        """Returns true if the position differe from the current position of
        the cursor"""
        pos = qt.QPoint(pos[0], pos[1])
        pos = widget.mapTo(widget.window(), pos)
        willMove = pos != self.__mousePos
        self.__mousePos = pos
        return willMove

    def mouseMove(self, widget, pos=None, delay=-1):
        """Override TestCaseQt to wait while MPL did not reveive the expected
        event"""
        pos = self._patchPos(widget, pos)
        willMove = self._checkMouseMove(widget, pos)
        hadMplEvents = self._haveMplEvent(widget, self.__mousePos)
        willHaveMplEvents = self._haveMplEvent(widget, pos)
        if (not hadMplEvents and not willHaveMplEvents) or not willMove:
            return TestCaseQt.mouseMove(self, widget, pos=pos, delay=delay)
        with self._waitForPlotEvent(widget, "move"):
            with self._waitForMplEvent(widget, "motion_notify_event"):
                TestCaseQt.mouseMove(self, widget, pos=pos, delay=delay)

    def mouseClick(self, widget, button, modifier=None, pos=None, delay=-1):
        """Override TestCaseQt to wait while MPL did not reveive the expected
        event"""
        pos = self._patchPos(widget, pos)
        self._checkMouseMove(widget, pos)
        if not self._haveMplEvent(widget, pos):
            return TestCaseQt.mouseClick(self, widget, button, modifier=modifier, pos=pos, delay=delay)
        with self._waitForPlotEvent(widget, "release"):
            with self._waitForMplEvent(widget, "button_release_event"):
                TestCaseQt.mouseClick(self, widget, button, modifier=modifier, pos=pos, delay=delay)

    def mousePress(self, widget, button, modifier=None, pos=None, delay=-1):
        """Override TestCaseQt to wait while MPL did not reveive the expected
        event"""
        pos = self._patchPos(widget, pos)
        self._checkMouseMove(widget, pos)
        if not self._haveMplEvent(widget, pos):
            return TestCaseQt.mousePress(self, widget, button, modifier=modifier, pos=pos, delay=delay)
        with self._waitForPlotEvent(widget, "press"):
            with self._waitForMplEvent(widget, "button_press_event"):
                TestCaseQt.mousePress(self, widget, button, modifier=modifier, pos=pos, delay=delay)

    def mouseRelease(self, widget, button, modifier=None, pos=None, delay=-1):
        """Override TestCaseQt to wait while MPL did not reveive the expected
        event"""
        pos = self._patchPos(widget, pos)
        self._checkMouseMove(widget, pos)
        if not self._haveMplEvent(widget, pos):
            return TestCaseQt.mouseRelease(self, widget, button, modifier=modifier, pos=pos, delay=delay)
        with self._waitForPlotEvent(widget, "release"):
            with self._waitForMplEvent(widget, "button_release_event"):
                TestCaseQt.mouseRelease(self, widget, button, modifier=modifier, pos=pos, delay=delay)


class TestPlotWidget(_PlotWidgetTest, ParametricTestCase):
    """Basic tests for PlotWidget"""

    def testShow(self):
        """Most basic test"""
        pass

    def testSetTitleLabels(self):
        """Set title and axes labels"""

        title, xlabel, ylabel = 'the title', 'x label', 'y label'
        self.plot.setGraphTitle(title)
        self.plot.getXAxis().setLabel(xlabel)
        self.plot.getYAxis().setLabel(ylabel)
        self.qapp.processEvents()

        self.assertEqual(self.plot.getGraphTitle(), title)
        self.assertEqual(self.plot.getXAxis().getLabel(), xlabel)
        self.assertEqual(self.plot.getYAxis().getLabel(), ylabel)

    def testChangeLimitsWithAspectRatio(self):
        def checkLimits(expectedXLim=None, expectedYLim=None,
                        expectedRatio=None):
            xlim = self.plot.getXAxis().getLimits()
            ylim = self.plot.getYAxis().getLimits()
            ratio = abs(xlim[1] - xlim[0]) / abs(ylim[1] - ylim[0])

            if expectedXLim is not None:
                self.assertEqual(expectedXLim, xlim)

            if expectedYLim is not None:
                self.assertEqual(expectedYLim, ylim)

            if expectedRatio is not None:
                self.assertTrue(
                    numpy.allclose(expectedRatio, ratio, atol=0.01))

        self.plot.setKeepDataAspectRatio()
        self.qapp.processEvents()
        xlim = self.plot.getXAxis().getLimits()
        ylim = self.plot.getYAxis().getLimits()
        defaultRatio = abs(xlim[1] - xlim[0]) / abs(ylim[1] - ylim[0])

        self.plot.getXAxis().setLimits(1., 10.)
        checkLimits(expectedXLim=(1., 10.), expectedRatio=defaultRatio)
        self.qapp.processEvents()
        checkLimits(expectedXLim=(1., 10.), expectedRatio=defaultRatio)

        self.plot.getYAxis().setLimits(1., 10.)
        checkLimits(expectedYLim=(1., 10.), expectedRatio=defaultRatio)
        self.qapp.processEvents()
        checkLimits(expectedYLim=(1., 10.), expectedRatio=defaultRatio)


class TestPlotImage(_PlotWidgetTest, ParametricTestCase):
    """Basic tests for addImage"""

    def setUp(self):
        super(TestPlotImage, self).setUp()

        self.plot.getYAxis().setLabel('Rows')
        self.plot.getXAxis().setLabel('Columns')

    def testPlotColormapTemperature(self):
        self.plot.setGraphTitle('Temp. Linear')

        colormap = Colormap(name='temperature',
                            normalization='linear',
                            vmin=None,
                            vmax=None)
        self.plot.addImage(DATA_2D, legend="image 1", colormap=colormap)

    def testPlotColormapGray(self):
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setGraphTitle('Gray Linear')

        colormap = Colormap(name='gray',
                            normalization='linear',
                            vmin=None,
                            vmax=None)
        self.plot.addImage(DATA_2D, legend="image 1", colormap=colormap)

    def testPlotColormapTemperatureLog(self):
        self.plot.setGraphTitle('Temp. Log')

        colormap = Colormap(name='temperature',
                            normalization=Colormap.LOGARITHM,
                            vmin=None,
                            vmax=None)
        self.plot.addImage(DATA_2D, legend="image 1", colormap=colormap)

    def testPlotRgbRgba(self):
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setGraphTitle('RGB + RGBA')

        rgb = numpy.array(
            (((0, 0, 0), (128, 0, 0), (255, 0, 0)),
             ((0, 128, 0), (0, 128, 128), (0, 128, 256))),
            dtype=numpy.uint8)

        self.plot.addImage(rgb, legend="rgb",
                           origin=(0, 0), scale=(10, 10),
                           replace=False, resetzoom=False)

        rgba = numpy.array(
            (((0, 0, 0, .5), (.5, 0, 0, 1), (1, 0, 0, .5)),
             ((0, .5, 0, 1), (0, .5, .5, 1), (0, 1, 1, .5))),
            dtype=numpy.float32)

        self.plot.addImage(rgba, legend="rgba",
                           origin=(5, 5), scale=(10, 10),
                           replace=False, resetzoom=False)

        self.plot.resetZoom()

    def testPlotColormapCustom(self):
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setGraphTitle('Custom colormap')

        colormap = Colormap(name=None,
                            normalization=Colormap.LINEAR,
                            vmin=None,
                            vmax=None,
                            colors=((0., 0., 0.), (1., 0., 0.),
                               (0., 1., 0.), (0., 0., 1.)))
        self.plot.addImage(DATA_2D, legend="image 1", colormap=colormap,
                           replace=False, resetzoom=False)

        colormap = Colormap(name=None,
                            normalization=Colormap.LINEAR,
                            vmin=None,
                            vmax=None,
                            colors=numpy.array(
                                ((0, 0, 0, 0), (0, 0, 0, 128),
                                 (128, 128, 128, 128), (255, 255, 255, 255)),
                                dtype=numpy.uint8))
        self.plot.addImage(DATA_2D, legend="image 2", colormap=colormap,
                           origin=(DATA_2D.shape[0], 0),
                           replace=False, resetzoom=False)
        self.plot.resetZoom()

    def testImageOriginScale(self):
        """Test of image with different origin and scale"""
        self.plot.setGraphTitle('origin and scale')

        tests = [  # (origin, scale)
            ((10, 20), (1, 1)),
            ((10, 20), (-1, -1)),
            ((-10, 20), (2, 1)),
            ((10, -20), (-1, -2)),
            (100, 2),
            (-100, (1, 1)),
            ((10, 20), 2),
            ]

        for origin, scale in tests:
            with self.subTest(origin=origin, scale=scale):
                self.plot.addImage(DATA_2D, origin=origin, scale=scale)

                try:
                    ox, oy = origin
                except TypeError:
                    ox, oy = origin, origin
                try:
                    sx, sy = scale
                except TypeError:
                    sx, sy = scale, scale
                xbounds = ox, ox + DATA_2D.shape[1] * sx
                ybounds = oy, oy + DATA_2D.shape[0] * sy

                # Check limits without aspect ratio
                xmin, xmax = self.plot.getXAxis().getLimits()
                ymin, ymax = self.plot.getYAxis().getLimits()
                self.assertEqual(xmin, min(xbounds))
                self.assertEqual(xmax, max(xbounds))
                self.assertEqual(ymin, min(ybounds))
                self.assertEqual(ymax, max(ybounds))

                # Check limits with aspect ratio
                self.plot.setKeepDataAspectRatio(True)
                xmin, xmax = self.plot.getXAxis().getLimits()
                ymin, ymax = self.plot.getYAxis().getLimits()
                self.assertTrue(xmin <= min(xbounds))
                self.assertTrue(xmax >= max(xbounds))
                self.assertTrue(ymin <= min(ybounds))
                self.assertTrue(ymax >= max(ybounds))

                self.plot.setKeepDataAspectRatio(False)  # Reset aspect ratio
                self.plot.clear()
                self.plot.resetZoom()

    def testPlotColormapDictAPI(self):
        """Test that the addImage API using a colormap dictionary is still
        working"""
        self.plot.setGraphTitle('Temp. Log')

        colormap = {
            'name': 'temperature',
            'normalization': 'log',
            'vmin': None,
            'vmax': None
        }
        self.plot.addImage(DATA_2D, legend="image 1", colormap=colormap)


class TestPlotCurve(_PlotWidgetTest):
    """Basic tests for addCurve."""

    # Test data sets
    xData = numpy.arange(1000)
    yData = -500 + 100 * numpy.sin(xData)
    xData2 = xData + 1000
    yData2 = xData - 1000 + 200 * numpy.random.random(1000)

    def setUp(self):
        super(TestPlotCurve, self).setUp()
        self.plot.setGraphTitle('Curve')
        self.plot.getYAxis().setLabel('Rows')
        self.plot.getXAxis().setLabel('Columns')

        self.plot.setActiveCurveHandling(False)

    def testPlotCurveColorFloat(self):
        color = numpy.array(numpy.random.random(3 * 1000),
                            dtype=numpy.float32).reshape(1000, 3)

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve 1",
                           replace=False, resetzoom=False,
                           color=color,
                           linestyle="", symbol="s")
        self.plot.addCurve(self.xData2, self.yData2,
                           legend="curve 2",
                           replace=False, resetzoom=False,
                           color='green', linestyle="-", symbol='o')
        self.plot.resetZoom()

    def testPlotCurveColorByte(self):
        color = numpy.array(255 * numpy.random.random(3 * 1000),
                            dtype=numpy.uint8).reshape(1000, 3)

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve 1",
                           replace=False, resetzoom=False,
                           color=color,
                           linestyle="", symbol="s")
        self.plot.addCurve(self.xData2, self.yData2,
                           legend="curve 2",
                           replace=False, resetzoom=False,
                           color='green', linestyle="-", symbol='o')
        self.plot.resetZoom()

    def testPlotCurveColors(self):
        color = numpy.array(numpy.random.random(3 * 1000),
                            dtype=numpy.float32).reshape(1000, 3)

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve 2",
                           replace=False, resetzoom=False,
                           color=color, linestyle="-", symbol='o')
        self.plot.resetZoom()


class TestPlotMarker(_PlotWidgetTest):
    """Basic tests for add*Marker"""

    def setUp(self):
        super(TestPlotMarker, self).setUp()
        self.plot.getYAxis().setLabel('Rows')
        self.plot.getXAxis().setLabel('Columns')

        self.plot.getXAxis().setAutoScale(False)
        self.plot.getYAxis().setAutoScale(False)
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setLimits(0., 100., -100., 100.)

    def testPlotMarkerX(self):
        self.plot.setGraphTitle('Markers X')

        markers = [
            (10., 'blue', False, False),
            (20., 'red', False, False),
            (40., 'green', True, False),
            (60., 'gray', True, True),
            (80., 'black', False, True),
        ]

        for x, color, select, drag in markers:
            name = str(x)
            if select:
                name += " sel."
            if drag:
                name += " drag"
            self.plot.addXMarker(x, name, name, color, select, drag)
        self.plot.resetZoom()

    def testPlotMarkerY(self):
        self.plot.setGraphTitle('Markers Y')

        markers = [
            (-50., 'blue', False, False),
            (-30., 'red', False, False),
            (0., 'green', True, False),
            (10., 'gray', True, True),
            (80., 'black', False, True),
        ]

        for y, color, select, drag in markers:
            name = str(y)
            if select:
                name += " sel."
            if drag:
                name += " drag"
            self.plot.addYMarker(y, name, name, color, select, drag)
        self.plot.resetZoom()

    def testPlotMarkerPt(self):
        self.plot.setGraphTitle('Markers Pt')

        markers = [
            (10., -50., 'blue', False, False),
            (40., -30., 'red', False, False),
            (50., 0., 'green', True, False),
            (50., 20., 'gray', True, True),
            (70., 50., 'black', False, True),
        ]
        for x, y, color, select, drag in markers:
            name = "{0},{1}".format(x, y)
            if select:
                name += " sel."
            if drag:
                name += " drag"
            self.plot.addMarker(x, y, name, name, color, select, drag)

        self.plot.resetZoom()

    def testPlotMarkerWithoutLegend(self):
        self.plot.setGraphTitle('Markers without legend')
        self.plot.getYAxis().setInverted(True)

        # Markers without legend
        self.plot.addMarker(10, 10)
        self.plot.addMarker(10, 20)
        self.plot.addMarker(40, 50, text='test', symbol=None)
        self.plot.addMarker(40, 50, text='test', symbol='+')
        self.plot.addXMarker(25)
        self.plot.addXMarker(35)
        self.plot.addXMarker(45, text='test')
        self.plot.addYMarker(55)
        self.plot.addYMarker(65)
        self.plot.addYMarker(75, text='test')

        self.plot.resetZoom()


# TestPlotItem ################################################################

class TestPlotItem(_PlotWidgetTest):
    """Basic tests for addItem."""

    # Polygon coordinates and color
    polygons = [  # legend, x coords, y coords, color
        ('triangle', numpy.array((10, 30, 50)),
         numpy.array((55, 70, 55)), 'red'),
        ('square', numpy.array((10, 10, 50, 50)),
         numpy.array((10, 50, 50, 10)), 'green'),
        ('star', numpy.array((60, 70, 80, 60, 80)),
         numpy.array((25, 50, 25, 40, 40)), 'blue'),
    ]

    # Rectangle coordinantes and color
    rectangles = [  # legend, x coords, y coords, color
        ('square 1', numpy.array((1., 10.)),
         numpy.array((1., 10.)), 'red'),
        ('square 2', numpy.array((10., 20.)),
         numpy.array((10., 20.)), 'green'),
        ('square 3', numpy.array((20., 30.)),
         numpy.array((20., 30.)), 'blue'),
        ('rect 1', numpy.array((1., 30.)),
         numpy.array((35., 40.)), 'black'),
        ('line h', numpy.array((1., 30.)),
         numpy.array((45., 45.)), 'darkRed'),
    ]

    def setUp(self):
        super(TestPlotItem, self).setUp()

        self.plot.getYAxis().setLabel('Rows')
        self.plot.getXAxis().setLabel('Columns')
        self.plot.getXAxis().setAutoScale(False)
        self.plot.getYAxis().setAutoScale(False)
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setLimits(0., 100., -100., 100.)

    def testPlotItemPolygonFill(self):
        self.plot.setGraphTitle('Item Fill')

        for legend, xList, yList, color in self.polygons:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="polygon", fill=True, color=color)
        self.plot.resetZoom()

    def testPlotItemPolygonNoFill(self):
        self.plot.setGraphTitle('Item No Fill')

        for legend, xList, yList, color in self.polygons:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="polygon", fill=False, color=color)
        self.plot.resetZoom()

    def testPlotItemRectangleFill(self):
        self.plot.setGraphTitle('Rectangle Fill')

        for legend, xList, yList, color in self.rectangles:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="rectangle", fill=True, color=color)
        self.plot.resetZoom()

    def testPlotItemRectangleNoFill(self):
        self.plot.setGraphTitle('Rectangle No Fill')

        for legend, xList, yList, color in self.rectangles:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="rectangle", fill=False, color=color)
        self.plot.resetZoom()


class TestPlotActiveCurveImage(_PlotWidgetTest):
    """Basic tests for active image handling"""

    def testActiveCurveAndLabels(self):
        # Active curve handling off, no label change
        self.plot.setActiveCurveHandling(False)
        self.plot.getXAxis().setLabel('XLabel')
        self.plot.getYAxis().setLabel('YLabel')
        self.plot.addCurve((1, 2), (1, 2))
        self.assertEqual(self.plot.getXAxis().getLabel(), 'XLabel')
        self.assertEqual(self.plot.getYAxis().getLabel(), 'YLabel')

        self.plot.addCurve((1, 2), (2, 3), xlabel='x1', ylabel='y1')
        self.assertEqual(self.plot.getXAxis().getLabel(), 'XLabel')
        self.assertEqual(self.plot.getYAxis().getLabel(), 'YLabel')

        self.plot.clear()
        self.assertEqual(self.plot.getXAxis().getLabel(), 'XLabel')
        self.assertEqual(self.plot.getYAxis().getLabel(), 'YLabel')

        # Active curve handling on, label changes
        self.plot.setActiveCurveHandling(True)
        self.plot.getXAxis().setLabel('XLabel')
        self.plot.getYAxis().setLabel('YLabel')

        # labels changed as active curve
        self.plot.addCurve((1, 2), (1, 2), legend='1',
                           xlabel='x1', ylabel='y1')
        self.assertEqual(self.plot.getXAxis().getLabel(), 'x1')
        self.assertEqual(self.plot.getYAxis().getLabel(), 'y1')

        # labels not changed as not active curve
        self.plot.addCurve((1, 2), (2, 3), legend='2')
        self.assertEqual(self.plot.getXAxis().getLabel(), 'x1')
        self.assertEqual(self.plot.getYAxis().getLabel(), 'y1')

        # labels changed
        self.plot.setActiveCurve('2')
        self.assertEqual(self.plot.getXAxis().getLabel(), 'XLabel')
        self.assertEqual(self.plot.getYAxis().getLabel(), 'YLabel')

        self.plot.setActiveCurve('1')
        self.assertEqual(self.plot.getXAxis().getLabel(), 'x1')
        self.assertEqual(self.plot.getYAxis().getLabel(), 'y1')

        self.plot.clear()
        self.assertEqual(self.plot.getXAxis().getLabel(), 'XLabel')
        self.assertEqual(self.plot.getYAxis().getLabel(), 'YLabel')

    def testActiveImageAndLabels(self):
        # Active image handling always on, no API for toggling it
        self.plot.getXAxis().setLabel('XLabel')
        self.plot.getYAxis().setLabel('YLabel')

        # labels changed as active curve
        self.plot.addImage(numpy.arange(100).reshape(10, 10), replace=False,
                           legend='1', xlabel='x1', ylabel='y1')
        self.assertEqual(self.plot.getXAxis().getLabel(), 'x1')
        self.assertEqual(self.plot.getYAxis().getLabel(), 'y1')

        # labels not changed as not active curve
        self.plot.addImage(numpy.arange(100).reshape(10, 10), replace=False,
                           legend='2')
        self.assertEqual(self.plot.getXAxis().getLabel(), 'x1')
        self.assertEqual(self.plot.getYAxis().getLabel(), 'y1')

        # labels changed
        self.plot.setActiveImage('2')
        self.assertEqual(self.plot.getXAxis().getLabel(), 'XLabel')
        self.assertEqual(self.plot.getYAxis().getLabel(), 'YLabel')

        self.plot.setActiveImage('1')
        self.assertEqual(self.plot.getXAxis().getLabel(), 'x1')
        self.assertEqual(self.plot.getYAxis().getLabel(), 'y1')

        self.plot.clear()
        self.assertEqual(self.plot.getXAxis().getLabel(), 'XLabel')
        self.assertEqual(self.plot.getYAxis().getLabel(), 'YLabel')


##############################################################################
# Log
##############################################################################

class TestPlotEmptyLog(_PlotWidgetTest):
    """Basic tests for log plot"""
    def testEmptyPlotTitleLabelsLog(self):
        self.plot.setGraphTitle('Empty Log Log')
        self.plot.getXAxis().setLabel('X')
        self.plot.getYAxis().setLabel('Y')
        self.plot.getXAxis()._setLogarithmic(True)
        self.plot.getYAxis()._setLogarithmic(True)
        self.plot.resetZoom()


class TestPlotAxes(TestCaseQt, ParametricTestCase):

    # Test data
    xData = numpy.arange(1, 10)
    yData = xData ** 2

    def setUp(self):
        super(TestPlotAxes, self).setUp()
        self.plot = PlotWidget()
        # It is not needed to display the plot
        # It saves a lot of time
        # self.plot.show()
        # self.qWaitForWindowExposed(self.plot)

    def tearDown(self):
        self.qapp.processEvents()
        self.plot.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.plot.close()
        del self.plot
        super(TestPlotAxes, self).tearDown()

    def testDefaultAxes(self):
        axis = self.plot.getXAxis()
        self.assertEqual(axis.getScale(), axis.LINEAR)
        axis = self.plot.getYAxis()
        self.assertEqual(axis.getScale(), axis.LINEAR)
        axis = self.plot.getYAxis(axis="right")
        self.assertEqual(axis.getScale(), axis.LINEAR)

    def testOldPlotAxis_getterSetter(self):
        """Test silx API prior to silx 0.6"""
        x = self.plot.getXAxis()
        y = self.plot.getYAxis()
        p = self.plot

        tests = [
            # setters
            (p.setGraphXLimits, (10, 20), x.getLimits, (10, 20)),
            (p.setGraphYLimits, (10, 20), y.getLimits, (10, 20)),
            (p.setGraphXLabel, "foox", x.getLabel, "foox"),
            (p.setGraphYLabel, "fooy", y.getLabel, "fooy"),
            (p.setYAxisInverted, True, y.isInverted, True),
            (p.setXAxisLogarithmic, True, x.getScale, x.LOGARITHMIC),
            (p.setYAxisLogarithmic, True, y.getScale, y.LOGARITHMIC),
            (p.setXAxisAutoScale, False, x.isAutoScale, False),
            (p.setYAxisAutoScale, False, y.isAutoScale, False),
            # getters
            (x.setLimits, (11, 20), p.getGraphXLimits, (11, 20)),
            (y.setLimits, (11, 20), p.getGraphYLimits, (11, 20)),
            (x.setLabel, "fooxx", p.getGraphXLabel, "fooxx"),
            (y.setLabel, "fooyy", p.getGraphYLabel, "fooyy"),
            (y.setInverted, False, p.isYAxisInverted, False),
            (x.setScale, x.LINEAR, p.isXAxisLogarithmic, False),
            (y.setScale, y.LINEAR, p.isYAxisLogarithmic, False),
            (x.setAutoScale, True, p.isXAxisAutoScale, True),
            (y.setAutoScale, True, p.isYAxisAutoScale, True),
        ]
        for testCase in tests:
            setter, value, getter, expected = testCase
            with self.subTest():
                if setter is not None:
                    if not isinstance(value, tuple):
                        value = (value, )
                    setter(*value)
                if getter is not None:
                    self.assertEqual(getter(), expected)

    @utils.test_logging(deprecation.depreclog.name, warning=2)
    def testOldPlotAxis_Logarithmic(self):
        """Test silx API prior to silx 0.6"""
        x = self.plot.getXAxis()
        y = self.plot.getYAxis()
        yright = self.plot.getYAxis(axis="right")

        listener = SignalListener()
        self.plot.sigSetXAxisLogarithmic.connect(listener.partial("x"))
        self.plot.sigSetYAxisLogarithmic.connect(listener.partial("y"))

        self.assertEqual(x.getScale(), x.LINEAR)
        self.assertEqual(y.getScale(), x.LINEAR)
        self.assertEqual(yright.getScale(), x.LINEAR)

        self.plot.setXAxisLogarithmic(True)
        self.assertEqual(x.getScale(), x.LOGARITHMIC)
        self.assertEqual(y.getScale(), x.LINEAR)
        self.assertEqual(yright.getScale(), x.LINEAR)
        self.assertEqual(self.plot.isXAxisLogarithmic(), True)
        self.assertEqual(self.plot.isYAxisLogarithmic(), False)
        self.assertEqual(listener.arguments(callIndex=-1), ("x", True))

        self.plot.setYAxisLogarithmic(True)
        self.assertEqual(x.getScale(), x.LOGARITHMIC)
        self.assertEqual(y.getScale(), x.LOGARITHMIC)
        self.assertEqual(yright.getScale(), x.LOGARITHMIC)
        self.assertEqual(self.plot.isXAxisLogarithmic(), True)
        self.assertEqual(self.plot.isYAxisLogarithmic(), True)
        self.assertEqual(listener.arguments(callIndex=-1), ("y", True))

        yright.setScale(yright.LINEAR)
        self.assertEqual(x.getScale(), x.LOGARITHMIC)
        self.assertEqual(y.getScale(), x.LINEAR)
        self.assertEqual(yright.getScale(), x.LINEAR)
        self.assertEqual(self.plot.isXAxisLogarithmic(), True)
        self.assertEqual(self.plot.isYAxisLogarithmic(), False)
        self.assertEqual(listener.arguments(callIndex=-1), ("y", False))

    @utils.test_logging(deprecation.depreclog.name, warning=2)
    def testOldPlotAxis_AutoScale(self):
        """Test silx API prior to silx 0.6"""
        x = self.plot.getXAxis()
        y = self.plot.getYAxis()
        yright = self.plot.getYAxis(axis="right")

        listener = SignalListener()
        self.plot.sigSetXAxisAutoScale.connect(listener.partial("x"))
        self.plot.sigSetYAxisAutoScale.connect(listener.partial("y"))

        self.assertEqual(x.isAutoScale(), True)
        self.assertEqual(y.isAutoScale(), True)
        self.assertEqual(yright.isAutoScale(), True)

        self.plot.setXAxisAutoScale(False)
        self.assertEqual(x.isAutoScale(), False)
        self.assertEqual(y.isAutoScale(), True)
        self.assertEqual(yright.isAutoScale(), True)
        self.assertEqual(self.plot.isXAxisAutoScale(), False)
        self.assertEqual(self.plot.isYAxisAutoScale(), True)
        self.assertEqual(listener.arguments(callIndex=-1), ("x", False))

        self.plot.setYAxisAutoScale(False)
        self.assertEqual(x.isAutoScale(), False)
        self.assertEqual(y.isAutoScale(), False)
        self.assertEqual(yright.isAutoScale(), False)
        self.assertEqual(self.plot.isXAxisAutoScale(), False)
        self.assertEqual(self.plot.isYAxisAutoScale(), False)
        self.assertEqual(listener.arguments(callIndex=-1), ("y", False))

        yright.setAutoScale(True)
        self.assertEqual(x.isAutoScale(), False)
        self.assertEqual(y.isAutoScale(), True)
        self.assertEqual(yright.isAutoScale(), True)
        self.assertEqual(self.plot.isXAxisAutoScale(), False)
        self.assertEqual(self.plot.isYAxisAutoScale(), True)
        self.assertEqual(listener.arguments(callIndex=-1), ("y", True))

    @utils.test_logging(deprecation.depreclog.name, warning=1)
    def testOldPlotAxis_Inverted(self):
        """Test silx API prior to silx 0.6"""
        x = self.plot.getXAxis()
        y = self.plot.getYAxis()
        yright = self.plot.getYAxis(axis="right")

        listener = SignalListener()
        self.plot.sigSetYAxisInverted.connect(listener.partial("y"))

        self.assertEqual(x.isInverted(), False)
        self.assertEqual(y.isInverted(), False)
        self.assertEqual(yright.isInverted(), False)

        self.plot.setYAxisInverted(True)
        self.assertEqual(x.isInverted(), False)
        self.assertEqual(y.isInverted(), True)
        self.assertEqual(yright.isInverted(), True)
        self.assertEqual(self.plot.isYAxisInverted(), True)
        self.assertEqual(listener.arguments(callIndex=-1), ("y", True))

        yright.setInverted(False)
        self.assertEqual(x.isInverted(), False)
        self.assertEqual(y.isInverted(), False)
        self.assertEqual(yright.isInverted(), False)
        self.assertEqual(self.plot.isYAxisInverted(), False)
        self.assertEqual(listener.arguments(callIndex=-1), ("y", False))

    def testLogXWithData(self):
        self.plot.setGraphTitle('Curve X: Log Y: Linear')
        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=True,
                           color='green', linestyle="-", symbol='o')
        axis = self.plot.getXAxis()
        axis.setScale(axis.LOGARITHMIC)

        self.assertEqual(axis.getScale(), axis.LOGARITHMIC)

    def testLogYWithData(self):
        self.plot.setGraphTitle('Curve X: Linear Y: Log')
        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=True,
                           color='green', linestyle="-", symbol='o')
        axis = self.plot.getYAxis()
        axis.setScale(axis.LOGARITHMIC)

        self.assertEqual(axis.getScale(), axis.LOGARITHMIC)
        axis = self.plot.getYAxis(axis="right")
        self.assertEqual(axis.getScale(), axis.LOGARITHMIC)

    def testLogYRightWithData(self):
        self.plot.setGraphTitle('Curve X: Linear Y: Log')
        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=True,
                           color='green', linestyle="-", symbol='o')
        axis = self.plot.getYAxis(axis="right")
        axis.setScale(axis.LOGARITHMIC)

        self.assertEqual(axis.getScale(), axis.LOGARITHMIC)
        axis = self.plot.getYAxis()
        self.assertEqual(axis.getScale(), axis.LOGARITHMIC)

    def testLimitsChanged_setLimits(self):
        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=False,
                           color='green', linestyle="-", symbol='o')
        listener = SignalListener()
        self.plot.getXAxis().sigLimitsChanged.connect(listener.partial(axis="x"))
        self.plot.getYAxis().sigLimitsChanged.connect(listener.partial(axis="y"))
        self.plot.getYAxis(axis="right").sigLimitsChanged.connect(listener.partial(axis="y2"))
        self.plot.setLimits(0, 1, 0, 1, 0, 1)
        # at least one event per axis
        self.assertEquals(len(set(listener.karguments(argumentName="axis"))), 3)

    def testLimitsChanged_resetZoom(self):
        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=False,
                           color='green', linestyle="-", symbol='o')
        listener = SignalListener()
        self.plot.getXAxis().sigLimitsChanged.connect(listener.partial(axis="x"))
        self.plot.getYAxis().sigLimitsChanged.connect(listener.partial(axis="y"))
        self.plot.getYAxis(axis="right").sigLimitsChanged.connect(listener.partial(axis="y2"))
        self.plot.resetZoom()
        # at least one event per axis
        self.assertEquals(len(set(listener.karguments(argumentName="axis"))), 3)

    def testLimitsChanged_setXLimit(self):
        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=False,
                           color='green', linestyle="-", symbol='o')
        listener = SignalListener()
        axis = self.plot.getXAxis()
        axis.sigLimitsChanged.connect(listener)
        axis.setLimits(20, 30)
        # at least one event per axis
        self.assertEquals(listener.arguments(callIndex=-1), (20.0, 30.0))
        self.assertEquals(axis.getLimits(), (20.0, 30.0))

    def testLimitsChanged_setYLimit(self):
        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=False,
                           color='green', linestyle="-", symbol='o')
        listener = SignalListener()
        axis = self.plot.getYAxis()
        axis.sigLimitsChanged.connect(listener)
        axis.setLimits(20, 30)
        # at least one event per axis
        self.assertEquals(listener.arguments(callIndex=-1), (20.0, 30.0))
        self.assertEquals(axis.getLimits(), (20.0, 30.0))

    def testLimitsChanged_setYRightLimit(self):
        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=False,
                           color='green', linestyle="-", symbol='o')
        listener = SignalListener()
        axis = self.plot.getYAxis(axis="right")
        axis.sigLimitsChanged.connect(listener)
        axis.setLimits(20, 30)
        # at least one event per axis
        self.assertEquals(listener.arguments(callIndex=-1), (20.0, 30.0))
        self.assertEquals(axis.getLimits(), (20.0, 30.0))

    def testScaleProxy(self):
        listener = SignalListener()
        y = self.plot.getYAxis()
        yright = self.plot.getYAxis(axis="right")
        y.sigScaleChanged.connect(listener.partial("left"))
        yright.sigScaleChanged.connect(listener.partial("right"))
        yright.setScale(yright.LOGARITHMIC)

        self.assertEquals(y.getScale(), y.LOGARITHMIC)
        events = listener.arguments()
        self.assertEquals(len(events), 2)
        self.assertIn(("left", y.LOGARITHMIC), events)
        self.assertIn(("right", y.LOGARITHMIC), events)

    def testAutoScaleProxy(self):
        listener = SignalListener()
        y = self.plot.getYAxis()
        yright = self.plot.getYAxis(axis="right")
        y.sigAutoScaleChanged.connect(listener.partial("left"))
        yright.sigAutoScaleChanged.connect(listener.partial("right"))
        yright.setAutoScale(False)

        self.assertEquals(y.isAutoScale(), False)
        events = listener.arguments()
        self.assertEquals(len(events), 2)
        self.assertIn(("left", False), events)
        self.assertIn(("right", False), events)

    def testInvertedProxy(self):
        listener = SignalListener()
        y = self.plot.getYAxis()
        yright = self.plot.getYAxis(axis="right")
        y.sigInvertedChanged.connect(listener.partial("left"))
        yright.sigInvertedChanged.connect(listener.partial("right"))
        yright.setInverted(True)

        self.assertEquals(y.isInverted(), True)
        events = listener.arguments()
        self.assertEquals(len(events), 2)
        self.assertIn(("left", True), events)
        self.assertIn(("right", True), events)

    def testAxesDisplayedFalse(self):
        """Test coverage on setAxesDisplayed(False)"""
        self.plot.setAxesDisplayed(False)

    def testAxesDisplayedTrue(self):
        """Test coverage on setAxesDisplayed(True)"""
        self.plot.setAxesDisplayed(True)


class TestPlotCurveLog(_PlotWidgetTest, ParametricTestCase):
    """Basic tests for addCurve with log scale axes"""

    # Test data
    xData = numpy.arange(1000) + 1
    yData = xData ** 2

    def _setLabels(self):
        self.plot.getXAxis().setLabel('X')
        self.plot.getYAxis().setLabel('X * X')

    def testPlotCurveLogX(self):
        self._setLabels()
        self.plot.getXAxis()._setLogarithmic(True)
        self.plot.setGraphTitle('Curve X: Log Y: Linear')

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=True,
                           color='green', linestyle="-", symbol='o')

    def testPlotCurveLogY(self):
        self._setLabels()
        self.plot.getYAxis()._setLogarithmic(True)

        self.plot.setGraphTitle('Curve X: Linear Y: Log')

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=True,
                           color='green', linestyle="-", symbol='o')

    def testPlotCurveLogXY(self):
        self._setLabels()
        self.plot.getXAxis()._setLogarithmic(True)
        self.plot.getYAxis()._setLogarithmic(True)

        self.plot.setGraphTitle('Curve X: Log Y: Log')

        self.plot.addCurve(self.xData, self.yData,
                           legend="curve",
                           replace=False, resetzoom=True,
                           color='green', linestyle="-", symbol='o')

    def testPlotCurveErrorLogXY(self):
        self.plot.getXAxis()._setLogarithmic(True)
        self.plot.getYAxis()._setLogarithmic(True)

        # Every second error leads to negative number
        errors = numpy.ones_like(self.xData)
        errors[::2] = self.xData[::2] + 1

        tests = [  # name, xerror, yerror
            ('xerror=3', 3, None),
            ('xerror=N array', errors, None),
            ('xerror=Nx1 array', errors.reshape(len(errors), 1), None),
            ('xerror=2xN array', numpy.array((errors, errors)), None),
            ('yerror=6', None, 6),
            ('yerror=N array', None, errors ** 2),
            ('yerror=Nx1 array', None, (errors ** 2).reshape(len(errors), 1)),
            ('yerror=2xN array', None, numpy.array((errors, errors)) ** 2),
        ]

        for name, xError, yError in tests:
            with self.subTest(name):
                self.plot.setGraphTitle(name)
                self.plot.addCurve(self.xData, self.yData,
                                   legend=name,
                                   xerror=xError, yerror=yError,
                                   replace=False, resetzoom=True,
                                   color='green', linestyle="-", symbol='o')

                self.qapp.processEvents()

                self.plot.clear()
                self.plot.resetZoom()
                self.qapp.processEvents()

    def testPlotCurveToggleLog(self):
        """Add a curve with negative data and toggle log axis"""
        arange = numpy.arange(1000) + 1
        tests = [  # name, xData, yData
            ('x>0, some negative y', arange, arange - 500),
            ('x>0, y<0', arange, -arange),
            ('some negative x, y>0', arange - 500, arange),
            ('x<0, y>0', -arange, arange),
            ('some negative x and y', arange - 500, arange - 500),
            ('x<0, y<0', -arange, -arange),
        ]

        for name, xData, yData in tests:
            with self.subTest(name):
                self.plot.addCurve(xData, yData, resetzoom=True)
                self.qapp.processEvents()

                # no log axis
                xLim = self.plot.getXAxis().getLimits()
                self.assertEqual(xLim, (min(xData), max(xData)))
                yLim = self.plot.getYAxis().getLimits()
                self.assertEqual(yLim, (min(yData), max(yData)))

                # x axis log
                self.plot.getXAxis()._setLogarithmic(True)
                self.qapp.processEvents()

                xLim = self.plot.getXAxis().getLimits()
                yLim = self.plot.getYAxis().getLimits()
                positives = xData > 0
                if numpy.any(positives):
                    self.assertTrue(numpy.allclose(
                        xLim, (min(xData[positives]), max(xData[positives]))))
                    self.assertEqual(
                        yLim, (min(yData[positives]), max(yData[positives])))
                else:  # No positive x in the curve
                    self.assertEqual(xLim, (1., 100.))
                    self.assertEqual(yLim, (1., 100.))

                # x axis and y axis log
                self.plot.getYAxis()._setLogarithmic(True)
                self.qapp.processEvents()

                xLim = self.plot.getXAxis().getLimits()
                yLim = self.plot.getYAxis().getLimits()
                positives = numpy.logical_and(xData > 0, yData > 0)
                if numpy.any(positives):
                    self.assertTrue(numpy.allclose(
                        xLim, (min(xData[positives]), max(xData[positives]))))
                    self.assertTrue(numpy.allclose(
                        yLim, (min(yData[positives]), max(yData[positives]))))
                else:  # No positive x and y in the curve
                    self.assertEqual(xLim, (1., 100.))
                    self.assertEqual(yLim, (1., 100.))

                # y axis log
                self.plot.getXAxis()._setLogarithmic(False)
                self.qapp.processEvents()

                xLim = self.plot.getXAxis().getLimits()
                yLim = self.plot.getYAxis().getLimits()
                positives = yData > 0
                if numpy.any(positives):
                    self.assertEqual(
                        xLim, (min(xData[positives]), max(xData[positives])))
                    self.assertTrue(numpy.allclose(
                        yLim, (min(yData[positives]), max(yData[positives]))))
                else:  # No positive y in the curve
                    self.assertEqual(xLim, (1., 100.))
                    self.assertEqual(yLim, (1., 100.))

                # no log axis
                self.plot.getYAxis()._setLogarithmic(False)
                self.qapp.processEvents()

                xLim = self.plot.getXAxis().getLimits()
                self.assertEqual(xLim, (min(xData), max(xData)))
                yLim = self.plot.getYAxis().getLimits()
                self.assertEqual(yLim, (min(yData), max(yData)))

                self.plot.clear()
                self.plot.resetZoom()
                self.qapp.processEvents()


class TestPlotImageLog(_PlotWidgetTest):
    """Basic tests for addImage with log scale axes."""

    def setUp(self):
        super(TestPlotImageLog, self).setUp()

        self.plot.getXAxis().setLabel('Columns')
        self.plot.getYAxis().setLabel('Rows')

    def testPlotColormapGrayLogX(self):
        self.plot.getXAxis()._setLogarithmic(True)
        self.plot.setGraphTitle('CMap X: Log Y: Linear')

        colormap = Colormap(name='gray',
                            normalization='linear',
                            vmin=None,
                            vmax=None)
        self.plot.addImage(DATA_2D, legend="image 1",
                           origin=(1., 1.), scale=(1., 1.),
                           replace=False, resetzoom=False, colormap=colormap)
        self.plot.resetZoom()

    def testPlotColormapGrayLogY(self):
        self.plot.getYAxis()._setLogarithmic(True)
        self.plot.setGraphTitle('CMap X: Linear Y: Log')

        colormap = Colormap(name='gray',
                            normalization='linear',
                            vmin=None,
                            vmax=None)
        self.plot.addImage(DATA_2D, legend="image 1",
                           origin=(1., 1.), scale=(1., 1.),
                           replace=False, resetzoom=False, colormap=colormap)
        self.plot.resetZoom()

    def testPlotColormapGrayLogXY(self):
        self.plot.getXAxis()._setLogarithmic(True)
        self.plot.getYAxis()._setLogarithmic(True)
        self.plot.setGraphTitle('CMap X: Log Y: Log')

        colormap = Colormap(name='gray',
                            normalization='linear',
                            vmin=None,
                            vmax=None)
        self.plot.addImage(DATA_2D, legend="image 1",
                           origin=(1., 1.), scale=(1., 1.),
                           replace=False, resetzoom=False, colormap=colormap)
        self.plot.resetZoom()

    def testPlotRgbRgbaLogXY(self):
        self.plot.getXAxis()._setLogarithmic(True)
        self.plot.getYAxis()._setLogarithmic(True)
        self.plot.setGraphTitle('RGB + RGBA X: Log Y: Log')

        rgb = numpy.array(
            (((0, 0, 0), (128, 0, 0), (255, 0, 0)),
             ((0, 128, 0), (0, 128, 128), (0, 128, 256))),
            dtype=numpy.uint8)

        self.plot.addImage(rgb, legend="rgb",
                           origin=(1, 1), scale=(10, 10),
                           replace=False, resetzoom=False)

        rgba = numpy.array(
            (((0, 0, 0, .5), (.5, 0, 0, 1), (1, 0, 0, .5)),
             ((0, .5, 0, 1), (0, .5, .5, 1), (0, 1, 1, .5))),
            dtype=numpy.float32)

        self.plot.addImage(rgba, legend="rgba",
                           origin=(5., 5.), scale=(10., 10.),
                           replace=False, resetzoom=False)
        self.plot.resetZoom()


class TestPlotMarkerLog(_PlotWidgetTest):
    """Basic tests for markers on log scales"""

    # Test marker parameters
    markers = [  # x, y, color, selectable, draggable
        (10., 10., 'blue', False, False),
        (20., 20., 'red', False, False),
        (40., 100., 'green', True, False),
        (40., 500., 'gray', True, True),
        (60., 800., 'black', False, True),
    ]

    def setUp(self):
        super(TestPlotMarkerLog, self).setUp()

        self.plot.getYAxis().setLabel('Rows')
        self.plot.getXAxis().setLabel('Columns')
        self.plot.getXAxis().setAutoScale(False)
        self.plot.getYAxis().setAutoScale(False)
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setLimits(1., 100., 1., 1000.)
        self.plot.getXAxis()._setLogarithmic(True)
        self.plot.getYAxis()._setLogarithmic(True)

    def testPlotMarkerXLog(self):
        self.plot.setGraphTitle('Markers X, Log axes')

        for x, _, color, select, drag in self.markers:
            name = str(x)
            if select:
                name += " sel."
            if drag:
                name += " drag"
            self.plot.addXMarker(x, name, name, color, select, drag)
        self.plot.resetZoom()

    def testPlotMarkerYLog(self):
        self.plot.setGraphTitle('Markers Y, Log axes')

        for _, y, color, select, drag in self.markers:
            name = str(y)
            if select:
                name += " sel."
            if drag:
                name += " drag"
            self.plot.addYMarker(y, name, name, color, select, drag)
        self.plot.resetZoom()

    def testPlotMarkerPtLog(self):
        self.plot.setGraphTitle('Markers Pt, Log axes')

        for x, y, color, select, drag in self.markers:
            name = "{0},{1}".format(x, y)
            if select:
                name += " sel."
            if drag:
                name += " drag"
            self.plot.addMarker(x, y, name, name, color, select, drag)
        self.plot.resetZoom()


class TestPlotItemLog(_PlotWidgetTest):
    """Basic tests for items with log scale axes"""

    # Polygon coordinates and color
    polygons = [  # legend, x coords, y coords, color
        ('triangle', numpy.array((10, 30, 50)),
         numpy.array((55, 70, 55)), 'red'),
        ('square', numpy.array((10, 10, 50, 50)),
         numpy.array((10, 50, 50, 10)), 'green'),
        ('star', numpy.array((60, 70, 80, 60, 80)),
         numpy.array((25, 50, 25, 40, 40)), 'blue'),
    ]

    # Rectangle coordinantes and color
    rectangles = [  # legend, x coords, y coords, color
        ('square 1', numpy.array((1., 10.)),
         numpy.array((1., 10.)), 'red'),
        ('square 2', numpy.array((10., 20.)),
         numpy.array((10., 20.)), 'green'),
        ('square 3', numpy.array((20., 30.)),
         numpy.array((20., 30.)), 'blue'),
        ('rect 1', numpy.array((1., 30.)),
         numpy.array((35., 40.)), 'black'),
        ('line h', numpy.array((1., 30.)),
         numpy.array((45., 45.)), 'darkRed'),
    ]

    def setUp(self):
        super(TestPlotItemLog, self).setUp()

        self.plot.getYAxis().setLabel('Rows')
        self.plot.getXAxis().setLabel('Columns')
        self.plot.getXAxis().setAutoScale(False)
        self.plot.getYAxis().setAutoScale(False)
        self.plot.setKeepDataAspectRatio(False)
        self.plot.setLimits(1., 100., 1., 100.)
        self.plot.getXAxis()._setLogarithmic(True)
        self.plot.getYAxis()._setLogarithmic(True)

    def testPlotItemPolygonLogFill(self):
        self.plot.setGraphTitle('Item Fill Log')

        for legend, xList, yList, color in self.polygons:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="polygon", fill=True, color=color)
        self.plot.resetZoom()

    def testPlotItemPolygonLogNoFill(self):
        self.plot.setGraphTitle('Item No Fill Log')

        for legend, xList, yList, color in self.polygons:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="polygon", fill=False, color=color)
        self.plot.resetZoom()

    def testPlotItemRectangleLogFill(self):
        self.plot.setGraphTitle('Rectangle Fill Log')

        for legend, xList, yList, color in self.rectangles:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="rectangle", fill=True, color=color)
        self.plot.resetZoom()

    def testPlotItemRectangleLogNoFill(self):
        self.plot.setGraphTitle('Rectangle No Fill Log')

        for legend, xList, yList, color in self.rectangles:
            self.plot.addItem(xList, yList, legend=legend,
                              replace=False,
                              shape="rectangle", fill=False, color=color)
        self.plot.resetZoom()


def suite():
    test_suite = unittest.TestSuite()
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTests(TestPlotWidget))
    test_suite.addTest(loadTests(TestPlotImage))
    test_suite.addTest(loadTests(TestPlotCurve))
    test_suite.addTest(loadTests(TestPlotMarker))
    test_suite.addTest(loadTests(TestPlotItem))
    test_suite.addTest(loadTests(TestPlotAxes))
    test_suite.addTest(loadTests(TestPlotEmptyLog))
    test_suite.addTest(loadTests(TestPlotCurveLog))
    test_suite.addTest(loadTests(TestPlotImageLog))
    test_suite.addTest(loadTests(TestPlotMarkerLog))
    test_suite.addTest(loadTests(TestPlotItemLog))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
