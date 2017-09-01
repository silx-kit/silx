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


import logging
import contextlib

from silx.gui.test.utils import TestCaseQt

from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.backends.BackendMatplotlib import BackendMatplotlibQt


logger = logging.getLogger(__name__)


class PlotWidgetTestCase(TestCaseQt):
    """Base class for tests of PlotWidget, not a TestCase in itself.

    plot attribute is the PlotWidget created for the test.
    """

    def __init__(self, methodName='runTest'):
        TestCaseQt.__init__(self, methodName=methodName)
        self.__mousePos = None

    def _createPlot(self):
        return PlotWidget()

    def setUp(self):
        super(PlotWidgetTestCase, self).setUp()
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
        super(PlotWidgetTestCase, self).tearDown()

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
        with self._waitForMplEvent(widget, "motion_notify_event"):
            TestCaseQt.mouseMove(self, widget, pos=pos, delay=delay)

    def mouseClick(self, widget, button, modifier=None, pos=None, delay=-1):
        """Override TestCaseQt to wait while MPL did not reveive the expected
        event"""
        pos = self._patchPos(widget, pos)
        self._checkMouseMove(widget, pos)
        if not self._haveMplEvent(widget, pos):
            return TestCaseQt.mouseClick(self, widget, button, modifier=modifier, pos=pos, delay=delay)
        with self._waitForMplEvent(widget, "button_release_event"):
            TestCaseQt.mouseClick(self, widget, button, modifier=modifier, pos=pos, delay=delay)

    def mousePress(self, widget, button, modifier=None, pos=None, delay=-1):
        """Override TestCaseQt to wait while MPL did not reveive the expected
        event"""
        pos = self._patchPos(widget, pos)
        self._checkMouseMove(widget, pos)
        if not self._haveMplEvent(widget, pos):
            return TestCaseQt.mousePress(self, widget, button, modifier=modifier, pos=pos, delay=delay)
        with self._waitForMplEvent(widget, "button_press_event"):
            TestCaseQt.mousePress(self, widget, button, modifier=modifier, pos=pos, delay=delay)

    def mouseRelease(self, widget, button, modifier=None, pos=None, delay=-1):
        """Override TestCaseQt to wait while MPL did not reveive the expected
        event"""
        pos = self._patchPos(widget, pos)
        self._checkMouseMove(widget, pos)
        if not self._haveMplEvent(widget, pos):
            return TestCaseQt.mouseRelease(self, widget, button, modifier=modifier, pos=pos, delay=delay)
        with self._waitForMplEvent(widget, "button_release_event"):
            TestCaseQt.mouseRelease(self, widget, button, modifier=modifier, pos=pos, delay=delay)
