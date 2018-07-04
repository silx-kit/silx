# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016=2017 European Synchrotron Radiation Facility
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
"""Tests of plot interaction, through a PlotWidget"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "01/09/2017"


import unittest
from silx.gui import qt
from .utils import PlotWidgetTestCase


class _SignalDump(object):
    """Callable object that store passed arguments in a list"""

    def __init__(self):
        self._received = []

    def __call__(self, *args):
        self._received.append(args)

    @property
    def received(self):
        """Return a shallow copy of the list of received arguments"""
        return list(self._received)


class TestSelectPolygon(PlotWidgetTestCase):
    """Test polygon selection interaction"""

    def _interactionModeChanged(self, source):
        """Check that source received in event is the correct one"""
        self.assertEqual(source, self)

    def _draw(self, polygon):
        """Draw a polygon in the plot

        :param polygon: List of points (x, y) of the polygon (closed)
        """
        plot = self.plot.getWidgetHandle()

        dump = _SignalDump()
        self.plot.sigPlotSignal.connect(dump)

        for pos in polygon:
            self.mouseMove(plot, pos=pos)
            self.mouseClick(plot, qt.Qt.LeftButton, pos=pos)

        self.plot.sigPlotSignal.disconnect(dump)
        return [args[0] for args in dump.received]

    def test(self):
        """Test draw polygons + events"""
        self.plot.sigInteractiveModeChanged.connect(
            self._interactionModeChanged)

        self.plot.setInteractiveMode(
            'draw', shape='polygon', label='test', source=self)
        interaction = self.plot.getInteractiveMode()

        self.assertEqual(interaction['mode'], 'draw')
        self.assertEqual(interaction['shape'], 'polygon')

        self.plot.sigInteractiveModeChanged.disconnect(
            self._interactionModeChanged)

        plot = self.plot.getWidgetHandle()
        xCenter, yCenter = plot.width() // 2, plot.height() // 2
        offset = min(plot.width(), plot.height()) // 10

        # Star polygon
        star = [(xCenter, yCenter + offset),
                (xCenter - offset, yCenter - offset),
                (xCenter + offset, yCenter),
                (xCenter - offset, yCenter),
                (xCenter + offset, yCenter - offset),
                (xCenter, yCenter + offset)]  # Close polygon

        # Draw while dumping signals
        events = self._draw(star)

        # Test last event
        drawEvents = [event for event in events
                      if event['event'].startswith('drawing')]
        self.assertEqual(drawEvents[-1]['event'], 'drawingFinished')
        self.assertEqual(len(drawEvents[-1]['points']), 6)

        # Large square
        largeSquare = [(xCenter - offset, yCenter - offset),
                       (xCenter + offset, yCenter - offset),
                       (xCenter + offset, yCenter + offset),
                       (xCenter - offset, yCenter + offset),
                       (xCenter - offset, yCenter - offset)]  # Close polygon

        # Draw while dumping signals
        events = self._draw(largeSquare)

        # Test last event
        drawEvents = [event for event in events
                      if event['event'].startswith('drawing')]
        self.assertEqual(drawEvents[-1]['event'], 'drawingFinished')
        self.assertEqual(len(drawEvents[-1]['points']), 5)

        # Rectangle too thin along X: Some points are ignored
        thinRectX = [(xCenter, yCenter - offset),
                     (xCenter, yCenter + offset),
                     (xCenter + 1, yCenter + offset),
                     (xCenter + 1, yCenter - offset)]  # Close polygon

        # Draw while dumping signals
        events = self._draw(thinRectX)

        # Test last event
        drawEvents = [event for event in events
                      if event['event'].startswith('drawing')]
        self.assertEqual(drawEvents[-1]['event'], 'drawingFinished')
        self.assertEqual(len(drawEvents[-1]['points']), 3)

        # Rectangle too thin along Y: Some points are ignored
        thinRectY = [(xCenter - offset, yCenter),
                     (xCenter + offset, yCenter),
                     (xCenter + offset, yCenter + 1),
                     (xCenter - offset, yCenter + 1)]  # Close polygon

        # Draw while dumping signals
        events = self._draw(thinRectY)

        # Test last event
        drawEvents = [event for event in events
                      if event['event'].startswith('drawing')]
        self.assertEqual(drawEvents[-1]['event'], 'drawingFinished')
        self.assertEqual(len(drawEvents[-1]['points']), 3)


def suite():
    test_suite = unittest.TestSuite()
    for TestClass in (TestSelectPolygon,):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
