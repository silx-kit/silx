#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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
"""Example demonstrating animated axis"""

import numpy
from silx.gui import qt
from silx.gui.plot import Plot1D
from silx.gui.plot.actions import control as control_actions


class TestAnimatedAxes(Plot1D):
    def __init__(self, parent=None):
        super(TestAnimatedAxes, self).__init__(parent=parent)
        self._createActions()
        self._i = 0
        self._timer = qt.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._toggleSequence()

        toolbar = qt.QToolBar(self)
        toolbar.addAction(control_actions.OpenGLAction(parent=toolbar, plot=self))
        self.addToolBar(toolbar)


    def _createActions(self):
        action = qt.QAction(self)
        action.setText("Start/stop sequence")
        action.triggered.connect(self._toggleSequence)
        action.setShortcut(" ")
        self.addAction(action)

        action = qt.QAction(self)
        action.setText("Create/remove curve1")
        action.triggered.connect(self._toggleCurve1)
        action.setShortcut("1")
        self.addAction(action)

        action = qt.QAction(self)
        action.setText("Create/remove curve1")
        action.triggered.connect(self._toggleCurve2)
        action.setShortcut("2")
        self.addAction(action)

        action = qt.QAction(self)
        action.setText("Create/remove curve1")
        action.triggered.connect(self._toggleCurve3)
        action.setShortcut("3")
        self.addAction(action)

        action = qt.QAction(self)
        action.setText("Create/remove curve1")
        action.triggered.connect(self._toggleCurve4)
        action.setShortcut("4")
        self.addAction(action)

        action = qt.QAction(self)
        action.setText("Create/remove curve5")
        action.triggered.connect(self._toggleCurve5)
        action.setShortcut("5")
        self.addAction(action)

    def _executeCommand(self, command):
        sequence = qt.QKeySequence(command)
        for action in self.actions():
            if action.shortcut() == sequence:
                action.trigger()

    def _tick(self):
        sequence = "12531415"
        command = sequence[self._i % len(sequence)]
        self._executeCommand(command)
        self._i += 1

    def _toggleSequence(self):
        if self._timer.isActive():
            self._timer.stop()
        else:
            self._timer.start(2000)

    def _toggleCurve1(self):
        legend = "curve1"
        curve = self.getCurve(legend)
        if curve is None:
            xx = numpy.arange(0, 50)
            yy = numpy.sin(xx) + 2
            self.addCurve(xx, yy, legend=legend)
        else:
            self.removeCurve(legend)
            self.resetZoom()

    def _toggleCurve2(self):
        legend = "curve2"
        curve = self.getCurve(legend)
        if curve is None:
            xx = numpy.arange(100, 200)
            yy = numpy.sin(xx) + 3
            self.addCurve(xx, yy, legend=legend)
        else:
            self.removeCurve(legend)
            self.resetZoom()

    def _toggleCurve3(self):
        legend = "curve3"
        curve = self.getCurve(legend)
        if curve is None:
            xx = numpy.arange(10, 100)
            yy = numpy.sin(xx) + -1
            self.addCurve(xx, yy, legend=legend)
        else:
            self.removeCurve(legend)
            self.resetZoom()

    def _toggleCurve4(self):
        legend = "curve4"
        curve = self.getCurve(legend)
        if curve is None:
            xx = numpy.arange(110, 130)
            yy = numpy.sin(xx) + 1.5
            self.addCurve(xx, yy, legend=legend, yaxis="right")
        else:
            self.removeCurve(legend)
            self.resetZoom()

    def _toggleCurve5(self):
        legend = "curve5"
        curve = self.getCurve(legend)
        if curve is None:
            xx = numpy.arange(-10, 30)
            yy = numpy.sin(xx) + 1.5
            self.addCurve(xx, yy, legend=legend, yaxis="right")
        else:
            self.removeCurve(legend)
            self.resetZoom()


def main():
    app = qt.QApplication([])
    plot = TestAnimatedAxes()
    plot.getView().setAnimated(True)
    plot.show()
    return app.exec_()


if __name__ == "__main__":
    main()
