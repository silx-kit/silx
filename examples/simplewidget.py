#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
"""This script shows a gallery of simple widgets provided by silx.

It shows the following widgets:

- :class:`~silx.gui.widgets.WaitingPushButton`:
  A button with a progress-like waiting animated icon.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "02/08/2018"

import sys
import functools
import numpy

from silx.gui import qt
from silx.gui.colors import Colormap
from silx.gui.widgets.WaitingPushButton import WaitingPushButton
from silx.gui.widgets.ThreadPoolPushButton import ThreadPoolPushButton
from silx.gui.widgets.RangeSlider import RangeSlider


class SimpleWidgetExample(qt.QMainWindow):
    """This windows shows simple widget provided by silx."""

    def __init__(self):
        """Constructor"""
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Silx simple widget example")

        main_panel = qt.QWidget(self)
        main_panel.setLayout(qt.QVBoxLayout())

        layout = main_panel.layout()
        layout.addWidget(qt.QLabel("WaitingPushButton"))
        layout.addWidget(self.createWaitingPushButton())
        layout.addWidget(self.createWaitingPushButton2())

        layout.addWidget(qt.QLabel("ThreadPoolPushButton"))
        layout.addWidget(self.createThreadPoolPushButton())

        layout.addWidget(qt.QLabel("RangeSlider"))
        layout.addWidget(self.createRangeSlider())
        layout.addWidget(self.createRangeSliderWithBackground())

        self.setCentralWidget(main_panel)

    def createWaitingPushButton(self):
        widget = WaitingPushButton(text="Push me and wait for ever")
        widget.clicked.connect(widget.swapWaiting)
        return widget

    def createWaitingPushButton2(self):
        widget = WaitingPushButton(text="Push me")
        widget.setDisabledWhenWaiting(False)
        widget.clicked.connect(widget.swapWaiting)
        return widget

    def printResult(self, result):
        print(result)

    def printError(self, result):
        print("Error")
        print(result)

    def printEvent(self, eventName, *args):
        print("Event %s: %s" % (eventName, args))

    def takesTimeToComputePow(self, a, b):
        qt.QThread.sleep(2)
        return a ** b

    def createThreadPoolPushButton(self):
        widget = ThreadPoolPushButton(text="Compute 2^16")
        widget.setCallable(self.takesTimeToComputePow, 2, 16)
        widget.succeeded.connect(self.printResult)
        widget.failed.connect(self.printError)
        return widget

    def createRangeSlider(self):
        widget = RangeSlider(self)
        widget.setRange(0, 500)
        widget.setValues(100, 400)
        widget.sigValueChanged.connect(functools.partial(self.printEvent, "sigValueChanged"))
        widget.sigPositionChanged.connect(functools.partial(self.printEvent, "sigPositionChanged"))
        widget.sigPositionCountChanged.connect(functools.partial(self.printEvent, "sigPositionCountChanged"))
        return widget

    def createRangeSliderWithBackground(self):
        widget = RangeSlider(self)
        widget.setRange(0, 500)
        widget.setValues(100, 400)
        background = numpy.sin(numpy.arange(250) / 250.0)
        background[0], background[-1] = background[-1], background[0]
        colormap = Colormap("viridis")
        widget.setGroovePixmapFromProfile(background, colormap)
        return widget


def main():
    """
    Main function
    """
    app = qt.QApplication([])
    sys.excepthook = qt.exceptionHandler
    window = SimpleWidgetExample()
    window.show()
    result = app.exec_()
    # remove ending warnings relative to QTimer
    app.deleteLater()
    sys.excepthook = sys.__excepthook__
    sys.exit(result)


main()
