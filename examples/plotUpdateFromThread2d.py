# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""This script illustrates the update of a :mod:`silx.gui.plot` widget from a thread.

The problem is that plot and GUI methods should be called from the main thread.
To safely update the plot from another thread, one need to make the update
asynchronously from the main thread.
In this example, this is achieved through a Qt signal.

In this example we create a subclass of :class:`~silx.gui.plot.PlotWindow.Plot1D`
that adds a thread-safe method to add curves:
:meth:`ThreadSafePlot1D.addCurveThreadSafe`.
This thread-safe method is then called from a thread to update the plot.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/09/2017"


import threading
import time

import numpy

from silx.gui import qt
from silx.gui.plot import Plot1D


class ThreadSafePlot1D(Plot1D):
    """Add a thread-safe :meth:`addCurveThreadSafe` method to Plot1D.
    """

    _sigAddCurve = qt.Signal(tuple, dict)
    """Signal used to perform addCurve in the main thread.

    It takes args and kwargs as arguments.
    """

    def __init__(self, parent=None):
        super(ThreadSafePlot1D, self).__init__(parent)
        # Connect the signal to the method actually calling addCurve
        self._sigAddCurve.connect(self.__addCurve)

    def __addCurve(self, args, kwargs):
        """Private method calling addCurve from _sigAddCurve"""
        self.addCurve(*args, **kwargs)

    def addCurveThreadSafe(self, *args, **kwargs):
        """Thread-safe version of :meth:`silx.gui.plot.Plot.addCurve`

        This method takes the same arguments as Plot.addCurve.

        WARNING: This method does not return a value as opposed to Plot.addCurve
        """
        self._sigAddCurve.emit(args, kwargs)


class UpdateThread(threading.Thread):
    """Thread updating the curve of a :class:`ThreadSafePlot1D`

    :param plot1d: The ThreadSafePlot1D to update."""

    def __init__(self, plot1d):
        self.plot1d = plot1d
        self.running = False
        super(UpdateThread, self).__init__()

    def start(self):
        """Start the update thread"""
        self.running = True
        super(UpdateThread, self).start()

    def run(self):
        """Method implementing thread loop that updates the plot"""
        while self.running:
            time.sleep(1)
            self.plot1d.addCurveThreadSafe(
                numpy.arange(1000), numpy.random.random(1000), resetzoom=False)

    def stop(self):
        """Stop the update thread"""
        self.running = False
        self.join(2)


def main():
    global app
    app = qt.QApplication([])

    # Create a ThreadSafePlot1D, set its limits and display it
    plot1d = ThreadSafePlot1D()
    plot1d.setLimits(0., 1000., 0., 1.)
    plot1d.show()

    # Create the thread that calls ThreadSafePlot1D.addCurveThreadSafe
    updateThread = UpdateThread(plot1d)
    updateThread.start()  # Start updating the plot

    app.exec_()

    updateThread.stop()  # Stop updating the plot


if __name__ == '__main__':
    main()
