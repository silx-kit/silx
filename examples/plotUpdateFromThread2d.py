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
"""This script illustrates the update of a :mod:`silx.gui.plot` widget from a
thread.

The problem is that plot and GUI methods should be called from the main thread.
To safely update the plot from another thread, one need to make the update
asynchronously from the main thread.
In this example, this is achieved through a Qt signal.

In this example we create a subclass of
:class:`~silx.gui.plot.PlotWindow.Plot2D`
that adds a thread-safe method to add images:
:meth:`ThreadSafePlot1D.addImageThreadSafe`.
This thread-safe method is then called from a thread to update the plot.

Update from 1d to 2d example by Hans Fangohr, European XFEL GmbH, 26 Feb 2018.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/09/2017"


import threading
import time

import numpy

from silx.gui import qt
from silx.gui.plot import Plot2D


Nx = 150
Ny = 50


class ThreadSafePlot2D(Plot2D):
    """Add a thread-safe :meth:`addImageThreadSafe` method to Plot2D.
    """

    _sigAddImage = qt.Signal(tuple, dict)
    """Signal used to perform addImage in the main thread.

    It takes args and kwargs as arguments.
    """

    def __init__(self, parent=None):
        super(ThreadSafePlot2D, self).__init__(parent)
        # Connect the signal to the method actually calling addImage
        self._sigAddImage.connect(self.__addImage)

    def __addImage(self, args, kwargs):
        """Private method calling addImage from _sigAddImage"""
        self.addImage(*args, **kwargs)

    def addImageThreadSafe(self, *args, **kwargs):
        """Thread-safe version of :meth:`silx.gui.plot.Plot.addImage`

        This method takes the same arguments as Plot.addImage.

        WARNING: This method does not return a value as opposed to
        Plot.addImage
        """
        self._sigAddImage.emit(args, kwargs)


class UpdateThread(threading.Thread):
    """Thread updating the image of a :class:`ThreadSafePlot2D`

    :param plot2d: The ThreadSafePlot2D to update."""

    def __init__(self, plot2d):
        self.plot2d = plot2d
        self.running = False
        super(UpdateThread, self).__init__()

    def start(self):
        """Start the update thread"""
        self.running = True
        super(UpdateThread, self).start()

    def run(self, pos={'x0': 0, 'y0': 0}):
        """Method implementing thread loop that updates the plot"""
        while self.running:
            time.sleep(1)
            # pixels in plot (defined at beginning of file)
            # Nx = 70
            # Ny = 50
            # width of peak
            sigma_x = 0.15
            sigma_y = 0.25
            # x and y positions
            x = numpy.linspace(-1.5, 1.5, Nx)
            y = numpy.linspace(-1.0, 1.0, Ny)
            xv, yv = numpy.meshgrid(x, y)
            signal = numpy.exp(- ((xv - pos['x0']) ** 2 / sigma_x ** 2
                                  + (yv - pos['y0']) ** 2 / sigma_y ** 2))
            # add noise
            signal += 0.3 * numpy.random.random(size=signal.shape)
            # random walk of center of peak ('drift')
            pos['x0'] += 0.05 * (numpy.random.random() - 0.5)
            pos['y0'] += 0.05 * (numpy.random.random() - 0.5)
            # plot the data
            self.plot2d.addImage(signal, replace=True, resetzoom=False)

    def stop(self):
        """Stop the update thread"""
        self.running = False
        self.join(2)


def main():
    global app
    app = qt.QApplication([])

    # Create a ThreadSafePlot2D, set its limits and display it
    plot2d = ThreadSafePlot2D()
    plot2d.setLimits(0, Nx, 0, Ny)
    plot2d.show()

    # Create the thread that calls ThreadSafePlot2D.addImageThreadSafe
    updateThread = UpdateThread(plot2d)
    updateThread.start()  # Start updating the plot

    app.exec_()

    updateThread.stop()  # Stop updating the plot


if __name__ == '__main__':
    main()
