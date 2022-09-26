# /*##########################################################################
#
# Copyright (c) 2017-2021 European Synchrotron Radiation Facility
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
"""This script illustrates the update of a :class:`~silx.gui.plot.Plot2D`
widget from a thread.

The problem is that plot and GUI methods should be called from the main thread.
To safely update the plot from another thread, one need to execute the update
asynchronously in the main thread.
In this example, this is achieved with
:func:`~silx.gui.utils.concurrent.submitToQtMainThread`.

In this example a thread calls submitToQtMainThread to update the image
of a plot.

Update from 1d to 2d example by Hans Fangohr, European XFEL GmbH, 26 Feb 2018.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/09/2017"


import threading
import time

import numpy

from silx.gui import qt
from silx.gui.utils import concurrent
from silx.gui.plot import Plot2D


Nx = 150
Ny = 50


class UpdateThread(threading.Thread):
    """Thread updating the image of a :class:`~sil.gui.plot.Plot2D`

    :param plot2d: The Plot2D to update."""

    def __init__(self, plot2d):
        self.plot2d = plot2d
        self.running = False
        self.future_result = None
        super(UpdateThread, self).__init__()

    def start(self):
        """Start the update thread"""
        self.running = True
        super(UpdateThread, self).start()

    def run(self, pos={'x0': 0, 'y0': 0}):
        """Method implementing thread loop that updates the plot

        It produces an image every 10 ms or so, and
        either updates the plot or skip the image
        """
        while self.running:
            time.sleep(0.01)

            # Create image
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

            # If previous frame was not added to the plot yet, skip this one
            if self.future_result is None or self.future_result.done():
                # plot the data asynchronously, and
                # keep a reference to the `future` object
                self.future_result = concurrent.submitToQtMainThread(
                    self.plot2d.addImage, signal, resetzoom=False)

    def stop(self):
        """Stop the update thread"""
        self.running = False
        self.join(2)


def main():
    global app
    app = qt.QApplication([])

    # Create a Plot2D, set its limits and display it
    plot2d = Plot2D()
    plot2d.getIntensityHistogramAction().setVisible(True)
    plot2d.setLimits(0, Nx, 0, Ny)
    plot2d.getDefaultColormap().setVRange(0., 1.5)
    plot2d.show()

    # Create the thread that calls submitToQtMainThread
    updateThread = UpdateThread(plot2d)
    updateThread.start()  # Start updating the plot

    app.exec()

    updateThread.stop()  # Stop updating the plot


if __name__ == '__main__':
    main()
