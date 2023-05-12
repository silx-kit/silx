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
"""This script illustrates the update of a :mod:`silx.gui.plot` widget from a thread.

The problem is that plot and GUI methods should be called from the main thread.
To safely update the plot from another thread, one need to execute the update
asynchronously in the main thread.
In this example, this is achieved with
:func:`~silx.gui.utils.concurrent.submitToQtMainThread`.

In this example a thread calls submitToQtMainThread to update the curve
of a plot.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/09/2017"


import threading
import time

import numpy

from silx.gui import qt
from silx.gui.utils import concurrent

from silx.gui.plot import Plot1D


class UpdateThread(threading.Thread):
    """Thread updating the curve of a :class:`~silx.gui.plot.Plot1D`

    :param plot1d: The Plot1D to update."""

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
            # Run plot update asynchronously
            concurrent.submitToQtMainThread(
                self.plot1d.addCurve,
                numpy.arange(1000),
                numpy.random.random(1000),
                resetzoom=False)

    def stop(self):
        """Stop the update thread"""
        self.running = False
        self.join(2)


def main():
    global app
    app = qt.QApplication([])

    # Create a Plot1D, set its limits and display it
    plot1d = Plot1D()
    plot1d.setLimits(0., 1000., 0., 1.)
    plot1d.show()

    # Create the thread that calls submitToQtMainThread
    updateThread = UpdateThread(plot1d)
    updateThread.start()  # Start updating the plot

    app.exec()

    updateThread.stop()  # Stop updating the plot


if __name__ == '__main__':
    main()
