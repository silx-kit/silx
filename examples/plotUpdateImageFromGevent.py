# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2022 European Synchrotron Radiation Facility
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
widget from a gevent coroutine.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/09/2017"


import threading
import time
import gevent
import logging

import numpy

from silx.gui import qt
from silx.gui.utils import concurrent
from silx.gui.plot import Plot2D

_logger = logging.getLogger(__name__)

Nx = 150
Ny = 50


def process_gevent():
    """Process gevent in case of QTimer triggering it."""
    try:
        gevent.sleep(0.01)
    except Exception:
        _logger.critical("Uncaught exception from gevent", exc_info=True)


def update_image(plot2d):
    """Update the image of a :class:`~sil.gui.plot.Plot2D`

    :param plot2d: The Plot2D to update."""

    pos = {'x0': 0, 'y0': 0}
    while True:
        gevent.sleep(0.01)

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

        plot2d.addImage(signal, resetzoom=False)


def main():
    global app
    app = qt.QApplication([])

    gevent_timer = qt.QTimer()
    gevent_timer.start(10)
    gevent_timer.timeout.connect(process_gevent)

    # Create a Plot2D, set its limits and display it
    plot2d = Plot2D()
    plot2d.getIntensityHistogramAction().setVisible(True)
    plot2d.setLimits(0, Nx, 0, Ny)
    plot2d.getDefaultColormap().setVRange(0., 1.5)
    plot2d.show()

    # Create the thread that calls submitToQtMainThread
    updater = gevent.spawn(update_image, plot2d)

    app.exec()

    updater.kill()  # Stop updating the plot
    updater.join()


if __name__ == '__main__':
    main()
