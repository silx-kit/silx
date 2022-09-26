#!/usr/bin/env python
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
"""
Example to show the use of :class:`~silx.gui.plot.ScatterView.ScatterView` widget.
"""

__license__ = "MIT"

import logging
from silx.gui.plot.ScatterView import ScatterView
from silx.gui import qt
import numpy
import scipy.signal


logging.basicConfig()
logger = logging.getLogger(__name__)


def createData():
    nbPoints = 200
    nbX = int(numpy.sqrt(nbPoints))
    nbY = nbPoints // nbX + 1

    # Motor position
    yy = numpy.atleast_2d(numpy.ones(nbY)).T
    xx = numpy.atleast_2d(numpy.ones(nbX))

    positionX = numpy.linspace(10, 50, nbX) * yy
    positionX = positionX.reshape(nbX * nbY)
    positionX = positionX + numpy.random.rand(len(positionX)) - 0.5

    positionY = numpy.atleast_2d(numpy.linspace(20, 60, nbY)).T * xx
    positionY = positionY.reshape(nbX * nbY)
    positionY = positionY + numpy.random.rand(len(positionY)) - 0.5

    # Diodes position
    lut = scipy.signal.gaussian(max(nbX, nbY), std=8) * 10
    yy, xx = numpy.ogrid[:nbY, :nbX]
    signal = lut[yy] * lut[xx]
    diode1 = numpy.random.poisson(signal * 10)
    diode1 = diode1.reshape(nbX * nbY)
    return positionX, positionY, diode1


def main(argv=None):
    """Display an image from a file in an :class:`ImageView` widget.

    :param argv: list of command line arguments or None (the default)
                 to use sys.argv.
    :type argv: list of str
    :return: Exit status code
    :rtype: int
    :raises IOError: if no image can be loaded from the file
    """
    import argparse
    import os.path

    global app  # QApplication must be global to avoid seg fault on quit
    app = qt.QApplication([])
    sys.excepthook = qt.exceptionHandler

    mainWindow = ScatterView()
    mainWindow.setAttribute(qt.Qt.WA_DeleteOnClose)
    xx, yy, value = createData()
    mainWindow.setData(x=xx, y=yy, value=value)
    mainWindow.show()
    mainWindow.setFocus(qt.Qt.OtherFocusReason)

    return app.exec()


if __name__ == "__main__":
    import sys
    sys.exit(main(argv=sys.argv[1:]))
