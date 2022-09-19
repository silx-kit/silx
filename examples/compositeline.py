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
Example to show the use of markers to draw head and tail of lines.
"""

__license__ = "MIT"

import logging
from silx.gui.plot import Plot1D
from silx.gui import qt
import numpy


logging.basicConfig()
logger = logging.getLogger(__name__)


def main(argv=None):
    """Display few lines with markers.
    """
    global app  # QApplication must be global to avoid seg fault on quit
    app = qt.QApplication([])
    sys.excepthook = qt.exceptionHandler

    mainWindow = Plot1D(backend="gl")
    mainWindow.setAttribute(qt.Qt.WA_DeleteOnClose)
    plot = mainWindow
    plot.setDataMargins(0.1, 0.1, 0.1, 0.1)

    plot.addCurve(x=[-10,0,0,-10,-10], y=[90,90,10,10,90], legend="box1", color="gray")
    plot.addCurve(x=[110,100,100,110,110], y=[90,90,10,10,90], legend="box2", color="gray")
    plot.addCurve(y=[-10,0,0,-10,-10], x=[90,90,10,10,90], legend="box3", color="gray")
    plot.addCurve(y=[110,100,100,110,110], x=[90,90,10,10,90], legend="box4", color="gray")

    def addLine(source, destination, symbolSource, symbolDestination, legend, color):
        line = numpy.array([source, destination]).T
        plot.addCurve(x=line[0,:], y=line[1,:], color=color, legend=legend)
        plot.addMarker(x=source[0], y=source[1], symbol=symbolSource, color=color)
        plot.addMarker(x=destination[0], y=destination[1], symbol=symbolDestination, color=color)

    addLine([0, 50], [100, 50], "caretleft", "caretright", "l1", "red")
    addLine([0, 30], [100, 30], "tickup", "tickdown", "l2", "blue")
    addLine([0, 70], [100, 70], "|", "|", "l3", "black")

    addLine([50, 0], [50, 100], "caretdown", "caretup", "l4", "red")
    addLine([30, 0], [30, 100], "tickleft", "tickright", "l5", "blue")
    addLine([70, 0], [70, 100], "_", "_", "l6", "black")

    mainWindow.setVisible(True)
    return app.exec()


if __name__ == "__main__":
    import sys
    sys.exit(main(argv=sys.argv[1:]))
