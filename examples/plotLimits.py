#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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
"""This script is an example to illustrate how to set range constraints on
plot axes.
"""

from silx.gui import qt
from silx.gui import plot
import numpy
import silx.test.utils


class ConstrainedViewPlot(qt.QMainWindow):

    def __init__(self):
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Plot with constrained axes")
        widget = qt.QWidget(self)
        self.setCentralWidget(widget)

        layout = qt.QGridLayout()
        widget.setLayout(layout)

        backend = "mpl"

        data = numpy.arange(100 * 100)
        data = (data % 100) / 5.0
        data = numpy.sin(data)
        data = silx.test.utils.add_gaussian_noise(data, mean=0.01)
        data.shape = 100, 100

        data1d = numpy.mean(data, axis=0)

        self.plot2d = plot.Plot2D(parent=widget, backend=backend)
        self.plot2d.setGraphTitle("A pixel can't be too big")
        self.plot2d.setInteractiveMode('pan')
        self.plot2d.addImage(data)
        self.plot2d.getXAxis().setRangeConstraints(minRange=10)
        self.plot2d.getYAxis().setRangeConstraints(minRange=10)

        self.plot2d2 = plot.Plot2D(parent=widget, backend=backend)
        self.plot2d2.setGraphTitle("The image can't be too small")
        self.plot2d2.setInteractiveMode('pan')
        self.plot2d2.addImage(data)
        self.plot2d2.getXAxis().setRangeConstraints(maxRange=200)
        self.plot2d2.getYAxis().setRangeConstraints(maxRange=200)

        self.plot1d = plot.Plot1D(parent=widget, backend=backend)
        self.plot1d.setGraphTitle("The curve is clamped into the view")
        self.plot1d.addCurve(x=numpy.arange(100), y=data1d, legend="mean")
        self.plot1d.getXAxis().setLimitsConstraints(minPos=0, maxPos=100)
        self.plot1d.getYAxis().setLimitsConstraints(minPos=data1d.min(), maxPos=data1d.max())

        self.plot1d2 = plot.Plot1D(parent=widget, backend=backend)
        self.plot1d2.setGraphTitle("Only clamp y-axis")
        self.plot1d2.setInteractiveMode('pan')
        self.plot1d2.addCurve(x=numpy.arange(100), y=data1d, legend="mean")
        self.plot1d2.getYAxis().setLimitsConstraints(minPos=data1d.min(), maxPos=data1d.max())

        layout.addWidget(self.plot2d, 0, 0)
        layout.addWidget(self.plot1d, 0, 1)
        layout.addWidget(self.plot2d2, 1, 0)
        layout.addWidget(self.plot1d2, 1, 1)


if __name__ == "__main__":
    app = qt.QApplication([])
    window = ConstrainedViewPlot()
    window.setVisible(True)
    app.exec()
