#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""This script is an example to illustrate how to use axis synchronization
tool.
"""

from silx.gui import qt
from silx.gui import plot
import numpy
import silx.test.utils
from silx.gui.plot.utils.axis import SyncAxes


class SyncPlot(qt.QMainWindow):

    def __init__(self):
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Plot with synchronized axes")
        widget = qt.QWidget(self)
        self.setCentralWidget(widget)

        layout = qt.QGridLayout()
        widget.setLayout(layout)

        backend = "mpl"
        self.plot2d = plot.Plot2D(parent=widget, backend=backend)
        self.plot2d.setInteractiveMode('pan')
        self.plot1d_x1 = plot.Plot1D(parent=widget, backend=backend)
        self.plot1d_x2 = plot.PlotWidget(parent=widget, backend=backend)
        self.plot1d_y1 = plot.Plot1D(parent=widget, backend=backend)
        self.plot1d_y2 = plot.PlotWidget(parent=widget, backend=backend)

        data = numpy.arange(100 * 100)
        data = (data % 100) / 5.0
        data = numpy.sin(data)
        data = silx.test.utils.add_gaussian_noise(data, mean=0.01)
        data.shape = 100, 100

        self.plot2d.addImage(data)
        self.plot1d_x1.addCurve(x=numpy.arange(100), y=numpy.mean(data, axis=0), legend="mean")
        self.plot1d_x1.addCurve(x=numpy.arange(100), y=numpy.max(data, axis=0), legend="max")
        self.plot1d_x1.addCurve(x=numpy.arange(100), y=numpy.min(data, axis=0), legend="min")
        self.plot1d_x2.addCurve(x=numpy.arange(100), y=numpy.std(data, axis=0))

        self.plot1d_y1.addCurve(y=numpy.arange(100), x=numpy.mean(data, axis=1), legend="mean")
        self.plot1d_y1.addCurve(y=numpy.arange(100), x=numpy.max(data, axis=1), legend="max")
        self.plot1d_y1.addCurve(y=numpy.arange(100), x=numpy.min(data, axis=1), legend="min")
        self.plot1d_y2.addCurve(y=numpy.arange(100), x=numpy.std(data, axis=1))

        self.constraint1 = SyncAxes([self.plot2d.getXAxis(), self.plot1d_x1.getXAxis(), self.plot1d_x2.getXAxis()])
        self.constraint2 = SyncAxes([self.plot2d.getYAxis(), self.plot1d_y1.getYAxis(), self.plot1d_y2.getYAxis()])
        self.constraint3 = SyncAxes([self.plot1d_x1.getYAxis(), self.plot1d_y1.getXAxis()])
        self.constraint4 = SyncAxes([self.plot1d_x2.getYAxis(), self.plot1d_y2.getXAxis()])

        layout.addWidget(self.plot2d, 0, 0)
        layout.addWidget(self.createCenteredLabel(u"↓↑"), 1, 0)
        layout.addWidget(self.plot1d_x1, 2, 0)
        layout.addWidget(self.createCenteredLabel(u"↓↑"), 3, 0)
        layout.addWidget(self.plot1d_x2, 4, 0)
        layout.addWidget(self.createCenteredLabel(u"→\n←"), 0, 1)
        layout.addWidget(self.plot1d_y1, 0, 2)
        layout.addWidget(self.createCenteredLabel(u"→\n←"), 0, 3)
        layout.addWidget(self.plot1d_y2, 0, 4)
        layout.addWidget(self.createCenteredLabel(u"↗↙"), 2, 2)
        layout.addWidget(self.createCenteredLabel(u"↗↙"), 4, 4)

    def createCenteredLabel(self, text):
        label = qt.QLabel(self)
        label.setAlignment(qt.Qt.AlignCenter)
        label.setText(text)
        return label


if __name__ == "__main__":
    app = qt.QApplication([])
    window = SyncPlot()
    window.setAttribute(qt.Qt.WA_DeleteOnClose, True)
    window.setVisible(True)
    app.exec_()
