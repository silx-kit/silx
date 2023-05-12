#!/usr/bin/env python
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
from silx.gui.plot import Plot2D
import numpy
import silx.test.utils
from silx.gui.plot.utils.axis import SyncAxes
from silx.gui.colors import Colormap


class SyncPlot(qt.QMainWindow):

    def __init__(self):
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Plot with synchronized axes")
        widget = qt.QWidget(self)
        self.setCentralWidget(widget)

        layout = qt.QGridLayout()
        widget.setLayout(layout)

        backend = "gl"
        plots = []

        data = numpy.arange(100 * 100)
        data = (data % 100) / 5.0
        data = numpy.sin(data)
        data.shape = 100, 100

        colormaps = ["gray", "red", "green", "blue"]
        for i in range(2 * 2):
            plot = Plot2D(parent=widget, backend=backend)
            plot.setInteractiveMode('pan')
            plot.setDefaultColormap(Colormap(colormaps[i]))
            noisyData = silx.test.utils.add_gaussian_noise(data, mean=i / 10.0)
            plot.addImage(noisyData)
            plots.append(plot)

        xAxis = [p.getXAxis() for p in plots]
        yAxis = [p.getYAxis() for p in plots]

        self.constraint1 = SyncAxes(xAxis,
                                    syncLimits=False,
                                    syncScale=True,
                                    syncDirection=True,
                                    syncCenter=True,
                                    syncZoom=True)
        self.constraint2 = SyncAxes(yAxis,
                                    syncLimits=False,
                                    syncScale=True,
                                    syncDirection=True,
                                    syncCenter=True,
                                    syncZoom=True)

        for i, plot in enumerate(plots):
            if i % 2 == 0:
                plot.setFixedWidth(400)
            else:
                plot.setFixedWidth(500)
            if i // 2 == 0:
                plot.setFixedHeight(400)
            else:
                plot.setFixedHeight(500)
            layout.addWidget(plot, i // 2, i % 2)

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
    app.exec()
