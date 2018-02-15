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
"""This script shows how to subclass :class:`~silx.gui.plot.PlotWidget` to tune its tools.

It subclasses a :class:`~silx.gui.plot.PlotWidget` and adds toolbars and
a colorbar by using pluggable widgets:

- QAction from :mod:`silx.gui.plot.actions`
- QToolButton from :mod:`silx.gui.plot.PlotToolButtons`
- QToolBar from :mod:`silx.gui.plot.PlotTools`
- :class:`silx.gui.plot.ColorBar.ColorBarWidget`
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/09/2017"

import numpy

from silx.gui import qt
import silx.gui.plot

from silx.gui.plot import actions  # QAction to use with PlotWidget
from silx.gui.plot import PlotToolButtons  # QToolButton to use with PlotWidget
from silx.gui.plot.PlotTools import LimitsToolBar
from silx.gui.plot.ColorBar import ColorBarWidget

class MyPlotWidget(silx.gui.plot.PlotWidget):
    """PlotWidget with an ad hoc toolbar and a colorbar"""

    def __init__(self, parent=None):
        super(MyPlotWidget, self).__init__(parent)

        # Add a tool bar to PlotWidget
        toolBar = qt.QToolBar("Plot Tools", self)
        self.addToolBar(toolBar)

        # Add actions from silx.gui.plot.action to the toolbar
        resetZoomAction = actions.control.ResetZoomAction(self, self)
        toolBar.addAction(resetZoomAction)

        # Add tool buttons from silx.gui.plot.PlotToolButtons
        aspectRatioButton = PlotToolButtons.AspectToolButton(
            parent=self, plot=self)
        toolBar.addWidget(aspectRatioButton)

        # Add limits tool bar from silx.gui.plot.PlotTools
        limitsToolBar = LimitsToolBar(parent=self, plot=self)
        self.addToolBar(qt.Qt.BottomToolBarArea, limitsToolBar)

        self._initColorBar()

    def _initColorBar(self):
        """Create the ColorBarWidget and add it to the PlotWidget"""

        # Add a colorbar on the right side
        colorBar = ColorBarWidget(parent=self, plot=self)

        # Make ColorBarWidget background white by changing its palette
        colorBar.setAutoFillBackground(True)
        palette = colorBar.palette()
        palette.setColor(qt.QPalette.Background, qt.Qt.white)
        palette.setColor(qt.QPalette.Window, qt.Qt.white)
        colorBar.setPalette(palette)

        # Add the ColorBarWidget by changing PlotWidget's central widget
        gridLayout = qt.QGridLayout()
        gridLayout.setSpacing(0)
        gridLayout.setContentsMargins(0, 0, 0, 0)
        plot = self.getWidgetHandle()  # Get the widget rendering the plot
        gridLayout.addWidget(plot, 0, 0)
        gridLayout.addWidget(colorBar, 0, 1)
        gridLayout.setRowStretch(0, 1)
        gridLayout.setColumnStretch(0, 1)
        centralWidget = qt.QWidget()
        centralWidget.setLayout(gridLayout)
        self.setCentralWidget(centralWidget)


def main():
    global app
    app = qt.QApplication([])

    # Create the ad hoc plot widget and change its default colormap
    plot = MyPlotWidget()
    plot.getDefaultColormap().setName('viridis')
    plot.show()

    # Add an image to the plot
    x = numpy.outer(
        numpy.linspace(-10, 10, 200), numpy.linspace(-10, 5, 150))
    image = numpy.sin(x) / x
    plot.addImage(image)

    app.exec_()


if __name__ == '__main__':
    main()
