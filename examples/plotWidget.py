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
"""This script shows how to create a custom window around a PlotWidget.

It subclasses :class:`QMainWindow`, uses a :class:`~silx.gui.plot.PlotWidget`
as its central widget and adds toolbars and a colorbar by using pluggable widgets:

- :class:`~silx.gui.plot.PlotWidget` from :mod:`silx.gui.plot`
- QToolBar from :mod:`silx.gui.plot.tools`
- QAction from :mod:`silx.gui.plot.actions`
- QToolButton from :mod:`silx.gui.plot.PlotToolButtons`
- :class:`silx.gui.plot.ColorBar.ColorBarWidget`
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "05/09/2017"

import numpy

from silx.gui import qt

from silx.gui.plot import PlotWidget
from silx.gui.plot import tools   # QToolbars to use with PlotWidget
from silx.gui.plot import actions  # QAction to use with PlotWidget
from silx.gui.plot import PlotToolButtons  # QToolButton to use with PlotWidget
from silx.gui.plot.ColorBar import ColorBarWidget


class MyPlotWindow(qt.QMainWindow):
    """QMainWindow with selected tools"""

    def __init__(self, parent=None):
        super(MyPlotWindow, self).__init__(parent)

        # Create a PlotWidget
        self._plot = PlotWidget(parent=self)

        # Create a colorbar linked with the PlotWidget
        colorBar = ColorBarWidget(parent=self, plot=self._plot)

        # Make ColorBarWidget background white by changing its palette
        colorBar.setAutoFillBackground(True)
        palette = colorBar.palette()
        palette.setColor(qt.QPalette.Background, qt.Qt.white)
        palette.setColor(qt.QPalette.Window, qt.Qt.white)
        colorBar.setPalette(palette)

        # Combine the ColorBarWidget and the PlotWidget as
        # this QMainWindow's central widget
        gridLayout = qt.QGridLayout()
        gridLayout.setSpacing(0)
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.addWidget(self._plot, 0, 0)
        gridLayout.addWidget(colorBar, 0, 1)
        gridLayout.setRowStretch(0, 1)
        gridLayout.setColumnStretch(0, 1)
        centralWidget = qt.QWidget(self)
        centralWidget.setLayout(gridLayout)
        self.setCentralWidget(centralWidget)

        # Add ready to use toolbar with zoom and pan interaction mode buttons
        interactionToolBar = tools.InteractiveModeToolBar(
            parent=self, plot=self._plot)
        self.addToolBar(interactionToolBar)
        # Add toolbar actions to activate keyboard shortcuts
        self.addActions(interactionToolBar.actions())

        # Add a new toolbar
        toolBar = qt.QToolBar("Plot Tools", self)
        self.addToolBar(toolBar)

        # Add actions from silx.gui.plot.action to the toolbar
        resetZoomAction = actions.control.ResetZoomAction(
            parent=self, plot=self._plot)
        toolBar.addAction(resetZoomAction)

        # Add tool buttons from silx.gui.plot.PlotToolButtons
        aspectRatioButton = PlotToolButtons.AspectToolButton(
            parent=self, plot=self._plot)
        toolBar.addWidget(aspectRatioButton)

        # Add ready to use toolbar with copy, save and print buttons
        outputToolBar = tools.OutputToolBar(parent=self, plot=self._plot)
        self.addToolBar(outputToolBar)
        # Add toolbar actions to activate keyboard shortcuts
        self.addActions(outputToolBar.actions())

        # Add limits tool bar from silx.gui.plot.PlotTools
        limitsToolBar = tools.LimitsToolBar(parent=self, plot=self._plot)
        self.addToolBar(qt.Qt.BottomToolBarArea, limitsToolBar)

    def getPlotWidget(self):
        """Returns the PlotWidget contains in this window"""
        return self._plot


def main():
    global app
    app = qt.QApplication([])

    # Create the ad hoc window containing a PlotWidget and associated tools
    window = MyPlotWindow()
    window.setAttribute(qt.Qt.WA_DeleteOnClose)
    window.show()

    # Change the default colormap
    plot = window.getPlotWidget()
    plot.getDefaultColormap().setName('viridis')

    # Add an image to the plot
    x = numpy.outer(
        numpy.linspace(-10, 10, 200), numpy.linspace(-10, 5, 150))
    image = numpy.sin(x) / x
    plot.addImage(image)

    app.exec_()


if __name__ == '__main__':
    main()
