#!/usr/bin/env python
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
"""This script illustrates the addition of a context menu to a
:class:`~silx.gui.plot.PlotWidget`.

This is done by adding a custom context menu to the plot area of PlotWidget:
- set the context menu policy of the plot area to Qt.CustomContextMenu.
- connect to the plot area customContextMenuRequested signal.

The same method works with :class:`~silx.gui.plot.PlotWindow.PlotWindow`,
:class:`~silx.gui.plot.PlotWindow.Plot1D` and
:class:`~silx.gui.plot.PlotWindow.Plot2D` widgets as they
inherit from :class:`~silx.gui.plot.PlotWidget`.

For more information on context menus, see Qt documentation.
"""

import numpy

from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.actions.control import ZoomBackAction, CrosshairAction
from silx.gui.plot.actions.io import SaveAction, PrintAction


class PlotWidgetWithContextMenu(PlotWidget):
    """This class adds a custom context menu to PlotWidget's plot area."""

    def __init__(self, *args, **kwargs):
        super(PlotWidgetWithContextMenu, self).__init__(*args, **kwargs)
        self.setWindowTitle('PlotWidget with a context menu')
        self.setGraphTitle('Right-click on the plot to access context menu')

        # Create QAction for the context menu once for all
        self._zoomBackAction = ZoomBackAction(plot=self, parent=self)
        self._crosshairAction = CrosshairAction(plot=self, parent=self)
        self._saveAction = SaveAction(plot=self, parent=self)
        self._printAction = PrintAction(plot=self, parent=self)

        # Retrieve PlotWidget's plot area widget
        plotArea = self.getWidgetHandle()

        # Set plot area custom context menu
        plotArea.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        plotArea.customContextMenuRequested.connect(self._contextMenu)

    def _contextMenu(self, pos):
        """Handle plot area customContextMenuRequested signal.

        :param QPoint pos: Mouse position relative to plot area
        """
        # Create the context menu
        menu = qt.QMenu(self)
        menu.addAction(self._zoomBackAction)
        menu.addSeparator()
        menu.addAction(self._crosshairAction)
        menu.addSeparator()
        menu.addAction(self._saveAction)
        menu.addAction(self._printAction)

        # Displaying the context menu at the mouse position requires
        # a global position.
        # The position received as argument is relative to PlotWidget's
        # plot area, and thus needs to be converted.
        plotArea = self.getWidgetHandle()
        globalPosition = plotArea.mapToGlobal(pos)
        menu.exec_(globalPosition)


# Start the QApplication
app = qt.QApplication([])  # Start QApplication
plot = PlotWidgetWithContextMenu()  # Create the widget

# Add content to the plot
x = numpy.linspace(0, 2 * numpy.pi, 1000)
plot.addCurve(x, numpy.sin(x), legend='sin')

# Show the widget and start the application
plot.show()
app.exec_()
