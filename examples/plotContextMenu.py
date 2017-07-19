#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""This script illustrates the addition of a context menu to a PlotWidget.

The context menu is added to the PlotWidget by inheriting from PlotWidget
and implementing QWidget.contextMenuEvent method.
For alternative ways of managing context menus, see Qt documentation.
"""
__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "19/07/2017"


import numpy

from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.actions import control, io


class PlotWidgetWithContextMenu(PlotWidget):
    """This class inherit from plot to add specific context menu."""

    def __init__(self, *args, **kwargs):
        super(PlotWidgetWithContextMenu, self).__init__(*args, **kwargs)
        self.setWindowTitle('PlotWidget with a context menu')
        self.setGraphTitle('Right-click on the plot to access context menu')

        # Create QAction for the context menu
        self._zoomBackAction = control.ZoomBackAction(plot=self, parent=self)
        self._zoomInAction = control.ZoomInAction(plot=self, parent=self)
        self._zoomOutAction = control.ZoomOutAction(plot=self, parent=self)
        self._saveAction = io.SaveAction(plot=self, parent=self)
        self._printAction = io.PrintAction(plot=self, parent=self)

    def contextMenuEvent(self, event):
        """Override QWidget.contextMenuEvent to implement the context menu"""
        menu = qt.QMenu(self)
        menu.addAction(self._zoomBackAction)
        menu.addAction(self._zoomInAction)
        menu.addAction(self._zoomOutAction)
        menu.addSeparator()
        menu.addAction(self._saveAction)
        menu.addAction(self._printAction)
        menu.exec_(event.globalPos())


# Start the QApplication
app = qt.QApplication([])  # Start QApplication
plot = PlotWidgetWithContextMenu(backend='gl')  # Create the widget

# Add content to the plot
x = numpy.linspace(0, 2 * numpy.pi, 1000)
plot.addCurve(x, numpy.sin(x), legend='sin')

# Show the widget and start the application
plot.show()
app.exec_()
