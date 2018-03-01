# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
"""A toolbar to display and edit limits of a PlotWidget
"""


from __future__ import division

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "16/10/2017"


from ... import qt
from ...widgets.FloatEdit import FloatEdit


class LimitsToolBar(qt.QToolBar):
    """QToolBar displaying and controlling the limits of a :class:`PlotWidget`.

    To run the following sample code, a QApplication must be initialized.
    First, create a PlotWindow:

    >>> from silx.gui.plot import PlotWindow
    >>> plot = PlotWindow()  # Create a PlotWindow to add the toolbar to

    Then, create the LimitsToolBar and add it to the PlotWindow.

    >>> from silx.gui import qt
    >>> from silx.gui.plot.tools import LimitsToolBar

    >>> toolbar = LimitsToolBar(plot=plot)  # Create the toolbar
    >>> plot.addToolBar(qt.Qt.BottomToolBarArea, toolbar)  # Add it to the plot
    >>> plot.show()  # To display the PlotWindow with the limits toolbar

    :param parent: See :class:`QToolBar`.
    :param plot: :class:`PlotWidget` instance on which to operate.
    :param str title: See :class:`QToolBar`.
    """

    def __init__(self, parent=None, plot=None, title='Limits'):
        super(LimitsToolBar, self).__init__(title, parent)
        assert plot is not None
        self._plot = plot
        self._plot.sigPlotSignal.connect(self._plotWidgetSlot)

        self._initWidgets()

    @property
    def plot(self):
        """The :class:`PlotWidget` the toolbar is attached to."""
        return self._plot

    def _initWidgets(self):
        """Create and init Toolbar widgets."""
        xMin, xMax = self.plot.getXAxis().getLimits()
        yMin, yMax = self.plot.getYAxis().getLimits()

        self.addWidget(qt.QLabel('Limits: '))
        self.addWidget(qt.QLabel(' X: '))
        self._xMinFloatEdit = FloatEdit(self, xMin)
        self._xMinFloatEdit.editingFinished[()].connect(
            self._xFloatEditChanged)
        self.addWidget(self._xMinFloatEdit)

        self._xMaxFloatEdit = FloatEdit(self, xMax)
        self._xMaxFloatEdit.editingFinished[()].connect(
            self._xFloatEditChanged)
        self.addWidget(self._xMaxFloatEdit)

        self.addWidget(qt.QLabel(' Y: '))
        self._yMinFloatEdit = FloatEdit(self, yMin)
        self._yMinFloatEdit.editingFinished[()].connect(
            self._yFloatEditChanged)
        self.addWidget(self._yMinFloatEdit)

        self._yMaxFloatEdit = FloatEdit(self, yMax)
        self._yMaxFloatEdit.editingFinished[()].connect(
            self._yFloatEditChanged)
        self.addWidget(self._yMaxFloatEdit)

    def _plotWidgetSlot(self, event):
        """Listen to :class:`PlotWidget` events."""
        if event['event'] not in ('limitsChanged',):
            return

        xMin, xMax = self.plot.getXAxis().getLimits()
        yMin, yMax = self.plot.getYAxis().getLimits()

        self._xMinFloatEdit.setValue(xMin)
        self._xMaxFloatEdit.setValue(xMax)
        self._yMinFloatEdit.setValue(yMin)
        self._yMaxFloatEdit.setValue(yMax)

    def _xFloatEditChanged(self):
        """Handle X limits changed from the GUI."""
        xMin, xMax = self._xMinFloatEdit.value(), self._xMaxFloatEdit.value()
        if xMax < xMin:
            xMin, xMax = xMax, xMin

        self.plot.getXAxis().setLimits(xMin, xMax)

    def _yFloatEditChanged(self):
        """Handle Y limits changed from the GUI."""
        yMin, yMax = self._yMinFloatEdit.value(), self._yMaxFloatEdit.value()
        if yMax < yMin:
            yMin, yMax = yMax, yMin

        self.plot.getYAxis().setLimits(yMin, yMax)
