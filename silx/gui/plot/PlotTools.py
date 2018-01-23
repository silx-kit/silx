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
"""Set of widgets to associate with a :class:'PlotWidget'.
"""

from __future__ import division

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "16/10/2017"


import logging
import numbers
import traceback
import weakref

import numpy

from .. import qt
from silx.gui.widgets.FloatEdit import FloatEdit

_logger = logging.getLogger(__name__)


# PositionInfo ################################################################

class PositionInfo(qt.QWidget):
    """QWidget displaying coords converted from data coords of the mouse.

    Provide this widget with a list of couple:

    - A name to display before the data
    - A function that takes (x, y) as arguments and returns something that
      gets converted to a string.
      If the result is a float it is converted with '%.7g' format.

    To run the following sample code, a QApplication must be initialized.
    First, create a PlotWindow and add a QToolBar where to place the
    PositionInfo widget.

    >>> from silx.gui.plot import PlotWindow
    >>> from silx.gui import qt

    >>> plot = PlotWindow()  # Create a PlotWindow to add the widget to
    >>> toolBar = qt.QToolBar()  # Create a toolbar to place the widget in
    >>> plot.addToolBar(qt.Qt.BottomToolBarArea, toolBar)  # Add it to plot

    Then, create the PositionInfo widget and add it to the toolbar.
    The PositionInfo widget is created with a list of converters, here
    to display polar coordinates of the mouse position.

    >>> import numpy
    >>> from silx.gui.plot.PlotTools import PositionInfo

    >>> position = PositionInfo(plot=plot, converters=[
    ...     ('Radius', lambda x, y: numpy.sqrt(x*x + y*y)),
    ...     ('Angle', lambda x, y: numpy.degrees(numpy.arctan2(y, x)))])
    >>> toolBar.addWidget(position)  # Add the widget to the toolbar
    <...>
    >>> plot.show()  # To display the PlotWindow with the position widget

    :param plot: The PlotWidget this widget is displaying data coords from.
    :param converters:
        List of 2-tuple: name to display and conversion function from (x, y)
        in data coords to displayed value.
        If None, the default, it displays X and Y.
    :param parent: Parent widget
    """

    def __init__(self, parent=None, plot=None, converters=None):
        assert plot is not None
        self._plotRef = weakref.ref(plot)

        super(PositionInfo, self).__init__(parent)

        if converters is None:
            converters = (('X', lambda x, y: x), ('Y', lambda x, y: y))

        self.autoSnapToActiveCurve = False
        """Toggle snapping use position to active curve.

        - True to snap used coordinates to the active curve if the active curve
          is displayed with symbols and mouse is close enough.
          If the mouse is not close to a point of the curve, values are
          displayed in red.
        - False (the default) to always use mouse coordinates.

        """

        self._fields = []  # To store (QLineEdit, name, function (x, y)->v)

        # Create a new layout with new widgets
        layout = qt.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # layout.setSpacing(0)

        # Create all QLabel and store them with the corresponding converter
        for name, func in converters:
            layout.addWidget(qt.QLabel('<b>' + name + ':</b>'))

            contentWidget = qt.QLabel()
            contentWidget.setText('------')
            contentWidget.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
            contentWidget.setFixedWidth(
                contentWidget.fontMetrics().width('##############'))
            layout.addWidget(contentWidget)
            self._fields.append((contentWidget, name, func))

        layout.addStretch(1)
        self.setLayout(layout)

        # Connect to Plot events
        plot.sigPlotSignal.connect(self._plotEvent)

    @property
    def plot(self):
        """The :class:`.PlotWindow` this widget is attached to."""
        return self._plotRef()

    def getConverters(self):
        """Return the list of converters as 2-tuple (name, function)."""
        return [(name, func) for _label, name, func in self._fields]

    def _plotEvent(self, event):
        """Handle events from the Plot.

        :param dict event: Plot event
        """
        if event['event'] == 'mouseMoved':
            x, y = event['x'], event['y']
            xPixel, yPixel = event['xpixel'], event['ypixel']
            self._updateStatusBar(x, y, xPixel, yPixel)

    def _updateStatusBar(self, x, y, xPixel, yPixel):
        """Update information from the status bar using the definitions.

        :param float x: Position-x in data
        :param float y: Position-y in data
        :param float xPixel: Position-x in pixels
        :param float yPixel: Position-y in pixels
        """
        styleSheet = "color: rgb(0, 0, 0);"  # Default style

        if self.autoSnapToActiveCurve and self.plot.getGraphCursor():
            # Check if near active curve with symbols.

            styleSheet = "color: rgb(255, 0, 0);"  # Style far from curve

            activeCurve = self.plot.getActiveCurve()
            if activeCurve:
                xData = activeCurve.getXData(copy=False)
                yData = activeCurve.getYData(copy=False)
                if activeCurve.getSymbol():  # Only handled if symbols on curve
                    closestIndex = numpy.argmin(
                        pow(xData - x, 2) + pow(yData - y, 2))

                    xClosest = xData[closestIndex]
                    yClosest = yData[closestIndex]

                    closestInPixels = self.plot.dataToPixel(
                        xClosest, yClosest, axis=activeCurve.getYAxis())
                    if closestInPixels is not None:
                        if (abs(closestInPixels[0] - xPixel) < 5 and
                                abs(closestInPixels[1] - yPixel) < 5):
                            # Update label style sheet
                            styleSheet = "color: rgb(0, 0, 0);"

                            # if close enough, wrap to data point coords
                            x, y = xClosest, yClosest

        for label, name, func in self._fields:
            label.setStyleSheet(styleSheet)

            try:
                value = func(x, y)
                text = self.valueToString(value)
                label.setText(text)
            except:
                label.setText('Error')
                _logger.error(
                    "Error while converting coordinates (%f, %f)"
                    "with converter '%s'" % (x, y, name))
                _logger.error(traceback.format_exc())

    def valueToString(self, value):
        if isinstance(value, (tuple, list)):
            value = [self.valueToString(v) for v in value]
            return ", ".join(value)
        elif isinstance(value, numbers.Real):
            # Use this for floats and int
            return '%.7g' % value
        else:
            # Fallback for other types
            return str(value)

# LimitsToolBar ##############################################################

class LimitsToolBar(qt.QToolBar):
    """QToolBar displaying and controlling the limits of a :class:`PlotWidget`.

    To run the following sample code, a QApplication must be initialized.
    First, create a PlotWindow:

    >>> from silx.gui.plot import PlotWindow
    >>> plot = PlotWindow()  # Create a PlotWindow to add the toolbar to

    Then, create the LimitsToolBar and add it to the PlotWindow.

    >>> from silx.gui import qt
    >>> from silx.gui.plot.PlotTools import LimitsToolBar

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
