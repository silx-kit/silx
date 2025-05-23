# /*##########################################################################
#
# Copyright (c) 2016-2023 European Synchrotron Radiation Facility
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
"""This module provides a widget displaying mouse coordinates in a PlotWidget.

It can be configured to provide more information.
"""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "16/10/2017"


import logging
import numbers
import traceback
import weakref

import numpy

from ... import qt
from .. import items
from ...widgets.ElidedLabel import ElidedLabel


_logger = logging.getLogger(__name__)


class _PositionInfoLabel(ElidedLabel):
    """QLabel with a default size larger than what is displayed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)

    def sizeHint(self):
        hint = super().sizeHint()
        width = self.fontMetrics().boundingRect("##############").width()
        return qt.QSize(max(hint.width(), width), hint.height())


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
    >>> from silx.gui.plot.tools import PositionInfo

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

    SNAP_THRESHOLD_DIST = 5

    def __init__(self, parent=None, plot=None, converters=None):
        assert plot is not None
        self._plotRef = weakref.ref(plot)
        self._snappingMode = self.SNAPPING_DISABLED

        super().__init__(parent)

        if converters is None:
            converters = (("X", lambda x, y: x), ("Y", lambda x, y: y))

        self._fields = []  # To store (QLineEdit, name, function (x, y)->v)

        # Create a new layout with new widgets
        layout = qt.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # layout.setSpacing(0)

        # Create all QLabel and store them with the corresponding converter
        for name, func in converters:
            layout.addWidget(qt.QLabel("<b>" + name + ":</b>"))

            contentWidget = _PositionInfoLabel(self)
            contentWidget.setText("------")
            layout.addWidget(contentWidget)
            self._fields.append((contentWidget, name, func))

        layout.addStretch(1)
        self.setLayout(layout)

        # Connect to Plot events
        plot.sigPlotSignal.connect(self._plotEvent)

    def getPlotWidget(self):
        """Returns the PlotWidget this widget is attached to or None.

        :rtype: Union[~silx.gui.plot.PlotWidget,None]
        """
        return self._plotRef()

    def getConverters(self):
        """Return the list of converters as 2-tuple (name, function)."""
        return [(name, func) for _label, name, func in self._fields]

    def _plotEvent(self, event):
        """Handle events from the Plot.

        :param dict event: Plot event
        """
        if event["event"] == "mouseMoved":
            x, y = event["x"], event["y"]
            xPixel, yPixel = event["xpixel"], event["ypixel"]
            self._updateStatusBar(x, y, xPixel, yPixel)

    def updateInfo(self):
        """Update displayed information"""
        plot = self.getPlotWidget()
        if plot is None:
            _logger.error(
                "Trying to update PositionInfo " "while PlotWidget no longer exists"
            )
            return

        widget = plot.getWidgetHandle()
        position = widget.mapFromGlobal(qt.QCursor.pos())
        xPixel, yPixel = position.x(), position.y()
        dataPos = plot.pixelToData(xPixel, yPixel, check=True)
        if dataPos is not None:  # Inside plot area
            x, y = dataPos
            self._updateStatusBar(x, y, xPixel, yPixel)

    def _updateStatusBar(self, x, y, xPixel, yPixel):
        """Update information from the status bar using the definitions.

        :param float x: Position-x in data
        :param float y: Position-y in data
        :param float xPixel: Position-x in pixels
        :param float yPixel: Position-y in pixels
        """
        plot = self.getPlotWidget()
        if plot is None:
            return

        styleSheet = ""  # Default style
        xData, yData = x, y

        snappingMode = self.getSnappingMode()

        # Snapping when crosshair either not requested or active
        if snappingMode & (self.SNAPPING_CURVE | self.SNAPPING_SCATTER) and (
            not (snappingMode & self.SNAPPING_CROSSHAIR) or plot.getGraphCursor()
        ):
            styleSheet = "color: rgb(255, 0, 0);"  # Style far from item

            if snappingMode & self.SNAPPING_ACTIVE_ONLY:
                selectedItems = []

                if snappingMode & self.SNAPPING_CURVE:
                    activeCurve = plot.getActiveCurve()
                    if activeCurve:
                        selectedItems.append(activeCurve)

                if snappingMode & self.SNAPPING_SCATTER:
                    activeScatter = plot.getActiveScatter()
                    if activeScatter:
                        selectedItems.append(activeScatter)

            else:
                kinds = []
                if snappingMode & self.SNAPPING_CURVE:
                    kinds.append(items.Curve)
                    kinds.append(items.Histogram)
                if snappingMode & self.SNAPPING_SCATTER:
                    kinds.append(items.Scatter)
                selectedItems = [
                    item
                    for item in plot.getItems()
                    if isinstance(item, tuple(kinds)) and item.isVisible()
                ]

            # Compute distance threshold
            window = plot.window()
            windowHandle = window.windowHandle()
            if windowHandle is not None:
                ratio = windowHandle.devicePixelRatio()
            else:
                ratio = qt.QGuiApplication.primaryScreen().devicePixelRatio()

            # Baseline squared distance threshold
            sqDistInPixels = (self.SNAP_THRESHOLD_DIST * ratio) ** 2

            for item in selectedItems:
                if snappingMode & self.SNAPPING_SYMBOLS_ONLY and (
                    not isinstance(item, items.SymbolMixIn) or not item.getSymbol()
                ):
                    # Only handled if item symbols are visible
                    continue

                if isinstance(item, items.Histogram):
                    result = item.pick(xPixel, yPixel)
                    if result is not None:  # Histogram picked
                        index = result.getIndices()[0]
                        edges = item.getBinEdgesData(copy=False)

                        # Snap to bin center and value
                        xData = 0.5 * (edges[index] + edges[index + 1])
                        yData = item.getValueData(copy=False)[index]

                        # Update label style sheet
                        styleSheet = ""
                        break

                else:  # Curve, Scatter
                    result = item.pick(xPixel, yPixel)
                    if result is None:
                        continue
                    indices = result.getIndices(copy=False)
                    if indices is None:
                        continue

                    if isinstance(item, items.YAxisMixIn):
                        axis = item.getYAxis()
                    else:
                        axis = "left"

                    xArray = item.getXData(copy=False)[indices]
                    yArray = item.getYData(copy=False)[indices]
                    pixelPositions = plot.dataToPixel(xArray, yArray, axis=axis)
                    if pixelPositions is None:
                        continue
                    sqDistances = (pixelPositions[0] - xPixel) ** 2 + (
                        pixelPositions[1] - yPixel
                    ) ** 2
                    if not numpy.any(numpy.isfinite(sqDistances)):
                        continue
                    closestIndex = numpy.nanargmin(sqDistances)
                    closestSqDistInPixels = sqDistances[closestIndex]

                    if closestSqDistInPixels <= sqDistInPixels:
                        # Update label style sheet
                        styleSheet = ""

                        # if close enough, snap to data point coord
                        xData, yData = xArray[closestIndex], yArray[closestIndex]
                        sqDistInPixels = closestSqDistInPixels

        for label, name, func in self._fields:
            label.setStyleSheet(styleSheet)

            try:
                value = func(xData, yData)
                text = self.valueToString(value)
                label.setText(text)
            except:
                label.setText("Error")
                _logger.error(
                    "Error while converting coordinates (%f, %f)"
                    "with converter '%s'" % (xPixel, yPixel, name)
                )
                _logger.error(traceback.format_exc())

    def valueToString(self, value):
        if isinstance(value, (tuple, list)):
            value = [self.valueToString(v) for v in value]
            return ", ".join(value)
        elif isinstance(value, numbers.Real):
            # Use this for floats and int
            return "%.7g" % value
        else:
            # Fallback for other types
            return str(value)

    # Snapping mode

    SNAPPING_DISABLED = 0
    """No snapping occurs"""

    SNAPPING_CROSSHAIR = 1 << 0
    """Snapping only enabled when crosshair cursor is enabled"""

    SNAPPING_ACTIVE_ONLY = 1 << 1
    """Snapping only enabled for active item"""

    SNAPPING_SYMBOLS_ONLY = 1 << 2
    """Snapping only when symbols are visible"""

    SNAPPING_CURVE = 1 << 3
    """Snapping on curves"""

    SNAPPING_SCATTER = 1 << 4
    """Snapping on scatter"""

    def setSnappingMode(self, mode):
        """Set the snapping mode.

        The mode is a mask.

        :param int mode: The mode to use
        """
        if mode != self._snappingMode:
            self._snappingMode = mode
            self.updateInfo()

    def getSnappingMode(self):
        """Returns the snapping mode as a mask

        :rtype: int
        """
        return self._snappingMode
