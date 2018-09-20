# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""Base class for Plot backends.

It documents the Plot backend API.

This API is a simplified version of PyMca PlotBackend API.
"""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"

import weakref
from ... import qt


# Names for setCursor
CURSOR_DEFAULT = 'default'
CURSOR_POINTING = 'pointing'
CURSOR_SIZE_HOR = 'size horizontal'
CURSOR_SIZE_VER = 'size vertical'
CURSOR_SIZE_ALL = 'size all'


class BackendBase(object):
    """Class defining the API a backend of the Plot should provide."""

    def __init__(self, plot, parent=None):
        """Init.

        :param Plot plot: The Plot this backend is attached to
        :param parent: The parent widget of the plot widget.
        """
        self.__xLimits = 1., 100.
        self.__yLimits = {'left': (1., 100.), 'right': (1., 100.)}
        self.__yAxisInverted = False
        self.__keepDataAspectRatio = False
        self._xAxisTimeZone = None
        self._axesDisplayed = True
        # Store a weakref to get access to the plot state.
        self._setPlot(plot)

        self.__zoomBackAction = None

    @property
    def _plot(self):
        """The plot this backend is attached to."""
        if self._plotRef is None:
            raise RuntimeError('This backend is not attached to a Plot')

        plot = self._plotRef()
        if plot is None:
            raise RuntimeError('This backend is no more attached to a Plot')
        return plot

    def _setPlot(self, plot):
        """Allow to set plot after init.

        Use with caution, basically **immediately** after init.
        """
        self._plotRef = weakref.ref(plot)

    # Default Qt context menu

    def contextMenuEvent(self, event):
        """Override QWidget.contextMenuEvent to implement the context menu"""
        if self.__zoomBackAction is None:
            from ..actions.control import ZoomBackAction  # Avoid cyclic import
            self.__zoomBackAction = ZoomBackAction(plot=self._plot,
                                                   parent=self._plot)
        menu = qt.QMenu(self)
        menu.addAction(self.__zoomBackAction)
        menu.exec_(event.globalPos())

    # Add methods

    def addCurve(self, x, y, legend,
                 color, symbol, linewidth, linestyle,
                 yaxis,
                 xerror, yerror, z, selectable,
                 fill, alpha, symbolsize):
        """Add a 1D curve given by x an y to the graph.

        :param numpy.ndarray x: The data corresponding to the x axis
        :param numpy.ndarray y: The data corresponding to the y axis
        :param str legend: The legend to be associated to the curve
        :param color: color(s) to be used
        :type color: string ("#RRGGBB") or (npoints, 4) unsigned byte array or
                     one of the predefined color names defined in colors.py
        :param str symbol: Symbol to be drawn at each (x, y) position::

            - ' ' or '' no symbol
            - 'o' circle
            - '.' point
            - ',' pixel
            - '+' cross
            - 'x' x-cross
            - 'd' diamond
            - 's' square

        :param float linewidth: The width of the curve in pixels
        :param str linestyle: Type of line::

            - ' ' or ''  no line
            - '-'  solid line
            - '--' dashed line
            - '-.' dash-dot line
            - ':'  dotted line

        :param str yaxis: The Y axis this curve belongs to in: 'left', 'right'
        :param xerror: Values with the uncertainties on the x values
        :type xerror: numpy.ndarray or None
        :param yerror: Values with the uncertainties on the y values
        :type yerror: numpy.ndarray or None
        :param int z: Layer on which to draw the cuve
        :param bool selectable: indicate if the curve can be selected
        :param bool fill: True to fill the curve, False otherwise
        :param float alpha: Curve opacity, as a float in [0., 1.]
        :param float symbolsize: Size of the symbol (if any) drawn
                                 at each (x, y) position.
        :returns: The handle used by the backend to univocally access the curve
        """
        return legend

    def addImage(self, data, legend,
                 origin, scale, z,
                 selectable, draggable,
                 colormap, alpha):
        """Add an image to the plot.

        :param numpy.ndarray data: (nrows, ncolumns) data or
                     (nrows, ncolumns, RGBA) ubyte array
        :param str legend: The legend to be associated to the image
        :param origin: (origin X, origin Y) of the data.
                       Default: (0., 0.)
        :type origin: 2-tuple of float
        :param scale: (scale X, scale Y) of the data.
                       Default: (1., 1.)
        :type scale: 2-tuple of float
        :param int z: Layer on which to draw the image
        :param bool selectable: indicate if the image can be selected
        :param bool draggable: indicate if the image can be moved
        :param ~silx.gui.colors.Colormap colormap: Colormap object to use.
            Ignored if data is RGB(A).
        :param float alpha: Opacity of the image, as a float in range [0, 1].
        :returns: The handle used by the backend to univocally access the image
        """
        return legend

    def addItem(self, x, y, legend, shape, color, fill, overlay, z):
        """Add an item (i.e. a shape) to the plot.

        :param numpy.ndarray x: The X coords of the points of the shape
        :param numpy.ndarray y: The Y coords of the points of the shape
        :param str legend: The legend to be associated to the item
        :param str shape: Type of item to be drawn in
                          hline, polygon, rectangle, vline, polylines
        :param str color: Color of the item
        :param bool fill: True to fill the shape
        :param bool overlay: True if item is an overlay, False otherwise
        :param int z: Layer on which to draw the item
        :returns: The handle used by the backend to univocally access the item
        """
        return legend

    def addMarker(self, x, y, legend, text, color,
                  selectable, draggable,
                  symbol, linestyle, linewidth, constraint):
        """Add a point, vertical line or horizontal line marker to the plot.

        :param float x: Horizontal position of the marker in graph coordinates.
                        If None, the marker is a horizontal line.
        :param float y: Vertical position of the marker in graph coordinates.
                        If None, the marker is a vertical line.
        :param str legend: Legend associated to the marker
        :param str text: Text associated to the marker (or None for no text)
        :param str color: Color to be used for instance 'blue', 'b', '#FF0000'
        :param bool selectable: indicate if the marker can be selected
        :param bool draggable: indicate if the marker can be moved
        :param str symbol: Symbol representing the marker.
            Only relevant for point markers where X and Y are not None.
            Value in:

            - 'o' circle
            - '.' point
            - ',' pixel
            - '+' cross
            - 'x' x-cross
            - 'd' diamond
            - 's' square
        :param str linestyle: Style of the line.
            Only relevant for line markers where X or Y is None.
            Value in:

            - ' '  no line
            - '-'  solid line
            - '--' dashed line
            - '-.' dash-dot line
            - ':'  dotted line
        :param float linewidth: Width of the line.
            Only relevant for line markers where X or Y is None.
        :param constraint: A function filtering marker displacement by
                           dragging operations or None for no filter.
                           This function is called each time a marker is
                           moved.
                           This parameter is only used if draggable is True.
        :type constraint: None or a callable that takes the coordinates of
                          the current cursor position in the plot as input
                          and that returns the filtered coordinates.
        :return: Handle used by the backend to univocally access the marker
        """
        return legend

    # Remove methods

    def remove(self, item):
        """Remove an existing item from the plot.

        :param item: A backend specific item handle returned by a add* method
        """
        pass

    # Interaction methods

    def setGraphCursorShape(self, cursor):
        """Set the cursor shape.

        To override in interactive backends.

        :param str cursor: Name of the cursor shape or None
        """
        pass

    def setGraphCursor(self, flag, color, linewidth, linestyle):
        """Toggle the display of a crosshair cursor and set its attributes.

        To override in interactive backends.

        :param bool flag: Toggle the display of a crosshair cursor.
        :param color: The color to use for the crosshair.
        :type color: A string (either a predefined color name in colors.py
                    or "#RRGGBB")) or a 4 columns unsigned byte array.
        :param int linewidth: The width of the lines of the crosshair.
        :param linestyle: Type of line::

                - ' ' no line
                - '-' solid line
                - '--' dashed line
                - '-.' dash-dot line
                - ':' dotted line

        :type linestyle: None or one of the predefined styles.
        """
        pass

    def pickItems(self, x, y, kinds):
        """Get a list of items at a pixel position.

        :param float x: The x pixel coord where to pick.
        :param float y: The y pixel coord where to pick.
        :param List[str] kind: List of item kinds to pick.
            Supported kinds: 'marker', 'curve', 'image'.
        :return: All picked items from back to front.
                 One dict per item,
                 with 'kind' key in 'curve', 'marker', 'image';
                 'legend' key, the item legend.
                 and for curves, 'xdata' and 'ydata' keys storing picked
                 position on the curve.
        :rtype: list of dict
        """
        return []

    # Update curve

    def setCurveColor(self, curve, color):
        """Set the color of a curve.

        :param curve: The curve handle
        :param str color: The color to use.
        """
        pass

    # Misc.

    def getWidgetHandle(self):
        """Return the widget this backend is drawing to."""
        return None

    def postRedisplay(self):
        """Trigger a :meth:`Plot.replot`.

        Default implementation triggers a synchronous replot if plot is dirty.
        This method should be overridden by the embedding widget in order to
        provide an asynchronous call to replot in order to optimize the number
        replot operations.
        """
        # This method can be deferred and it might happen that plot has been
        # destroyed in between, especially with unittests

        plot = self._plotRef()
        if plot is not None and plot._getDirtyPlot():
            plot.replot()

    def replot(self):
        """Redraw the plot."""
        pass

    def saveGraph(self, fileName, fileFormat, dpi):
        """Save the graph to a file (or a StringIO)

        At least "png", "svg" are supported.

        :param fileName: Destination
        :type fileName: String or StringIO or BytesIO
        :param str fileFormat: String specifying the format
        :param int dpi: The resolution to use or None.
        """
        pass

    # Graph labels

    def setGraphTitle(self, title):
        """Set the main title of the plot.

        :param str title: Title associated to the plot
        """
        pass

    def setGraphXLabel(self, label):
        """Set the X axis label.

        :param str label: label associated to the plot bottom X axis
        """
        pass

    def setGraphYLabel(self, label, axis):
        """Set the left Y axis label.

        :param str label: label associated to the plot left Y axis
        :param str axis: The axis for which to get the limits: left or right
        """
        pass

    # Graph limits

    def setLimits(self, xmin, xmax, ymin, ymax, y2min=None, y2max=None):
        """Set the limits of the X and Y axes at once.

        :param float xmin: minimum bottom axis value
        :param float xmax: maximum bottom axis value
        :param float ymin: minimum left axis value
        :param float ymax: maximum left axis value
        :param float y2min: minimum right axis value
        :param float y2max: maximum right axis value
        """
        self.__xLimits = xmin, xmax
        self.__yLimits['left'] = ymin, ymax
        if y2min is not None and y2max is not None:
            self.__yLimits['right'] = y2min, y2max

    def getGraphXLimits(self):
        """Get the graph X (bottom) limits.

        :return:  Minimum and maximum values of the X axis
        """
        return self.__xLimits

    def setGraphXLimits(self, xmin, xmax):
        """Set the limits of X axis.

        :param float xmin: minimum bottom axis value
        :param float xmax: maximum bottom axis value
        """
        self.__xLimits = xmin, xmax

    def getGraphYLimits(self, axis):
        """Get the graph Y (left) limits.

        :param str axis: The axis for which to get the limits: left or right
        :return: Minimum and maximum values of the Y axis
        """
        return self.__yLimits[axis]

    def setGraphYLimits(self, ymin, ymax, axis):
        """Set the limits of the Y axis.

        :param float ymin: minimum left axis value
        :param float ymax: maximum left axis value
        :param str axis: The axis for which to get the limits: left or right
        """
        self.__yLimits[axis] = ymin, ymax

    # Graph axes


    def getXAxisTimeZone(self):
        """Returns tzinfo that is used if the X-Axis plots date-times.

        None means the datetimes are interpreted as local time.

        :rtype: datetime.tzinfo of None.
        """
        return self._xAxisTimeZone

    def setXAxisTimeZone(self, tz):
        """Sets tzinfo that is used if the X-Axis plots date-times.

        Use None to let the datetimes be interpreted as local time.

        :rtype: datetime.tzinfo of None.
        """
        self._xAxisTimeZone = tz

    def isXAxisTimeSeries(self):
        """Return True if the X-axis scale shows datetime objects.

        :rtype: bool
        """
        raise NotImplementedError()

    def setXAxisTimeSeries(self, isTimeSeries):
        """Set whether the X-axis is a time series

        :param bool flag: True to switch to time series, False for regular axis.
        """
        raise NotImplementedError()

    def setXAxisLogarithmic(self, flag):
        """Set the X axis scale between linear and log.

        :param bool flag: If True, the bottom axis will use a log scale
        """
        pass

    def setYAxisLogarithmic(self, flag):
        """Set the Y axis scale between linear and log.

        :param bool flag: If True, the left axis will use a log scale
        """
        pass

    def setYAxisInverted(self, flag):
        """Invert the Y axis.

        :param bool flag: If True, put the vertical axis origin on the top
        """
        self.__yAxisInverted = bool(flag)

    def isYAxisInverted(self):
        """Return True if left Y axis is inverted, False otherwise."""
        return self.__yAxisInverted

    def isKeepDataAspectRatio(self):
        """Returns whether the plot is keeping data aspect ratio or not."""
        return self.__keepDataAspectRatio

    def setKeepDataAspectRatio(self, flag):
        """Set whether to keep data aspect ratio or not.

        :param flag:  True to respect data aspect ratio
        :type flag: Boolean, default True
        """
        self.__keepDataAspectRatio = bool(flag)

    def setGraphGrid(self, which):
        """Set grid.

        :param which: None to disable grid, 'major' for major grid,
                     'both' for major and minor grid
        """
        pass

    # Data <-> Pixel coordinates conversion

    def dataToPixel(self, x, y, axis):
        """Convert a position in data space to a position in pixels
        in the widget.

        :param float x: The X coordinate in data space.
        :param float y: The Y coordinate in data space.
        :param str axis: The Y axis to use for the conversion
                         ('left' or 'right').
        :returns: The corresponding position in pixels or
                  None if the data position is not in the displayed area.
        :rtype: A tuple of 2 floats: (xPixel, yPixel) or None.
        """
        raise NotImplementedError()

    def pixelToData(self, x, y, axis, check):
        """Convert a position in pixels in the widget to a position in
        the data space.

        :param float x: The X coordinate in pixels.
        :param float y: The Y coordinate in pixels.
        :param str axis: The Y axis to use for the conversion
                         ('left' or 'right').
        :param bool check: True to check if the coordinates are in the
                           plot area.
        :returns: The corresponding position in data space or
                  None if the pixel position is not in the plot area.
        :rtype: A tuple of 2 floats: (xData, yData) or None.
        """
        raise NotImplementedError()

    def getPlotBoundsInPixels(self):
        """Plot area bounds in widget coordinates in pixels.

        :return: bounds as a 4-tuple of int: (left, top, width, height)
        """
        raise NotImplementedError()

    def setAxesDisplayed(self, displayed):
        """Display or not the axes.

        :param bool displayed: If `True` axes are displayed. If `False` axes
            are not anymore visible and the margin used for them is removed.
        """
        self._axesDisplayed = displayed

    def isAxesDisplayed(self):
        """private because in some case it is possible that one of the two axes
        are displayed and not the other.
        This only check status set to axes from the public API
        """
        return self._axesDisplayed
