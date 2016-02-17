#/*##########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
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
#############################################################################*/
__author__ = "V.A. Sole - ESRF Data Analysis"
__contact__ = "sole@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
This module can be used for testing purposes as well as an abstract class for
implementing Plot backends.

TODO: Still to be worked out: handling of the right vertical axis.

PlotBackend Functions (Functions marked by (*) only needed for handling images)

- addCurve
- addImage (*)
- addItem (*)
- clear
- clearCurves
- clearImages (*)
- clearMarkers
- enableActiveCurveHandling
- getDefaultColormap (*)
- getDrawMode
- getGraphXLabel
- getGraphXLimits
- getGraphYLabel
- getGraphYLimits
- getGraphTitle
- getSupportedColormaps (*)
- getWidgetHandle
- insertMarker
- insertXMarker
- insertYMarker
- invertYAxis
- isDrawModeEnabled
- isKeepDataAspectRatio
- isXAxisAutoScale
- isYAxisAutoScale
- keepDataAspectRatio(*)
- removeCurve
- removeImage (*)
- removeMarker
- resetZoom
- replot
- replot_
- saveGraph
- setActiveCurve
- setActiveImage (*)
- setCallback
- setDefaultColormap (*)
- setDrawModeEnabled
- setGraphTitle
- setGraphXLabel
- setGraphXLimits
- setGraphYLabel
- setGraphYLimits
- setLimits
- setXAxisAutoScale
- setXAxisLogarithmic
- setYAxisAutoScale
- setYAxisLogarithmic
- setZoomModeEnabled
- showGrid

PlotBackend "signals/events"

All the events pass via the callback_function supplied.

They consist on a dictionnary in which the 'event' key is mandatory.

The following keys will be present or not depending on the type of event, but
if present, their meaning should be:

KEY - Meaning

- button - "left", "right", "middle"
- label - The label or legend associated to the item associated to the event
- type - The type of item associated to event ('curve', 'marker', ...)
- x - Bottom axis value in graph coordenates
- y - Vertical axis value in graph coordenates
- xpixel - x position in pixel coordenates
- ypixel - y position in pixel coordenates
- xdata - Horizontal graph coordinate associated to the item
- ydata - Vertical graph coordinate associated to the item


drawingFinished

    It looks as it should export xdata, ydata and type.

    The information will depend on the type of item being drawn:

    - line - two points in graph and pixel coordinates
    - hline - one point in graph and pixel coordinates
    - vline - one point in graph and pixel coordinates
    - rectangle - four points in graph and pixel coordinates, x, y, width, height
    - polygone - n points in graph and pixel coordinates
    - ellipse - four points in graph and pixel coordinates?
    - circle - four points in graph and pixel coordinates, center and radius
    - parameters - Parameters passed to setDrawMode when enabling it

hover
    Emitted the mouse pass over an item with hover notification (markers)

imageClicked, curveClicked
    usefull for pop-up menus associated to the click using the xpixel, ypixel
    or to set a curve active using the label and type keys

markerMoving
    Additional keys:

    - draggable - True if it is a movable marker (it should be True)
    - selectable - True if the marker can be selected

markerMoved
    Additional keys:

    - draggable - True if it is a movable marker (it should be True)
    - selectable - True if the marker can be selected
    - xdata, ydata - Final position of the marker

markerClicked
    Additional keys:

    - draggable - True if it is a movable marker
    - selectable - True if the marker can be selected (it should be True)

mouseMoved
    Export the mouse position in pixel and graph coordenates

mouseClicked
    Emitted on mouse release when not zooming, nor drawing, nor picking

mouseDoubleClicked
    Emitted on mouse release when not zooming, nor drawing, nor picking

limitsChanged
    Emitted when limits of visible plot area are changed.
    This can results from user interaction or API calls.

    Keys:

    - source: id of the widget that emitted this event.
    - xdata: Range of X in graph coordinates: (xMin, xMax).
    - ydata: Range of Y in graph coordinates: (yMin, yMax).
    - y2data: Range of right axis in graph coordinates (y2Min, y2Max) or None.
"""

DEBUG = 0

from . import Colors

class PlotBackend(object):

    COLORDICT = Colors.COLORDICT
    """
    Dictionnary of predefined colors
    """

    def __init__(self, parent=None):
        self._callback = self._dummyCallback
        self._parent = parent
        self._zoomEnabled = True
        self._drawModeEnabled = False
        self._xAutoScale = True
        self._yAutoScale = True
        self.setGraphXLimits(0., 100.)
        self.setGraphYLimits(0., 100.)
        self._activeCurveHandling = False
        self.setActiveCurveColor("#000000")

    def addCurve(self, x, y, legend=None, info=None, replace=False, replot=True,
                 color=None, symbol=None, linewidth=None, linestyle=None,
                 xlabel=None, ylabel=None, yaxis=None,
                 xerror=None, yerror=None, z=1, selectable=True, **kw):
        """
        Add the 1D curve given by x an y to the graph.

        :param x: The data corresponding to the x axis
        :type x: list or numpy.ndarray
        :param y: The data corresponding to the y axis
        :type y: list or numpy.ndarray
        :param legend: The legend to be associated to the curve
        :type legend: string or None
        :param info: Dictionary of information associated to the curve
        :type info: dict or None
        :param replace: Flag to indicate if already existing curves are to be deleted
        :type replace: boolean default False
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        :param color: color(s) to be used
        :type color: string ("#RRGGBB") or (npoints, 4) unsigned byte array or
                     one of the predefined color names defined in Colors.py
        :param symbol: Symbol to be drawn at each (x, y) position::

            - 'o' circle
            - '.' point
            - ',' pixel
            - '+' cross
            - 'x' x-cross
            - 'd' diamond
            - 's' square

        :type symbol: None or one of the predefined symbols
        :param linewidth: The width of the curve in pixels (Default: 1).
        :type linewidth: None or float.
        :param linestyle: Type of line::

            - ' '  no line
            - '-'  solid line
            - '--' dashed line
            - '-.' dash-dot line
            - ':'  dotted line

        :type linestyle: None or one of the predefined styles.

        :param xlabel: Label associated to the X axis when the curve is active
        :type xlabel: string
        :param ylabel: Label associated to the Y axis when the curve is active
        :type ylabel: string
        :param yaxis: Anything different from "right" is equivalent to "left"
        :type yaxis: string or None
        :param xerror: Values with the uncertainties on the x values
        :type xlabel: array
        :param yerror: Values with the uncertainties on the y values
        :type ylabel: array
        :param z: level at which the curve is to be located (to allow overlays).
        :type z: A number bigger than or equal to zero (default: one)
        :param selectable: indicate if the curve can be picked.
        :type selectable: boolean default: True
        :returns: The legend/handle used by the backend to univocally access it.
        """
        print("PlotBackend addCurve not implemented")
        return legend

    def addImage(self, data, legend=None, info=None,
                    replace=True, replot=True,
                    xScale=None, yScale=None, z=0,
                    selectable=False, draggable=False,
                    colormap=None, **kw):
        """
        :param data: (nrows, ncolumns) data or (nrows, ncolumns, RGBA) ubyte array
        :type data: numpy.ndarray
        :param legend: The legend to be associated to the curve
        :type legend: string or None
        :param info: Dictionary of information associated to the image
        :type info: dict or None
        :param replace: Flag to indicate if already existing images are to be deleted
        :type replace: boolean default True
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        :param xScale: Two floats defining the x scale
        :type xScale: list or numpy.ndarray
        :param yScale: Two floats defining the y scale
        :type yScale: list or numpy.ndarray
        :param z: level at which the image is to be located (to allow overlays).
        :type z: A number bigger than or equal to zero (default)
        :param selectable: Flag to indicate if the image can be selected
        :type selectable: boolean, default False
        :param draggable: Flag to indicate if the image can be moved
        :type draggable: boolean, default False
        :param colormap: Dictionary describing the colormap to use (or None)
        :type colormap: Dictionnary or None (default). Ignored if data is RGB(A)
        :returns: The legend/handle used by the backend to univocally access it.
        """
        print("PlotBackend addImage not implemented")
        return legend

    def addItem(self, xList, yList, legend=None, info=None,
                                 replace=False, replot=True,
                                shape="polygon", fill=True, **kw):
        """
        :param shape: Type of item to be drawn
        :type shape: string, default polygon
        """
        print("PlotBackend addItem not implemented")
        return legend

    def clear(self):
        """
        Clear all curvers and other items from the plot
        """
        print("PlotBackend clear not implemented")
        return

    def clearCurves(self):
        """
        Clear all curves from the plot. Not the markers!!
        """
        print("PlotBackend clearCurves not implemented")
        return

    def clearImages(self):
        """
        Clear all images from the plot. Not the curves or markers.
        """
        print("PlotBackend clearImages not implemented")
        return

    def clearMarkers(self):
        """
        Clear all markers from the plot. Not the curves!!
        """
        print("PlotBackend clearMarkers not implemented")
        return

    def _dummyCallback(self, ddict):
        """
        Default callback
        """
        print("PlotBackend default callback called")
        print(ddict)

    def dataToPixel(self, x=None, y=None, axis="left"):
        """
        Convert a position in data space to a position in pixels in the widget.

        :param float x: The X coordinate in data space. If None (default)
                            the middle position of the displayed data is used.
        :param float y: The Y coordinate in data space. If None (default)
                            the middle position of the displayed data is used.
        :param str axis: The Y axis to use for the conversion
                         ('left' or 'right').
        :returns: The corresponding position in pixels or
                  None if the data position is not in the displayed area.
        :rtype: A tuple of 2 floats: (xPixel, yPixel) or None.
        """
        assert axis in ("left", "right")
        print("PlotBackend dataToPixel not implemented")
        return

    def enableActiveCurveHandling(self, flag=True):
        if flag:
            self._activeCurveHandling = True
        else:
            self._activeCurveHandling = False

    def getBaseVectors(self):
        """Returns the coordinate in the orthogonal plot of the X and Y unit
        vectors of the data.

        :return: X and Y data unit vectors in orthogonal plot coordinates
        :rtype: 2-tuple of 2-tuple of float: (xx, xy), (yx, yy)
        """
        print("PlotBackend getBaseVectors not implemented")

    def getGraphCursor(self):
        """
        Returns the current state of the crosshair cursor.

        :return: None if the crosshair cursor is not active,
                 else a tuple (color, linewidth, linestyle).
        """
        print("PlotBackend getGraphCursor not implemented")
        return None

    def getDefaultColormap(self):
        """
        Return the colormap that will be applied by the backend to an image
        if no colormap is applied to it.

        A colormap is a dictionnary with the keys:

        - name: string
        - normalization: string (linear, log)
        - autoscale: boolean
        - vmin: float, minimum value
        - vmax: float, maximum value
        - colors: integer (typically 256)
        """
        print("PlotBackend getDefaultColormap called")
        return {'name': 'gray', 'normalization':'linear',
                'autoscale':True, 'vmin':0.0, 'vmax':1.0,
                'colors':256}

    def getDrawMode(self):
        """
        Return a dictionnary (or None) with the parameters passed when setting
        the draw mode.

        - shape: The shape being drawn
        - label: Associated text (or None)

        and any other info passed to setDrawMode
        """
        print("PlotBackend getDrawMode not implemented")
        return None

    def getGraphTitle(self):
        """
        Get the graph title.
        :return:  string
        """
        print("PlotBackend getGraphTitle not implemented")
        return ""

    def getGraphXLimits(self):
        """
        Get the graph X (bottom) limits.
        :return:  Minimum and maximum values of the X axis
        """
        print("Get the graph X (bottom) limits")
        return self._xMin, self._xMax

    def getGraphXLabel(self):
        """
        Get the graph X (bottom) label.
        :return:  string
        """
        print("PlotBackend getGraphXLabel not implemented")
        return "X"

    def getGraphYLimits(self, axis="left"):
        """
        Get the graph Y (left) limits.

        :param axis: The axis for which to get the limits
        :type axis: str, either "left" (default) or "right"
        :return:  Minimum and maximum values of the Y axis
        """
        print("Get the graph Y (left) limits")
        assert axis in ("left", "right")
        if axis == "left":
            return self._yMin, self._yMax
        else:
            return self._yRightMin, self._yRightMax

    def getGraphYLabel(self):
        """
        Get the graph Y (left) label.
        :return:  string
        """
        print("PlotBackend getGraphYLabel not implemented")
        return "Y"

    def getSupportedColormaps(self):
        """
        Get a list of strings with the colormap names supported by the backend.
        The list should at least contain and start by:
        ['gray', 'reversed gray', 'temperature', 'red', 'green', 'blue']
        """
        return ['gray', 'reversed gray', 'temperature', 'red', 'green', 'blue']

    def getWidgetHandle(self):
        """
        :return: Backend widget or None if the backend inherits from widget.
        """
        return None

    def insertMarker(self, x, y, legend=None, text=None, color='k',
                      selectable=False, draggable=False, replot=True,
                      symbol=None, constraint=None,
                      **kw):
        """
        :param x: Horizontal position of the marker in graph coordenates
        :type x: float
        :param y: Vertical position of the marker in graph coordenates
        :type y: float
        :param legend: Legend associated to the marker to identify it
        :type legend: string
        :param text: Text associated to the marker
        :type text: string or None
        :param color: Color to be used for instance 'blue', 'b', '#FF0000'
        :type color: string, default 'k' (black)
        :param selectable: Flag to indicate if the marker can be selected
        :type selectable: boolean, default False
        :param draggable: Flag to indicate if the marker can be moved
        :type draggable: boolean, default False
        :param replot: Flag to indicate if the plot is to be updated
        :type replot: boolean, default True
        :param str symbol: Symbol representing the marker in:

            - 'o' circle
            - '.' point
            - ',' pixel
            - '+' cross
            - 'x' x-cross
            - 'd' diamond
            - 's' square

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
        print("PlotBackend insertMarker not implemented")
        return legend

    def insertXMarker(self, x, legend=None, text=None, color='k',
                      selectable=False, draggable=False, replot=True,
                      **kw):
        """
        :param x: Horizontal position of the marker in graph coordenates
        :type x: float
        :param legend: Legend associated to the marker to identify it
        :type legend: string
        :param text: Text associated to the marker
        :type text: string or None
        :param color: Color to be used for instance 'blue', 'b', '#FF0000'
        :type color: string, default 'k' (black)
        :param selectable: Flag to indicate if the marker can be selected
        :type selectable: boolean, default False
        :param draggable: Flag to indicate if the marker can be moved
        :type draggable: boolean, default False
        :param replot: Flag to indicate if the plot is to be updated
        :type replot: boolean, default True
        :return: Handle used by the backend to univocally access the marker
        """
        print("PlotBackend insertXMarker not implemented")
        return legend

    def insertYMarker(self, y, legend=None, text=None, color='k',
                      selectable=False, draggable=False, replot=True,
                      **kw):
        """
        :param y: Vertical position of the marker in graph coordenates
        :type y: float
        :param legend: Legend associated to the marker to identify it
        :type legend: string
        :param text: Text associated to the marker
        :type text: string or None
        :param color: Color to be used for instance 'blue', 'b', '#FF0000'
        :type color: string, default 'k' (black)
        :param selectable: Flag to indicate if the marker can be selected
        :type selectable: boolean, default False
        :param draggable: Flag to indicate if the marker can be moved
        :type draggable: boolean, default False
        :param replot: Flag to indicate if the plot is to be updated
        :type replot: boolean, default True
        :return: Handle used by the backend to univocally access the marker
        """
        print("PlotBackend insertYMarker not implemented")
        return legend

    def invertYAxis(self, flag=True):
        """
        :param flag: If True, put the vertical axis origin on plot top left
        :type flag: boolean
        """
        print("PlotBackend invertYAxis not implemented")

    def isDefaultBaseVectors(self):
        """Returns True if axes have the default basis, False otherwise.

        The default basis is x horizontal, y vertical.
        """
        return True

    def isDrawModeEnabled(self):
        """
        :return: True if user can draw
        """
        print("PlotBackend isDrawModeEnabled not implemented")
        return False

    def isKeepDataAspectRatio(self):
        """Returns whether the plot is keeping data aspect ratio or not.

        :return: True if keeping data aspect ratio else False.
        """
        print("PlotBackend isKeepDataAspectRatio not implemented")
        return False

    def isXAxisAutoScale(self):
        """
        :return: True if bottom axis is automatically adjusting the scale
        """
        print("PlotBackend isXAxisAutoScale not implemented")
        return True

    def isYAxisAutoScale(self):
        """
        :return: True if left axis is automatically adjusting the scale
        """
        print("PlotBackend isYAxisAutoScale not implemented")
        return True

    def isYAxisInverted(self):
        """
        :return: True if left axis is inverted
        """
        print("PlotBackend isYAxisInverted not implemented")

    def isZoomModeEnabled(self):
        """
        :return: True if user can zoom
        """
        print("PlotBackend isZoomModeEnabled not implemented")
        return True


    def keepDataAspectRatio(self, flag=True):
        """
        :param flag:  True to respect data aspect ratio
        :type flag: Boolean, default True
        """
        print("PlotBackend keepDataAspectRatio not implemented")

    def pixelToData(self, x=None, y=None, axis="left"):
        """
        Convert a position in pixels in the widget to a position in
        the data space.

        :param float x: The X coordinate in pixels. If None (default)
                            the center of the widget is used.
        :param float y: The Y coordinate in pixels. If None (default)
                            the center of the widget is used.
        :param str axis: The Y axis to use for the conversion
                         ('left' or 'right').
        :returns: The corresponding position in data space or
                  None if the pixel position is not in the plot area.
        :rtype: A tuple of 2 floats: (xData, yData) or None.
        """
        assert axis in ("left", "right")
        print("PlotBackend pixelToData not implemented")
        return

    def removeCurve(self, legend, replot=True):
        """
        Remove the curve associated to the supplied legend from the graph.
        The graph will be updated if replot is true.

        :param legend: The legend associated to the curve to be deleted
        :type legend: string or handle
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        """
        print("PlotBackend removeCurve not implemented")
        return

    def removeImage(self, legend, replot=True):
        """
        Remove the image associated to the supplied legend from the graph.
        The graph will be updated if replot is true.

        :param legend: The legend associated to the image to be deleted
        :type legend: string or handle
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        """
        print("PlotBackend removeImage not implemented")
        return

    def removeItem(self, legend, replot=True):
        print("PlotBackend removeItem not implemented")
        return

    def removeMarker(self, label, replot=True):
        """
        Remove the marker associated to the supplied handle from the graph.
        The graph will be updated if replot is true.

        :param label: The handle/label associated to the curve to be deleted
        :type label: string or handle
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        """
        print("PlotBackend removeMarker not implemented")

    def resetZoom(self, dataMargins=None):
        """
        Autoscale any axis that is in autoscale mode.
        Keep current limits on axes not in autoscale mode

        Extra margins can be added around the data inside the plot area.
        Margins are given as one ratio of the data range per limit of the
        data (xMin, xMax, yMin and yMax limits).
        For log scale, extra margins are applied in log10 of the data.

        :param dataMargins: Ratios of margins to add around the data inside
                            the plot area for each side (Default: no margins).
        :type dataMargins: A 4-tuple of float as (xMin, xMax, yMin, yMax).
        """
        print("PlotBackend resetZoom not implemented")

    def replot(self):
        """
        Update plot. If replot is a reserved word of the used backend, it can
        be implemented as replot_
        """
        print("PlotBackend replot not implemented")

    def saveGraph(self, fileName, fileFormat='svg', dpi=None, **kw):
        """
        :param fileName: Destination
        :type fileName: String or StringIO or BytesIO
        :param fileFormat:  String specifying the format
        :type fileFormat: String (default 'svg')
        """
        print("PlotBackend saveGraph not implemented")

    def setActiveCurve(self, legend, replot=True):
        """
        Make the curve identified by the supplied legend active curve.

        :param legend: The legend associated to the curve
        :type legend: string
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        """
        print("PlotBackend setActiveCurve not implemented")
        return

    def setActiveCurveColor(self, color="#000000"):
        self._activeCurveColor = color

    def setActiveImage(self, legend, replot=True):
        """
        Make the image identified by the supplied legend active.

        :param legend: The legend associated to the image
        :type legend: string
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        """
        if DEBUG:
            print("PlotBackend setActiveImage not implemented")
        return

    def setBaseVectors(self, x=(1., 0.), y=(0., 1.)):
        """Set the data coordinates relative to the orthogonal plot area.

        Useful for non-orthogonal axes.

        :param x: (x, y) coords on the X data base vector in orthogonal coords.
        :type x: 2-tuple of float
        :param y: (x, y) coords of the Y data base vector in orthogonal coords.
        :type y: 2-tuple of float
        """
        print("PlotBackend setBaseVectors not implemented")

    def setCallback(self, callback_function):
        """
        :param callback_function: function accepting a dictionnary as input to handle the graph events
        :type callback_function: callable
        """
        self._callback = callback_function

    def setGraphCursor(self, flag=None, color=None,
                       linewidth=None, linestyle=None):
        """
        Toggle the display of a crosshair cursor and set its attributes.

        :param bool flag: Toggle the display of a crosshair cursor.
                          The crosshair cursor is hidden by default.
        :param color: The color to use for the crosshair.
        :type color: A string (either a predefined color name in Colors.py
                     or "#RRGGBB")) or a 4 columns unsigned byte array.
                     Default is black.
        :param int linewidth: The width of the lines of the crosshair.
                              Default is 1.
        :param linestyle: Type of line::

            - ' '  no line
            - '-'  solid line
            - '--' dashed line
            - '-.' dash-dot line
            - ':'  dotted line

        :type linestyle: None or one of the predefined styles.
        """
        print("PlotBackend setGraphCursor not implemented")

    def setDefaultColormap(self, colormap=None):
        """
        Sets the colormap that will be applied by the backend to an image
        if no colormap is applied to it.

        A colormap is a dictionnary with the keys:

        :type name: string
        :type normalization: string (linear, log)
        :type autoscale: boolean
        :type vmin: float, minimum value
        :type vmax: float, maximum value
        :type colors: integer (typically 256)

        If None is passed, the backend will reset to its default colormap.
        """
        print("PlotBackend setDefaultColormap not implemented")
        return

    def setDrawModeEnabled(self, flag=True, shape="polygon", label=None,
                           color=None, **kw):
        """
        Zoom and drawing are not compatible and cannot be enabled simultanelously

        :param flag: Enable drawing mode disabling zoom and picking mode
        :type flag: boolean, default True
        :param shape: Type of item to be drawn (line, hline, vline, rectangle...)
        :type shape: string (default polygon)
        :param label: Associated text (for identifying the signals)
        :type label: string, default None
        :param color: The color to use to draw the selection area
        :type color: string ("#RRGGBB") or 4 column unsigned byte array or
                     one of the predefined color names defined in Colors.py
        """
        if flag:
            self._drawModeEnabled = True
            #cannot draw and zoom simultaneously
            self.setZoomModeEnabled(False)
        else:
            self._drawModeEnabled = False
        print("PlotBackend setDrawModeEnabled not implemented")

    def setGraphTitle(self, title=""):
        """
        :param title: Title associated to the plot
        :type title: string, default is an empty string
        """
        print("PlotBackend setTitle not implemented")

    def setGraphXLabel(self, label="X"):
        """
        :param label: label associated to the plot bottom axis
        :type label: string, default is 'X'
        """
        print("PlotBackend setGraphXLabel not implemented")

    def setGraphXLimits(self, xmin, xmax):
        """
        :param xmin: minimum bottom axis value
        :type xmin: float
        :param xmax: maximum bottom axis value
        :type xmax: float
        """
        self._xMin = xmin
        self._xMax = xmax
        print("PlotBackend setGraphXLimits not implemented")

    def setGraphYLabel(self, label="Y"):
        """
        :param label: label associated to the plot left axis
        :type label: string, default is 'Y'
        """
        print("PlotBackend setGraphYLabel not implemented")

    def setGraphYLimits(self, ymin, ymax, axis="left"):
        """
        :param ymin: minimum left axis value
        :type ymin: float
        :param ymax: maximum left axis value
        :type ymax: float
        :param axis: The axis for which to set the limits
        :type axis: str, either "left" (default) or "right"
        """
        assert axis in ("left", "right")
        if axis == "left":
            self._yMin = ymin
            self._yMax = ymax
        else:
            self._yRightMin = ymin
            self._yRightMax = ymax
        print("PlotBackend setGraphYLimits not implemented")

    def setLimits(self, xmin, xmax, ymin, ymax):
        """
        Convenience method

        :param xmin: minimum bottom axis value
        :type xmin: float
        :param xmax: maximum bottom axis value
        :type xmax: float
        :param ymin: minimum left axis value
        :type ymin: float
        :param ymax: maximum left axis value
        :type ymax: float
        """
        self.setGraphXLimits(xmin, xmax)
        self.setGraphYLimits(ymin, ymax)

    def setXAxisAutoScale(self, flag=True):
        """
        :param flag: If True, the bottom axis will adjust scale on zomm reset
        :type flag: boolean, default True
        """
        if flag:
            self._xAutoScale = True
        else:
            self._xAutoScale = False
        print("PlotBackend setXAxisAutoScale not implemented")

    def setXAxisLogarithmic(self, flag=True):
        """
        :param flag: If True, the bottom axis will use a log scale
        :type flag: boolean, default True
        """
        print("PlotBackend setXAxisLogarithmic not implemented")

    def setYAxisAutoScale(self, flag=True):
        """
        :param flag: If True, the left axis will adjust scale on zomm reset
        :type flag: boolean, default True
        """
        if flag:
            self._yAutoScale = True
        else:
            self._yAutoScale = False
        print("PlotBackend setYAxisAutoScale not implemented")

    def setYAxisLogarithmic(self, flag):
        """
        :param flag: If True, the left axis will use a log scale
        :type flag: boolean
        """
        print("PlotBackend setYAxisLogarithmic not implemented")

    def setZoomModeEnabled(self, flag=True, color=None):
        """
        Zoom and drawing cannot be simultaneously enabled.

        :param flag: If True, the user can zoom.
        :type flag: boolean, default True
        :param color: The color to use to draw the selection area.
                      Default 'black"
        :param color: The (optional) color to use to draw the selection area.
        :type color: string ("#RRGGBB") or 4 column unsigned byte array or
                     one of the predefined color names defined in Colors.py
        """
        if flag:
            self._zoomEnabled = True
            #cannot draw and zoom simultaneously
            self.setDrawModeEnabled(False)
        else:
            self._zoomEnabled = True
        print("PlotBackend setZoomModeEnabled not implemented")

    def showGrid(self, flag=True):
        """
        :param flag: If True, the grid will be shown.
        :type flag: boolean, default True
        """
        print("PlotBackend showGrid not implemented")

def main():
    import numpy
    from .Plot1D import Plot1D
    x = numpy.arange(100.)
    y = x * x
    plot = Plot1D()
    plot.addCurve(x, y, "dummy")
    plot.addCurve(x + 100, -x * x)
    print("X Limits = ", plot.getGraphXLimits())
    print("Y Limits = ", plot.getGraphYLimits())
    print("All curves = ", plot.getAllCurves())
    plot.removeCurve("dummy")
    print("All curves = ", plot.getAllCurves())

if __name__ == "__main__":
    main()
