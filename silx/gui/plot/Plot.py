# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
# ###########################################################################*/
"""Plot API for 1D and 2D data.

The :class:`Plot` implements the plot API initially provided in PyMca.


Colormap
--------

The :class:`Plot` uses a dictionary to describe a colormap.
This dictionary has the following keys:

- 'name': str, name of the colormap. Available colormap are returned by
          :meth:`Plot.getSupportedColormaps`.
          At least 'gray', 'reversed gray', 'temperature',
          'red', 'green', 'blue' are supported.
- 'normalization': Either 'linear' or 'log'
- 'autoscale': bool, True to get bounds from the min and max of the
               data, False to use [vmin, vmax]
- 'vmin': float, min value, ignored if autoscale is True
- 'vmax': float, max value, ignored if autoscale is True
- 'colors': optional, custom colormap.
            Nx3 or Nx4 numpy array of RGB(A) colors,
            either uint8 or float in [0, 1].
            If 'name' is None, then this array is used as the colormap.


Plot Events
-----------

The Plot sends some event to the registered callback
(See :meth:`Plot.setCallback`).
Those events are sent as a dictionary with a key 'event' describing the kind
of event.

Drawing events
..............

'drawingProgress' and 'drawingFinished' events are sent during drawing
interaction (See :meth:`Plot.setInteractiveMode`).

- 'event': 'drawingProgress' or 'drawingFinished'
- 'parameters': dict of parameters used by the drawing mode.
                It has the following keys: 'shape', 'label', 'color'.
                See :meth:`Plot.setInteractiveMode`.
- 'points': Points (x, y) in data coordinates of the drawn shape.
            For 'hline' and 'vline', it is the 2 points defining the line.
            For 'line' and 'rectangle', it is the coordinates of the start
            drawing point and the latest drawing point.
            For 'polygon', it is the coordinates of all points of the shape.
- 'type': The type of drawing in 'line', 'hline', 'polygon', 'rectangle',
          'vline'.
- 'xdata' and 'ydata': X coords and Y coords of shape points in data
                       coordinates (as in 'points').

When the type is 'rectangle', the following additional keys are provided:

- 'x' and 'y': The origin of the rectangle in data coordinates
- 'widht' and 'height': The size of the rectangle in data coordinates


Mouse events
............

'mouseMoved', 'mouseClicked' and 'mouseDoubleClicked' events are sent for
mouse events.

They provide the following keys:

- 'event': 'mouseMoved', 'mouseClicked' or 'mouseDoubleClicked'
- 'button': the mouse button that was pressed in 'left', 'middle', 'right'
- 'x' and 'y': The mouse position in data coordinates
- 'xpixel' and 'ypixel': The mouse position in pixels


Marker events
.............

'hover', 'markerClicked', 'markerMoving' and 'markerMoved' events are
sent during interaction with markers.

'hover' is sent when the mouse cursor is over a marker.
'markerClicker' is sent when the user click on a selectable marker.
'markerMoving' and 'markerMoved' are sent when a draggable marker is moved.

They provide the following keys:

- 'event': 'hover', 'markerClicked', 'markerMoving' or 'markerMoved'
- 'button': the mouse button that is pressed in 'left', 'middle', 'right'
- 'draggable': True if the marker is draggable, False otherwise
- 'label': The legend associated with the clicked image or curve
- 'selectable': True if the marker is selectable, False otherwise
- 'type': 'marker'
- 'x' and 'y': The mouse position in data coordinates
- 'xdata' and 'ydata': The marker position in data coordinates

'markerClicked' and 'markerMoving' events have a 'xpixel' and a 'ypixel'
additional keys, that provide the mouse position in pixels.


Image and curve events
......................

'curveClicked' and 'imageClicked' events are sent when a selectable curve
or image is clicked.

Both share the following keys:

- 'event': 'curveClicked' or 'imageClicked'
- 'button': the mouse button that was pressed in 'left', 'middle', 'right'
- 'label': The legend associated with the clicked image or curve
- 'type': The type of item in 'curve', 'image'
- 'x' and 'y': The clicked position in data coordinates
- 'xpixel' and 'ypixel': The clicked position in pixels

'curveClicked' events have a 'xdata' and a 'ydata' additional keys, that
provide the coordinates of the picked points of the curve.
There can be more than one point of the curve being picked, and if a line of
the curve is picked, only the first point of the line is included in the list.

'imageClicked' have a 'col' and a 'row' additional keys, that provide
the column and row index in the image array that was clicked.


Limits changed events
.....................

'limitsChanged' events are sent when the limits of the plot are changed.
This can results from user interaction or API calls.

It provides the following keys:

- 'event': 'limitsChanged'
- 'source': id of the widget that emitted this event.
- 'xdata': Range of X in graph coordinates: (xMin, xMax).
- 'ydata': Range of Y in graph coordinates: (yMin, yMax).
- 'y2data': Range of right axis in graph coordinates (y2Min, y2Max) or None.

Plot state change events
........................

The following events are emitted when the plot is modified.
They provide the new state:

- 'setGraphCursor' event with a 'state' key (bool)
- 'setGraphGrid' event with a 'which' key (str), see :meth:`setGraphGrid`
- 'setKeepDataAspectRatio' event with a 'state' key (bool)
- 'setXAxisAutoScale' event with a 'state' key (bool)
- 'setXAxisLogarithmic' event with a 'state' key (bool)
- 'setYAxisAutoScale' event with a 'state' key (bool)
- 'setYAxisInverted' event with a 'state' key (bool)
- 'setYAxisLogarithmic' event with a 'state' key (bool)

A 'contentChanged' event is triggered when the content of the plot is updated.
It provides the following keys:

- 'action': The change of the plot: 'add' or 'remove'
- 'kind': The kind of primitive changed: 'curve', 'image', 'item' or 'marker'
- 'legend': The legend of the primitive changed.

'activeCurveChanged' and 'activeImageChanged' events with the following keys:

- 'legend': Name (str) of the current active item or None if no active item.
- 'previous': Name (str) of the previous active item or None if no item was
              active. It is the same as 'legend' if 'updated' == True
- 'updated': (bool) True if active item name did not changed,
             but active item data or style was updated.

'interactiveModeChanged' event with a 'source' key identifying the object
setting the interactive mode.
"""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "23/02/2016"


from collections import Iterable, OrderedDict, namedtuple
import logging

import numpy

# Import matplotlib backend here to init matplotlib our way
from .BackendMatplotlib import BackendMatplotlibQt
from . import Colors
from . import PlotInteraction
from . import PlotEvents
from . import _utils


_logger = logging.getLogger(__name__)


_COLORDICT = Colors.COLORDICT
_COLORLIST = [_COLORDICT['black'],
              _COLORDICT['blue'],
              _COLORDICT['red'],
              _COLORDICT['green'],
              _COLORDICT['pink'],
              _COLORDICT['yellow'],
              _COLORDICT['brown'],
              _COLORDICT['cyan'],
              _COLORDICT['magenta'],
              _COLORDICT['orange'],
              _COLORDICT['violet'],
              # _COLORDICT['bluegreen'],
              _COLORDICT['grey'],
              _COLORDICT['darkBlue'],
              _COLORDICT['darkRed'],
              _COLORDICT['darkGreen'],
              _COLORDICT['darkCyan'],
              _COLORDICT['darkMagenta'],
              _COLORDICT['darkYellow'],
              _COLORDICT['darkBrown']]


"""
Object returned when requesting the data range.
"""
_PlotDataRange = namedtuple('PlotDataRange',
                            ['x', 'y', 'yright'])


class Plot(object):
    """This class implements the plot API initially provided in PyMca.

    Supported backends:

    - 'matplotlib' and 'mpl': Matplotlib with Qt.
    - 'none': No backend, to run headless for testing purpose.

    :param parent: The parent widget of the plot (Default: None)
    :param backend: The backend to use. A str in:
                    'matplotlib', 'mpl', 'none'
                    or a :class:`BackendBase.BackendBase` class
    """

    defaultBackend = 'matplotlib'
    """Class attribute setting the default backend for all instances."""

    colorList = _COLORLIST
    colorDict = _COLORDICT

    def __init__(self, parent=None, backend=None):
        self._autoreplot = False
        self._dirty = False

        if backend is None:
            backend = self.defaultBackend

        if hasattr(backend, "__call__"):
            self._backend = backend(self, parent)

        elif hasattr(backend, "lower"):
            lowerCaseString = backend.lower()
            if lowerCaseString in ("matplotlib", "mpl"):
                backendClass = BackendMatplotlibQt
            elif lowerCaseString == 'none':
                from .BackendBase import BackendBase as backendClass
            else:
                raise ValueError("Backend not supported %s" % backend)
            self._backend = backendClass(self, parent)

        else:
            raise ValueError("Backend not supported %s" % str(backend))

        super(Plot, self).__init__()

        self.setCallback()  # set _callback

        # Items handling
        self._curves = OrderedDict()
        self._hiddenCurves = set()

        self._images = OrderedDict()
        self._markers = OrderedDict()
        self._items = OrderedDict()

        self._dataRange = False

        # line types
        self._styleList = ['-', '--', '-.', ':']

        self._colorIndex = 0
        self._styleIndex = 0

        self._activeCurveHandling = True
        self._activeCurve = None
        self._activeCurveColor = "#000000"
        self._activeImage = None

        # default properties
        self._cursorConfiguration = None

        self._logY = False
        self._logX = False
        self._xAutoScale = True
        self._yAutoScale = True
        self._grid = None

        # Store default labels provided to setGraph[X|Y]Label
        self._defaultLabels = {'x': '', 'y': '', 'yright': ''}
        # Store currently displayed labels
        # Current label can differ from input one with active curve handling
        self._currentLabels = {'x': '', 'y': '', 'yright': ''}

        self.setGraphTitle()
        self.setGraphXLabel()
        self.setGraphYLabel()
        self.setGraphYLabel('', axis='right')

        self.setDefaultColormap()  # Init default colormap

        self.setDefaultPlotPoints(False)
        self.setDefaultPlotLines(True)

        self._eventHandler = PlotInteraction.PlotInteraction(self)
        self._eventHandler.setInteractiveMode('zoom', color=(0., 0., 0., 1.))

        self._pressedButtons = []  # Currently pressed mouse buttons

        self._defaultDataMargins = (0., 0., 0., 0.)

        # Only activate autoreplot at the end
        # This avoids errors when loaded in Qt designer
        self._dirty = False
        self._autoreplot = True

    def _getDirtyPlot(self):
        """Return the plot dirty flag.

        If False, the plot has not changed since last replot.
        If True, the full plot need to be redrawn.
        If 'overlay', only the overlay has changed since last replot.

        It can be accessed by backend to check the dirty state.

        :return: False, True, 'overlay'
        """
        return self._dirty

    def _setDirtyPlot(self, overlayOnly=False):
        """Mark the plot as needing redraw

        :param bool overlayOnly: True to redraw only the overlay,
                                 False to redraw everything
        """
        wasDirty = self._dirty

        if not self._dirty and overlayOnly:
            self._dirty = 'overlay'
        else:
            self._dirty = True

        if self._autoreplot and not wasDirty:
            self._backend.postRedisplay()

    def _invalidateDataRange(self):
        """
        Notifies this Plot instance that the range has changed and will have
        to be recomputed.
        """
        self._dataRange = False

    def _updateDataRange(self):
        """
        Recomputes the range of the data displayed on this Plot.
        """
        # already available
        if self._dataRange is not False:
            return self._dataRange

        xMin = yMinLeft = yMinRight = float('nan')
        xMax = yMaxLeft = yMaxRight = float('nan')

        for curve, info in self._curves.items():
            # using numpy's separate min and max is faster than
            # a pure python minmax.
            if info['xmin'] is not None:
                xMin = numpy.nanmin([xMin, info['xmin']])
            if info['xmax'] is not None:
                xMax = numpy.nanmax([xMax, info['xmax']])

            if info['params']['yaxis'] == 'left':
                if info['ymin'] is not None:
                    yMinLeft = numpy.nanmin([yMinLeft, info['ymin']])
                if info['ymax'] is not None:
                    yMaxLeft = numpy.nanmax([yMaxLeft, info['ymax']])
            else:
                if info['ymin'] is not None:
                    yMinRight = numpy.nanmin([yMinRight, info['ymin']])
                if info['ymax'] is not None:
                    yMaxRight = numpy.nanmax([yMaxRight, info['ymax']])

        if not self.isXAxisLogarithmic() and not self.isYAxisLogarithmic():
            for image, info in self._images.items():
                if info['data'] is not None:
                    height, width = info['data'].shape[:2]
                    params = info['params']
                    origin = params['origin']
                    scale = params['scale']
                    # Taking care of scale might be < 0
                    xCoords = [origin[0], origin[0] + width * scale[0]]
                    xMin = numpy.nanmin([xMin] + xCoords)
                    xMax = numpy.nanmax([xMax] + xCoords)
                    # Taking care of scale might be < 0
                    yCoords = [origin[1], origin[1] + height * scale[1]]
                    yMinLeft = numpy.nanmin([yMinLeft] + yCoords)
                    yMaxLeft = numpy.nanmax([yMaxLeft] + yCoords)

        lGetRange = (lambda x, y:
                     None if numpy.isnan(x) and numpy.isnan(y) else (x, y))
        xRange = lGetRange(xMin, xMax)
        yLeftRange = lGetRange(yMinLeft, yMaxLeft)
        yRightRange = lGetRange(yMinRight, yMaxRight)

        self._dataRange = _PlotDataRange(x=xRange,
                                         y=yLeftRange,
                                         yright=yRightRange)

        return self._dataRange

    def getDataRange(self):
        """
        Returns this Plot's data range.

        :return: a namedtuple with the following members :
                x, y (left y axis), yright. Each member is a tuple (min, max)
                or None if no data is associated with the axis.
        :rtype: namedtuple
        """
        return self._updateDataRange()

    # Add

    # add * input arguments management:
    # If an arg is set, then use it.
    # Else:
    #     If a curve with the same legend exists, then use its arg value
    #     Else, use a default value.
    # Store used value.
    # This value is used when curve is updated either internally or by user.

    def addCurve(self, x, y, legend=None, info=None,
                 replace=False, replot=None,
                 color=None, symbol=None,
                 linewidth=None, linestyle=None,
                 xlabel=None, ylabel=None, yaxis=None,
                 xerror=None, yerror=None, z=None, selectable=None,
                 fill=None, resetzoom=True, **kw):
        """Add a 1D curve given by x an y to the graph.

        Curves are uniquely identified by their legend.
        To add multiple curves, call :meth:`addCurve` multiple times with
        different legend argument.
        To replace/update an existing curve, call :meth:`addCurve` with the
        existing curve legend.

        When curve parameters are not provided, if a curve with the
        same legend is displayed in the plot, its parameters are used.

        :param numpy.ndarray x: The data corresponding to the x coordinates
        :param numpy.ndarray y: The data corresponding to the y coordinates
        :param str legend: The legend to be associated to the curve (or None)
        :param info: User-defined information associated to the curve
        :param bool replace: True (the default) to delete already existing
                             curves
        :param color: color(s) to be used
        :type color: str ("#RRGGBB") or (npoints, 4) unsigned byte array or
                     one of the predefined color names defined in Colors.py
        :param str symbol: Symbol to be drawn at each (x, y) position::

            - 'o' circle
            - '.' point
            - ',' pixel
            - '+' cross
            - 'x' x-cross
            - 'd' diamond
            - 's' square
            - None (the default) to use default symbol

        :param float linewidth: The width of the curve in pixels (Default: 1).
        :param str linestyle: Type of line::

            - ' '  no line
            - '-'  solid line
            - '--' dashed line
            - '-.' dash-dot line
            - ':'  dotted line
            - None (the default) to use default line style

        :param str xlabel: Label to show on the X axis when the curve is active
                           or None to keep default axis label.
        :param str ylabel: Label to show on the Y axis when the curve is active
                           or None to keep default axis label.
        :param str yaxis: The Y axis this curve is attached to.
                          Either 'left' (the default) or 'right'
        :param xerror: Values with the uncertainties on the x values
        :type xerror: A float, or a numpy.ndarray of float32.
                      If it is an array, it can either be a 1D array of
                      same length as the data or a 2D array with 2 rows
                      of same length as the data: row 0 for positive errors,
                      row 1 for negative errors.
        :param yerror: Values with the uncertainties on the y values
        :type yerror: A float, or a numpy.ndarray of float32. See xerror.
        :param int z: Layer on which to draw the curve (default: 1)
                      This allows to control the overlay.
        :param bool selectable: Indicate if the curve can be selected.
                                (Default: True)
        :param bool fill: True to fill the curve, False otherwise (default).
        :param bool resetzoom: True (the default) to reset the zoom.
        :returns: The key string identify this curve
        """
        # Take care of input parameters: check/conversion, default value

        if replot is not None:
            _logger.warning(
                'addCurve deprecated replot argument, use resetzoom instead')
            resetzoom = replot and resetzoom

        if kw:
            _logger.warning('addCurve: deprecated extra arguments')

        legend = "Unnamed curve 1.1" if legend is None else str(legend)

        # Check/Convert input arguments

        # Convert to arrays (not forcing type) in order to avoid
        # problems at unexpected places: missing min or max attributes, problem
        # when using numpy.nonzero on lists, ...
        x = numpy.asarray(x)
        y = numpy.asarray(y)

        # TODO check color

        assert symbol in ('o', '.', ',', '+', 'x', 'd', 's', '', None)

        assert linestyle in ('', ' ', '-', '--', '-.', ':', None)

        if xlabel is not None:
            xlabel = str(xlabel)

        if ylabel is not None:
            ylabel = str(ylabel)

        assert yaxis in (None, 'left', 'right')

        # TODO check xerror, yerror

        if xerror is not None:
            xerror = numpy.asarray(xerror)

        if yerror is not None:
            yerror = numpy.asarray(yerror)

        if z is not None:
            z = int(z)

        if selectable is not None:
            selectable = bool(selectable)

        if fill is not None:
            fill = bool(fill)

        # Store all params with defaults in a dict to treat them at once
        params = {
            'info': info, 'color': color,
            'symbol': symbol, 'linewidth': linewidth, 'linestyle': linestyle,
            'xlabel': xlabel, 'ylabel': ylabel, 'yaxis': yaxis,
            'xerror': xerror, 'yerror': yerror, 'z': z,
            'selectable': selectable, 'fill': fill
        }

        # Check if curve is previously active
        wasActive = self.getActiveCurve(just_legend=True) == legend

        # First, try to get defaults from existing curve with same name
        previousCurve = self._curves.get(legend, None)
        if previousCurve is not None:
            defaults = previousCurve['params']

        else:  # If no existing curve use default values
            default_color, default_linestyle = self._getColorAndStyle()
            defaults = {
                'info': None, 'color': default_color,
                'symbol': self._defaultPlotPoints,
                'linewidth': 1,
                'linestyle': default_linestyle,
                'xlabel': None, 'ylabel': None, 'yaxis': 'left',
                'xerror': None, 'yerror': None, 'z': 1,
                'selectable': True, 'fill': False
            }

        # If a parameter is not given as argument, use its default value
        for key in defaults:
            if params[key] is None:
                params[key] = defaults[key]

        # Add: replace, filter data, add

        # This must be done after getting params from existing curve
        if replace:
            self.remove(kind='curve')
        else:
            # Remove previous curve from backend
            # but not from _curves and hiddenCurves to keep its place
            # This is a subset of self.remove(legend, kind='curve')
            if legend in self._curves:
                handle = self._curves[legend]['handle']
                if handle is not None:
                    self._backend.remove(handle)
                    self._curves[legend]['handle'] = None
                    self._setDirtyPlot()

        # Filter-out values <= 0
        xFiltered, yFiltered, color, xerror, yerror = self._logFilterData(
            x, y, params['color'], params['xerror'], params['yerror'],
            self.isXAxisLogarithmic(), self.isYAxisLogarithmic())

        if len(xFiltered) and not self.isCurveHidden(legend):
            handle = self._backend.addCurve(xFiltered, yFiltered, legend,
                                            color=color,
                                            symbol=params['symbol'],
                                            linestyle=params['linestyle'],
                                            linewidth=params['linewidth'],
                                            yaxis=params['yaxis'],
                                            xerror=xerror,
                                            yerror=yerror,
                                            z=params['z'],
                                            selectable=params['selectable'],
                                            fill=params['fill'])
            self._setDirtyPlot()

            # caching the min and max values for the getDataRange method.
            xMin = numpy.nanmin(xFiltered)
            xMax = numpy.nanmax(xFiltered)
            yMin = numpy.nanmin(yFiltered)
            yMax = numpy.nanmax(yFiltered)

        else:
            # The curve has no points or is hidden
            handle = None
            xMin, xMax, yMin, yMax = None, None, None, None

        self._curves[legend] = {
            'handle': handle, 'x': x, 'y': y, 'params': params,
            'xmin': xMin, 'xmax': xMax, 'ymin': yMin, 'ymax': yMax
        }

        self._invalidateDataRange()

        self.notify(
            'contentChanged', action='add', kind='curve', legend=legend)

        if len(self._curves) == 1 or wasActive:
            self.setActiveCurve(legend)

        if resetzoom:
            # We ask for a zoom reset in order to handle the plot scaling
            # if the user does not want that, autoscale of the different
            # axes has to be set to off.
            self.resetZoom()

        return legend

    def addImage(self, data, legend=None, info=None,
                 replace=True, replot=None,
                 xScale=None, yScale=None, z=None,
                 selectable=False, draggable=False,
                 colormap=None, pixmap=None,
                 xlabel=None, ylabel=None,
                 origin=None, scale=None,
                 resetzoom=True, **kw):
        """Add a 2D dataset or an image to the plot.

        It displays either an array of data using a colormap or a RGB(A) image.

        Images are uniquely identified by their legend.
        To add multiple images, call :meth:`addImage` multiple times with
        different legend argument.
        To replace/update an existing image, call :meth:`addImage` with the
        existing image legend.

        When image parameters are not provided, if an image with the
        same legend is displayed in the plot, its parameters are used.

        :param numpy.ndarray data: (nrows, ncolumns) data or
                                   (nrows, ncolumns, RGBA) ubyte array
        :param str legend: The legend to be associated to the image (or None)
        :param info: User-defined information associated to the image
        :param bool replace: True (default) to delete already existing images
        :param int z: Layer on which to draw the image (default: 0)
                      This allows to control the overlay.
        :param bool selectable: Indicate if the image can be selected.
                                (default: False)
        :param bool draggable: Indicate if the image can be moved.
                               (default: False)
        :param dict colormap: Description of the colormap to use (or None)
                              This is ignored if data is a RGB(A) image.
                              See :mod:`Plot` for the documentation
                              of the colormap dict.
        :param pixmap: Pixmap representation of the data (if any)
        :type pixmap: (nrows, ncolumns, RGBA) ubyte array or None (default)
        :param str xlabel: X axis label to show when this curve is active,
                           or None to keep default axis label.
        :param str ylabel: Y axis label to show when this curve is active,
                           or None to keep default axis label.
        :param origin: (origin X, origin Y) of the data.
                       It is possible to pass a single float if both
                       coordinates are equal.
                       Default: (0., 0.)
        :type origin: float or 2-tuple of float
        :param scale: (scale X, scale Y) of the data.
                      It is possible to pass a single float if both
                      coordinates are equal.
                      Default: (1., 1.)
        :type scale: float or 2-tuple of float
        :param bool resetzoom: True (the default) to reset the zoom.
        :returns: The key string identify this image
        """
        # Take care of input parameters: check/conversion, default value

        if xScale is not None or yScale is not None:
            _logger.warning(
                'addImage deprecated xScale and yScale arguments,'
                'use origin, scale arguments instead.')
            if origin is None and scale is None:
                origin = xScale[0], yScale[0]
                scale = xScale[1], yScale[1]
            else:
                _logger.warning(
                    'addCurve: xScale, yScale and origin, scale arguments'
                    ' are conflicting. xScale and yScale are ignored.'
                    ' Use only origin, scale arguments.')

        if replot is not None:
            _logger.warning(
                'addImage deprecated replot argument, use resetzoom instead')
            resetzoom = replot and resetzoom

        if kw:
            _logger.warning('addImage: deprecated extra arguments')

        legend = "Unnamed Image 1.1" if legend is None else str(legend)

        # Check/Convert input arguments
        data = numpy.asarray(data)

        if origin is not None:
            if isinstance(origin, Iterable):
                origin = float(origin[0]), float(origin[1])
            else:  # single value origin
                origin = float(origin), float(origin)

        if scale is not None:
            if isinstance(scale, Iterable):
                scale = float(scale[0]), float(scale[1])
            else:  # single value scale
                scale = float(scale), float(scale)

        if z is not None:
            z = int(z)

        if selectable is not None:
            selectable = bool(selectable)

        if draggable is not None:
            draggable = bool(draggable)

        if pixmap is not None:
            pixmap = numpy.asarray(pixmap)

        if xlabel is not None:
            xlabel = str(xlabel)

        if ylabel is not None:
            ylabel = str(ylabel)

        # Store all params with defaults in a dict to treat them at once
        params = {
            'info': info, 'origin': origin, 'scale': scale, 'z': z,
            'selectable': selectable, 'draggable': draggable,
            'colormap': colormap,
            'xlabel': xlabel, 'ylabel': ylabel
        }

        # First, try to get defaults from existing curve with same name
        previousImage = self._images.get(legend, None)
        if previousImage is not None:
            defaults = previousImage['params']

        else:  # If no existing image use default values
            defaults = {
                'info': None, 'origin': (0., 0.), 'scale': (1., 1.), 'z': 0,
                'selectable': False, 'draggable': False,
                'colormap': self.getDefaultColormap(),
                'xlabel': None, 'ylabel': None
            }

        # If a parameter is not given as argument, use its default value
        for key in defaults:
            if params[key] is None:
                params[key] = defaults[key]

        # Check if curve is previously active
        wasActive = self.getActiveImage(just_legend=True) == legend

        # Add: replace, filter data, add

        if replace:
            self.remove(kind='image')
        else:
            # Remove previous image from backend
            # but not from _images to keep its place
            # This is a subset of self.remove(legend, kind='image')
            if legend in self._images:
                handle = self._images[legend]['handle']
                if handle is not None:
                    self._backend.remove(handle)
                    self._images[legend]['handle'] = None
                    self._setDirtyPlot()

        if self.isXAxisLogarithmic() or self.isYAxisLogarithmic():
            _logger.info('Hide image while axes has log scale.')

        if (data is not None and not self.isXAxisLogarithmic() and
                not self.isYAxisLogarithmic()):
            if pixmap is not None:
                dataToSend = pixmap
            else:
                dataToSend = data

            handle = self._backend.addImage(dataToSend, legend=legend,
                                            origin=params['origin'],
                                            scale=params['scale'],
                                            z=params['z'],
                                            selectable=params['selectable'],
                                            draggable=params['draggable'],
                                            colormap=params['colormap'])
            self._setDirtyPlot()
        else:
            handle = None  # data is None or log scale

        self._images[legend] = {
            'handle': handle,
            'data': data,
            'pixmap': pixmap,
            'params': params
        }

        if len(self._images) == 1 or wasActive:
            self.setActiveImage(legend)

        self._invalidateDataRange()

        self.notify(
            'contentChanged', action='add', kind='image', legend=legend)

        if resetzoom:
            # We ask for a zoom reset in order to handle the plot scaling
            # if the user does not want that, autoscale of the different
            # axes has to be set to off.
            self.resetZoom()

        return legend

    def addItem(self, xdata, ydata, legend=None, info=None,
                replace=False,
                shape="polygon", color='black', fill=True,
                overlay=False, z=None, **kw):
        """Add an item (i.e. a shape) to the plot.

        Items are uniquely identified by their legend.
        To add multiple items, call :meth:`addItem` multiple times with
        different legend argument.
        To replace/update an existing item, call :meth:`addItem` with the
        existing item legend.

        :param numpy.ndarray xdata: The X coords of the points of the shape
        :param numpy.ndarray ydata: The Y coords of the points of the shape
        :param str legend: The legend to be associated to the item
        :param info: User-defined information associated to the item
        :param bool replace: True (default) to delete already existing images
        :param str shape: Type of item to be drawn in
                          hline, polygon (the default), rectangle, vline,
                          polylines
        :param str color: Color of the item, e.g., 'blue', 'b', '#FF0000'
                          (Default: 'black')
        :param bool fill: True (the default) to fill the shape
        :param bool overlay: True if item is an overlay (Default: False).
                             This allows for rendering optimization if this
                             item is changed often.
        :param int z: Layer on which to draw the item (default: 2)
        :returns: The key string identify this item
        """
        # expected to receive the same parameters as the signal

        if kw:
            _logger.warning('addItem deprecated parameters: %s', str(kw))

        legend = "Unnamed Item 1.1" if legend is None else str(legend)

        z = int(z) if z is not None else 2

        if replace:
            self.remove(kind='item')
        else:
            self.remove(legend, kind='item')

        handle = self._backend.addItem(xdata, ydata, legend=legend,
                                       shape=shape, color=color,
                                       fill=fill, overlay=overlay, z=z)
        self._setDirtyPlot(overlayOnly=overlay)

        self._items[legend] = {'handle': handle, 'overlay': overlay}

        self.notify('contentChanged', action='add', kind='item', legend=legend)

        return legend

    def addXMarker(self, x, legend=None,
                   text=None,
                   color=None,
                   selectable=False,
                   draggable=False,
                   constraint=None,
                   **kw):
        """Add a vertical line marker to the plot.

        Markers are uniquely identified by their legend.
        As opposed to curves, images and items, two calls to
        :meth:`addXMarker` without legend argument adds two markers with
        different identifying legends.

        :param float x: Position of the marker on the X axis in data
                        coordinates
        :param str legend: Legend associated to the marker to identify it
        :param str text: Text to display on the marker.
        :param str color: Color of the marker, e.g., 'blue', 'b', '#FF0000'
                          (Default: 'black')
        :param bool selectable: Indicate if the marker can be selected.
                                (default: False)
        :param bool draggable: Indicate if the marker can be moved.
                               (default: False)
        :param constraint: A function filtering marker displacement by
                           dragging operations or None for no filter.
                           This function is called each time a marker is
                           moved.
                           This parameter is only used if draggable is True.
        :type constraint: None or a callable that takes the coordinates of
                          the current cursor position in the plot as input
                          and that returns the filtered coordinates.
        :return: The key string identify this marker
        """
        if kw:
            _logger.warning(
                'addXMarker deprecated extra parameters: %s', str(kw))

        return self._addMarker(x=x, y=None, legend=legend,
                               text=text, color=color,
                               selectable=selectable, draggable=draggable,
                               symbol=None, constraint=constraint)

    def addYMarker(self, y,
                   legend=None,
                   text=None,
                   color=None,
                   selectable=False,
                   draggable=False,
                   constraint=None,
                   **kw):
        """Add a horizontal line marker to the plot.

        Markers are uniquely identified by their legend.
        As opposed to curves, images and items, two calls to
        :meth:`addYMarker` without legend argument adds two markers with
        different identifying legends.

        :param float y: Position of the marker on the Y axis in data
                        coordinates
        :param str legend: Legend associated to the marker to identify it
        :param str text: Text to display next to the marker.
        :param str color: Color of the marker, e.g., 'blue', 'b', '#FF0000'
                          (Default: 'black')
        :param bool selectable: Indicate if the marker can be selected.
                                (default: False)
        :param bool draggable: Indicate if the marker can be moved.
                               (default: False)
        :param constraint: A function filtering marker displacement by
                           dragging operations or None for no filter.
                           This function is called each time a marker is
                           moved.
                           This parameter is only used if draggable is True.
        :type constraint: None or a callable that takes the coordinates of
                          the current cursor position in the plot as input
                          and that returns the filtered coordinates.
        :return: The key string identify this marker
        """
        if kw:
            _logger.warning(
                'addYMarker deprecated extra parameters: %s', str(kw))

        return self._addMarker(x=None, y=y, legend=legend,
                               text=text, color=color,
                               selectable=selectable, draggable=draggable,
                               symbol=None, constraint=constraint)

    def addMarker(self, x, y, legend=None,
                  text=None,
                  color=None,
                  selectable=False,
                  draggable=False,
                  symbol='+',
                  constraint=None,
                  **kw):
        """Add a point marker to the plot.

        Markers are uniquely identified by their legend.
        As opposed to curves, images and items, two calls to
        :meth:`addMarker` without legend argument adds two markers with
        different identifying legends.

        :param float x: Position of the marker on the X axis in data
                        coordinates
        :param float y: Position of the marker on the Y axis in data
                        coordinates
        :param str legend: Legend associated to the marker to identify it
        :param str text: Text to display next to the marker
        :param str color: Color of the marker, e.g., 'blue', 'b', '#FF0000'
                          (Default: 'black')
        :param bool selectable: Indicate if the marker can be selected.
                                (default: False)
        :param bool draggable: Indicate if the marker can be moved.
                               (default: False)
        :param str symbol: Symbol representing the marker in::

            - 'o' circle
            - '.' point
            - ',' pixel
            - '+' cross (the default)
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
        :return: The key string identify this marker
        """
        if kw:
            _logger.warning(
                'addMarker deprecated extra parameters: %s', str(kw))

        if x is None:
            xmin, xmax = self.getGraphXLimits()
            x = 0.5 * (xmax + xmin)

        if y is None:
            ymin, ymax = self.getGraphYLimits()
            y = 0.5 * (ymax + ymin)

        return self._addMarker(x=x, y=y, legend=legend,
                               text=text, color=color,
                               selectable=selectable, draggable=draggable,
                               symbol=symbol, constraint=constraint)

    def _addMarker(self, x, y, legend,
                   text, color,
                   selectable, draggable,
                   symbol, constraint):
        """Common method for adding point, vline and hline marker.

        See :meth:`addMarker` for argument documentation.
        """
        if legend is None:
            legend = "Unnamed Marker 0"
            i = 1
            while legend in self._markers:
                legend = "Unnamed Marker %d" % i
                i += 1

        if color is None:
            color = self.colorDict['black']
        elif color in self.colorDict:
            color = self.colorDict[color]

        if constraint is not None and not callable(constraint):
            # Then it must be a string
            if hasattr(constraint, 'lower'):
                if constraint.lower().startswith('h'):
                    constraint = lambda xData, yData: (xData, y)
                elif constraint.lower().startswith('v'):
                    constraint = lambda xData, yData: (x, yData)
                else:
                    raise ValueError(
                        "Unsupported constraint name: %s" % constraint)
            else:
                raise ValueError("Unsupported constraint")

        # Apply constraint to provided position
        if draggable and constraint is not None:
            x, y = constraint(x, y)

        if legend in self._markers:
            self.remove(legend, kind='marker')

        handle = self._backend.addMarker(
            x=x, y=y, legend=legend, text=text, color=color,
            selectable=selectable, draggable=draggable,
            symbol=symbol, constraint=constraint,
            overlay=draggable)

        self._markers[legend] = {'handle': handle, 'params': {
            'x': x, 'y': y,
            'text': text, 'color': color,
            'selectable': selectable, 'draggable': draggable,
            'symbol': symbol, 'constraint': constraint}
        }

        self._setDirtyPlot(overlayOnly=draggable)

        self.notify(
            'contentChanged', action='add', kind='marker', legend=legend)

        return legend

    # Hide

    def isCurveHidden(self, legend):
        """Returns True if the curve associated to legend is hidden, else False

        :param str legend: The legend key identifying the curve
        :return: True if the associated curve is hidden, False otherwise
        """
        return legend in self._hiddenCurves

    def hideCurve(self, legend, flag=True, replot=None):
        """Show/Hide the curve associated to legend.

        Even when hidden, the curve is kept in the list of curves.

        :param str legend: The legend associated to the curve to be hidden
        :param bool flag: True (default) to hide the curve, False to show it
        """
        if replot is not None:
            _logger.warning('hideCurve deprecated replot parameter')

        if legend not in self._curves:
            _logger.warning('Curve not in plot: %s', legend)
            return

        if flag:
            handle = self._curves[legend]['handle']
            if handle is not None:
                self._backend.remove(handle)
                self._curves[legend]['handle'] = None

            self._hiddenCurves.add(legend)
        else:
            self._hiddenCurves.discard(legend)
            curve = self._curves[legend]
            self.addCurve(curve['x'], curve['y'], legend, resetzoom=False,
                          **curve['params'])

        self._setDirtyPlot()

    # Remove

    ITEM_KINDS = 'curve', 'image', 'item', 'marker'

    def remove(self, legend=None, kind=ITEM_KINDS):
        """Remove one or all element(s) of the given legend and kind.

        Examples:

        - ``remove()`` clears the plot
        - ``remove(kind='curve')`` removes all curves from the plot
        - ``remove('myCurve', kind='curve')`` removes the curve with
          legend 'myCurve' from the plot.
        - ``remove('myImage, kind='image')`` removes the image with
          legend 'myImage' from the plot.
        - ``remove('myImage')`` removes elements (for instance curve, image,
          item and marker) with legend 'myImage'.

        :param str legend: The legend associated to the element to remove,
                           or None to remove
        :param kind: The kind of elements to remove from the plot.
                     In: 'all', 'curve', 'image', 'item', 'marker'.
                     By default, it removes all kind of elements.
        :type kind: str or tuple of str to specify multiple kinds.
        """
        if kind is 'all':  # Replace all by tuple of all kinds
            kind = self.ITEM_KINDS

        if kind in self.ITEM_KINDS:  # Kind is a str, make it a tuple
            kind = (kind,)

        if legend is None:  # This is a clear
            # Clear each given kind
            for aKind in kind:
                if aKind == 'curve':
                    # Copy as _curves gets changed
                    for legend in list(self._curves):
                        self.remove(legend, kind='curve')
                    self._curves = OrderedDict()
                    self._hiddenCurves = set()
                    self._colorIndex = 0
                    self._styleIndex = 0

                elif aKind == 'image':
                    # Copy as _images gets changed
                    for legend in list(self._images):
                        self.remove(legend, kind='image')
                    self._images = OrderedDict()

                elif aKind == 'item':
                    # Copy as _items gets changed
                    for legend in list(self._items):
                        self.remove(legend, kind='item')
                    self._items = OrderedDict()

                elif aKind == 'marker':
                    # Copy as _markers gets changed
                    for legend in list(self._markers):
                        self.remove(legend, kind='marker')
                    self._markers = OrderedDict()
                else:
                    _logger.warning('remove: Unhandled item kind %s', aKind)

        else:  # This is removing a single element
            # Remove each given kind
            for aKind in kind:
                if aKind == 'curve':
                    self._hiddenCurves.discard(legend)

                    if legend in self._curves:
                        if self.getActiveCurve(just_legend=True) == legend:
                            # Reset active curve
                            self.setActiveCurve(None)

                        handle = self._curves[legend]['handle']
                        if handle is not None:
                            self._backend.remove(handle)
                            self._setDirtyPlot()
                        del self._curves[legend]

                        if not self._curves:
                            self._colorIndex = 0
                            self._styleIndex = 0

                        self._invalidateDataRange()
                        self.notify('contentChanged', action='remove',
                                    kind='curve', legend=legend)

                elif aKind == 'image':
                    if legend in self._images:
                        if self.getActiveImage(just_legend=True) == legend:
                            # Reset active image
                            self.setActiveImage(None)

                        handle = self._images[legend]['handle']
                        if handle is not None:
                            self._backend.remove(handle)
                            self._setDirtyPlot()
                        del self._images[legend]

                        self._invalidateDataRange()
                        self.notify('contentChanged', action='remove',
                                    kind='image', legend=legend)

                elif aKind == 'item':
                    item = self._items.pop(legend, None)
                    if item is not None:
                        if item['handle'] is not None:
                            self._backend.remove(item['handle'])
                            self._setDirtyPlot(overlayOnly=item['overlay'])

                        self.notify('contentChanged', action='remove',
                                    kind='item', legend=legend)

                elif aKind == 'marker':
                    marker = self._markers.pop(legend, None)
                    if marker is not None:
                        if marker['handle'] is not None:
                            self._backend.remove(marker['handle'])
                            self._setDirtyPlot(
                                overlayOnly=marker['params']['draggable'])

                        self.notify('contentChanged', action='remove',
                                    kind='marker', legend=legend)

                else:
                    _logger.warning('remove: Unhandled item kind %s', aKind)

    def removeCurve(self, legend):
        """Remove the curve associated to legend from the graph.

        :param str legend: The legend associated to the curve to be deleted
        """
        if legend is None:
            return
        self.remove(legend, kind='curve')

    def removeImage(self, legend):
        """Remove the image associated to legend from the graph.

        :param str legend: The legend associated to the image to be deleted
        """
        if legend is None:
            return
        self.remove(legend, kind='image')

    def removeItem(self, legend):
        """Remove the item associated to legend from the graph.

        :param str legend: The legend associated to the item to be deleted
        """
        if legend is None:
            return
        self.remove(legend, kind='item')

    def removeMarker(self, legend):
        """Remove the marker associated to legend from the graph.

        :param str legend: The legend associated to the marker to be deleted
        """
        if legend is None:
            return
        self.remove(legend, kind='marker')

    # Clear

    def clear(self):
        """Remove everything from the plot."""
        self.remove()

    def clearCurves(self):
        """Remove all the curves from the plot."""
        self.remove(kind='curve')

    def clearImages(self):
        """Remove all the images from the plot."""
        self.remove(kind='image')

    def clearItems(self):
        """Remove all the items from the plot. """
        self.remove(kind='item')

    def clearMarkers(self):
        """Remove all the markers from the plot."""
        self.remove(kind='marker')

    # Interaction

    def getGraphCursor(self):
        """Returns the state of the crosshair cursor.

        See :meth:`setGraphCursor`.

        :return: None if the crosshair cursor is not active,
                 else a tuple (color, linewidth, linestyle).
        """
        return self._cursorConfiguration

    def setGraphCursor(self, flag=False, color='black',
                       linewidth=1, linestyle='-'):
        """Toggle the display of a crosshair cursor and set its attributes.

        :param bool flag: Toggle the display of a crosshair cursor.
                          The crosshair cursor is hidden by default.
        :param color: The color to use for the crosshair.
        :type color: A string (either a predefined color name in Colors.py
                    or "#RRGGBB")) or a 4 columns unsigned byte array
                    (Default: black).
        :param int linewidth: The width of the lines of the crosshair
                    (Default: 1).
        :param str linestyle: Type of line::

                - ' ' no line
                - '-' solid line (the default)
                - '--' dashed line
                - '-.' dash-dot line
                - ':' dotted line
        """
        if flag:
            self._cursorConfiguration = color, linewidth, linestyle
        else:
            self._cursorConfiguration = None

        self._backend.setGraphCursor(flag=flag, color=color,
                                     linewidth=linewidth, linestyle=linestyle)
        self._setDirtyPlot()
        self.notify('setGraphCursor',
                    state=self._cursorConfiguration is not None)

    def pan(self, direction, factor=0.1):
        """Pan the graph in the given direction by the given factor.

        Warning: Pan of right Y axis not implemented!

        :param str direction: One of 'up', 'down', 'left', 'right'.
        :param float factor: Proportion of the range used to pan the graph.
                             Must be strictly positive.
        """
        assert direction in ('up', 'down', 'left', 'right')
        assert factor > 0.

        if direction in ('left', 'right'):
            xFactor = factor if direction == 'right' else - factor
            xMin, xMax = self.getGraphXLimits()

            xMin, xMax = _utils.applyPan(xMin, xMax, xFactor,
                                         self.isXAxisLogarithmic())
            self.setGraphXLimits(xMin, xMax)

        else:  # direction in ('up', 'down')
            sign = -1. if self.isYAxisInverted() else 1.
            yFactor = sign * (factor if direction == 'up' else -factor)
            yMin, yMax = self.getGraphYLimits()
            yIsLog = self.isYAxisLogarithmic()

            yMin, yMax = _utils.applyPan(yMin, yMax, yFactor, yIsLog)
            self.setGraphYLimits(yMin, yMax, axis='left')

            y2Min, y2Max = self.getGraphYLimits(axis='right')

            y2Min, y2Max = _utils.applyPan(y2Min, y2Max, yFactor, yIsLog)
            self.setGraphYLimits(y2Min, y2Max, axis='right')

    # Active Curve/Image

    def isActiveCurveHandling(self):
        """Returns True if active curve selection is enabled."""
        return self._activeCurveHandling

    def setActiveCurveHandling(self, flag=True):
        """Enable/Disable active curve selection.

        :param bool flag: True (the default) to enable active curve selection.
        """
        if not flag:
            self.setActiveCurve(None)  # Reset active curve

        self._activeCurveHandling = bool(flag)

    def getActiveCurveColor(self):
        """Get the color used to display the currently active curve.

        See :meth:`setActiveCurveColor`.
        """
        return self._activeCurveColor

    def setActiveCurveColor(self, color="#000000"):
        """Set the color to use to display the currently active curve.

        :param str color: Color of the active curve,
                          e.g., 'blue', 'b', '#FF0000' (Default: 'black')
        """
        if color is None:
            color = "black"
        if color in self.colorDict:
            color = self.colorDict[color]
        self._activeCurveColor = color

    def getActiveCurve(self, just_legend=False):
        """Return the currently active curve.

        It returns None in case of not having an active curve.
        Default output has the form: [x, y, legend, info, params]
        where params is a dictionary containing curve parameters.

        Warning: Returned values MUST not be modified.
        Make a copy if you need to modify them.

        :param bool just_legend: True to get the legend of the curve,
                                 False (the default) to get the curve data
                                 and info.
        :return: legend of the active curve or [x, y, legend, info, params]
        :rtype: str or list
        """
        if not self.isActiveCurveHandling():
            return None

        if self._activeCurve not in self._curves:
            self._activeCurve = None

        if self._activeCurve is None:
            return None

        if just_legend:
            return self._activeCurve
        else:
            curve = self._curves[self._activeCurve]
            return (curve['x'], curve['y'], self._activeCurve,
                    curve['params']['info'] or {}, curve['params'])

    def setActiveCurve(self, legend, replot=None):
        """Make the curve associated to legend the active curve.

        :param str legend: The legend associated to the curve
                           or None to have no active curve.
        """
        if replot is not None:
            _logger.warning('setActiveCurve deprecated replot parameter')

        if not self.isActiveCurveHandling():
            return

        xLabel = self._defaultLabels['x']
        yLabel = self._defaultLabels['y']
        yRightLabel = self._defaultLabels['yright']

        oldActiveCurveLegend = self.getActiveCurve(just_legend=True)
        if oldActiveCurveLegend:  # Reset previous active curve
            handle = self._curves[oldActiveCurveLegend]['handle']
            if handle is not None:
                self._backend.setActiveCurve(handle, False)

        if legend is None:
            self._activeCurve = None
        else:
            legend = str(legend)
            if legend not in self._curves:
                _logger.warning("This curve does not exist: %s", legend)
                self._activeCurve = None
            else:
                self._activeCurve = legend

                handle = self._curves[self._activeCurve]['handle']
                if handle is not None:
                    self._backend.setActiveCurve(handle, True,
                                                 self.getActiveCurveColor())

                curveParams = self._curves[self._activeCurve]['params']

                if curveParams['xlabel'] is not None:
                    xLabel = curveParams['xlabel']
                if curveParams['ylabel'] is not None:
                    if curveParams['yaxis'] == 'left':
                        yLabel = curveParams['ylabel']
                    else:
                        yRightLabel = curveParams['ylabel']

        # Store current labels and update plot
        self._currentLabels['x'] = xLabel
        self._currentLabels['y'] = yLabel
        self._currentLabels['yright'] = yRightLabel

        self._backend.setGraphXLabel(xLabel)
        self._backend.setGraphYLabel(yLabel, axis='left')
        self._backend.setGraphYLabel(yRightLabel, axis='right')

        self._setDirtyPlot()

        if oldActiveCurveLegend is not None or self._activeCurve is not None:
            self.notify('activeCurveChanged',
                        updated=oldActiveCurveLegend != self._activeCurve,
                        previous=oldActiveCurveLegend,
                        legend=self._activeCurve)

        return self._activeCurve

    def getActiveImage(self, just_legend=False):
        """Returns the currently active image.

        It returns None in case of not having an active image.

        Default output has the form: [data, legend, info, pixmap, params]
        where params is a dictionnary containing image parameters.

        Warning: Returned values MUST not be modified.
        Make a copy if you need to modify them.

        :param bool just_legend: True to get the legend of the image,
                                 False (the default) to get the image data
                                 and info.
        :return: legend of active image or [data, legend, info, pixmap, params]
        :rtype: str or list
        """
        if self._activeImage not in self._images:
            self._activeImage = None

        if just_legend:
            return self._activeImage

        if self._activeImage is None:
            return None
        else:
            image = self._images[self._activeImage]
            return (image['data'], self._activeImage,
                    image['params']['info'] or {}, image['pixmap'],
                    image['params'])

    def setActiveImage(self, legend, replot=None):
        """Make the image associated to legend the active image.

        :param str legend: The legend associated to the image
                           or None to have no active image.
        """
        if replot is not None:
            _logger.warning('setActiveImage deprecated replot parameter')

        xLabel = self._defaultLabels['x']
        yLabel = self._defaultLabels['y']

        oldActiveImageLegend = self.getActiveImage(just_legend=True)

        if legend is None:
            self._activeImage = None
        else:
            legend = str(legend)
            if legend not in self._images:
                _logger.warning(
                    "setActiveImage: This image does not exist: %s", legend)
                self._activeImage = None
            else:
                self._activeImage = legend

                imageParams = self._images[self._activeImage]['params']

                if imageParams['xlabel'] is not None:
                    xLabel = imageParams['xlabel']
                if imageParams['ylabel'] is not None:
                    yLabel = imageParams['ylabel']

        # Store current labels and update plot
        self._currentLabels['x'] = xLabel
        self._currentLabels['y'] = yLabel

        self._backend.setGraphXLabel(xLabel)
        self._backend.setGraphYLabel(yLabel, axis='left')

        if oldActiveImageLegend is not None or self._activeImage is not None:
            self.notify('activeImageChanged',
                        updated=oldActiveImageLegend != self._activeImage,
                        previous=oldActiveImageLegend,
                        legend=self._activeImage)

        return self._activeImage

    # Getters

    def getAllCurves(self, just_legend=False, withhidden=False):
        """Returns all curves legend or info and data.

        It returns an empty list in case of not having any curve.

        If just_legend is False, it returns a list of the form:
            [[xvalues0, yvalues0, legend0, info0, params0],
             [xvalues1, yvalues1, legend1, info1, params1],
             [...],
             [xvaluesn, yvaluesn, legendn, infon, paramsn]]
        If just_legend is True, it returns a list of the form:
            [legend0, legend1, ..., legendn]

        Warning: Returned values MUST not be modified.
        Make a copy if you need to modify them.

        :param bool just_legend: True to get the legend of the curves,
                                 False (the default) to get the curves' data
                                 and info.
        :param bool withhidden: False (default) to skip hidden curves.
        :return: list of legends or list of [x, y, legend, info, params]
        :rtype: list of str or list of list
        """
        output = []
        for key in self._curves:
            if not withhidden and self.isCurveHidden(key):
                continue
            if just_legend:
                output.append(key)
            else:
                curve = self._curves[key]
                output.append((curve['x'], curve['y'], key,
                               curve['params']['info'] or {}, curve['params']))
        return output

    def getCurve(self, legend=None):
        """Get the data and info of a specific curve.

        It returns None in case no matching curve is found.

        Warning: Returned values MUST not be modified.
        Make a copy if you need to modify them.

        :param str legend:
            The legend identifying the curve.
            If not provided or None (the default), the active curve is returned
            or if there is no active curve, the lastest updated curve that is
            not hidden.
            is returned if there are curves in the plot.
        :return: None or list [x, y, legend, parameters]
        """
        if legend is None:
            legend = self.getActiveCurve(just_legend=True)
            if legend is None and self._curves:
                # There is no active curve, but there is some curves:
                # get one that is not hidden
                for curveLegend in reversed(list(self._curves.keys())):
                    if curveLegend not in self._hiddenCurves:
                        legend = curveLegend
                        break

        if legend is not None and legend in self._curves:
            curve = self._curves[legend]
            return (curve['x'], curve['y'], legend,
                    curve['params']['info'] or {}, curve['params'])
        else:
            return None

    def getImage(self, legend=None):
        """Get the data and info of a specific image.

        It returns None in case no matching image is found.

        Warning: Returned values MUST not be modified.
        Make a copy if you need to modify them.

        :param str legend:
            The legend identifying the image.
            If not provided or None (the default), the active image is returned
            or if there is no active image, the lastest updated image
            is returned if there are images in the plot.
        :return: None or list [image, legend, info, pixmap, params]
        """
        if legend is None:
            legend = self.getActiveImage(just_legend=True)
            if legend is None and self._images:
                # There is no active image, but there is some images: get one
                legend = list(self._images.keys())[-1]

        if legend is not None and legend in self._images:
            image = self._images[legend]
            return (image['data'], legend, image['params']['info'] or {},
                    image['pixmap'], image['params'])
        else:
            return None

    # Limits

    def _notifyLimitsChanged(self):
        """Send an event when plot area limits are changed."""
        xRange = self.getGraphXLimits()
        yRange = self.getGraphYLimits(axis='left')
        y2Range = self.getGraphYLimits(axis='right')
        event = PlotEvents.prepareLimitsChangedSignal(
            id(self.getWidgetHandle()), xRange, yRange, y2Range)
        self.notify(**event)

    def getGraphXLimits(self):
        """Get the graph X (bottom) limits.

        :return: Minimum and maximum values of the X axis
        """
        return self._backend.getGraphXLimits()

    def setGraphXLimits(self, xmin, xmax, replot=None):
        """Set the graph X (bottom) limits.

        :param float xmin: minimum bottom axis value
        :param float xmax: maximum bottom axis value
        """
        if replot is not None:
            _logger.warning('setGraphXLimits deprecated replot parameter')

        # Deal with incorrect values
        if xmax < xmin:
            _logger.warning('setGraphXLimits xmax < xmin, inverting limits.')
            xmin, xmax = xmax, xmin
        elif xmax == xmin:
            _logger.warning('setGraphXLimits xmax == xmin, expanding limits.')
            if xmin == 0.:
                xmin, xmax = -0.1, 0.1
            else:
                xmin, xmax = xmin * 1.1, xmax * 0.9

        self._backend.setGraphXLimits(xmin, xmax)
        self._setDirtyPlot()

        self._notifyLimitsChanged()

    def getGraphYLimits(self, axis='left'):
        """Get the graph Y limits.

        :param str axis: The axis for which to get the limits:
                         Either 'left' or 'right'
        :return: Minimum and maximum values of the X axis
        """
        assert axis in ('left', 'right')
        return self._backend.getGraphYLimits(axis)

    def setGraphYLimits(self, ymin, ymax, axis='left', replot=None):
        """Set the graph Y limits.

        :param float xmin: minimum bottom axis value
        :param float xmax: maximum bottom axis value
        :param str axis: The axis for which to get the limits:
                         Either 'left' or 'right'
        """
        if replot is not None:
            _logger.warning('setGraphYLimits deprecated replot parameter')

        # Deal with incorrect values
        if ymax < ymin:
            _logger.warning('setGraphYLimits ymax < ymin, inverting limits.')
            ymin, ymax = ymax, ymin
        elif ymax == ymin:
            _logger.warning('setGraphXLimits ymax == ymin, expanding limits.')
            if ymin == 0.:
                ymin, ymax = -0.1, 0.1
            else:
                ymin, ymax = ymin * 1.1, ymax * 0.9

        assert axis in ('left', 'right')
        self._backend.setGraphYLimits(ymin, ymax, axis)
        self._setDirtyPlot()

        self._notifyLimitsChanged()

    def setLimits(self, xmin, xmax, ymin, ymax, y2min=None, y2max=None):
        """Set the limits of the X and Y axes at once.

        If y2min or y2max is None, the right Y axis limits are not updated.

        :param float xmin: minimum bottom axis value
        :param float xmax: maximum bottom axis value
        :param float ymin: minimum left axis value
        :param float ymax: maximum left axis value
        :param float y2min: minimum right axis value or None (the default)
        :param float y2max: maximum right axis value or None (the default)
        """
        # Deal with incorrect values
        if xmax < xmin:
            _logger.warning('setLimits xmax < xmin, inverting limits.')
            xmin, xmax = xmax, xmin
        elif xmax == xmin:
            _logger.warning('setLimits xmax == xmin, expanding limits.')
            if xmin == 0.:
                xmin, xmax = -0.1, 0.1
            else:
                xmin, xmax = xmin * 1.1, xmax * 0.9

        if ymax < ymin:
            _logger.warning('setLimits ymax < ymin, inverting limits.')
            ymin, ymax = ymax, ymin
        elif ymax == ymin:
            _logger.warning('setLimits ymax == ymin, expanding limits.')
            if ymin == 0.:
                ymin, ymax = -0.1, 0.1
            else:
                ymin, ymax = ymin * 1.1, ymax * 0.9

        if y2min is None or y2max is None:
            # if one limit is None, both are ignored
            y2min, y2max = None, None
        else:
            if y2max < y2min:
                _logger.warning('setLimits y2max < y2min, inverting limits.')
                y2min, y2max = y2max, y2min
            elif y2max == y2min:
                _logger.warning('setLimits y2max == y2min, expanding limits.')
                if y2min == 0.:
                    y2min, y2max = -0.1, 0.1
                else:
                    y2min, y2max = y2min * 1.1, y2max * 0.9

        self._backend.setLimits(xmin, xmax, ymin, ymax, y2min, y2max)
        self._setDirtyPlot()
        self._notifyLimitsChanged()

    # Title and labels

    def getGraphTitle(self):
        """Return the plot main title as a str."""
        return self._graphTitle

    def setGraphTitle(self, title=""):
        """Set the plot main title.

        :param str title: Main title of the plot (default: '')
        """
        self._graphTitle = str(title)
        self._backend.setGraphTitle(title)
        self._setDirtyPlot()

    def getGraphXLabel(self):
        """Return the current X axis label as a str."""
        return self._defaultLabels['x']

    def setGraphXLabel(self, label="X"):
        """Set the plot X axis label.

        The provided label can be temporarily replaced by the X label of the
        active curve if any.

        :param str label: The X axis label (default: 'X')
        """
        self._defaultLabels['x'] = label
        self._currentLabels['x'] = label
        self._backend.setGraphXLabel(label)
        self._setDirtyPlot()

    def getGraphYLabel(self, axis='left'):
        """Return the current Y axis label as a str.

        :param str axis: The Y axis for which to get the label (left or right)
        """
        assert axis in ('left', 'right')

        return self._currentLabels['y' if axis == 'left' else 'yright']

    def setGraphYLabel(self, label="Y", axis='left'):
        """Set the plot Y axis label.

        The provided label can be temporarily replaced by the Y label of the
        active curve if any.

        :param str label: The Y axis label (default: 'Y')
        :param str axis: The Y axis for which to set the label (left or right)
        """
        assert axis in ('left', 'right')

        if axis == 'left':
            self._defaultLabels['y'] = label
            self._currentLabels['y'] = label
        else:
            self._defaultLabels['yright'] = label
            self._currentLabels['yright'] = label

        self._backend.setGraphYLabel(label, axis=axis)
        self._setDirtyPlot()

    # Axes

    def setYAxisInverted(self, flag=True):
        """Set the Y axis orientation.

        :param bool flag: True for Y axis going from top to bottom,
                          False for Y axis going from bottom to top
        """
        flag = bool(flag)
        self._backend.setYAxisInverted(flag)
        self._setDirtyPlot()
        self.notify('setYAxisInverted', state=flag)

    def isYAxisInverted(self):
        """Return True if Y axis goes from top to bottom, False otherwise."""
        return self._backend.isYAxisInverted()

    def isXAxisLogarithmic(self):
        """Return True if X axis scale is logarithmic, False if linear."""
        return self._logX

    def setXAxisLogarithmic(self, flag):
        """Set the bottom X axis scale (either linear or logarithmic).

        :param bool flag: True to use a logarithmic scale, False for linear.
        """
        if bool(flag) == self._logX:
            return
        self._logX = bool(flag)

        if self._logX:  # Switch to log scale
            for image in self._images.values():
                if image['handle'] is not None:
                    self._backend.remove(image['handle'])
                    image['handle'] = None

            for curve in self._curves.values():
                handle = curve['handle']
                if handle is not None:
                    self._backend.remove(handle)
                    curve['handle'] = None

            # matplotlib 1.5 crashes if the log set is made before
            # the call to self._update()
            # TODO: Decide what is better for other backends
            if (hasattr(self._backend, "matplotlibVersion") and
                    self._backend.matplotlibVersion >= "1.5"):
                self._update()
                self._backend.setXAxisLogarithmic(self._logX)
            else:
                self._backend.setXAxisLogarithmic(self._logX)
                self._update()
        else:
                self._backend.setXAxisLogarithmic(self._logX)
                self._update()

        self._invalidateDataRange()
        self._setDirtyPlot()
        self.resetZoom()
        self.notify('setXAxisLogarithmic', state=self._logX)

    def isYAxisLogarithmic(self):
        """Return True if Y axis scale is logarithmic, False if linear."""
        return self._logY

    def setYAxisLogarithmic(self, flag):
        """Set the Y axes scale (either linear or logarithmic).

        :param bool flag: True to use a logarithmic scale, False for linear.
        """
        if bool(flag) == self._logY:
            return
        self._logY = bool(flag)

        if self._logY:  # Switch to log scale
            for image in self._images.values():
                if image['handle'] is not None:
                    self._backend.remove(image['handle'])
                    image['handle'] = None

            for curve in self._curves.values():
                handle = curve['handle']
                if handle is not None:
                    self._backend.remove(handle)
                    curve['handle'] = None

            # matplotlib 1.5 crashes if the log set is made before
            # the call to self._update()
            # TODO: Decide what is better for other backends
            if (hasattr(self._backend, "matplotlibVersion") and
                    self._backend.matplotlibVersion >= "1.5"):
                self._update()
                self._backend.setYAxisLogarithmic(self._logY)
            else:
                self._backend.setYAxisLogarithmic(self._logY)
                self._update()
        else:
                self._backend.setYAxisLogarithmic(self._logY)
                self._update()

        self._invalidateDataRange()
        self._setDirtyPlot()
        self.resetZoom()
        self.notify('setYAxisLogarithmic', state=self._logY)

    def isXAxisAutoScale(self):
        """Return True if X axis is automatically adjusting its limits."""
        return self._xAutoScale

    def setXAxisAutoScale(self, flag=True):
        """Set the X axis limits adjusting behavior of :meth:`resetZoom`.

        :param bool flag: True to resize limits automatically,
                          False to disable it.
        """
        self._xAutoScale = bool(flag)
        self.notify('setXAxisAutoScale', state=self._xAutoScale)

    def isYAxisAutoScale(self):
        """Return True if Y axes are automatically adjusting its limits."""
        return self._yAutoScale

    def setYAxisAutoScale(self, flag=True):
        """Set the Y axis limits adjusting behavior of :meth:`resetZoom`.

        :param bool flag: True to resize limits automatically,
                          False to disable it.
        """
        self._yAutoScale = bool(flag)
        self.notify('setYAxisAutoScale', state=self._yAutoScale)

    def isKeepDataAspectRatio(self):
        """Returns whether the plot is keeping data aspect ratio or not."""
        return self._backend.isKeepDataAspectRatio()

    def setKeepDataAspectRatio(self, flag=True):
        """Set whether the plot keeps data aspect ratio or not.

        :param bool flag: True to respect data aspect ratio
        """
        flag = bool(flag)
        self._backend.setKeepDataAspectRatio(flag=flag)
        self._setDirtyPlot()
        self.resetZoom()
        self.notify('setKeepDataAspectRatio', state=flag)

    def getGraphGrid(self):
        """Return the current grid mode, either None, 'major' or 'both'.

        See :meth:`setGraphGrid`.
        """
        return self._grid

    def setGraphGrid(self, which=True):
        """Set the type of grid to display.

        :param which: None or False to disable the grid,
                      'major' or True for grid on major ticks (the default),
                      'both' for grid on both major and minor ticks.
        :type which: str of bool
        """
        assert which in (None, True, False, 'both', 'major')
        if not which:
            which = None
        elif which is True:
            which = 'major'
        self._grid = which
        self._backend.setGraphGrid(which)
        self._setDirtyPlot()
        self.notify('setGraphGrid', which=str(which))

    # Defaults

    def isDefaultPlotPoints(self):
        """Return True if default Curve symbol is 'o', False for no symbol."""
        return self._defaultPlotPoints == 'o'

    def setDefaultPlotPoints(self, flag):
        """Set the default symbol of all curves.

        When called, this reset the symbol of all existing curves.

        :param bool flag: True to use 'o' as the default curve symbol,
                          False to use no symbol.
        """
        self._defaultPlotPoints = 'o' if flag else ''

        # Reset symbol of all curves
        for curve in self._curves.values():
            curve['params']['symbol'] = self._defaultPlotPoints

        if self._curves:
            self._update()
            self._setDirtyPlot()

    def isDefaultPlotLines(self):
        """Return True for line as default line style, False for no line."""
        return self._plotLines

    def setDefaultPlotLines(self, flag):
        """Toggle the use of lines as the default curve line style.

        :param bool flag: True to use a line as the default line style,
                          False to use no line as the default line style.
        """
        self._plotLines = bool(flag)

        # Reset linestyle of all curves
        for curve in self._curves.values():
            curve['params']['linestyle'] = '-' if self._plotLines else ' '

        if self._curves:
            self._update()
            self._setDirtyPlot()

    def getDefaultColormap(self):
        """Return the default colormap used by :meth:`addImage` as a dict.

        See :mod:`Plot` for the documentation of the colormap dict.
        """
        return self._defaultColormap.copy()

    def setDefaultColormap(self, colormap=None):
        """Set the default colormap used by :meth:`addImage`.

        Setting the default colormap do not change any currently displayed
        image.
        It only affects future calls to :meth:`addImage` without the colormap
        parameter.

        :param dict colormap: The description of the default colormap, or
                            None to set the colormap to a linear autoscale
                            gray colormap.
                            See :mod:`Plot` for the documentation
                            of the colormap dict.
        """
        if colormap is None:
            colormap = {'name': 'gray', 'normalization': 'linear',
                        'autoscale': True, 'vmin': 0.0, 'vmax': 1.0}
        self._defaultColormap = colormap.copy()

    def getSupportedColormaps(self):
        """Get the supported colormap names as a tuple of str.

        The list should at least contain and start by:
        ('gray', 'reversed gray', 'temperature', 'red', 'green', 'blue')
        """
        return self._backend.getSupportedColormaps()

    def _getColorAndStyle(self):
        color = self.colorList[self._colorIndex]
        style = self._styleList[self._styleIndex]

        # Loop over color and then styles
        self._colorIndex += 1
        if self._colorIndex >= len(self.colorList):
            self._colorIndex = 0
            self._styleIndex = (self._styleIndex + 1) % len(self._styleList)

        # If color is the one of active curve, take the next one
        if color == self.getActiveCurveColor():
            color, style = self._getColorAndStyle()

        if not self._plotLines:
            style = ' '

        return color, style

    # Misc.

    def getWidgetHandle(self):
        """Return the widget the plot is displayed in.

        This widget is owned by the backend.
        """
        return self._backend.getWidgetHandle()

    def notify(self, event, **kwargs):
        """Send an event to the listeners.

        Event are passed to the registered callback as a dict with an 'event'
        key for backward compatibility with PyMca.

        :param str event: The type of event
        :param kwargs: The information of the event.
        """
        eventDict = kwargs.copy()
        eventDict['event'] = event
        self._callback(eventDict)

    def setCallback(self, callbackFunction=None):
        """Attach a listener to the backend.

        Limitation: Only one listener at a time.

        :param callbackFunction: function accepting a dictionnary as input
                                 to handle the graph events
                                 If None (default), use a default listener.
        """
        # TODO allow multiple listeners, keep a weakref on it
        # allow register listener by event type
        if callbackFunction is None:
            callbackFunction = self.graphCallback
        self._callback = callbackFunction

    def graphCallback(self, ddict=None):
        """This callback is going to receive all the events from the plot.

        Those events will consist on a dictionnary and among the dictionnary
        keys the key 'event' is mandatory to describe the type of event.
        This default implementation only handles setting the active curve.
        """

        if ddict is None:
            ddict = {}
        _logger.debug("Received dict keys = %s", str(ddict.keys()))
        _logger.debug(str(ddict))
        if ddict['event'] in ["legendClicked", "curveClicked"]:
            if ddict['button'] == "left":
                self.setActiveCurve(ddict['label'])

    def saveGraph(self, filename, fileFormat=None, dpi=None, **kw):
        """Save a snapshot of the plot.

        Supported file formats: "png", "svg", "pdf", "ps", "eps",
        "tif", "tiff", "jpeg", "jpg".

        :param filename: Destination
        :type filename: str, StringIO or BytesIO
        :param str fileFormat:  String specifying the format
        :return: False if cannot save the plot, True otherwise
        """
        if kw:
            _logger.warning('Extra parameters ignored: %s', str(kw))

        if fileFormat is None:
            if not hasattr(filename, 'lower'):
                _logger.warning(
                    'saveGraph cancelled, cannot define file format.')
                return False
            else:
                fileFormat = (filename.split(".")[-1]).lower()

        supportedFormats = ("png", "svg", "pdf", "ps", "eps",
                            "tif", "tiff", "jpeg", "jpg")

        if fileFormat not in supportedFormats:
            _logger.warning('Unsupported format %s', fileFormat)
            return False
        else:
            self._backend.saveGraph(filename,
                                    fileFormat=fileFormat,
                                    dpi=dpi)
            return True

    def getDataMargins(self):
        """Get the default data margin ratios, see :meth:`setDataMargins`.

        :return: The margin ratios for each side (xMin, xMax, yMin, yMax).
        :rtype: A 4-tuple of floats.
        """
        return self._defaultDataMargins

    def setDataMargins(self, xMinMargin=0., xMaxMargin=0.,
                       yMinMargin=0., yMaxMargin=0.):
        """Set the default data margins to use in :meth:`resetZoom`.

        Set the default ratios of margins (as floats) to add around the data
        inside the plot area for each side.
        """
        self._defaultDataMargins = (xMinMargin, xMaxMargin,
                                    yMinMargin, yMaxMargin)

    def getAutoReplot(self):
        """Return True if replot is automatically handled, False otherwise.

        See :meth`setAutoReplot`.
        """
        return self._autoreplot

    def setAutoReplot(self, autoreplot=True):
        """Set automatic replot mode.

        When enabled, the plot is redrawn automatically when changed.
        When disabled, the plot is not redrawn when its content change.
        Instead, it :meth:`replot` must be called.

        :param bool autoreplot: True to enable it (default),
                                False to disable it.
        """
        self._autoreplot = bool(autoreplot)

        # If the plot is dirty before enabling autoreplot,
        # then _backend.postRedisplay will never be called from _setDirtyPlot
        if self._autoreplot and self._getDirtyPlot():
            self._backend.postRedisplay()

    def replot(self):
        """Redraw the plot immediately."""
        self._backend.replot()
        self._dirty = False  # reset dirty flag

    def resetZoom(self, dataMargins=None):
        """Reset the plot limits to the bounds of the data and redraw the plot.

        It automatically scale limits of axes that are in autoscale mode
        (See :meth:`setXAxisAutoScale`, :meth:`setYAxisAutoScale`).
        It keeps current limits on axes that are not in autoscale mode.

        Extra margins can be added around the data inside the plot area.
        Margins are given as one ratio of the data range per limit of the
        data (xMin, xMax, yMin and yMax limits).
        For log scale, extra margins are applied in log10 of the data.

        :param dataMargins: Ratios of margins to add around the data inside
                            the plot area for each side (Default: no margins).
        :type dataMargins: A 4-tuple of float as (xMin, xMax, yMin, yMax).
        """
        if dataMargins is None:
            dataMargins = self._defaultDataMargins

        xlim = self.getGraphXLimits()
        ylim = self.getGraphYLimits(axis='left')
        y2lim = self.getGraphYLimits(axis='right')

        self._backend.resetZoom(dataMargins)
        self._setDirtyPlot()

        if (xlim != self.getGraphXLimits() or
                ylim != self.getGraphYLimits(axis='left') or
                y2lim != self.getGraphYLimits(axis='right')):
            self._notifyLimitsChanged()

    # Internal

    @staticmethod
    def _logFilterData(x, y, color, xerror, yerror, xLog, yLog):
        """Filter out values with x or y <= 0 on log axes

        All arrays are expected to have the same length.

        :param x: The x coords.
        :param y: The y coords.
        :param color: The addCurve color arg (might not be an array).
        :param xerror: The addCuve xerror arg (might not be an array).
        :param yerror: The addCuve yerror arg (might not be an array).
        :param bool xLog: True to filter arrays according to X coords.
        :param bool yLog: True to filter arrays according to Y coords.
        :return: The filter arrays or unchanged object if
        :rtype: (x, y, color, xerror, yerror)
        """
        if xLog and yLog:
            idx = numpy.nonzero((x > 0) & (y > 0))[0]
        elif yLog:
            idx = numpy.nonzero(y > 0)[0]
        elif xLog:
            idx = numpy.nonzero(x > 0)[0]
        else:
            return x, y, color, xerror, yerror

        x = numpy.take(x, idx)
        y = numpy.take(y, idx)

        if isinstance(color, numpy.ndarray) and len(color) == len(x):
            # Nx(3 or 4) array (do not change RGBA color defined as an array)
            color = numpy.take(color, idx, axis=0)

        if isinstance(xerror, numpy.ndarray):
            if len(xerror) == len(x):
                # N or Nx1 array
                xerror = numpy.take(xerror, idx, axis=0)
            elif len(xerror) == 2 and len(xerror.shape) == 2:
                # 2xN array (+/- error)
                xerror = xerror[:, idx]

        if isinstance(yerror, numpy.ndarray):
            if len(yerror) == len(y):
                # N or Nx1 array
                yerror = numpy.take(yerror, idx, axis=0)
            elif len(yerror) == 2 and len(yerror.shape) == 2:
                # 2xN array (+/- error)
                yerror = yerror[:, idx]

        return x, y, color, xerror, yerror

    def _update(self):
        _logger.debug("_update called")

        # curves
        activeCurve = self.getActiveCurve(just_legend=True)
        curves = list(self._curves)
        for legend in curves:
            curve = self._curves[legend]
            self.addCurve(curve['x'], curve['y'], legend, resetzoom=False,
                          **curve['params'])

        if len(curves):
            if activeCurve not in curves:
                activeCurve = curves[0]
        else:
            activeCurve = None
        self.setActiveCurve(activeCurve)

        # images
        if not self.isXAxisLogarithmic() and not self.isYAxisLogarithmic():
            for legend in list(self._images):  # Copy has images is changed
                image = self._images[legend]
                self.addImage(image['data'], legend,
                              replace=False, resetzoom=False,
                              pixmap=image['pixmap'], **image['params'])

    # Coord conversion

    def dataToPixel(self, x=None, y=None, axis="left", check=True):
        """Convert a position in data coordinates to a position in pixels.

        :param float x: The X coordinate in data space. If None (default)
                        the middle position of the displayed data is used.
        :param float y: The Y coordinate in data space. If None (default)
                        the middle position of the displayed data is used.
        :param str axis: The Y axis to use for the conversion
                         ('left' or 'right').
        :param bool check: True to return None if outside displayed area,
                           False to convert to pixels anyway
        :returns: The corresponding position in pixels or
                  None if the data position is not in the displayed area and
                  check is True.
        :rtype: A tuple of 2 floats: (xPixel, yPixel) or None.
        """
        assert axis in ("left", "right")

        xmin, xmax = self.getGraphXLimits()
        ymin, ymax = self.getGraphYLimits(axis=axis)

        if x is None:
            x = 0.5 * (xmax + xmin)
        if y is None:
            y = 0.5 * (ymax + ymin)

        if check:
            if x > xmax or x < xmin:
                return None

            if y > ymax or y < ymin:
                return None

        return self._backend.dataToPixel(x, y, axis=axis)

    def pixelToData(self, x, y, axis="left", check=False):
        """Convert a position in pixels to a position in data coordinates.

        :param float x: The X coordinate in pixels. If None (default)
                            the center of the widget is used.
        :param float y: The Y coordinate in pixels. If None (default)
                            the center of the widget is used.
        :param str axis: The Y axis to use for the conversion
                         ('left' or 'right').
        :param bool check: Toggle checking if pixel is in plot area.
                           If False, this method never returns None.
        :returns: The corresponding position in data space or
                  None if the pixel position is not in the plot area.
        :rtype: A tuple of 2 floats: (xData, yData) or None.
        """
        assert axis in ("left", "right")
        return self._backend.pixelToData(x, y, axis=axis, check=check)

    def getPlotBoundsInPixels(self):
        """Plot area bounds in widget coordinates in pixels.

        :return: bounds as a 4-tuple of int: (left, top, width, height)
        """
        return self._backend.getPlotBoundsInPixels()

    # Interaction support

    def setGraphCursorShape(self, cursor=None):
        """Set the cursor shape.

        :param str cursor: Name of the cursor shape
        """
        self._backend.setGraphCursorShape(cursor)

    def _pickMarker(self, x, y, test=None):
        """Pick a marker at the given position.

        To use for interaction implementation.

        :param float x: X position in pixels.
        :param float y: Y position in pixels.
        :param test: A callable to call for each picked marker to filter
                     picked markers. If None (default), do not filter markers.
        """
        if test is None:
            test = lambda marker: True

        markers = self._backend.pickItems(x, y)
        markers = [item for item in markers if item['kind'] == 'marker']

        for item in reversed(markers):
            legend = item['legend']
            params = self._getMarker(legend)
            if params is not None and test(params):
                params['legend'] = legend
                return params
        return None

    def _moveMarker(self, legend, x, y):
        """Move a marker to a position.

        To use for interaction implementation.

        :param str legend: The legend associated to the marker.
        :param float x: The new X position of the marker in data coordinates.
        :param float y: The new Y position of the marker in data coordinates.
        """
        params = self._getMarker(legend)
        if params is not None:
            if params['x'] is not None:
                params['x'] = x
            if params['y'] is not None:
                params['y'] = y
            self._addMarker(**params)

    def _getMarker(self, legend):
        """Get the parameters of a marker

        :param str legend: The legend of the marker to retrieve
        :return: A copy of the parameters the marker has been created with
        :rtype: dict or None if marker does not exist
        """
        marker = self._markers.get(legend, None)
        if marker is None:
            return None
        else:
            # Return a shallow copy
            params = marker['params'].copy()
            params['legend'] = legend
            return params

    def _pickImageOrCurve(self, x, y, test=None):
        """Pick an image or a curve at the given position.

        To use for interaction implementation.

        :param float x: X position in pixels.
        :param float y: Y position in pixels.
        :param test: A callable to call for each picked item to filter
                     picked items. If None (default), do not filter items.
        """
        if test is None:
            test = lambda item: True

        items = self._backend.pickItems(x, y)
        items = [item for item in items if item['kind'] in ['curve', 'image']]

        for item in reversed(items):
            kind, legend = item['kind'], item['legend']
            if kind == 'curve':
                curve = self._curves.get(legend, None)
                if curve is not None:
                    params = curve['params'].copy()  # shallow copy
                    if test(params):
                        params['legend'] = legend
                        return kind, params, item['xdata'], item['ydata']

            elif kind == 'image':
                image = self._images.get(legend, None)
                if image is not None:
                    params = image['params'].copy()  # shallow copy
                    if test(params):
                        params['legend'] = legend
                        return kind, params, None

            else:
                _logger.warning('Unsupported kind: %s', kind)

        return None

    def _moveImage(self, legend, dx, dy):
        """Move an image to a position.

        To use for interaction implementation.

        :param str legend: The legend associated to the image.
        :param float dx: The X offset to apply to the image in data coords.
        :param float dy: The Y offset to apply to the image in data coords.
        """
        # TODO: poor implementation, better to do move image in backend...
        image = self._images[legend]
        params = image['params'].copy()
        params['origin'] = params['origin'][0] + dx, params['origin'][1] + dy
        self.addImage(image['data'], legend,
                      replace=False, resetzoom=False,
                      pixmap=image['pixmap'], **params)

    # User event handling #

    def _isPositionInPlotArea(self, x, y):
        """Project position in pixel to the closest point in the plot area

        :param float x: X coordinate in widget coordinate (in pixel)
        :param float y: Y coordinate in widget coordinate (in pixel)
        :return: (x, y) in widget coord (in pixel) in the plot area
        """
        left, top, width, height = self.getPlotBoundsInPixels()
        xPlot = _utils.clamp(x, left, left + width)
        yPlot = _utils.clamp(y, top, top + height)
        return xPlot, yPlot

    def onMousePress(self, xPixel, yPixel, btn):
        """Handle mouse press event.

        :param float xPixel: X mouse position in pixels
        :param float yPixel: Y mouse position in pixels
        :param str btn: Mouse button in 'left', 'middle', 'right'
        """
        if self._isPositionInPlotArea(xPixel, yPixel) == (xPixel, yPixel):
            self._pressedButtons.append(btn)
            self._eventHandler.handleEvent('press', xPixel, yPixel, btn)

    def onMouseMove(self, xPixel, yPixel):
        """Handle mouse move event.

        :param float xPixel: X mouse position in pixels
        :param float yPixel: Y mouse position in pixels
        """
        inXPixel, inYPixel = self._isPositionInPlotArea(xPixel, yPixel)
        isCursorInPlot = inXPixel == xPixel and inYPixel == yPixel

        if isCursorInPlot:
            # Signal mouse move event
            dataPos = self.pixelToData(inXPixel, inYPixel)
            assert dataPos is not None

            btn = self._pressedButtons[-1] if self._pressedButtons else None
            event = PlotEvents.prepareMouseSignal(
                'mouseMoved', btn, dataPos[0], dataPos[1], xPixel, yPixel)
            self.notify(**event)

        # Either button was pressed in the plot or cursor is in the plot
        if isCursorInPlot or self._pressedButtons:
            self._eventHandler.handleEvent('move', inXPixel, inYPixel)

    def onMouseRelease(self, xPixel, yPixel, btn):
        """Handle mouse release event.

        :param float xPixel: X mouse position in pixels
        :param float yPixel: Y mouse position in pixels
        :param str btn: Mouse button in 'left', 'middle', 'right'
        """
        try:
            self._pressedButtons.remove(btn)
        except ValueError:
            pass
        else:
            xPixel, yPixel = self._isPositionInPlotArea(xPixel, yPixel)
            self._eventHandler.handleEvent('release', xPixel, yPixel, btn)

    def onMouseWheel(self, xPixel, yPixel, angleInDegrees):
        """Handle mouse wheel event.

        :param float xPixel: X mouse position in pixels
        :param float yPixel: Y mouse position in pixels
        :param float angleInDegrees: Angle corresponding to wheel motion.
                                     Positive for movement away from the user,
                                     negative for movement toward the user.
        """
        if self._isPositionInPlotArea(xPixel, yPixel) == (xPixel, yPixel):
            self._eventHandler.handleEvent(
                'wheel', xPixel, yPixel, angleInDegrees)

    # Interaction modes #

    def getInteractiveMode(self):
        """Returns the current interactive mode as a dict.

        The returned dict contains at least the key 'mode'.
        Mode can be: 'draw', 'pan', 'select', 'zoom'.
        It can also contains extra keys (e.g., 'color') specific to a mode
        as provided to :meth:`setInteractiveMode`.
        """
        return self._eventHandler.getInteractiveMode()

    def setInteractiveMode(self, mode, color='black',
                           shape='polygon', label=None,
                           zoomOnWheel=True, source=None):
        """Switch the interactive mode.

        :param str mode: The name of the interactive mode.
                         In 'draw', 'pan', 'select', 'zoom'.
        :param color: Only for 'draw' and 'zoom' modes.
                      Color to use for drawing selection area. Default black.
        :type color: Color description: The name as a str or
                     a tuple of 4 floats.
        :param str shape: Only for 'draw' mode. The kind of shape to draw.
                          In 'polygon', 'rectangle', 'line', 'vline', 'hline',
                          'freeline'.
                          Default is 'polygon'.
        :param str label: Only for 'draw' mode, sent in drawing events.
        :param bool zoomOnWheel: Toggle zoom on wheel support
        :param source: A user-defined object (typically the caller object)
                       that will be send in the interactiveModeChanged event,
                       to identify which object required a mode change.
                       Default: None
        """
        self._eventHandler.setInteractiveMode(mode, color, shape, label)
        self._eventHandler.zoomOnWheel = zoomOnWheel

        self.notify(
            'interactiveModeChanged', source=source)

    # Deprecated #

    def isDrawModeEnabled(self):
        """Deprecated, use :meth:`getInteractiveMode` instead.

        Return True if the current interactive state is drawing."""
        _logger.warning(
            'isDrawModeEnabled deprecated, use getInteractiveMode instead')
        return self.getInteractiveMode()['mode'] == 'draw'

    def setDrawModeEnabled(self, flag=True, shape='polygon', label=None,
                           color=None, **kwargs):
        """Deprecated, use :meth:`setInteractiveMode` instead.

        Set the drawing mode if flag is True and its parameters.

        If flag is False, only item selection is enabled.

        Warning: Zoom and drawing are not compatible and cannot be enabled
        simultanelously.

        :param bool flag: True to enable drawing and disable zoom and select.
        :param str shape: Type of item to be drawn in:
                          hline, vline, rectangle, polygon (default)
        :param str label: Associated text for identifying draw signals
        :param color: The color to use to draw the selection area
        :type color: string ("#RRGGBB") or 4 column unsigned byte array or
                     one of the predefined color names defined in Colors.py
        """
        _logger.warning(
            'setDrawModeEnabled deprecated, use setInteractiveMode instead')

        if kwargs:
            _logger.warning('setDrawModeEnabled ignores additional parameters')

        if color is None:
            color = 'black'

        if flag:
            self.setInteractiveMode('draw', shape=shape,
                                    label=label, color=color)
        elif self.getInteractiveMode()['mode'] == 'draw':
            self.setInteractiveMode('select')

    def getDrawMode(self):
        """Deprecated, use :meth:`getInteractiveMode` instead.

        Return the draw mode parameters as a dict of None.

        It returns None if the interactive moed is not a drawing mode,
        otherwise, it returns a dict containing the drawing mode parameters
        as provided to :meth:`setDrawModeEnabled`.
        """
        _logger.warning(
            'getDrawMode deprecated, use getInteractiveMode instead')
        mode = self.getInteractiveMode()
        return mode if mode['mode'] == 'draw' else None

    def isZoomModeEnabled(self):
        """Deprecated, use :meth:`getInteractiveMode` instead.

        Return True if the current interactive state is zooming."""
        _logger.warning(
            'isZoomModeEnabled deprecated, use getInteractiveMode instead')
        return self.getInteractiveMode()['mode'] == 'zoom'

    def setZoomModeEnabled(self, flag=True, color=None):
        """Deprecated, use :meth:`setInteractiveMode` instead.

        Set the zoom mode if flag is True, else item selection is enabled.

        Warning: Zoom and drawing are not compatible and cannot be enabled
        simultanelously

        :param bool flag: If True, enable zoom and select mode.
        :param color: The color to use to draw the selection area.
                      (Default: 'black')
        :param color: The color to use to draw the selection area
        :type color: string ("#RRGGBB") or 4 column unsigned byte array or
                     one of the predefined color names defined in Colors.py
        """
        _logger.warning(
            'setZoomModeEnabled deprecated, use setInteractiveMode instead')
        if color is None:
            color = 'black'

        if flag:
            self.setInteractiveMode('zoom', color=color)
        elif self.getInteractiveMode()['mode'] == 'zoom':
            self.setInteractiveMode('select')

    def insertMarker(self, *args, **kwargs):
        """Deprecated, use :meth:`addMarker` instead."""
        _logger.warning(
            'insertMarker deprecated, use addMarker instead.')
        return self.addMarker(*args, **kwargs)

    def insertXMarker(self, *args, **kwargs):
        """Deprecated, use :meth:`addXMarker` instead."""
        _logger.warning(
            'insertXMarker deprecated, use addXMarker instead.')
        return self.addXMarker(*args, **kwargs)

    def insertYMarker(self, *args, **kwargs):
        """Deprecated, use :meth:`addYMarker` instead."""
        _logger.warning(
            'insertYMarker deprecated, use addYMarker instead.')
        return self.addYMarker(*args, **kwargs)

    def isActiveCurveHandlingEnabled(self):
        """Deprecated, use :meth:`isActiveCurveHandling` instead."""
        _logger.warning(
            'isActiveCurveHandlingEnabled deprecated, '
            'use isActiveCurveHandling instead.')
        return self.isActiveCurveHandling()

    def enableActiveCurveHandling(self, *args, **kwargs):
        """Deprecated, use :meth:`setActiveCurveHandling` instead."""
        _logger.warning(
            'enableActiveCurveHandling deprecated, '
            'use setActiveCurveHandling instead.')
        return self.setActiveCurveHandling(*args, **kwargs)

    def invertYAxis(self, *args, **kwargs):
        """Deprecated, use :meth:`setYAxisInverted` instead."""
        _logger.warning('invertYAxis deprecated, '
                        'use setYAxisInverted instead.')
        return self.setYAxisInverted(*args, **kwargs)

    def showGrid(self, flag=True):
        """Deprecated, use :meth:`setGraphGrid` instead."""
        _logger.warning("showGrid deprecated, use setGraphGrid instead")
        if flag in (0, False):
            flag = None
        elif flag in (1, True):
            flag = 'major'
        else:
            flag = 'both'
        return self.setGraphGrid(flag)

    def keepDataAspectRatio(self, *args, **kwargs):
        """Deprecated, use :meth:`setKeepDataAspectRatio`."""
        _logger.warning('keepDataAspectRatio deprecated,'
                        'use setKeepDataAspectRatio instead')
        return self.setKeepDataAspectRatio(*args, **kwargs)
