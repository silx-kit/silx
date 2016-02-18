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
#
# ###########################################################################*/
"""Plot API for 1D and 2D data.

The :class:`Plot` implements the plot API initially provided in PyMca.
"""

__authors__ = ["V.A. Sole - ESRF Data Analysis", "T. Vincent"]
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"


from collections import OrderedDict
import logging
import math
import numpy

from .. import qt

from . import BackendBase
from . import Colors
from . import PlotInteraction
from . import PlotEvents

from .MatplotlibBackend import MatplotlibBackend


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

# PyQtGraph symbols ['o', 's', 't', 'd', '+', 'x']

# Matplotlib symbols:
# "." 	point
# "," 	pixel
# "o" 	circle
# "v" 	triangle_down
# "^" 	triangle_up
# "<" 	triangle_left
# ">" 	triangle_right
# "1" 	tri_down
# "2" 	tri_up
# "3" 	tri_left
# "4" 	tri_right
# "8" 	octagon
# "s" 	square
# "p" 	pentagon
# "*" 	star
# "h" 	hexagon1
# "H" 	hexagon2
# "+" 	plus
# "x" 	x
# "D" 	diamond
# "d" 	thin_diamond
# "|" 	vline
# "_" 	hline
# "None" 	nothing
# None 	nothing
# " " 	nothing
# "" 	nothing
#


def clamp(value, min_=0., max_=1.):  # TODO move elsewhere
    if value < min_:
        return min_
    elif value > max_:
        return max_
    else:
        return value


class Plot(object):
    # give the possibility to set the default backend for all instances
    # via a class attribute.
    defaultBackend = MatplotlibBackend

    colorList = _COLORLIST
    colorDict = _COLORDICT

    def __init__(self, parent=None, backend=None, callback=None):
        self._parent = parent
        self._dirty = True

        if backend is None:
            backend = self.defaultBackend

        if hasattr(backend, "__call__"):
            # to be called
            self._plot = backend(self, parent)
        elif isinstance(backend, BackendBase.BackendBase):
            self._plot = backend
            self._plot._setPlot(self)
        elif hasattr(backend, "lower"):
            lowerCaseString = backend.lower()
            if lowerCaseString in ["matplotlib", "mpl"]:
                from .MatplotlibBackend import MatplotlibBackend as be
            # elif lowerCaseString in ["gl", "opengl"]:
            #     from .backends.OpenGLBackend import OpenGLBackend as be
            # elif lowerCaseString in ["pyqtgraph"]:
            #     from .backends.PyQtGraphBackend import PyQtGraphBackend as be
            # elif lowerCaseString in ["glut"]:
            #     from .backends.GLUTOpenGLBackend import \
            #         GLUTOpenGLBackend as be
            # elif lowerCaseString in ["osmesa", "mesa"]:
            #     from .backends.OSMesaGLBackend import OSMesaGLBackend as be
            else:
                raise ValueError("Backend not understood %s" % backend)
            self._plot = be(self, parent)

        super(Plot, self).__init__()

        self.setCallback(callback)  # set _callback

        # Items handling
        self._curves = OrderedDict()
        self._hiddenCurves = set()

        self._images = OrderedDict()
        self._markers = OrderedDict()
        self._items = OrderedDict()

        self._selectionAreas = set()

        # line types
        self._styleList = ['-', '--', '-.', ':']

        self._colorIndex = 0
        self._styleIndex = 0

        self._activeCurveHandling = True
        self._activeCurve = None
        self._activeCurveColor = "#000000"
        self._activeImage = None

        # default properties
        self._logY = False
        self._logX = False
        self._xAutoScale = True
        self._yAutoScale = True

        self.setGraphTitle('')
        self.setGraphXLabel('')
        self.setGraphYLabel('')

        self.setDefaultColormap()  # Init default colormap

        self.setDefaultPlotPoints(False)
        self.setDefaultPlotLines(True)

        self._eventHandler = PlotInteraction.PlotInteraction(self)
        self._eventHandler.setInteractiveMode('zoom', color=(0., 0., 0., 1.))

        self._pressedButtons = []  # Currently pressed mouse buttons

        self._defaultDataMargins = (0., 0., 0., 0.)

    def _dirtyPlot(self):
        self._dirty = True

    # Private stuff called from elsewhere....

    PLUGINS_DIR = None  # TODO useful?

    # TODO called from ProfileScanWidget
    @property
    def _curveList(self):
        _logger.warning('depreacted: access a private member of Plot.py')
        return list(self._curves)

    # TODO called from PlotWindow
    def _getAllLimits(self):
        """
        Internal method to retrieve the limits based on the curves, not
        on the plot. It might be of use to reset the zoom when one of the
        X or Y axes is not set to autoscale.
        """
        _logger.warning('depreacted: access a private member of Plot.py')

        if not self._curves:
            return 0.0, 0.0, 100., 100.

        # Init to infinity values
        xmin, ymin = float('inf'), float('inf')
        xmax, ymax = - float('inf'), - float('inf')

        for curve in self._curves.values():
            x, y = curve['x'], curve['y']

            xmin = min(xmin, x.min())
            ymin = min(ymin, y.min())
            xmax = max(xmax, x.max())
            ymax = max(ymax, y.max())

        return xmin, ymin, xmax, ymax

    ##########################

    # Add

    # add * input arguments management:
    # If an arg is set, then use it.
    # Else:
    #     If a curve with the same legend exists, then use its arg value
    #     Else, use a default value.
    # Store used value.
    # This value is used when curve is updated either internally or by user.

    def addCurve(self, x, y, legend=None, info=None,
                 replace=False, replot=True,
                 color=None, symbol=None, linestyle=None,
                 xlabel=None, ylabel=None, yaxis=None,
                 xerror=None, yerror=None, z=None, selectable=None,
                 fill=None, **kw):
        # Take care of input parameters: check/conversion, default value

        if kw:
            _logger.warning('addCurve: deprecated extra arguments')

        legend = "Unnamed curve 1.1" if legend is None else str(legend)

        # Check/Convert input arguments

        # Convert to arrays (not forcing type) in order to avoid
        # problems at unexpected places: missing min or max attributes, problem
        # when using numpy.nonzero on lists, ...
        x = numpy.asarray(x)
        y = numpy.asarray(y)

        # TODO color

        # assert symbol in (None, '', ' ', 'o')  # TODO complete

        # assert linestyle in (None, '', ' ', '-')  # TODO complete

        if xlabel is not None:
            xlabel = str(xlabel)

        if ylabel is not None:
            ylabel = str(ylabel)

        assert yaxis in (None, 'left', 'right')

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
            'symbol': symbol, 'linestyle': linestyle,
            'xlabel': xlabel, 'ylabel': ylabel, 'yaxis': yaxis,
            'xerror': xerror, 'yerror': yerror, 'z': z,
            'selectable': selectable, 'fill': fill
        }

        # First, try to get defaults from existing curve with same name
        previousCurve = self._curves.get(legend, None)
        if previousCurve is not None:
            defaults = previousCurve['params']

        else:  # If no existing curve use default values
            # TODO What to do with x and y, xerror, yerror
            default_color, default_linestyle = self._getColorAndStyle()
            defaults = {
                'info': None, 'color': default_color,
                'symbol': self._defaultPlotPoints,
                'linestyle': default_linestyle,
                'xlabel': 'X', 'ylabel': 'Y', 'yaxis': 'left',
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
            self.clearCurves(replot=False)
        else:
            self.removeCurve(legend, replot=False)

        # Filter-out values <= 0
        x, y, color, xerror, yerror = self._logFilterData(
            x, y, params['color'], params['xerror'], params['yerror'],
            self.isXAxisLogarithmic(), self.isYAxisLogarithmic())

        if len(x) and not self.isCurveHidden(legend):
            curveHandle = self._plot.addCurve(x, y, legend,
                                              color=color,
                                              symbol=params['symbol'],
                                              linestyle=params['linestyle'],
                                              linewidth=1,
                                              yaxis=params['yaxis'],
                                              xerror=xerror,
                                              yerror=yerror,
                                              z=params['z'],
                                              selectable=params['selectable'],
                                              fill=params['fill'])
            self._dirtyPlot()
        else:
            curveHandle = None  # The curve has no points or is hidden

        self._curves[legend] = {
            'handle': curveHandle, 'x': x, 'y': y, 'params': params
        }

        if len(self._curves) == 1:
            self.setActiveCurve(legend)

        if replot:
            # We ask for a zoom reset in order to handle the plot scaling
            # if the user does not want that, autoscale of the different
            # axes has to be set to off.
            self.resetZoom()
            # self.replot()

        return legend

    def addImage(self, data, legend=None, info=None,
                 replace=True, replot=True,
                 xScale=None, yScale=None, z=None,
                 selectable=False, draggable=False,
                 colormap=None, pixmap=None,
                 xlabel=None, ylabel=None, **kw):
        """
        :param data: (nrows, ncolumns) data or
                     (nrows, ncolumns, RGBA) ubyte array
        :type data: numpy.ndarray
        :param legend: The legend to be associated to the curve
        :type legend: string or None
        :param info: Dictionary of information associated to the image
        :type info: dict or None
        :param replace: indicate if already existing images are to be deleted
        :type replace: boolean default True
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        :param xScale: Two floats defining the x scale
        :type xScale: list or numpy.ndarray
        :param yScale: Two floats defining the y scale
        :type yScale: list or numpy.ndarray
        :param z: level at which the image is located (to allow overlays)
        :type z: A number bigger than or equal to zero (default)
        :param selectable: Flag to indicate if the image can be selected
        :type selectable: boolean, default False
        :param draggable: Flag to indicate if the image can be moved
        :type draggable: boolean, default False
        :param colormap: Dictionary describing the colormap to use (or None)
        :type colormap: dict or None (default). Ignored if data is RGB(A)
        :param pixmap: Pixmap representation of the data (if any)
        :type pixmap: (nrows, ncolumns, RGBA) ubyte array or None (default)
        :returns: The legend used by the backend to univocally access it.
        """
        # Take care of input parameters: check/conversion, default value

        if kw:
            _logger.warning('addImage: deprecated extra arguments')

        if pixmap is not None:
            _logger.warning('addImage: deprecated pixmap argument')

        legend = "Unnamed Image 1.1" if legend is None else str(legend)

        # Check/Convert input arguments
        data = numpy.asarray(data)

        if xScale is not None:
            xScale = float(xScale[0]), float(xScale[1])

        if yScale is not None:
            yScale = float(yScale[0]), float(yScale[1])

        if z is not None:
            z = int(z)

        if selectable is not None:
            selectable = bool(selectable)

        if draggable is not None:
            draggable = bool(draggable)

        if pixmap is not None:  # TODO remove this from the API!
            pixmap = numpy.asarray(pixmap)

        if xlabel is not None:
            xlabel = str(xlabel)

        if ylabel is not None:
            ylabel = str(ylabel)

        # Store all params with defaults in a dict to treat them at once
        params = {
            'info': info, 'xScale': xScale, 'yScale': yScale, 'z': z,
            'selectable': selectable, 'draggable': draggable,
            'colormap': colormap,
            'xlabel': xlabel, 'ylabel': ylabel
            # TODO xlabel, ylabel is not used by active image!!
        }

        # First, try to get defaults from existing curve with same name
        previousImage = self._images.get(legend, None)
        if previousImage is not None:
            defaults = previousImage['params']

        else:  # If no existing curve use default values
            defaults = {
                'info': None, 'xScale': (0., 1.), 'yScale': (0., 1.), 'z': 0,
                'selectable': False, 'draggable': False,
                'colormap': self.getDefaultColormap(),
                'xlabel': 'Column', 'ylabel': 'Row'
            }

        # If a parameter is not given as argument, use its default value
        for key in defaults:
            if params[key] is None:
                params[key] = defaults[key]

        # Add: replace, filter data, add

        if replace:
            self.clearImages(replot=False)
        else:
            self.removeImage(legend, replot=False)

        if self.isXAxisLogarithmic() or self.isYAxisLogarithmic():
            _logger.warning('Hide image while axes has log scale.')

        if (data is not None and not self.isXAxisLogarithmic() and
                not self.isYAxisLogarithmic()):
            if pixmap is not None:
                dataToSend = pixmap
            else:
                dataToSend = data

            imageHandle = self._plot.addImage(dataToSend, legend=legend,
                                              xScale=params['xScale'],
                                              yScale=params['yScale'],
                                              z=params['z'],
                                              selectable=params['selectable'],
                                              draggable=params['draggable'],
                                              colormap=params['colormap'])
            self._dirtyPlot()
        else:
            imageHandle = None  # data is None or log scale

        self._images[legend] = {
            'handle': imageHandle,
            'data': data,
            'pixmap': pixmap,
            'params': params
        }

        if len(self._images) == 1:
            self.setActiveImage(legend)

        if replot:
            # We ask for a zoom reset in order to handle the plot scaling
            # if the user does not want that, autoscale of the different
            # axes has to be set to off.
            self.resetZoom()
            # self.replot()
        return legend

    def addItem(self, xdata, ydata, legend=None, info=None,
                replot=True, replace=False,
                shape="polygon", color='black', fill=True,
                overlay=False, **kw):
        # expected to receive the same parameters as the signal

        if kw:
            _logger.warning('Ignoring extra parameters %s', str(kw))

        legend = "Unnamed Item 1.1" if legend is None else str(legend)

        if replace:
            self.clearItems(replot=False)
        else:
            self.removeItem(legend, replot=False)

        item = self._plot.addItem(xdata, ydata, legend=legend,
                                  shape=shape, color=color,
                                  fill=fill, overlay=overlay)

        self._items[legend] = {'handle': item, 'overlay': overlay}

        if not overlay:
            self._dirtyPlot()

        if replot:
            self.replot()

        return legend

    def insertXMarker(self, x, legend=None,
                      text=None,
                      color=None,
                      selectable=False,
                      draggable=False,
                      **kw):
        if kw:
            _logger.warning('Extra parameters ignored: %s', str(kw))

        if text is None:
            text = kw.get("label", None)
            if text is not None:
                _logger.warning(
                    "insertXMarker deprecation: Use 'text' instead of 'label'")

        if color is None:
            color = self.colorDict['black']
        elif color in self.colorDict:
            color = self.colorDict[color]

        if legend is None:
            i = 0
            while legend in self._markers:
                legend = "Unnamed X Marker %d" % i
                i += 1

        if legend in self._markers:
            self.removeMarker(legend, replot=False)

        self._markers[legend] = self._plot.addXMarker(
            x, legend, text=text, color=color,
            selectable=selectable, draggable=draggable)
        self._dirtyPlot()

        return legend

    def insertYMarker(self, y,
                      legend=None,
                      text=None,
                      color=None,
                      selectable=False,
                      draggable=False,
                      **kw):
        if kw:
            _logger.warning('Extra parameters ignored: %s', str(kw))

        if text is None:
            text = kw.get("label", None)
            if text is not None:
                _logger.warning(
                    "insertYMarker deprecation: Use 'text' instead of 'label'")

        if color is None:
            color = self.colorDict['black']
        elif color in self.colorDict:
            color = self.colorDict[color]

        if legend is None:
            i = 0
            while legend in self._markers:
                legend = "Unnamed Y Marker %d" % i
                i += 1

        if legend in self._markers:
            self.removeMarker(legend, replot=False)

        self._markers[legend] = self._plot.addYMarker(
            y, legend=legend, text=text, color=color,
            selectable=selectable, draggable=draggable)
        self._dirtyPlot()

        return legend

    def insertMarker(self, x, y, legend=None,
                     text=None,
                     color=None,
                     selectable=False,
                     draggable=False,
                     symbol='+',
                     constraint=None,
                     **kw):
        if kw:
            _logger.warning('Extra parameters ignored: %s', str(kw))

        if text is None and 'label' in kw:
            text = kw['label']
            _logger.warning(
                "deprecation warning: Use 'text' instead of 'label'")

        if x is None:
            xmin, xmax = self.getGraphXLimits()
            x = 0.5 * (xmax + xmin)

        if y is None:
            ymin, ymax = self.getGraphYLimits()
            y = 0.5 * (ymax + ymin)

        if legend is None:
            i = 0
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
            self.removeMarker(legend, replot=False)

        self._markers[legend] = self._plot.addMarker(
            x, y, legend=legend, text=text, color=color,
            selectable=selectable, draggable=draggable,
            symbol=symbol, constraint=constraint)
        self._dirtyPlot()

        return legend

    # Hide

    def isCurveHidden(self, legend):
        return legend in self._hiddenCurves

    def hideCurve(self, legend, flag=True, replot=True):
        if legend not in self._curves:
            _logger.warning('Curve not in plot: %s', legend)
            return

        if flag:
            handle = self._curves[legend]['handle']
            if handle is not None:
                self._plot.remove(handle)
                self._curves[legend]['handle'] = None

            self._hiddenCurves.add(legend)
        else:
            self._hiddenCurves.discard(legend)
            curve = self._curves[legend]
            self.addCurve(curve['x'], curve['y'], legend, replot=False,
                          **curve['params'])

        self._dirtyPlot()

        if replot:
            self.replot()

    # Remove

    def removeCurve(self, legend, replot=True):
        """
        Remove the curve associated to the supplied legend from the graph.
        The graph will be updated if replot is true.
        :param legend: The legend associated to the curve to be deleted
        :type legend: string or None
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        """
        if legend is None:
            return

        self._hiddenCurves.discard(legend)

        if legend in self._curves:
            handle = self._curves[legend]['handle']
            if handle is not None:
                self._plot.remove(handle)
                self._dirtyPlot()
            del self._curves[legend]

        if not self._curves:
            self._colorIndex = 0
            self._styleIndex = 0

        if replot:
            self.replot()

    def removeImage(self, legend, replot=True):
        """
        Remove the image associated to the supplied legend from the graph.
        The graph will be updated if replot is true.
        :param legend: The legend associated to the image to be deleted
        :type legend: string or handle
        :param replot: Flag to indicate plot is to be immediately updated
        :type replot: boolean default True
        """
        if legend is None:
            return

        if legend in self._images:
            handle = self._images[legend]['handle']
            if handle is not None:
                self._plot.remove(handle)
                self._dirtyPlot()
            del self._images[legend]

        if replot:
            self.replot()

    def removeItem(self, legend, replot=True):
        if legend is None:
            return

        item = self._items.pop(legend, None)
        if item is not None and item['handle'] is not None:
            self._plot.remove(item['handle'])
            if not item['overlay']:
                self._dirtyPlot()

        if replot:
            self.replot()

    def removeMarker(self, marker, replot=True):
        handle = self._markers.pop(marker, None)
        if handle is not None:
            self._plot.remove(handle)
            self._dirtyPlot()

        if replot:
            self.replot()

    # Clear

    def clear(self, replot=True):
        self.clearCurves(replot=False)
        self.clearMarkers(replot=False)
        self.clearImages(replot=False)
        self.clearItems(replot=False)

        self._plot.clear()
        self._dirtyPlot()

        if replot:
            self.replot()

    def clearCurves(self, replot=True):
        for legend in list(self._curves):  # Copy as _curves gets changed
            self.removeCurve(legend, replot=False)
        self._curves = OrderedDict()
        self._hiddenCurves = set()
        self._colorIndex = 0
        self._styleIndex = 0

        if replot:
            self.replot()

    def clearImages(self, replot=True):
        """Clear all images from the plot.

        Not the curves or markers.
        """
        for legend in list(self._images):  # Copy as _images gets changed
            self.removeImage(legend, replot=False)
        self._images = OrderedDict()

        if replot:
            self.replot()

    def clearItems(self, replot=True):
        for legend in list(self._items):  # Copy as _items gets changed
            self.removeItem(legend, replot=False)
        self._items = OrderedDict()

        if replot:
            self.replot()

    def clearMarkers(self, replot=True):
        for legend in list(self._markers):  # Copy as _markers gets changed
            self.removeMarker(legend, replot=False)
        self._markers = OrderedDict()

        if replot:
            self.replot()

    # Interaction

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

                - ' ' no line
                - '-' solid line
                - '--' dashed line
                - '-.' dash-dot line
                - ':' dotted line

        :type linestyle: None or one of the predefined styles.
        """
        self._plot.setGraphCursor(flag=flag, color=color,
                                  linewidth=linewidth, linestyle=linestyle)

    def getGraphCursor(self):
        """
        Returns the current state of the crosshair cursor.

        :return: None if the crosshair cursor is not active,
                 else a tuple (color, linewidth, linestyle).
        """
        return self._plot.getGraphCursor()

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

            xMin, xMax = _applyPan(xMin, xMax, xFactor,
                                   self.isXAxisLogarithmic())
            self.setGraphXLimits(xMin, xMax)

        else:  # direction in ('up', 'down')
            sign = -1. if self.isYAxisInverted() else 1.
            yFactor = sign * (factor if direction == 'up' else -factor)
            yMin, yMax = self.getGraphYLimits()
            yIsLog = self.isYAxisLogarithmic()

            yMin, yMax = _applyPan(yMin, yMax, yFactor, yIsLog)
            self.setGraphYLimits(yMin, yMax)

            # TODO handle second Y axis

        self.replot()

    # Active Curve/Image

    def isActiveCurveHandlingEnabled(self):
        return self._activeCurveHandling

    def enableActiveCurveHandling(self, flag=True):
        if not flag:
            self.setActiveCurve(None)  # Reset active curve

        self._activeCurveHandling = bool(flag)

    def getActiveCurveColor(self):
        return self._activeCurveColor

    def setActiveCurveColor(self, color="#000000"):
        if color is None:
            color = "black"
        if color in self.colorDict:
            color = self.colorDict[color]
        self._activeCurveColor = color

    def getActiveCurve(self, just_legend=False):
        """
        :param just_legend: Flag to specify the type of output required
        :type just_legend: boolean
        :return: legend of the active curve or list [x, y, legend, info]
        :rtype: string or list

        Function to access the graph currently active curve.
        It returns None in case of not having an active curve.

        Default output has the form:
            xvalues, yvalues, legend, dict
            where dict is a dictionnary containing curve info.
            For the time being, only the plot labels associated to the
            curve are warranted to be present under the keys xlabel, ylabel.

        If just_legend is True:
            The legend of the active curve (or None) is returned.
        """
        if not self.isActiveCurveHandlingEnabled():
            return None

        if self._activeCurve not in self._curves:
            self._activeCurve = None

        if self._activeCurve is None:
            return None

        if just_legend:
            return self._activeCurve
        else:
            curve = self._curves[self._activeCurve]
            return curve['x'], curve['y'], self._activeCurve, curve['params']

    def setActiveCurve(self, legend, replot=True):
        """Make the curve with the specified legend the active curve.

        :param str legend: The legend associated to the curve
                           or None to have no active curve.
        """
        if not self.isActiveCurveHandlingEnabled():
            return

        xLabel = self._xLabel
        yLabel = self._yLabel

        oldActiveCurve = self.getActiveCurve()
        if oldActiveCurve:  # Reset previous active curve
            handle = self._curves[oldActiveCurve[2]]['handle']
            if handle is not None:
                self._plot.setCurveColor(handle, oldActiveCurve[3]['color'])
                self._dirtyPlot()

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
                    self._plot.setCurveColor(handle,
                                             self.getActiveCurveColor())
                    self._dirtyPlot()

                activeCurve = self.getActiveCurve()
                xLabel = activeCurve[3]['xlabel']
                yLabel = activeCurve[3]['ylabel']  # TODO y2 axis case

        # Store current labels and update plot
        self._currentXLabel = xLabel
        self._plot.setGraphXLabel(xLabel)
        self._dirtyPlot()
        self._currentYLabel = yLabel
        self._plot.setGraphYLabel(yLabel)  # TODO handle y2 axis
        self._dirtyPlot()

        if replot:
            self.replot()

        return self._activeCurve

    def getActiveImage(self, just_legend=False):
        """
        Function to access the plot currently active image.
        It returns None in case of not having an active image.

        Default output has the form:
            data, legend, dict, pixmap
            where dict is a dictionnary containing image info.
            For the time being, only the plot labels associated to the
            image are warranted to be present under the keys xlabel, ylabel.

        If just_legend is True:
            The legend of the active imagee (or None) is returned.

        :param just_legend: Flag to specify the type of output required
        :type just_legend: boolean
        :return: legend of active image or list [data, legend, info, pixmap]
        :rtype: string or list
        """
        if self._activeImage not in self._images:
            self._activeImage = None

        if just_legend:
            return self._activeImage

        if self._activeImage is None:
            return None
        else:
            image = self._images[self._activeImage]
            return image['x'], image['y'], self._activeImage, image['params']

    def setActiveImage(self, legend, replot=True):
        """Funtion to request the plot window to set the image with the
        specified legend as the active image.

        :param legend: The legend associated to the image
        :type legend: string
        """
        if legend is None:
            self._activeImage = None
        else:
            legend = str(legend)
            if legend not in self._images:
                _logger.warning(
                    "setActiveImage: This image does not exist: %s", legend)
                self._activeCurve = None
            else:
                self._activeImage = legend

        if replot:
            self.replot()

        return self._activeImage

    # Getters

    def getAllCurves(self, just_legend=False):
        """
        :param just_legend: Flag to specify the type of output required
        :type just_legend: boolean
        :return: legend of the curves or list [[x, y, legend, info], ...]
        :rtype: list of strings or list of curves

        It returns an empty list in case of not having any curve.
        If just_legend is False:
            It returns a list of the form:
                [[xvalues0, yvalues0, legend0, dict0],
                 [xvalues1, yvalues1, legend1, dict1],
                 [...],
                 [xvaluesn, yvaluesn, legendn, dictn]]
            or just an empty list.
        If just_legend is True:
            It returns a list of the form:
                [legend0, legend1, ..., legendn]
            or just an empty list.
        """
        output = []
        for key in self._curves:
            if self.isCurveHidden(key):
                continue
            if just_legend:
                output.append(key)
            else:
                curve = self._curves[key]
                output.append((curve['x'], curve['y'], key, curve['params']))
        return output

    def getCurve(self, legend):
        """
        :param legend: legend associated to the curve
        :type legend: boolean
        :return: list [x, y, legend, info]
        :rtype: list

        Function to access the graph specified curve.
        It returns None in case of not having the curve.

        Default output has the form:
            xvalues, yvalues, legend, info
            where info is a dictionnary containing curve info.
            For the time being, only the plot labels associated to the
            curve are warranted to be present under the keys xlabel, ylabel.
        """
        if legend in self._curves:
            curve = self._curves[legend]
            return curve['x'], curve['y'], legend, curve['params']
        else:
            return None

    # TODO actually not used...
    def getMonotonicCurves(self):
        """
        Convenience method that calls getAllCurves and makes sure that all of
        the X values are strictly increasing.

        :return: It returns a list of the form:
                [[xvalues0, yvalues0, legend0, dict0],
                 [xvalues1, yvalues1, legend1, dict1],
                 [...],
                 [xvaluesn, yvaluesn, legendn, dictn]]
        """
        allCurves = self.getAllCurves() * 1
        for i in range(len(allCurves)):
            curve = allCurves[i]
            x, y, legend, info = curve[0:4]
            if self.isCurveHidden(legend):
                continue
            # Sort
            idx = numpy.argsort(x, kind='mergesort')
            xproc = numpy.take(x, idx)
            yproc = numpy.take(y, idx)
            # Ravel, Increase
            xproc = xproc.ravel()
            idx = numpy.nonzero((xproc[1:] > xproc[:-1]))[0]
            xproc = numpy.take(xproc, idx)
            yproc = numpy.take(yproc, idx)
            allCurves[i][0:2] = x, y
        return allCurves

    def getImage(self, legend):
        """
        :param legend: legend associated to the curve
        :type legend: boolean
        :return: list [image, legend, info, pixmap]
        :rtype: list

        Function to access the graph currently active curve.
        It returns None in case of not having an active curve.

        Default output has the form:
            image, legend, info, pixmap
            where info is a dictionnary containing image information.
        """
        if legend in self._images:
            image = self._images[legend]
            return image['x'], image['y'], legend, image['params']
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
        self.notify(event)

    def getGraphXLimits(self):
        """Get the graph X (bottom) limits.

        :return:  Minimum and maximum values of the X axis
        """
        return self._plot.getGraphXLimits()

    def setGraphXLimits(self, xmin, xmax, replot=False):
        self._plot.setGraphXLimits(xmin, xmax)
        self._dirtyPlot()

        self._notifyLimitsChanged()

        if replot:
            self.replot()

    def getGraphYLimits(self, axis='left'):
        """Get the graph Y (left) limits.

        :param str axis: The axis for which to get the limits: left or right
        :return:  Minimum and maximum values of the X axis
        """
        assert axis in ('left', 'right')
        return self._plot.getGraphYLimits(axis)

    def setGraphYLimits(self, ymin, ymax, replot=False):
        self._plot.setGraphYLimits(ymin, ymax)
        self._dirtyPlot()

        self._notifyLimitsChanged()

        if replot:
            self.replot()

    def setLimits(self, xmin, xmax, ymin, ymax, y2min=None, y2max=None):
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if ymax < ymin:
            ymin, ymax = ymax, ymin

        if y2min is None or y2max is None:
            # if one limit is None, both are ignored
            y2min, y2max = None, None
        elif y2max < y2min:
                y2min, y2max = y2max, y2min

        self._plot.setLimits(xmin, xmax, ymin, ymax, y2min, y2max)
        self._dirtyPlot()
        self._notifyLimitsChanged()

    # Title and labels

    def getGraphTitle(self):
        return self._graphTitle

    def setGraphTitle(self, title=""):
        self._graphTitle = str(title)
        self._plot.setGraphTitle(title)
        self._dirtyPlot()

    def getGraphXLabel(self):
        return self._currentXLabel

    def setGraphXLabel(self, label="X"):
        self._xLabel = label
        # Current label can differ from input one with active curve handling
        self._currentXLabel = label
        self._plot.setGraphXLabel(label)
        self._dirtyPlot()

    def getGraphYLabel(self):
        return self._currentYLabel

    def setGraphYLabel(self, label="Y"):
        self._yLabel = label
        # Current label can differ from input one with active curve handling
        self._currentYLabel = label
        self._plot.setGraphYLabel(label)
        self._dirtyPlot()

    # Axes

    def invertYAxis(self, flag=True):
        self._plot.invertYAxis(flag)
        self._dirtyPlot()

    def isYAxisInverted(self):
        return self._plot.isYAxisInverted()

    def isXAxisLogarithmic(self):
        return self._logX

    def setXAxisLogarithmic(self, flag):
        if bool(flag) == self._logX:
            return
        self._logX = bool(flag)

        if self._logX:  # Switch to log scale
            for image in self._images.values():
                if image['handle'] is not None:
                    self._plot.remove(image['handle'])
                    image['handle'] = None

            for curve in self._curves.values():
                handle = curve['handle']
                if handle is not None:
                    self._plot.remove(handle)
                    curve['handle'] = None

            # matplotlib 1.5 crashes if the log set is made before
            # the call to self._update()
            # TODO: Decide what is better for other backends
            if (hasattr(self._plot, "matplotlibVersion") and
                    self._plot.matplotlibVersion >= "1.5"):
                self._update()
                self._plot.setXAxisLogarithmic(self._logX)
            else:
                self._plot.setXAxisLogarithmic(self._logX)
                self._update()
        else:
                self._plot.setXAxisLogarithmic(self._logX)
                self._update()

        self._dirtyPlot()
        self.replot()

    def isYAxisLogarithmic(self):
        return self._logY

    def setYAxisLogarithmic(self, flag):
        if bool(flag) == self._logY:
            return
        self._logY = bool(flag)

        if self._logY:  # Switch to log scale
            for image in self._images.values():
                if image['handle'] is not None:
                    self._plot.remove(image['handle'])
                    image['handle'] = None

            for curve in self._curves.values():
                handle = curve['handle']
                if handle is not None:
                    self._plot.remove(handle)
                    curve['handle'] = None

            # matplotlib 1.5 crashes if the log set is made before
            # the call to self._update()
            # TODO: Decide what is better for other backends
            if (hasattr(self._plot, "matplotlibVersion") and
                    self._plot.matplotlibVersion >= "1.5"):
                self._update()
                self._plot.setYAxisLogarithmic(self._logY)
            else:
                self._plot.setYAxisLogarithmic(self._logY)
                self._update()
        else:
                self._plot.setYAxisLogarithmic(self._logY)
                self._update()

        self._dirtyPlot()
        self.replot()

    def isXAxisAutoScale(self):
        return self._xAutoScale

    def setXAxisAutoScale(self, flag=True):
        self._xAutoScale = bool(flag)

    def isYAxisAutoScale(self):
        return self._yAutoScale

    def setYAxisAutoScale(self, flag=True):
        self._yAutoScale = flag

    def isKeepDataAspectRatio(self):
        return self._plot.isKeepDataAspectRatio()

    def keepDataAspectRatio(self, flag=True):
        """
        :param flag:  True to respect data aspect ratio
        :type flag: Boolean, default True
        """
        self._plot.keepDataAspectRatio(flag=flag)
        self._dirtyPlot()
        self.resetZoom()

    def showGrid(self, flag=True):
        _logger.debug("Plot showGrid called")
        self._plot.showGrid(flag)
        self._dirtyPlot()
        self.replot()

    # Defaults

    def setDefaultPlotPoints(self, flag):
        self._defaultPlotPoints = 'o' if flag else ''

        # Reset symbol of all curves
        for curve in self._curves:
            curve['params']['symbol'] = self._defaultPlotPoints

        if self._curves:
            self._update()
            self._dirtyPlot()
            self.replot()

    def setDefaultPlotLines(self, flag):
        self._plotLines = bool(flag)

        if self._curves:
            self._update()
            self._dirtyPlot()
            self.replot()

    def getDefaultColormap(self):
        """
        Return the colormap that will be applied by the backend to an image
        if no colormap is applied to it.
        A colormap is a dictionnary with the keys:
        :type name: string
        :type normalization: string (linear, log)
        :type autoscale: boolean
        :type vmin: float, minimum value
        :type vmax: float, maximum value
        :type colors: integer (typically 256)
        """
        return self._defaultColormap

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
        if colormap is None:
            colormap = {'name': 'gray', 'normalization': 'linear',
                        'autoscale': True, 'vmin': 0.0, 'vmax': 1.0,
                        'colors': 256}
        self._defaultColormap = colormap

    def getSupportedColormaps(self):
        """Get a list of strings with the supported colormap names.

        The list should at least contain and start by:
        ['gray', 'reversed gray', 'temperature', 'red', 'green', 'blue']
        """
        return self._plot.getSupportedColormaps()

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
        return self._plot.getWidgetHandle()

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
        """
        This callback is going to receive all the events from the plot.
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

    def saveGraph(self, filename, fileFormat='svg', dpi=None, **kw):
        """
        :param fileName: Destination
        :type fileName: String or StringIO or BytesIO
        :param fileFormat:  String specifying the format
        :type fileFormat: String (default 'svg')
        """
        if kw:
            _logger.warning('Extra parameters ignored: %s', str(kw))

        return self._plot.saveGraph(filename,
                                    fileFormat=fileFormat,
                                    dpi=dpi)

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

    def replot(self):
        _logger.debug("replot called")
        self._plot.replot(overlayOnly=not self._dirty)
        self._dirty = False  # reset dirty flag

    def resetZoom(self, dataMargins=None):
        if dataMargins is None:
            dataMargins = self._defaultDataMargins
        self._plot.resetZoom(dataMargins)
        self._dirtyPlot()
        self.replot()

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
            self.addCurve(curve['x'], curve['y'], legend, replot=False,
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
                              replace=False, replot=False,
                              pixmap=image['pixmap'], **image['params'])

    # Coord conversion

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
        return self._plot.dataToPixel(x, y, axis=axis)

    def pixelToData(self, x=None, y=None, axis="left", check=False):
        """
        Convert a position in pixels in the widget to a position in
        the data space.

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
        return self._plot.pixelToData(x, y, axis=axis, check=check)

    def getPlotBoundsInPixels(self):
        """Plot area bounds in widget coordinates in pixels.

        :return: bounds as a 4-tuple of int: (left, top, width, height)
        """
        return self._plot.getPlotBoundsInPixels()

    # Interaction support

    def notify(self, event):
        """Send an event to the listeners.

        The event dict must at least contains an 'event' key which store the
        type of event.
        The other keys are event specific.

        :param dict event: The information of the event.
        """
        self._callback(event)

    def setSelectionArea(self, points, fill, color, name=''):
        """Set a polygon selection area overlaid on the plot.
        Multiple simultaneous areas are supported through the name parameter.

        :param points: The 2D coordinates of the points of the polygon
        :type points: An iterable of (x, y) coordinates
        :param str fill: The fill mode: 'hatch', 'solid' or None
        :param color: RGBA color to use
        :type color: list or tuple of 4 float in the range [0, 1]
        :param name: The key associated with this selection area
        """
        points = numpy.asarray(points)

        # TODO Not very nice, but as is for now
        legend = '__SELECTION_AREA__' + name

        fill = bool(fill)  # TODO not very nice either

        # TODO make it an overlay
        self.addItem(points[:, 0], points[:, 1], legend=legend,
                     replace=False, replot=False,
                     shape='polygon', color=color, fill=fill,
                     overlay=True)
        self._selectionAreas.add(legend)

    def resetSelectionArea(self):
        """Remove all selection areas set by setSelectionArea."""
        for legend in self._selectionAreas:
            self.removeItem(legend, replot=False)
        self._selectionAreas = set()

    # TODO move this to backend?
    _QT_CURSORS = {
        PlotInteraction.CURSOR_DEFAULT: qt.Qt.ArrowCursor,
        PlotInteraction.CURSOR_POINTING: qt.Qt.PointingHandCursor,
        PlotInteraction.CURSOR_SIZE_HOR: qt.Qt.SizeHorCursor,
        PlotInteraction.CURSOR_SIZE_VER: qt.Qt.SizeVerCursor,
        PlotInteraction.CURSOR_SIZE_ALL: qt.Qt.SizeAllCursor,
    }

    def setCursor(self, cursor=PlotInteraction.CURSOR_DEFAULT):
        """Set the cursor shape.

        :param str cursor: Name of the cursor shape
        """
        cursor = self._QT_CURSORS[cursor]
        self.getWidgetHandle().setCursor(qt.QCursor(cursor))

    def pickMarker(self, *args, **kwargs):
        return None  # TODO

    def pickImageOrCurve(self, *args, **kwargs):
        return None  # TODO

    # User event handling #

    def _isPositionInPlotArea(self, x, y):
        """Project position in pixel to the closest point in the plot area

        :param float x: X coordinate in widget coordinate (in pixel)
        :param float y: Y coordinate in widget coordinate (in pixel)
        :return: (x, y) in widget coord (in pixel) in the plot area
        """
        left, top, width, height = self.getPlotBoundsInPixels()
        xPlot = clamp(x, left, left + width)
        yPlot = clamp(y, top, top + height)
        return xPlot, yPlot

    def onMousePress(self, xPixel, yPixel, btn):
        if self._isPositionInPlotArea(xPixel, yPixel) == (xPixel, yPixel):
            self._pressedButtons.append(btn)
            self._eventHandler.handleEvent('press', xPixel, yPixel, btn)

    def onMouseMove(self, xPixel, yPixel):
        inXPixel, inYPixel = self._isPositionInPlotArea(xPixel, yPixel)
        isCursorInPlot = inXPixel == xPixel and inYPixel == yPixel

        # crosshair stuff
        # previousMousePosInPixels = self._mousePosInPixels
        # self._mousePosInPixels = (xPixel, yPixel) if isCursorInPlot else None
        # if (self._crosshairCursor is not None and
        #        previousMousePosInPixels != self._crosshairCursor):
        #    # Avoid replot when cursor remains outside plot area
        #    self.replot()

        if isCursorInPlot:
            # Signal mouse move event
            dataPos = self.pixelToData(inXPixel, inYPixel)
            assert dataPos is not None

            btn = self._pressedButtons[-1] if self._pressedButtons else None
            event = PlotEvents.prepareMouseSignal(
                'mouseMoved', btn, dataPos[0], dataPos[1], xPixel, yPixel)
            self.notify(event)

        # Either button was pressed in the plot or cursor is in the plot
        if isCursorInPlot or self._pressedButtons:
            self._eventHandler.handleEvent('move', inXPixel, inYPixel)

    def onMouseRelease(self, xPixel, yPixel, btn):
        try:
            self._pressedButtons.remove(btn)
        except ValueError:
            pass
        else:
            xPixel, yPixel = self._isPositionInPlotArea(xPixel, yPixel)
            self._eventHandler.handleEvent('release', xPixel, yPixel, btn)

    def onMouseWheel(self, xPixel, yPixel, angleInDegrees):
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
                           shape='polygon', label=None):
        """Switch the interactive mode.

        :param str mode: The name of the interactive mode.
                         In 'draw', 'pan', 'select', 'zoom'.
        :param color: Only for 'draw' and 'zoom' modes.
                      Color to use for drawing selection area. Default black.
        :type color: Color description: The name as a str or
                     a tuple of 4 floats.
        :param str shape: Only for 'draw' mode. The kind of shape to draw.
                          In 'polygon', 'rectangle', 'line', 'vline', 'hline'.
                          Default is 'polygon'.
        :param str label: Only for 'draw' mode.
        """
        self._eventHandler.setInteractiveMode(mode, color, shape, label)

    # TODO deprecate all the following

    def isDrawModeEnabled(self):
        return self.getInteractiveMode()['mode'] == 'draw'

    def setDrawModeEnabled(self, flag=True, shape='polygon', label=None,
                           color=None, **kwargs):
        """Zoom and drawing are not compatible and cannot be enabled
        simultanelously

        :param flag: Enable drawing mode disabling zoom and picking mode
        :type flag: boolean, default True
        :param shape: Type of item to be drawn in:
                      hline, vline, rectangle, polygon
        :type shape: string (default polygon)
        :param str label: Associated text for identifying draw signals
        :param color: The color to use to draw the selection area
        :type color: string ("#RRGGBB") or 4 column unsigned byte array or
                     one of the predefined color names defined in Colors.py
        """
        if kwargs:
            _logger.warning('setDrawModeEnabled ignores additional parameters')

        if flag:
            self.setInteractiveMode('draw', shape=shape,
                                    label=label, color=color)
        elif self.getInteractiveMode()['mode'] == 'draw':
            self.setInteractiveMode('select')

    def getDrawMode(self):
        """
        Return a dictionnary (or None) with the parameters passed when setting
        the draw mode.
        :key shape: The shape being drawn
        :key label: Associated text (or None)
        and any other info
        """
        mode = self.getInteractiveMode()
        return mode if mode['mode'] == 'draw' else None

    def isZoomModeEnabled(self):
        return self.getInteractiveMode()['mode'] == 'zoom'

    def setZoomModeEnabled(self, flag=True, color=None):
        """Zoom and drawing are not compatible and cannot be enabled
        simultanelously

        :param flag: If True, the user can zoom.
        :type flag: boolean, default True
        :param color: The color to use to draw the selection area.
                      Default 'black"
        :param color: The color to use to draw the selection area
        :type color: string ("#RRGGBB") or 4 column unsigned byte array or
                     one of the predefined color names defined in Colors.py
        """
        if flag:
            self.setInteractiveMode('zoom', color=color)
        elif self.getInteractiveMode()['mode'] == 'zoom':
            self.setInteractiveMode('select')


def _applyPan(min_, max_, panFactor, isLog10):
    """Returns a new range with applied panning.

    Moves the range according to panFactor.
    If isLog10 is True, converts to log10 before moving.

    :param float min_: Min value of the data range to pan.
    :param float max_: Max value of the data range to pan.
                       Must be >= min_.
    :param float panFactor: Signed proportion of the range to use for pan.
    :param bool isLog10: True if log10 scale, False if linear scale.
    :return: New min and max value with pan applied.
    :rtype: 2-tuple of float.
    """
    if isLog10 and min_ > 0.:
        # Negative range and log scale can happen with matplotlib
        logMin, logMax = math.log10(min_), math.log10(max_)
        logOffset = panFactor * (logMax - logMin)
        newMin = pow(10., logMin + logOffset)
        newMax = pow(10., logMax + logOffset)

        # Takes care of out-of-range values
        if newMin > 0. and newMax < float('inf'):
            min_, max_ = newMin, newMax

    else:
        offset = panFactor * (max_ - min_)
        newMin, newMax = min_ + offset, max_ + offset

        # Takes care of out-of-range values
        if newMin > - float('inf') and newMax < float('inf'):
            min_, max_ = newMin, newMax
    return min_, max_
