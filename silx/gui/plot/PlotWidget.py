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
# ###########################################################################*/
"""Qt widget providing plot API for 1D and 2D data.

The :class:`PlotWidget` implements the plot API initially provided in PyMca.
"""

from __future__ import division


__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "12/10/2018"


from collections import OrderedDict, namedtuple
from contextlib import contextmanager
import datetime as dt
import itertools
import logging

import numpy

import silx
from silx.utils.weakref import WeakMethodProxy
from silx.utils import deprecation
from silx.utils.property import classproperty
from silx.utils.deprecation import deprecated
# Import matplotlib backend here to init matplotlib our way
from .backends.BackendMatplotlib import BackendMatplotlibQt

from ..colors import Colormap
from .. import colors
from . import PlotInteraction
from . import PlotEvents
from .LimitsHistory import LimitsHistory
from . import _utils

from . import items
from .items.curve import CurveStyle
from .items.axis import TickMode  # noqa

from .. import qt
from ._utils.panzoom import ViewConstraints
from ...gui.plot._utils.dtime_ticklayout import timestamp

_logger = logging.getLogger(__name__)


_COLORDICT = colors.COLORDICT
_COLORLIST = silx.config.DEFAULT_PLOT_CURVE_COLORS

"""
Object returned when requesting the data range.
"""
_PlotDataRange = namedtuple('PlotDataRange',
                            ['x', 'y', 'yright'])


class PlotWidget(qt.QMainWindow):
    """Qt Widget providing a 1D/2D plot.

    This widget is a QMainWindow.
    This class implements the plot API initially provided in PyMca.

    Supported backends:

    - 'matplotlib' and 'mpl': Matplotlib with Qt.
    - 'opengl' and 'gl': OpenGL backend (requires PyOpenGL and OpenGL >= 2.1)
    - 'none': No backend, to run headless for testing purpose.

    :param parent: The parent of this widget or None (default).
    :param backend: The backend to use, in:
                    'matplotlib' (default), 'mpl', 'opengl', 'gl', 'none'
                    or a :class:`BackendBase.BackendBase` class
    :type backend: str or :class:`BackendBase.BackendBase`
    """

    # TODO: Can be removed for silx 0.10
    @classproperty
    @deprecation.deprecated(replacement="silx.config.DEFAULT_PLOT_BACKEND", since_version="0.8", skip_backtrace_count=2)
    def DEFAULT_BACKEND(self):
        """Class attribute setting the default backend for all instances."""
        return silx.config.DEFAULT_PLOT_BACKEND

    colorList = _COLORLIST
    colorDict = _COLORDICT

    sigPlotSignal = qt.Signal(object)
    """Signal for all events of the plot.

    The signal information is provided as a dict.
    See the :ref:`plot signal documentation page <plot_signal>` for
    information about the content of the dict
    """

    sigSetKeepDataAspectRatio = qt.Signal(bool)
    """Signal emitted when plot keep aspect ratio has changed"""

    sigSetGraphGrid = qt.Signal(str)
    """Signal emitted when plot grid has changed"""

    sigSetGraphCursor = qt.Signal(bool)
    """Signal emitted when plot crosshair cursor has changed"""

    sigSetPanWithArrowKeys = qt.Signal(bool)
    """Signal emitted when pan with arrow keys has changed"""

    _sigAxesVisibilityChanged = qt.Signal(bool)
    """Signal emitted when the axes visibility changed"""

    sigContentChanged = qt.Signal(str, str, str)
    """Signal emitted when the content of the plot is changed.

    It provides the following information:

    - action: The change of the plot: 'add' or 'remove'
    - kind: The kind of primitive changed:
      'curve', 'image', 'scatter', 'histogram', 'item' or 'marker'
    - legend: The legend of the primitive changed.
    """

    sigActiveCurveChanged = qt.Signal(object, object)
    """Signal emitted when the active curve has changed.

    It provides the following information:

    - previous: The legend of the previous active curve or None
    - legend: The legend of the new active curve or None if no curve is active
    """

    sigActiveImageChanged = qt.Signal(object, object)
    """Signal emitted when the active image has changed.

    It provides the following information:

    - previous: The legend of the previous active image or None
    - legend: The legend of the new active image or None if no image is active
    """

    sigActiveScatterChanged = qt.Signal(object, object)
    """Signal emitted when the active Scatter has changed.

    It provides the following information:

    - previous: The legend of the previous active scatter or None
    - legend: The legend of the new active image or None if no image is active
    """

    sigInteractiveModeChanged = qt.Signal(object)
    """Signal emitted when the interactive mode has changed

    It provides the source as passed to :meth:`setInteractiveMode`.
    """

    sigItemAdded = qt.Signal(items.Item)
    """Signal emitted when an item was just added to the plot

    It provides the added item.
    """

    sigItemAboutToBeRemoved = qt.Signal(items.Item)
    """Signal emitted right before an item is removed from the plot.

    It provides the item that will be removed.
    """

    sigVisibilityChanged = qt.Signal(bool)
    """Signal emitted when the widget becomes visible (or invisible).
    This happens when the widget is hidden or shown.

    It provides the visible state.
    """

    def __init__(self, parent=None, backend=None,
                 legends=False, callback=None, **kw):
        self._autoreplot = False
        self._dirty = False
        self._cursorInPlot = False
        self.__muteActiveItemChanged = False

        if kw:
            _logger.warning(
                'deprecated: __init__ extra arguments: %s', str(kw))
        if legends:
            _logger.warning('deprecated: __init__ legend argument')
        if callback:
            _logger.warning('deprecated: __init__ callback argument')

        self._panWithArrowKeys = True
        self._viewConstrains = None

        super(PlotWidget, self).__init__(parent)
        if parent is not None:
            # behave as a widget
            self.setWindowFlags(qt.Qt.Widget)
        else:
            self.setWindowTitle('PlotWidget')

        if backend is None:
            backend = silx.config.DEFAULT_PLOT_BACKEND

        if hasattr(backend, "__call__"):
            self._backend = backend(self, parent)

        elif hasattr(backend, "lower"):
            lowerCaseString = backend.lower()
            if lowerCaseString in ("matplotlib", "mpl"):
                backendClass = BackendMatplotlibQt
            elif lowerCaseString in ('gl', 'opengl'):
                from .backends.BackendOpenGL import BackendOpenGL
                backendClass = BackendOpenGL
            elif lowerCaseString == 'none':
                from .backends.BackendBase import BackendBase as backendClass
            else:
                raise ValueError("Backend not supported %s" % backend)
            self._backend = backendClass(self, parent)

        else:
            raise ValueError("Backend not supported %s" % str(backend))

        self.setCallback()  # set _callback

        # Items handling
        self._content = OrderedDict()
        self._contentToUpdate = []  # Used as an OrderedSet

        self._dataRange = None

        # line types
        self._styleList = ['-', '--', '-.', ':']
        self._colorIndex = 0
        self._styleIndex = 0

        self._activeCurveSelectionMode = "atmostone"
        self._activeCurveStyle = CurveStyle(color='#000000')
        self._activeLegend = {'curve': None, 'image': None,
                              'scatter': None}

        # default properties
        self._cursorConfiguration = None

        self._xAxis = items.XAxis(self)
        self._yAxis = items.YAxis(self)
        self._yRightAxis = items.YRightAxis(self, self._yAxis)

        self._grid = None
        self._graphTitle = ''

        self.setGraphTitle()
        self.setGraphXLabel()
        self.setGraphYLabel()
        self.setGraphYLabel('', axis='right')

        self.setDefaultColormap()  # Init default colormap

        self.setDefaultPlotPoints(False)
        self.setDefaultPlotLines(True)

        self._limitsHistory = LimitsHistory(self)

        self._eventHandler = PlotInteraction.PlotInteraction(self)
        self._eventHandler.setInteractiveMode('zoom', color=(0., 0., 0., 1.))

        self._pressedButtons = []  # Currently pressed mouse buttons

        self._defaultDataMargins = (0., 0., 0., 0.)

        # Only activate autoreplot at the end
        # This avoids errors when loaded in Qt designer
        self._dirty = False
        self._autoreplot = True

        widget = self.getWidgetHandle()
        if widget is not None:
            self.setCentralWidget(widget)
        else:
            _logger.info("PlotWidget backend does not support widget")

        self.setFocusPolicy(qt.Qt.StrongFocus)
        self.setFocus(qt.Qt.OtherFocusReason)

        # Set default limits
        self.setGraphXLimits(0., 100.)
        self.setGraphYLimits(0., 100., axis='right')
        self.setGraphYLimits(0., 100., axis='left')

    # TODO: Can be removed for silx 0.10
    @staticmethod
    @deprecation.deprecated(replacement="silx.config.DEFAULT_PLOT_BACKEND", since_version="0.8", skip_backtrace_count=2)
    def setDefaultBackend(backend):
        """Set system wide default plot backend.

        .. versionadded:: 0.6

        :param backend: The backend to use, in:
                        'matplotlib' (default), 'mpl', 'opengl', 'gl', 'none'
                        or a :class:`BackendBase.BackendBase` class
        """
        silx.config.DEFAULT_PLOT_BACKEND = backend

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

        if self._autoreplot and not wasDirty and self.isVisible():
            self._backend.postRedisplay()

    def showEvent(self, event):
        if self._autoreplot and self._dirty:
            self._backend.postRedisplay()
        super(PlotWidget, self).showEvent(event)
        self.sigVisibilityChanged.emit(True)

    def hideEvent(self, event):
        super(PlotWidget, self).hideEvent(event)
        self.sigVisibilityChanged.emit(False)

    def _invalidateDataRange(self):
        """
        Notifies this PlotWidget instance that the range has changed
        and will have to be recomputed.
        """
        self._dataRange = None

    def _updateDataRange(self):
        """
        Recomputes the range of the data displayed on this PlotWidget.
        """
        xMin = yMinLeft = yMinRight = float('nan')
        xMax = yMaxLeft = yMaxRight = float('nan')

        for item in self._content.values():
            if item.isVisible():
                bounds = item.getBounds()
                if bounds is not None:
                    xMin = numpy.nanmin([xMin, bounds[0]])
                    xMax = numpy.nanmax([xMax, bounds[1]])
                    # Take care of right axis
                    if (isinstance(item, items.YAxisMixIn) and
                            item.getYAxis() == 'right'):
                        yMinRight = numpy.nanmin([yMinRight, bounds[2]])
                        yMaxRight = numpy.nanmax([yMaxRight, bounds[3]])
                    else:
                        yMinLeft = numpy.nanmin([yMinLeft, bounds[2]])
                        yMaxLeft = numpy.nanmax([yMaxLeft, bounds[3]])

        def lGetRange(x, y):
            return None if numpy.isnan(x) and numpy.isnan(y) else (x, y)
        xRange = lGetRange(xMin, xMax)
        yLeftRange = lGetRange(yMinLeft, yMaxLeft)
        yRightRange = lGetRange(yMinRight, yMaxRight)

        self._dataRange = _PlotDataRange(x=xRange,
                                         y=yLeftRange,
                                         yright=yRightRange)

    def getDataRange(self):
        """
        Returns this PlotWidget's data range.

        :return: a namedtuple with the following members :
                x, y (left y axis), yright. Each member is a tuple (min, max)
                or None if no data is associated with the axis.
        :rtype: namedtuple
        """
        if self._dataRange is None:
            self._updateDataRange()
        return self._dataRange

    # Content management

    @staticmethod
    def _itemKey(item):
        """Build the key of given :class:`Item` in the plot

        :param Item item: The item to make the key from
        :return: (legend, kind)
        :rtype: (str, str)
        """
        if isinstance(item, items.Curve):
            kind = 'curve'
        elif isinstance(item, items.ImageBase):
            kind = 'image'
        elif isinstance(item, items.Scatter):
            kind = 'scatter'
        elif isinstance(item, (items.Marker,
                               items.XMarker, items.YMarker)):
            kind = 'marker'
        elif isinstance(item, items.Shape):
            kind = 'item'
        elif isinstance(item, items.Histogram):
            kind = 'histogram'
        else:
            raise ValueError('Unsupported item type %s' % type(item))

        return item.getLegend(), kind

    def _add(self, item):
        """Add the given :class:`Item` to the plot.

        :param Item item: The item to append to the plot content
        """
        key = self._itemKey(item)
        if key in self._content:
            raise RuntimeError('Item already in the plot')

        # Add item to plot
        self._content[key] = item
        item._setPlot(self)
        if item.isVisible():
            self._itemRequiresUpdate(item)
        if isinstance(item, items.DATA_ITEMS):
            self._invalidateDataRange()  # TODO handle this automatically

        self._notifyContentChanged(item)
        self.sigItemAdded.emit(item)

    def _notifyContentChanged(self, item):
        legend, kind = self._itemKey(item)
        self.notify('contentChanged', action='add', kind=kind, legend=legend)

    def _remove(self, item):
        """Remove the given :class:`Item` from the plot.

        :param Item item: The item to remove from the plot content
        """
        key = self._itemKey(item)
        if key not in self._content:
            raise RuntimeError('Item not in the plot')

        self.sigItemAboutToBeRemoved.emit(item)

        legend, kind = key

        if kind in self._ACTIVE_ITEM_KINDS:
            if self._getActiveItem(kind) == item:
                # Reset active item
                self._setActiveItem(kind, None)

        # Remove item from plot
        self._content.pop(key)
        if item in self._contentToUpdate:
            self._contentToUpdate.remove(item)
        if item.isVisible():
            self._setDirtyPlot(overlayOnly=item.isOverlay())
        if item.getBounds() is not None:
            self._invalidateDataRange()
        item._removeBackendRenderer(self._backend)
        item._setPlot(None)

        if (kind == 'curve' and not self.getAllCurves(just_legend=True,
                                                      withhidden=True)):
            self._resetColorAndStyle()

        self.notify('contentChanged', action='remove',
                    kind=kind, legend=legend)

    def _itemRequiresUpdate(self, item):
        """Called by items in the plot for asynchronous update

        :param Item item: The item that required update
        """
        assert item.getPlot() == self
        # Pu item at the end of the list
        if item in self._contentToUpdate:
            self._contentToUpdate.remove(item)
        self._contentToUpdate.append(item)
        self._setDirtyPlot(overlayOnly=item.isOverlay())

    @contextmanager
    def _muteActiveItemChangedSignal(self):
        self.__muteActiveItemChanged = True
        yield
        self.__muteActiveItemChanged = False

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
                 fill=None, resetzoom=True,
                 histogram=None, copy=True, **kw):
        """Add a 1D curve given by x an y to the graph.

        Curves are uniquely identified by their legend.
        To add multiple curves, call :meth:`addCurve` multiple times with
        different legend argument.
        To replace an existing curve, call :meth:`addCurve` with the
        existing curve legend.
        If you want to display the curve values as an histogram see the
        histogram parameter or :meth:`addHistogram`.

        When curve parameters are not provided, if a curve with the
        same legend is displayed in the plot, its parameters are used.

        :param numpy.ndarray x: The data corresponding to the x coordinates.
          If you attempt to plot an histogram you can set edges values in x.
          In this case len(x) = len(y) + 1.
          If x contains datetime objects the XAxis tickMode is set to
          TickMode.TIME_SERIES.
        :param numpy.ndarray y: The data corresponding to the y coordinates
        :param str legend: The legend to be associated to the curve (or None)
        :param info: User-defined information associated to the curve
        :param bool replace: True (the default) to delete already existing
                             curves
        :param color: color(s) to be used
        :type color: str ("#RRGGBB") or (npoints, 4) unsigned byte array or
                     one of the predefined color names defined in colors.py
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
        :param str histogram: if not None then the curve will be draw as an
            histogram. The step for each values of the curve can be set to the
            left, center or right of the original x curve values.
            If histogram is not None and len(x) == len(y)+1 then x is directly
            take as edges of the histogram.
            Type of histogram::

            - None (default)
            - 'left'
            - 'right'
            - 'center'
        :param bool copy: True make a copy of the data (default),
                          False to use provided arrays.
        :returns: The key string identify this curve
        """
        # Deprecation warnings
        if replot is not None:
            _logger.warning(
                'addCurve deprecated replot argument, use resetzoom instead')
            resetzoom = replot and resetzoom

        if kw:
            _logger.warning('addCurve: deprecated extra arguments')

        # This is an histogram, use addHistogram
        if histogram is not None:
            histoLegend = self.addHistogram(histogram=y,
                                            edges=x,
                                            legend=legend,
                                            color=color,
                                            fill=fill,
                                            align=histogram,
                                            copy=copy)
            histo = self.getHistogram(histoLegend)

            histo.setInfo(info)
            if linewidth is not None:
                histo.setLineWidth(linewidth)
            if linestyle is not None:
                histo.setLineStyle(linestyle)
            if xlabel is not None:
                _logger.warning(
                    'addCurve: Histogram does not support xlabel argument')
            if ylabel is not None:
                _logger.warning(
                    'addCurve: Histogram does not support ylabel argument')
            if yaxis is not None:
                histo.setYAxis(yaxis)
            if z is not None:
                histo.setZValue(z)
            if selectable is not None:
                _logger.warning(
                    'addCurve: Histogram does not support selectable argument')

            return

        legend = 'Unnamed curve 1.1' if legend is None else str(legend)

        # Check if curve was previously active
        wasActive = self.getActiveCurve(just_legend=True) == legend

        if replace:
            self._resetColorAndStyle()

        # Create/Update curve object
        curve = self.getCurve(legend)
        mustBeAdded = curve is None
        if curve is None:
            # No previous curve, create a default one and add it to the plot
            curve = items.Curve() if histogram is None else items.Histogram()
            curve._setLegend(legend)
            # Set default color, linestyle and symbol
            default_color, default_linestyle = self._getColorAndStyle()
            curve.setColor(default_color)
            curve.setLineStyle(default_linestyle)
            curve.setSymbol(self._defaultPlotPoints)

        # Do not emit sigActiveCurveChanged,
        # it will be sent once with _setActiveItem
        with self._muteActiveItemChangedSignal():
            # Override previous/default values with provided ones
            curve.setInfo(info)
            if color is not None:
                curve.setColor(color)
            if symbol is not None:
                curve.setSymbol(symbol)
            if linewidth is not None:
                curve.setLineWidth(linewidth)
            if linestyle is not None:
                curve.setLineStyle(linestyle)
            if xlabel is not None:
                curve._setXLabel(xlabel)
            if ylabel is not None:
                curve._setYLabel(ylabel)
            if yaxis is not None:
                curve.setYAxis(yaxis)
            if z is not None:
                curve.setZValue(z)
            if selectable is not None:
                curve._setSelectable(selectable)
            if fill is not None:
                curve.setFill(fill)

            # Set curve data
            # If errors not provided, reuse previous ones
            # TODO: Issue if size of data change but not that of errors
            if xerror is None:
                xerror = curve.getXErrorData(copy=False)
            if yerror is None:
                yerror = curve.getYErrorData(copy=False)

            # Convert x to timestamps so that the internal representation
            # remains floating points. The user is expected to set the axis'
            # tickMode to TickMode.TIME_SERIES and, if necessary, set the axis
            # to the correct time zone.
            if len(x) > 0 and isinstance(x[0], dt.datetime):
                x = [timestamp(d) for d in x]

            curve.setData(x, y, xerror, yerror, copy=copy)

        if replace:  # Then remove all other curves
            for c in self.getAllCurves(withhidden=True):
                if c is not curve:
                    self._remove(c)

        if mustBeAdded:
            self._add(curve)
        else:
            self._notifyContentChanged(curve)

        if wasActive:
            self.setActiveCurve(curve.getLegend())
        elif self.getActiveCurveSelectionMode() == "legacy":
            if self.getActiveCurve(just_legend=True) is None:
                if len(self.getAllCurves(just_legend=True,
                                     withhidden=False)) == 1:
                    if curve.isVisible():
                        self.setActiveCurve(curve.getLegend())

        if resetzoom:
            # We ask for a zoom reset in order to handle the plot scaling
            # if the user does not want that, autoscale of the different
            # axes has to be set to off.
            self.resetZoom()

        return legend

    def addHistogram(self,
                     histogram,
                     edges,
                     legend=None,
                     color=None,
                     fill=None,
                     align='center',
                     resetzoom=True,
                     copy=True):
        """Add an histogram to the graph.

        This is NOT computing the histogram, this method takes as parameter
        already computed histogram values.

        Histogram are uniquely identified by their legend.
        To add multiple histograms, call :meth:`addHistogram` multiple times
        with different legend argument.

        When histogram parameters are not provided, if an histogram with the
        same legend is displayed in the plot, its parameters are used.

        :param numpy.ndarray histogram: The values of the histogram.
        :param numpy.ndarray edges:
            The bin edges of the histogram.
            If histogram and edges have the same length, the bin edges
            are computed according to the align parameter.
        :param str legend:
            The legend to be associated to the histogram (or None)
        :param color: color to be used
        :type color: str ("#RRGGBB") or RGB unsigned byte array or
                     one of the predefined color names defined in colors.py
        :param bool fill: True to fill the curve, False otherwise (default).
        :param str align:
            In case histogram values and edges have the same length N,
            the N+1 bin edges are computed according to the alignment in:
            'center' (default), 'left', 'right'.
        :param bool resetzoom: True (the default) to reset the zoom.
        :param bool copy: True make a copy of the data (default),
                          False to use provided arrays.
        :returns: The key string identify this histogram
        """
        legend = 'Unnamed histogram' if legend is None else str(legend)

        # Create/Update histogram object
        histo = self.getHistogram(legend)
        mustBeAdded = histo is None
        if histo is None:
            # No previous histogram, create a default one and
            # add it to the plot
            histo = items.Histogram()
            histo._setLegend(legend)
            histo.setColor(self._getColorAndStyle()[0])

        # Override previous/default values with provided ones
        if color is not None:
            histo.setColor(color)
        if fill is not None:
            histo.setFill(fill)

        # Set histogram data
        histo.setData(histogram, edges, align=align, copy=copy)

        if mustBeAdded:
            self._add(histo)
        else:
            self._notifyContentChanged(histo)

        if resetzoom:
            # We ask for a zoom reset in order to handle the plot scaling
            # if the user does not want that, autoscale of the different
            # axes has to be set to off.
            self.resetZoom()

        return legend

    def addImage(self, data, legend=None, info=None,
                 replace=False, replot=None,
                 xScale=None, yScale=None, z=None,
                 selectable=None, draggable=None,
                 colormap=None, pixmap=None,
                 xlabel=None, ylabel=None,
                 origin=None, scale=None,
                 resetzoom=True, copy=True, **kw):
        """Add a 2D dataset or an image to the plot.

        It displays either an array of data using a colormap or a RGB(A) image.

        Images are uniquely identified by their legend.
        To add multiple images, call :meth:`addImage` multiple times with
        different legend argument.
        To replace/update an existing image, call :meth:`addImage` with the
        existing image legend.

        When image parameters are not provided, if an image with the
        same legend is displayed in the plot, its parameters are used.

        :param numpy.ndarray data:
            (nrows, ncolumns) data or
            (nrows, ncolumns, RGBA) ubyte array
            Note: boolean values are converted to int8.
        :param str legend: The legend to be associated to the image (or None)
        :param info: User-defined information associated to the image
        :param bool replace:
            True to delete already existing images (Default: False).
        :param int z: Layer on which to draw the image (default: 0)
                      This allows to control the overlay.
        :param bool selectable: Indicate if the image can be selected.
                                (default: False)
        :param bool draggable: Indicate if the image can be moved.
                               (default: False)
        :param colormap: Colormap object to use (or None).
                         This is ignored if data is a RGB(A) image.
        :type colormap: Union[~silx.gui.colors.Colormap, dict]
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
        :param bool copy: True make a copy of the data (default),
                          False to use provided arrays.
        :returns: The key string identify this image
        """
        # Deprecation warnings
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

        # Check if image was previously active
        wasActive = self.getActiveImage(just_legend=True) == legend

        data = numpy.array(data, copy=False)
        assert data.ndim in (2, 3)

        image = self.getImage(legend)
        if image is not None and image.getData(copy=False).ndim != data.ndim:
            # Update a data image with RGBA image or the other way around:
            # Remove previous image
            # In this case, we don't retrieve defaults from the previous image
            self._remove(image)
            image = None

        mustBeAdded = image is None
        if image is None:
            # No previous image, create a default one and add it to the plot
            if data.ndim == 2:
                image = items.ImageData()
                image.setColormap(self.getDefaultColormap())
            else:
                image = items.ImageRgba()
            image._setLegend(legend)

        # Do not emit sigActiveImageChanged,
        # it will be sent once with _setActiveItem
        with self._muteActiveItemChangedSignal():
            # Override previous/default values with provided ones
            image.setInfo(info)
            if origin is not None:
                image.setOrigin(origin)
            if scale is not None:
                image.setScale(scale)
            if z is not None:
                image.setZValue(z)
            if selectable is not None:
                image._setSelectable(selectable)
            if draggable is not None:
                image._setDraggable(draggable)
            if colormap is not None and isinstance(image, items.ColormapMixIn):
                if isinstance(colormap, dict):
                    image.setColormap(Colormap._fromDict(colormap))
                else:
                    assert isinstance(colormap, Colormap)
                    image.setColormap(colormap)
            if xlabel is not None:
                image._setXLabel(xlabel)
            if ylabel is not None:
                image._setYLabel(ylabel)

            if data.ndim == 2:
                image.setData(data, alternative=pixmap, copy=copy)
            else:  # RGB(A) image
                if pixmap is not None:
                    _logger.warning(
                        'addImage: pixmap argument ignored when data is RGB(A)')
                image.setData(data, copy=copy)

        if replace:
            for img in self.getAllImages():
                if img is not image:
                    self._remove(img)

        if mustBeAdded:
            self._add(image)
        else:
            self._notifyContentChanged(image)

        if len(self.getAllImages()) == 1 or wasActive:
            self.setActiveImage(legend)

        if resetzoom:
            # We ask for a zoom reset in order to handle the plot scaling
            # if the user does not want that, autoscale of the different
            # axes has to be set to off.
            self.resetZoom()

        return legend

    def addScatter(self, x, y, value, legend=None, colormap=None,
                   info=None, symbol=None, xerror=None, yerror=None,
                   z=None, copy=True):
        """Add a (x, y, value) scatter to the graph.

        Scatters are uniquely identified by their legend.
        To add multiple scatters, call :meth:`addScatter` multiple times with
        different legend argument.
        To replace/update an existing scatter, call :meth:`addScatter` with the
        existing scatter legend.

        When scatter parameters are not provided, if a scatter with the
        same legend is displayed in the plot, its parameters are used.

        :param numpy.ndarray x: The data corresponding to the x coordinates.
        :param numpy.ndarray y: The data corresponding to the y coordinates
        :param numpy.ndarray value: The data value associated with each point
        :param str legend: The legend to be associated to the scatter (or None)
        :param ~silx.gui.colors.Colormap colormap:
            Colormap object to be used for the scatter (or None)
        :param info: User-defined information associated to the curve
        :param str symbol: Symbol to be drawn at each (x, y) position::

            - 'o' circle
            - '.' point
            - ',' pixel
            - '+' cross
            - 'x' x-cross
            - 'd' diamond
            - 's' square
            - None (the default) to use default symbol

        :param xerror: Values with the uncertainties on the x values
        :type xerror: A float, or a numpy.ndarray of float32.
                      If it is an array, it can either be a 1D array of
                      same length as the data or a 2D array with 2 rows
                      of same length as the data: row 0 for positive errors,
                      row 1 for negative errors.
        :param yerror: Values with the uncertainties on the y values
        :type yerror: A float, or a numpy.ndarray of float32. See xerror.
        :param int z: Layer on which to draw the scatter (default: 1)
                      This allows to control the overlay.

        :param bool copy: True make a copy of the data (default),
                          False to use provided arrays.
        :returns: The key string identify this scatter
        """
        legend = 'Unnamed scatter 1.1' if legend is None else str(legend)

        # Check if scatter was previously active
        wasActive = self._getActiveItem(kind='scatter',
                                        just_legend=True) == legend

        # Create/Update curve object
        scatter = self._getItem(kind='scatter', legend=legend)
        mustBeAdded = scatter is None
        if scatter is None:
            # No previous scatter, create a default one and add it to the plot
            scatter = items.Scatter()
            scatter._setLegend(legend)
            scatter.setColormap(self.getDefaultColormap())

        # Do not emit sigActiveScatterChanged,
        # it will be sent once with _setActiveItem
        with self._muteActiveItemChangedSignal():
            # Override previous/default values with provided ones
            scatter.setInfo(info)
            if symbol is not None:
                scatter.setSymbol(symbol)
            if z is not None:
                scatter.setZValue(z)
            if colormap is not None:
                if isinstance(colormap, dict):
                    scatter.setColormap(Colormap._fromDict(colormap))
                else:
                    assert isinstance(colormap, Colormap)
                    scatter.setColormap(colormap)

            # Set scatter data
            # If errors not provided, reuse previous ones
            if xerror is None:
                xerror = scatter.getXErrorData(copy=False)
                if xerror is not None and len(xerror) != len(x):
                    xerror = None
            if yerror is None:
                yerror = scatter.getYErrorData(copy=False)
                if yerror is not None and len(yerror) != len(y):
                    yerror = None

            scatter.setData(x, y, value, xerror, yerror, copy=copy)

        if mustBeAdded:
            self._add(scatter)
        else:
            self._notifyContentChanged(scatter)

        if len(self._getItems(kind="scatter")) == 1 or wasActive:
            self._setActiveItem('scatter', scatter.getLegend())

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

        item = items.Shape(shape)
        item._setLegend(legend)
        item.setInfo(info)
        item.setColor(color)
        item.setFill(fill)
        item.setOverlay(overlay)
        item.setZValue(z)
        item.setPoints(numpy.array((xdata, ydata)).T)

        self._add(item)

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
            xmin, xmax = self._xAxis.getLimits()
            x = 0.5 * (xmax + xmin)

        if y is None:
            ymin, ymax = self._yAxis.getLimits()
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
        assert (x, y) != (None, None)

        if legend is None:  # Find an unused legend
            markerLegends = self._getAllMarkers(just_legend=True)
            for index in itertools.count():
                legend = "Unnamed Marker %d" % index
                if legend not in markerLegends:
                    break  # Keep this legend
        legend = str(legend)

        if x is None:
            markerClass = items.YMarker
        elif y is None:
            markerClass = items.XMarker
        else:
            markerClass = items.Marker

        # Create/Update marker object
        marker = self._getMarker(legend)
        if marker is not None and not isinstance(marker, markerClass):
            _logger.warning('Adding marker with same legend'
                            ' but different type replaces it')
            self._remove(marker)
            marker = None

        mustBeAdded = marker is None
        if marker is None:
            # No previous marker, create one
            marker = markerClass()
            marker._setLegend(legend)

        if text is not None:
            marker.setText(text)
        if color is not None:
            marker.setColor(color)
        if selectable is not None:
            marker._setSelectable(selectable)
        if draggable is not None:
            marker._setDraggable(draggable)
        if symbol is not None:
            marker.setSymbol(symbol)

        # TODO to improve, but this ensure constraint is applied
        marker.setPosition(x, y)
        if constraint is not None:
            marker._setConstraint(constraint)
        marker.setPosition(x, y)

        if mustBeAdded:
            self._add(marker)
        else:
            self._notifyContentChanged(marker)

        return legend

    # Hide

    def isCurveHidden(self, legend):
        """Returns True if the curve associated to legend is hidden, else False

        :param str legend: The legend key identifying the curve
        :return: True if the associated curve is hidden, False otherwise
        """
        curve = self._getItem('curve', legend)
        return curve is not None and not curve.isVisible()

    def hideCurve(self, legend, flag=True, replot=None):
        """Show/Hide the curve associated to legend.

        Even when hidden, the curve is kept in the list of curves.

        :param str legend: The legend associated to the curve to be hidden
        :param bool flag: True (default) to hide the curve, False to show it
        """
        if replot is not None:
            _logger.warning('hideCurve deprecated replot parameter')

        curve = self._getItem('curve', legend)
        if curve is None:
            _logger.warning('Curve not in plot: %s', legend)
            return

        isVisible = not flag
        if isVisible != curve.isVisible():
            curve.setVisible(isVisible)

    # Remove

    ITEM_KINDS = 'curve', 'image', 'scatter', 'item', 'marker', 'histogram'
    """List of supported kind of items in the plot."""

    _ACTIVE_ITEM_KINDS = 'curve', 'scatter', 'image'
    """List of item's kind which have a active item."""

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
                     See :attr:`ITEM_KINDS`.
                     By default, it removes all kind of elements.
        :type kind: str or tuple of str to specify multiple kinds.
        """
        if kind is 'all':  # Replace all by tuple of all kinds
            kind = self.ITEM_KINDS

        if kind in self.ITEM_KINDS:  # Kind is a str, make it a tuple
            kind = (kind,)

        for aKind in kind:
            assert aKind in self.ITEM_KINDS

        if legend is None:  # This is a clear
            # Clear each given kind
            for aKind in kind:
                for legend in self._getItems(
                        kind=aKind, just_legend=True, withhidden=True):
                    self.remove(legend=legend, kind=aKind)

        else:  # This is removing a single element
            # Remove each given kind
            for aKind in kind:
                item = self._getItem(aKind, legend)
                if item is not None:
                    self._remove(item)

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
        :type color: A string (either a predefined color name in colors.py
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
            xMin, xMax = self._xAxis.getLimits()

            xMin, xMax = _utils.applyPan(xMin, xMax, xFactor,
                                         self._xAxis.getScale() == self._xAxis.LOGARITHMIC)
            self._xAxis.setLimits(xMin, xMax)

        else:  # direction in ('up', 'down')
            sign = -1. if self._yAxis.isInverted() else 1.
            yFactor = sign * (factor if direction == 'up' else -factor)
            yMin, yMax = self._yAxis.getLimits()
            yIsLog = self._yAxis.getScale() == self._yAxis.LOGARITHMIC

            yMin, yMax = _utils.applyPan(yMin, yMax, yFactor, yIsLog)
            self._yAxis.setLimits(yMin, yMax)

            y2Min, y2Max = self._yRightAxis.getLimits()

            y2Min, y2Max = _utils.applyPan(y2Min, y2Max, yFactor, yIsLog)
            self._yRightAxis.setLimits(y2Min, y2Max)

    # Active Curve/Image

    def isActiveCurveHandling(self):
        """Returns True if active curve selection is enabled.

        :rtype: bool
        """
        return self.getActiveCurveSelectionMode() != 'none'

    def setActiveCurveHandling(self, flag=True):
        """Enable/Disable active curve selection.

        :param bool flag: True to enable 'atmostone' active curve selection,
            False to disable active curve selection.
        """
        self.setActiveCurveSelectionMode('atmostone' if flag else 'none')

    def getActiveCurveStyle(self):
        """Returns the current style applied to active curve

        :rtype: CurveStyle
        """
        return self._activeCurveStyle

    def setActiveCurveStyle(self,
                            color=None,
                            linewidth=None,
                            linestyle=None,
                            symbol=None,
                            symbolsize=None):
        """Set the style of active curve

        :param color: Color
        :param Union[str,None] linestyle: Style of the line
        :param Union[float,None] linewidth: Width of the line
        :param Union[str,None] symbol: Symbol of the markers
        :param Union[float,None] symbolsize: Size of the symbols
        """
        self._activeCurveStyle = CurveStyle(color=color,
                                            linewidth=linewidth,
                                            linestyle=linestyle,
                                            symbol=symbol,
                                            symbolsize=symbolsize)
        curve = self.getActiveCurve()
        if curve is not None:
            curve.setHighlightedStyle(self.getActiveCurveStyle())

    @deprecated(replacement="getActiveCurveStyle", since_version="0.9")
    def getActiveCurveColor(self):
        """Get the color used to display the currently active curve.

        See :meth:`setActiveCurveColor`.
        """
        return self._activeCurveStyle.getColor()

    @deprecated(replacement="setActiveCurveStyle", since_version="0.9")
    def setActiveCurveColor(self, color="#000000"):
        """Set the color to use to display the currently active curve.

        :param str color: Color of the active curve,
                          e.g., 'blue', 'b', '#FF0000' (Default: 'black')
        """
        if color is None:
            color = "black"
        if color in self.colorDict:
            color = self.colorDict[color]
        self.setActiveCurveStyle(color=color)

    def getActiveCurve(self, just_legend=False):
        """Return the currently active curve.

        It returns None in case of not having an active curve.

        :param bool just_legend: True to get the legend of the curve,
                                 False (the default) to get the curve data
                                 and info.
        :return: Active curve's legend or corresponding
                 :class:`.items.Curve`
        :rtype: str or :class:`.items.Curve` or None
        """
        if not self.isActiveCurveHandling():
            return None

        return self._getActiveItem(kind='curve', just_legend=just_legend)

    def setActiveCurve(self, legend, replot=None):
        """Make the curve associated to legend the active curve.

        :param legend: The legend associated to the curve
                       or None to have no active curve.
        :type legend: str or None
        """
        if replot is not None:
            _logger.warning('setActiveCurve deprecated replot parameter')

        if not self.isActiveCurveHandling():
            return
        if legend is None and self.getActiveCurveSelectionMode() == "legacy":
            _logger.info(
                'setActiveCurve(None) ignored due to active curve selection mode')
            return

        return self._setActiveItem(kind='curve', legend=legend)

    def setActiveCurveSelectionMode(self, mode):
        """Sets the current selection mode.

        :param str mode: The active curve selection mode to use.
            It can be: 'legacy', 'atmostone' or 'none'.
        """
        assert mode in ('legacy', 'atmostone', 'none')

        if mode != self._activeCurveSelectionMode:
            self._activeCurveSelectionMode = mode
            if mode == 'none':  # reset active curve
                self._setActiveItem(kind='curve', legend=None)

            elif mode == 'legacy' and self.getActiveCurve() is None:
                # Select an active curve
                curves = self.getAllCurves(just_legend=False,
                                           withhidden=False)
                if len(curves) == 1:
                    if curves[0].isVisible():
                        self.setActiveCurve(curves[0].getLegend())

    def getActiveCurveSelectionMode(self):
        """Returns the current selection mode.

        It can be "atmostone", "legacy" or "none".

        :rtype: str
        """
        return self._activeCurveSelectionMode

    def getActiveImage(self, just_legend=False):
        """Returns the currently active image.

        It returns None in case of not having an active image.

        :param bool just_legend: True to get the legend of the image,
                                 False (the default) to get the image data
                                 and info.
        :return: Active image's legend or corresponding image object
        :rtype: str, :class:`.items.ImageData`, :class:`.items.ImageRgba`
                or None
        """
        return self._getActiveItem(kind='image', just_legend=just_legend)

    def setActiveImage(self, legend, replot=None):
        """Make the image associated to legend the active image.

        :param str legend: The legend associated to the image
                           or None to have no active image.
        """
        if replot is not None:
            _logger.warning('setActiveImage deprecated replot parameter')

        return self._setActiveItem(kind='image', legend=legend)

    def _getActiveItem(self, kind, just_legend=False):
        """Return the currently active item of that kind if any

        :param str kind: Type of item: 'curve', 'scatter' or 'image'
        :param bool just_legend: True to get the legend,
                                 False (default) to get the item
        :return: legend or item or None if no active item
        """
        assert kind in self._ACTIVE_ITEM_KINDS

        if self._activeLegend[kind] is None:
            return None

        if (self._activeLegend[kind], kind) not in self._content:
            self._activeLegend[kind] = None
            return None

        if just_legend:
            return self._activeLegend[kind]
        else:
            return self._getItem(kind, self._activeLegend[kind])

    def _setActiveItem(self, kind, legend):
        """Make the curve associated to legend the active curve.

        :param str kind: Type of item: 'curve' or 'image'
        :param legend: The legend associated to the curve
                       or None to have no active curve.
        :type legend: str or None
        """
        assert kind in self._ACTIVE_ITEM_KINDS

        xLabel = None
        yLabel = None
        yRightLabel = None

        oldActiveItem = self._getActiveItem(kind=kind)

        if oldActiveItem is not None:  # Stop listening previous active image
            oldActiveItem.sigItemChanged.disconnect(self._activeItemChanged)

        # Curve specific: Reset highlight of previous active curve
        if kind == 'curve' and oldActiveItem is not None:
            oldActiveItem.setHighlighted(False)

        if legend is None:
            self._activeLegend[kind] = None
        else:
            legend = str(legend)
            item = self._getItem(kind, legend)
            if item is None:
                _logger.warning("This %s does not exist: %s", kind, legend)
                self._activeLegend[kind] = None
            else:
                self._activeLegend[kind] = legend

                # Curve specific: handle highlight
                if kind == 'curve':
                    item.setHighlightedStyle(self.getActiveCurveStyle())
                    item.setHighlighted(True)

                if isinstance(item, items.LabelsMixIn):
                    if item.getXLabel() is not None:
                        xLabel = item.getXLabel()
                    if item.getYLabel() is not None:
                        if (isinstance(item, items.YAxisMixIn) and
                                item.getYAxis() == 'right'):
                            yRightLabel = item.getYLabel()
                        else:
                            yLabel = item.getYLabel()

                # Start listening new active item
                item.sigItemChanged.connect(self._activeItemChanged)

        # Store current labels and update plot
        self._xAxis._setCurrentLabel(xLabel)
        self._yAxis._setCurrentLabel(yLabel)
        self._yRightAxis._setCurrentLabel(yRightLabel)

        self._setDirtyPlot()

        activeLegend = self._activeLegend[kind]
        if oldActiveItem is not None or activeLegend is not None:
            if oldActiveItem is None:
                oldActiveLegend = None
            else:
                oldActiveLegend = oldActiveItem.getLegend()
            self.notify(
                'active' + kind[0].upper() + kind[1:] + 'Changed',
                updated=oldActiveLegend != activeLegend,
                previous=oldActiveLegend,
                legend=activeLegend)

        return activeLegend

    def _activeItemChanged(self, type_):
        """Listen for active item changed signal and broadcast signal

        :param item.ItemChangedType type_: The type of item change
        """
        if not self.__muteActiveItemChanged:
            item = self.sender()
            if item is not None:
                legend, kind = self._itemKey(item)
                self.notify(
                    'active' + kind[0].upper() + kind[1:] + 'Changed',
                    updated=False,
                    previous=legend,
                    legend=legend)

    # Getters

    def getItems(self):
        """Returns the list of items in the plot

        :rtype: List[silx.gui.plot.items.Item]
        """
        return tuple(self._content.values())

    def getAllCurves(self, just_legend=False, withhidden=False):
        """Returns all curves legend or info and data.

        It returns an empty list in case of not having any curve.

        If just_legend is False, it returns a list of :class:`items.Curve`
        objects describing the curves.
        If just_legend is True, it returns a list of curves' legend.

        :param bool just_legend: True to get the legend of the curves,
                                 False (the default) to get the curves' data
                                 and info.
        :param bool withhidden: False (default) to skip hidden curves.
        :return: list of curves' legend or :class:`.items.Curve`
        :rtype: list of str or list of :class:`.items.Curve`
        """
        return self._getItems(kind='curve',
                              just_legend=just_legend,
                              withhidden=withhidden)

    def getCurve(self, legend=None):
        """Get the object describing a specific curve.

        It returns None in case no matching curve is found.

        :param str legend:
            The legend identifying the curve.
            If not provided or None (the default), the active curve is returned
            or if there is no active curve, the latest updated curve that is
            not hidden is returned if there are curves in the plot.
        :return: None or :class:`.items.Curve` object
        """
        return self._getItem(kind='curve', legend=legend)

    def getAllImages(self, just_legend=False):
        """Returns all images legend or objects.

        It returns an empty list in case of not having any image.

        If just_legend is False, it returns a list of :class:`items.ImageBase`
        objects describing the images.
        If just_legend is True, it returns a list of legends.

        :param bool just_legend: True to get the legend of the images,
                                 False (the default) to get the images'
                                 object.
        :return: list of images' legend or :class:`.items.ImageBase`
        :rtype: list of str or list of :class:`.items.ImageBase`
        """
        return self._getItems(kind='image',
                              just_legend=just_legend,
                              withhidden=True)

    def getImage(self, legend=None):
        """Get the object describing a specific image.

        It returns None in case no matching image is found.

        :param str legend:
            The legend identifying the image.
            If not provided or None (the default), the active image is returned
            or if there is no active image, the latest updated image
            is returned if there are images in the plot.
        :return: None or :class:`.items.ImageBase` object
        """
        return self._getItem(kind='image', legend=legend)

    def getScatter(self, legend=None):
        """Get the object describing a specific scatter.

        It returns None in case no matching scatter is found.

        :param str legend:
            The legend identifying the scatter.
            If not provided or None (the default), the active scatter is
            returned or if there is no active scatter, the latest updated
            scatter is returned if there are scatters in the plot.
        :return: None or :class:`.items.Scatter` object
        """
        return self._getItem(kind='scatter', legend=legend)

    def getHistogram(self, legend=None):
        """Get the object describing a specific histogram.

        It returns None in case no matching histogram is found.

        :param str legend:
            The legend identifying the histogram.
            If not provided or None (the default), the latest updated scatter
            is returned if there are histograms in the plot.
        :return: None or :class:`.items.Histogram` object
        """
        return self._getItem(kind='histogram', legend=legend)

    def _getItems(self, kind=ITEM_KINDS, just_legend=False, withhidden=False):
        """Retrieve all items of a kind in the plot

        :param kind: The kind of elements to retrieve from the plot.
                     See :attr:`ITEM_KINDS`.
                     By default, it removes all kind of elements.
        :type kind: str or tuple of str to specify multiple kinds.
        :param str kind: Type of item: 'curve' or 'image'
        :param bool just_legend: True to get the legend of the curves,
                                 False (the default) to get the curves' data
                                 and info.
        :param bool withhidden: False (default) to skip hidden curves.
        :return: list of legends or item objects
        """
        if kind is 'all':  # Replace all by tuple of all kinds
            kind = self.ITEM_KINDS

        if kind in self.ITEM_KINDS:  # Kind is a str, make it a tuple
            kind = (kind,)

        for aKind in kind:
            assert aKind in self.ITEM_KINDS

        output = []
        for (legend, type_), item in self._content.items():
            if type_ in kind and (withhidden or item.isVisible()):
                output.append(legend if just_legend else item)
        return output

    def _getItem(self, kind, legend=None):
        """Get an item from the plot: either an image or a curve.

        Returns None if no match found.

        :param str kind: Type of item to retrieve,
                         see :attr:`ITEM_KINDS`.
        :param str legend: Legend of the item or
                           None to get active or last item
        :return: Object describing the item or None
        """
        assert kind in self.ITEM_KINDS

        if legend is not None:
            return self._content.get((legend, kind), None)
        else:
            if kind in self._ACTIVE_ITEM_KINDS:
                item = self._getActiveItem(kind=kind)
                if item is not None:  # Return active item if available
                    return item
            # Return last visible item if any
            allItems = self._getItems(
                kind=kind, just_legend=False, withhidden=False)
            return allItems[-1] if allItems else None

    # Limits

    def _notifyLimitsChanged(self, emitSignal=True):
        """Send an event when plot area limits are changed."""
        xRange = self._xAxis.getLimits()
        yRange = self._yAxis.getLimits()
        y2Range = self._yRightAxis.getLimits()
        if emitSignal:
            axes = self.getXAxis(), self.getYAxis(), self.getYAxis(axis="right")
            ranges = xRange, yRange, y2Range
            for axis, limits in zip(axes, ranges):
                axis.sigLimitsChanged.emit(*limits)
        event = PlotEvents.prepareLimitsChangedSignal(
            id(self.getWidgetHandle()), xRange, yRange, y2Range)
        self.notify(**event)

    def getLimitsHistory(self):
        """Returns the object handling the history of limits of the plot"""
        return self._limitsHistory

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
        self._xAxis.setLimits(xmin, xmax)

    def getGraphYLimits(self, axis='left'):
        """Get the graph Y limits.

        :param str axis: The axis for which to get the limits:
                         Either 'left' or 'right'
        :return: Minimum and maximum values of the X axis
        """
        assert axis in ('left', 'right')
        yAxis = self._yAxis if axis == 'left' else self._yRightAxis
        return yAxis.getLimits()

    def setGraphYLimits(self, ymin, ymax, axis='left', replot=None):
        """Set the graph Y limits.

        :param float ymin: minimum bottom axis value
        :param float ymax: maximum bottom axis value
        :param str axis: The axis for which to get the limits:
                         Either 'left' or 'right'
        """
        if replot is not None:
            _logger.warning('setGraphYLimits deprecated replot parameter')
        assert axis in ('left', 'right')
        yAxis = self._yAxis if axis == 'left' else self._yRightAxis
        return yAxis.setLimits(ymin, ymax)

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
        axis = self.getXAxis()
        xmin, xmax = axis._checkLimits(xmin, xmax)
        axis = self.getYAxis()
        ymin, ymax = axis._checkLimits(ymin, ymax)

        if y2min is None or y2max is None:
            # if one limit is None, both are ignored
            y2min, y2max = None, None
        else:
            axis = self.getYAxis(axis="right")
            y2min, y2max = axis._checkLimits(y2min, y2max)

        if self._viewConstrains:
            view = self._viewConstrains.normalize(xmin, xmax, ymin, ymax)
            xmin, xmax, ymin, ymax = view

        self._backend.setLimits(xmin, xmax, ymin, ymax, y2min, y2max)
        self._setDirtyPlot()
        self._notifyLimitsChanged()

    def _getViewConstraints(self):
        """Return the plot object managing constaints on the plot view.

        :rtype: ViewConstraints
        """
        if self._viewConstrains is None:
            self._viewConstrains = ViewConstraints()
        return self._viewConstrains

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
        return self._xAxis.getLabel()

    def setGraphXLabel(self, label="X"):
        """Set the plot X axis label.

        The provided label can be temporarily replaced by the X label of the
        active curve if any.

        :param str label: The X axis label (default: 'X')
        """
        self._xAxis.setLabel(label)

    def getGraphYLabel(self, axis='left'):
        """Return the current Y axis label as a str.

        :param str axis: The Y axis for which to get the label (left or right)
        """
        assert axis in ('left', 'right')
        yAxis = self._yAxis if axis == 'left' else self._yRightAxis
        return yAxis.getLabel()

    def setGraphYLabel(self, label="Y", axis='left'):
        """Set the plot Y axis label.

        The provided label can be temporarily replaced by the Y label of the
        active curve if any.

        :param str label: The Y axis label (default: 'Y')
        :param str axis: The Y axis for which to set the label (left or right)
        """
        assert axis in ('left', 'right')
        yAxis = self._yAxis if axis == 'left' else self._yRightAxis
        return yAxis.setLabel(label)

    # Axes

    def getXAxis(self):
        """Returns the X axis

        .. versionadded:: 0.6

        :rtype: :class:`.items.Axis`
        """
        return self._xAxis

    def getYAxis(self, axis="left"):
        """Returns an Y axis

        .. versionadded:: 0.6

        :param str axis: The Y axis to return
                         ('left' or 'right').
        :rtype: :class:`.items.Axis`
        """
        assert(axis in ["left", "right"])
        return self._yAxis if axis == "left" else self._yRightAxis

    def setAxesDisplayed(self, displayed):
        """Display or not the axes.

        :param bool displayed: If `True` axes are displayed. If `False` axes
            are not anymore visible and the margin used for them is removed.
        """
        self._backend.setAxesDisplayed(displayed)
        self._setDirtyPlot()
        self._sigAxesVisibilityChanged.emit(displayed)

    def _isAxesDisplayed(self):
        return self._backend.isAxesDisplayed()

    @property
    @deprecated(since_version='0.6')
    def sigSetYAxisInverted(self):
        """Signal emitted when Y axis orientation has changed"""
        return self._yAxis.sigInvertedChanged

    @property
    @deprecated(since_version='0.6')
    def sigSetXAxisLogarithmic(self):
        """Signal emitted when X axis scale has changed"""
        return self._xAxis._sigLogarithmicChanged

    @property
    @deprecated(since_version='0.6')
    def sigSetYAxisLogarithmic(self):
        """Signal emitted when Y axis scale has changed"""
        return self._yAxis._sigLogarithmicChanged

    @property
    @deprecated(since_version='0.6')
    def sigSetXAxisAutoScale(self):
        """Signal emitted when X axis autoscale has changed"""
        return self._xAxis.sigAutoScaleChanged

    @property
    @deprecated(since_version='0.6')
    def sigSetYAxisAutoScale(self):
        """Signal emitted when Y axis autoscale has changed"""
        return self._yAxis.sigAutoScaleChanged

    def setYAxisInverted(self, flag=True):
        """Set the Y axis orientation.

        :param bool flag: True for Y axis going from top to bottom,
                          False for Y axis going from bottom to top
        """
        self._yAxis.setInverted(flag)

    def isYAxisInverted(self):
        """Return True if Y axis goes from top to bottom, False otherwise."""
        return self._yAxis.isInverted()

    def isXAxisLogarithmic(self):
        """Return True if X axis scale is logarithmic, False if linear."""
        return self._xAxis._isLogarithmic()

    def setXAxisLogarithmic(self, flag):
        """Set the bottom X axis scale (either linear or logarithmic).

        :param bool flag: True to use a logarithmic scale, False for linear.
        """
        self._xAxis._setLogarithmic(flag)

    def isYAxisLogarithmic(self):
        """Return True if Y axis scale is logarithmic, False if linear."""
        return self._yAxis._isLogarithmic()

    def setYAxisLogarithmic(self, flag):
        """Set the Y axes scale (either linear or logarithmic).

        :param bool flag: True to use a logarithmic scale, False for linear.
        """
        self._yAxis._setLogarithmic(flag)

    def isXAxisAutoScale(self):
        """Return True if X axis is automatically adjusting its limits."""
        return self._xAxis.isAutoScale()

    def setXAxisAutoScale(self, flag=True):
        """Set the X axis limits adjusting behavior of :meth:`resetZoom`.

        :param bool flag: True to resize limits automatically,
                          False to disable it.
        """
        self._xAxis.setAutoScale(flag)

    def isYAxisAutoScale(self):
        """Return True if Y axes are automatically adjusting its limits."""
        return self._yAxis.isAutoScale()

    def setYAxisAutoScale(self, flag=True):
        """Set the Y axis limits adjusting behavior of :meth:`resetZoom`.

        :param bool flag: True to resize limits automatically,
                          False to disable it.
        """
        self._yAxis.setAutoScale(flag)

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
        self._forceResetZoom()
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
        curves = self.getAllCurves(just_legend=False, withhidden=True)

        if curves:
            for curve in curves:
                curve.setSymbol(self._defaultPlotPoints)

    def isDefaultPlotLines(self):
        """Return True for line as default line style, False for no line."""
        return self._plotLines

    def setDefaultPlotLines(self, flag):
        """Toggle the use of lines as the default curve line style.

        :param bool flag: True to use a line as the default line style,
                          False to use no line as the default line style.
        """
        self._plotLines = bool(flag)

        linestyle = '-' if self._plotLines else ' '

        # Reset linestyle of all curves
        curves = self.getAllCurves(withhidden=True)

        if curves:
            for curve in curves:
                curve.setLineStyle(linestyle)

    def getDefaultColormap(self):
        """Return the default colormap used by :meth:`addImage`.

        :rtype: ~silx.gui.colors.Colormap
        """
        return self._defaultColormap

    def setDefaultColormap(self, colormap=None):
        """Set the default colormap used by :meth:`addImage`.

        Setting the default colormap do not change any currently displayed
        image.
        It only affects future calls to :meth:`addImage` without the colormap
        parameter.

        :param ~silx.gui.colors.Colormap colormap:
            The description of the default colormap, or
            None to set the colormap to a linear
            autoscale gray colormap.
        """
        if colormap is None:
            colormap = Colormap(name=silx.config.DEFAULT_COLORMAP_NAME,
                                normalization='linear',
                                vmin=None,
                                vmax=None)
        if isinstance(colormap, dict):
            self._defaultColormap = Colormap._fromDict(colormap)
        else:
            assert isinstance(colormap, Colormap)
            self._defaultColormap = colormap
        self.notify('defaultColormapChanged')

    @staticmethod
    def getSupportedColormaps():
        """Get the supported colormap names as a tuple of str.

        The list contains at least:
        ('gray', 'reversed gray', 'temperature', 'red', 'green', 'blue',
        'magma', 'inferno', 'plasma', 'viridis')
        """
        return Colormap.getSupportedColormaps()

    def _resetColorAndStyle(self):
        self._colorIndex = 0
        self._styleIndex = 0

    def _getColorAndStyle(self):
        color = self.colorList[self._colorIndex]
        style = self._styleList[self._styleIndex]

        # Loop over color and then styles
        self._colorIndex += 1
        if self._colorIndex >= len(self.colorList):
            self._colorIndex = 0
            self._styleIndex = (self._styleIndex + 1) % len(self._styleList)

        # If color is the one of active curve, take the next one
        if colors.rgba(color) == self.getActiveCurveStyle().getColor():
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
        """Send an event to the listeners and send signals.

        Event are passed to the registered callback as a dict with an 'event'
        key for backward compatibility with PyMca.

        :param str event: The type of event
        :param kwargs: The information of the event.
        """
        eventDict = kwargs.copy()
        eventDict['event'] = event
        self.sigPlotSignal.emit(eventDict)

        if event == 'setKeepDataAspectRatio':
            self.sigSetKeepDataAspectRatio.emit(kwargs['state'])
        elif event == 'setGraphGrid':
            self.sigSetGraphGrid.emit(kwargs['which'])
        elif event == 'setGraphCursor':
            self.sigSetGraphCursor.emit(kwargs['state'])
        elif event == 'contentChanged':
            self.sigContentChanged.emit(
                kwargs['action'], kwargs['kind'], kwargs['legend'])
        elif event == 'activeCurveChanged':
            self.sigActiveCurveChanged.emit(
                kwargs['previous'], kwargs['legend'])
        elif event == 'activeImageChanged':
            self.sigActiveImageChanged.emit(
                kwargs['previous'], kwargs['legend'])
        elif event == 'activeScatterChanged':
            self.sigActiveScatterChanged.emit(
                kwargs['previous'], kwargs['legend'])
        elif event == 'interactiveModeChanged':
            self.sigInteractiveModeChanged.emit(kwargs['source'])

        eventDict = kwargs.copy()
        eventDict['event'] = event
        self._callback(eventDict)

    def setCallback(self, callbackFunction=None):
        """Attach a listener to the backend.

        Limitation: Only one listener at a time.

        :param callbackFunction: function accepting a dictionary as input
                                 to handle the graph events
                                 If None (default), use a default listener.
        """
        # TODO allow multiple listeners
        # allow register listener by event type
        if callbackFunction is None:
            callbackFunction = WeakMethodProxy(self.graphCallback)
        self._callback = callbackFunction

    def graphCallback(self, ddict=None):
        """This callback is going to receive all the events from the plot.

        Those events will consist on a dictionary and among the dictionary
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
                qt.QToolTip.showText(self.cursor().pos(), ddict['label'])
        elif ddict['event'] == 'mouseClicked' and ddict['button'] == 'left':
            self.setActiveCurve(None)

    def saveGraph(self, filename, fileFormat=None, dpi=None, **kw):
        """Save a snapshot of the plot.

        Supported file formats depends on the backend in use.
        The following file formats are always supported: "png", "svg".
        The matplotlib backend supports more formats:
        "pdf", "ps", "eps", "tiff", "jpeg", "jpg".

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
        for item in self._contentToUpdate:
            item._update(self._backend)
        self._contentToUpdate = []
        self._backend.replot()
        self._dirty = False  # reset dirty flag

    def _forceResetZoom(self, dataMargins=None):
        """Reset the plot limits to the bounds of the data and redraw the plot.

        This method forces a reset zoom and does not check axis autoscale.

        Extra margins can be added around the data inside the plot area
        (see :meth:`setDataMargins`).
        Margins are given as one ratio of the data range per limit of the
        data (xMin, xMax, yMin and yMax limits).
        For log scale, extra margins are applied in log10 of the data.

        :param dataMargins: Ratios of margins to add around the data inside
                            the plot area for each side (default: no margins).
        :type dataMargins: A 4-tuple of float as (xMin, xMax, yMin, yMax).
        """
        if dataMargins is None:
            dataMargins = self._defaultDataMargins

        # Get data range
        ranges = self.getDataRange()
        xmin, xmax = (1., 100.) if ranges.x is None else ranges.x
        ymin, ymax = (1., 100.) if ranges.y is None else ranges.y
        if ranges.yright is None:
            ymin2, ymax2 = None, None
        else:
            ymin2, ymax2 = ranges.yright

        # Add margins around data inside the plot area
        newLimits = list(_utils.addMarginsToLimits(
            dataMargins,
            self._xAxis._isLogarithmic(),
            self._yAxis._isLogarithmic(),
            xmin, xmax, ymin, ymax, ymin2, ymax2))

        if self.isKeepDataAspectRatio():
            # Use limits with margins to keep ratio
            xmin, xmax, ymin, ymax = newLimits[:4]

            # Compute bbox wth figure aspect ratio
            plotWidth, plotHeight = self.getPlotBoundsInPixels()[2:]
            if plotWidth > 0 and plotHeight > 0:
                plotRatio = plotHeight / plotWidth
                dataRatio = (ymax - ymin) / (xmax - xmin)
                if dataRatio < plotRatio:
                    # Increase y range
                    ycenter = 0.5 * (ymax + ymin)
                    yrange = (xmax - xmin) * plotRatio
                    newLimits[2] = ycenter - 0.5 * yrange
                    newLimits[3] = ycenter + 0.5 * yrange

                elif dataRatio > plotRatio:
                    # Increase x range
                    xcenter = 0.5 * (xmax + xmin)
                    xrange_ = (ymax - ymin) / plotRatio
                    newLimits[0] = xcenter - 0.5 * xrange_
                    newLimits[1] = xcenter + 0.5 * xrange_

        self.setLimits(*newLimits)

    def resetZoom(self, dataMargins=None):
        """Reset the plot limits to the bounds of the data and redraw the plot.

        It automatically scale limits of axes that are in autoscale mode
        (see :meth:`getXAxis`, :meth:`getYAxis` and :meth:`Axis.setAutoScale`).
        It keeps current limits on axes that are not in autoscale mode.

        Extra margins can be added around the data inside the plot area
        (see :meth:`setDataMargins`).
        Margins are given as one ratio of the data range per limit of the
        data (xMin, xMax, yMin and yMax limits).
        For log scale, extra margins are applied in log10 of the data.

        :param dataMargins: Ratios of margins to add around the data inside
                            the plot area for each side (default: no margins).
        :type dataMargins: A 4-tuple of float as (xMin, xMax, yMin, yMax).
        """
        xLimits = self._xAxis.getLimits()
        yLimits = self._yAxis.getLimits()
        y2Limits = self._yRightAxis.getLimits()

        xAuto = self._xAxis.isAutoScale()
        yAuto = self._yAxis.isAutoScale()

        # With log axes, autoscale if limits are <= 0
        # This avoids issues with toggling log scale with matplotlib 2.1.0
        if self._xAxis.getScale() == self._xAxis.LOGARITHMIC and xLimits[0] <= 0:
            xAuto = True
        if self._yAxis.getScale() == self._yAxis.LOGARITHMIC and (yLimits[0] <= 0 or y2Limits[0] <= 0):
            yAuto = True

        if not xAuto and not yAuto:
            _logger.debug("Nothing to autoscale")
        else:  # Some axes to autoscale
            self._forceResetZoom(dataMargins=dataMargins)

            # Restore limits for axis not in autoscale
            if not xAuto and yAuto:
                self.setGraphXLimits(*xLimits)
            elif xAuto and not yAuto:
                if y2Limits is not None:
                    self.setGraphYLimits(
                        y2Limits[0], y2Limits[1], axis='right')
                if yLimits is not None:
                    self.setGraphYLimits(yLimits[0], yLimits[1], axis='left')

        if (xLimits != self._xAxis.getLimits() or
                yLimits != self._yAxis.getLimits() or
                y2Limits != self._yRightAxis.getLimits()):
            self._notifyLimitsChanged()

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

        xmin, xmax = self._xAxis.getLimits()
        yAxis = self.getYAxis(axis=axis)
        ymin, ymax = yAxis.getLimits()

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
            def test(mark):
                return True

        markers = self._backend.pickItems(x, y, kinds=('marker',))
        legends = [m['legend'] for m in markers if m['kind'] == 'marker']

        for legend in reversed(legends):
            marker = self._getMarker(legend)
            if marker is not None and test(marker):
                return marker
        return None

    def _getAllMarkers(self, just_legend=False):
        """Returns all markers' legend or objects

        :param bool just_legend: True to get the legend of the markers,
                                 False (the default) to get marker objects.
        :return: list of legend of list of marker objects
        :rtype: list of str or list of marker objects
        """
        return self._getItems(
            kind='marker', just_legend=just_legend, withhidden=True)

    def _getMarker(self, legend=None):
        """Get the object describing a specific marker.

        It returns None in case no matching marker is found

        :param str legend: The legend of the marker to retrieve
        :rtype: None of marker object
        """
        return self._getItem(kind='marker', legend=legend)

    def _pickImageOrCurve(self, x, y, test=None):
        """Pick an image or a curve at the given position.

        To use for interaction implementation.

        :param float x: X position in pixels
        :param float y: Y position in pixels
        :param test: A callable to call for each picked item to filter
                     picked items. If None (default), do not filter items.
        """
        if test is None:
            def test(i):
                return True

        allItems = self._backend.pickItems(x, y, kinds=('curve', 'image'))
        allItems = [item for item in allItems
                    if item['kind'] in ['curve', 'image']]

        for item in reversed(allItems):
            kind, legend = item['kind'], item['legend']
            if kind == 'curve':
                curve = self.getCurve(legend)
                if curve is not None and test(curve):
                    return kind, curve, item['indices']

            elif kind == 'image':
                image = self.getImage(legend)
                if image is not None and test(image):
                    return kind, image, None

            else:
                _logger.warning('Unsupported kind: %s', kind)

        return None

    def _pick(self, x, y):
        """Pick items in the plot at given position.

        :param float x: X position in pixels
        :param float y: Y position in pixels
        :return: Iterable of (plot item, indices) at picked position.
            Items are ordered from back to front.
        """
        items = []

        # Convert backend result to plot items
        for itemInfo in self._backend.pickItems(
                x, y, kinds=('marker', 'curve', 'image')):
            kind, legend = itemInfo['kind'], itemInfo['legend']

            if kind in ('marker', 'image'):
                item = self._getItem(kind=kind, legend=legend)
                indices = None  # TODO compute indices for images

            else:  # backend kind == 'curve'
                for kind in ('curve', 'histogram', 'scatter'):
                    item = self._getItem(kind=kind, legend=legend)
                    if item is not None:
                        indices = itemInfo['indices']
                        break
                else:
                    _logger.error(
                        'Cannot find corresponding picked item')
                    continue
            items.append((item, indices))

        return tuple(items)

    # User event handling #

    def _isPositionInPlotArea(self, x, y):
        """Project position in pixel to the closest point in the plot area

        :param float x: X coordinate in widget coordinate (in pixel)
        :param float y: Y coordinate in widget coordinate (in pixel)
        :return: (x, y) in widget coord (in pixel) in the plot area
        """
        left, top, width, height = self.getPlotBoundsInPixels()
        xPlot = numpy.clip(x, left, left + width)
        yPlot = numpy.clip(y, top, top + height)
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

        if self._cursorInPlot != isCursorInPlot:
            self._cursorInPlot = isCursorInPlot
            self._eventHandler.handleEvent(
                'enter' if self._cursorInPlot else 'leave')

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

    def onMouseLeaveWidget(self):
        """Handle mouse leave widget event."""
        if self._cursorInPlot:
            self._cursorInPlot = False
            self._eventHandler.handleEvent('leave')

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
                           zoomOnWheel=True, source=None, width=None):
        """Switch the interactive mode.

        :param str mode: The name of the interactive mode.
                         In 'draw', 'pan', 'select', 'select-draw', 'zoom'.
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
        :param float width: Width of the pencil. Only for draw pencil mode.
        """
        self._eventHandler.setInteractiveMode(mode, color, shape, label, width)
        self._eventHandler.zoomOnWheel = zoomOnWheel

        self.notify(
            'interactiveModeChanged', source=source)

    # Panning with arrow keys

    def isPanWithArrowKeys(self):
        """Returns whether or not panning the graph with arrow keys is enable.

        See :meth:`setPanWithArrowKeys`.
        """
        return self._panWithArrowKeys

    def setPanWithArrowKeys(self, pan=False):
        """Enable/Disable panning the graph with arrow keys.

        This grabs the keyboard.

        :param bool pan: True to enable panning, False to disable.
        """
        pan = bool(pan)
        panHasChanged = self._panWithArrowKeys != pan

        self._panWithArrowKeys = pan
        if not self._panWithArrowKeys:
            self.setFocusPolicy(qt.Qt.NoFocus)
        else:
            self.setFocusPolicy(qt.Qt.StrongFocus)
            self.setFocus(qt.Qt.OtherFocusReason)

        if panHasChanged:
            self.sigSetPanWithArrowKeys.emit(pan)

    # Dict to convert Qt arrow key code to direction str.
    _ARROWS_TO_PAN_DIRECTION = {
        qt.Qt.Key_Left: 'left',
        qt.Qt.Key_Right: 'right',
        qt.Qt.Key_Up: 'up',
        qt.Qt.Key_Down: 'down'
    }

    def keyPressEvent(self, event):
        """Key event handler handling panning on arrow keys.

        Overrides base class implementation.
        """
        key = event.key()
        if self._panWithArrowKeys and key in self._ARROWS_TO_PAN_DIRECTION:
            self.pan(self._ARROWS_TO_PAN_DIRECTION[key], factor=0.1)

            # Send a mouse move event to the plot widget to take into account
            # that even if mouse didn't move on the screen, it moved relative
            # to the plotted data.
            qapp = qt.QApplication.instance()
            event = qt.QMouseEvent(
                qt.QEvent.MouseMove,
                self.getWidgetHandle().mapFromGlobal(qt.QCursor.pos()),
                qt.Qt.NoButton,
                qapp.mouseButtons(),
                qapp.keyboardModifiers())
            qapp.sendEvent(self.getWidgetHandle(), event)

        else:
            # Only call base class implementation when key is not handled.
            # See QWidget.keyPressEvent for details.
            super(PlotWidget, self).keyPressEvent(event)

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
        simultaneously.

        :param bool flag: True to enable drawing and disable zoom and select.
        :param str shape: Type of item to be drawn in:
                          hline, vline, rectangle, polygon (default)
        :param str label: Associated text for identifying draw signals
        :param color: The color to use to draw the selection area
        :type color: string ("#RRGGBB") or 4 column unsigned byte array or
                     one of the predefined color names defined in colors.py
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

        It returns None if the interactive mode is not a drawing mode,
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
        simultaneously

        :param bool flag: If True, enable zoom and select mode.
        :param color: The color to use to draw the selection area.
                      (Default: 'black')
        :param color: The color to use to draw the selection area
        :type color: string ("#RRGGBB") or 4 column unsigned byte array or
                     one of the predefined color names defined in colors.py
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
        """Deprecated, use :meth:`Axis.setInverted` instead."""
        _logger.warning('invertYAxis deprecated, '
                        'use getYAxis().setInverted instead.')
        return self.getYAxis().setInverted(*args, **kwargs)

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
