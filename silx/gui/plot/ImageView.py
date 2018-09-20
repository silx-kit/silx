# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2018 European Synchrotron Radiation Facility
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
"""QWidget displaying a 2D image with histograms on its sides.

The :class:`ImageView` implements this widget, and
:class:`ImageViewMainWindow` provides a main window with additional toolbar
and status bar.

Basic usage of :class:`ImageView` is through the following methods:

- :meth:`ImageView.getColormap`, :meth:`ImageView.setColormap` to update the
  default colormap to use and update the currently displayed image.
- :meth:`ImageView.setImage` to update the displayed image.

For an example of use, see `imageview.py` in :ref:`sample-code`.
"""

from __future__ import division


__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "26/04/2018"


import logging
import numpy

import silx
from .. import qt

from . import items, PlotWindow, PlotWidget, actions
from ..colors import Colormap
from ..colors import cursorColorForColormap
from .tools import LimitsToolBar
from .Profile import ProfileToolBar


_logger = logging.getLogger(__name__)


# RadarView ###################################################################

class RadarView(qt.QGraphicsView):
    """Widget presenting a synthetic view of a 2D area and
    the current visible area.

    Coordinates are as in QGraphicsView:
    x goes from left to right and y goes from top to bottom.
    This widget preserves the aspect ratio of the areas.

    The 2D area and the visible area can be set with :meth:`setDataRect`
    and :meth:`setVisibleRect`.
    When the visible area has been dragged by the user, its new position
    is signaled by the *visibleRectDragged* signal.

    It is possible to invert the direction of the axes by using the
    :meth:`scale` method of QGraphicsView.
    """

    visibleRectDragged = qt.Signal(float, float, float, float)
    """Signals that the visible rectangle has been dragged.

    It provides: left, top, width, height in data coordinates.
    """

    _DATA_PEN = qt.QPen(qt.QColor('white'))
    _DATA_BRUSH = qt.QBrush(qt.QColor('light gray'))
    _VISIBLE_PEN = qt.QPen(qt.QColor('red'))
    _VISIBLE_PEN.setWidth(2)
    _VISIBLE_PEN.setCosmetic(True)
    _VISIBLE_BRUSH = qt.QBrush(qt.QColor(0, 0, 0, 0))
    _TOOLTIP = 'Radar View:\nRed contour: Visible area\nGray area: The image'

    _PIXMAP_SIZE = 256

    class _DraggableRectItem(qt.QGraphicsRectItem):
        """RectItem which signals its change through visibleRectDragged."""
        def __init__(self, *args, **kwargs):
            super(RadarView._DraggableRectItem, self).__init__(
                *args, **kwargs)

            self._previousCursor = None
            self.setFlag(qt.QGraphicsItem.ItemIsMovable)
            self.setFlag(qt.QGraphicsItem.ItemSendsGeometryChanges)
            self.setAcceptHoverEvents(True)
            self._ignoreChange = False
            self._constraint = 0, 0, 0, 0

        def setConstraintRect(self, left, top, width, height):
            """Set the constraint rectangle for dragging.

            The coordinates are in the _DraggableRectItem coordinate system.

            This constraint only applies to modification through interaction
            (i.e., this constraint is not applied to change through API).

            If the _DraggableRectItem is smaller than the constraint rectangle,
            the _DraggableRectItem remains within the constraint rectangle.
            If the _DraggableRectItem is wider than the constraint rectangle,
            the constraint rectangle remains within the _DraggableRectItem.
            """
            self._constraint = left, left + width, top, top + height

        def setPos(self, *args, **kwargs):
            """Overridden to ignore changes from API in itemChange."""
            self._ignoreChange = True
            super(RadarView._DraggableRectItem, self).setPos(*args, **kwargs)
            self._ignoreChange = False

        def moveBy(self, *args, **kwargs):
            """Overridden to ignore changes from API in itemChange."""
            self._ignoreChange = True
            super(RadarView._DraggableRectItem, self).moveBy(*args, **kwargs)
            self._ignoreChange = False

        def itemChange(self, change, value):
            """Callback called before applying changes to the item."""
            if (change == qt.QGraphicsItem.ItemPositionChange and
                    not self._ignoreChange):
                # Makes sure that the visible area is in the data
                # or that data is in the visible area if area is too wide
                x, y = value.x(), value.y()
                xMin, xMax, yMin, yMax = self._constraint

                if self.rect().width() <= (xMax - xMin):
                    if x < xMin:
                        value.setX(xMin)
                    elif x > xMax - self.rect().width():
                        value.setX(xMax - self.rect().width())
                else:
                    if x > xMin:
                        value.setX(xMin)
                    elif x < xMax - self.rect().width():
                        value.setX(xMax - self.rect().width())

                if self.rect().height() <= (yMax - yMin):
                    if y < yMin:
                        value.setY(yMin)
                    elif y > yMax - self.rect().height():
                        value.setY(yMax - self.rect().height())
                else:
                    if y > yMin:
                        value.setY(yMin)
                    elif y < yMax - self.rect().height():
                        value.setY(yMax - self.rect().height())

                if self.pos() != value:
                    # Notify change through signal
                    views = self.scene().views()
                    assert len(views) == 1
                    views[0].visibleRectDragged.emit(
                        value.x() + self.rect().left(),
                        value.y() + self.rect().top(),
                        self.rect().width(),
                        self.rect().height())

                return value

            return super(RadarView._DraggableRectItem, self).itemChange(
                change, value)

        def hoverEnterEvent(self, event):
            """Called when the mouse enters the rectangle area"""
            self._previousCursor = self.cursor()
            self.setCursor(qt.Qt.OpenHandCursor)

        def hoverLeaveEvent(self, event):
            """Called when the mouse leaves the rectangle area"""
            if self._previousCursor is not None:
                self.setCursor(self._previousCursor)
                self._previousCursor = None

    def __init__(self, parent=None):
        self._scene = qt.QGraphicsScene()
        self._dataRect = self._scene.addRect(0, 0, 1, 1,
                                             self._DATA_PEN,
                                             self._DATA_BRUSH)
        self._visibleRect = self._DraggableRectItem(0, 0, 1, 1)
        self._visibleRect.setPen(self._VISIBLE_PEN)
        self._visibleRect.setBrush(self._VISIBLE_BRUSH)
        self._scene.addItem(self._visibleRect)

        super(RadarView, self).__init__(self._scene, parent)
        self.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(qt.Qt.NoFocus)
        self.setStyleSheet('border: 0px')
        self.setToolTip(self._TOOLTIP)

    def sizeHint(self):
        # """Overridden to avoid sizeHint to depend on content size."""
        return self.minimumSizeHint()

    def wheelEvent(self, event):
        # """Overridden to disable vertical scrolling with wheel."""
        event.ignore()

    def resizeEvent(self, event):
        # """Overridden to fit current content to new size."""
        self.fitInView(self._scene.itemsBoundingRect(), qt.Qt.KeepAspectRatio)
        super(RadarView, self).resizeEvent(event)

    def setDataRect(self, left, top, width, height):
        """Set the bounds of the data rectangular area.

        This sets the coordinate system.
        """
        self._dataRect.setRect(left, top, width, height)
        self._visibleRect.setConstraintRect(left, top, width, height)
        self.fitInView(self._scene.itemsBoundingRect(), qt.Qt.KeepAspectRatio)

    def setVisibleRect(self, left, top, width, height):
        """Set the visible rectangular area.

        The coordinates are relative to the data rect.
        """
        self._visibleRect.setRect(0, 0, width, height)
        self._visibleRect.setPos(left, top)
        self.fitInView(self._scene.itemsBoundingRect(), qt.Qt.KeepAspectRatio)


# ImageView ###################################################################

class ImageView(PlotWindow):
    """Display a single image with horizontal and vertical histograms.

    Use :meth:`setImage` to control the displayed image.
    This class also provides the :class:`silx.gui.plot.Plot` API.

    The :class:`ImageView` inherits from :class:`.PlotWindow` (which provides
    the toolbars) and also exposes :class:`.PlotWidget` API for further
    plot control (plot title, axes labels, aspect ratio, ...).

    :param parent: The parent of this widget or None.
    :param backend: The backend to use for the plot (default: matplotlib).
                    See :class:`.PlotWidget` for the list of supported backend.
    :type backend: str or :class:`BackendBase.BackendBase`
    """

    HISTOGRAMS_COLOR = 'blue'
    """Color to use for the side histograms."""

    HISTOGRAMS_HEIGHT = 200
    """Height in pixels of the side histograms."""

    IMAGE_MIN_SIZE = 200
    """Minimum size in pixels of the image area."""

    # Qt signals
    valueChanged = qt.Signal(float, float, float)
    """Signals that the data value under the cursor has changed.

    It provides: row, column, data value.

    When the cursor is over an histogram, either row or column is Nan
    and the provided data value is the histogram value
    (i.e., the sum along the corresponding row/column).
    Row and columns are either Nan or integer values.
    """

    def __init__(self, parent=None, backend=None):
        self._imageLegend = '__ImageView__image' + str(id(self))
        self._cache = None  # Store currently visible data information
        self._updatingLimits = False

        super(ImageView, self).__init__(parent=parent, backend=backend,
                                        resetzoom=True, autoScale=False,
                                        logScale=False, grid=False,
                                        curveStyle=False, colormap=True,
                                        aspectRatio=True, yInverted=True,
                                        copy=True, save=True, print_=True,
                                        control=False, position=False,
                                        roi=False, mask=True)
        if parent is None:
            self.setWindowTitle('ImageView')

        if silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION == 'downward':
            self.getYAxis().setInverted(True)

        self._initWidgets(backend)

        self.profile = ProfileToolBar(plot=self)
        """"Profile tools attached to this plot.

        See :class:`silx.gui.plot.PlotTools.ProfileToolBar`
        """

        self.addToolBar(self.profile)

        # Sync PlotBackend and ImageView
        self._updateYAxisInverted()

    def _initWidgets(self, backend):
        """Set-up layout and plots."""
        self._histoHPlot = PlotWidget(backend=backend)
        self._histoHPlot.getWidgetHandle().setMinimumHeight(
            self.HISTOGRAMS_HEIGHT)
        self._histoHPlot.getWidgetHandle().setMaximumHeight(
            self.HISTOGRAMS_HEIGHT)
        self._histoHPlot.setInteractiveMode('zoom')
        self._histoHPlot.sigPlotSignal.connect(self._histoHPlotCB)

        self.setPanWithArrowKeys(True)

        self.setInteractiveMode('zoom')  # Color set in setColormap
        self.sigPlotSignal.connect(self._imagePlotCB)
        self.getYAxis().sigInvertedChanged.connect(self._updateYAxisInverted)
        self.sigActiveImageChanged.connect(self._activeImageChangedSlot)

        self._histoVPlot = PlotWidget(backend=backend)
        self._histoVPlot.getWidgetHandle().setMinimumWidth(
            self.HISTOGRAMS_HEIGHT)
        self._histoVPlot.getWidgetHandle().setMaximumWidth(
            self.HISTOGRAMS_HEIGHT)
        self._histoVPlot.setInteractiveMode('zoom')
        self._histoVPlot.sigPlotSignal.connect(self._histoVPlotCB)

        self._radarView = RadarView()
        self._radarView.visibleRectDragged.connect(self._radarViewCB)

        layout = qt.QGridLayout()
        layout.addWidget(self.getWidgetHandle(), 0, 0)
        layout.addWidget(self._histoVPlot.getWidgetHandle(), 0, 1)
        layout.addWidget(self._histoHPlot.getWidgetHandle(), 1, 0)
        layout.addWidget(self._radarView, 1, 1)

        layout.setColumnMinimumWidth(0, self.IMAGE_MIN_SIZE)
        layout.setColumnStretch(0, 1)
        layout.setColumnMinimumWidth(1, self.HISTOGRAMS_HEIGHT)
        layout.setColumnStretch(1, 0)

        layout.setRowMinimumHeight(0, self.IMAGE_MIN_SIZE)
        layout.setRowStretch(0, 1)
        layout.setRowMinimumHeight(1, self.HISTOGRAMS_HEIGHT)
        layout.setRowStretch(1, 0)

        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        centralWidget = qt.QWidget(self)
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def _dirtyCache(self):
        self._cache = None

    def _updateHistograms(self):
        """Update histograms content using current active image."""
        activeImage = self.getActiveImage()
        if activeImage is not None:
            wasUpdatingLimits = self._updatingLimits
            self._updatingLimits = True

            data = activeImage.getData(copy=False)
            origin = activeImage.getOrigin()
            scale = activeImage.getScale()
            height, width = data.shape

            xMin, xMax = self.getXAxis().getLimits()
            yMin, yMax = self.getYAxis().getLimits()

            # Convert plot area limits to image coordinates
            # and work in image coordinates (i.e., in pixels)
            xMin = int((xMin - origin[0]) / scale[0])
            xMax = int((xMax - origin[0]) / scale[0])
            yMin = int((yMin - origin[1]) / scale[1])
            yMax = int((yMax - origin[1]) / scale[1])

            if (xMin < width and xMax >= 0 and
                    yMin < height and yMax >= 0):
                # The image is at least partly in the plot area
                # Get the visible bounds in image coords (i.e., in pixels)
                subsetXMin = 0 if xMin < 0 else xMin
                subsetXMax = (width if xMax >= width else xMax) + 1
                subsetYMin = 0 if yMin < 0 else yMin
                subsetYMax = (height if yMax >= height else yMax) + 1

                if (self._cache is None or
                        subsetXMin != self._cache['dataXMin'] or
                        subsetXMax != self._cache['dataXMax'] or
                        subsetYMin != self._cache['dataYMin'] or
                        subsetYMax != self._cache['dataYMax']):
                    # The visible area of data has changed, update histograms

                    # Rebuild histograms for visible area
                    visibleData = data[subsetYMin:subsetYMax,
                                       subsetXMin:subsetXMax]
                    histoHVisibleData = numpy.sum(visibleData, axis=0)
                    histoVVisibleData = numpy.sum(visibleData, axis=1)

                    self._cache = {
                        'dataXMin': subsetXMin,
                        'dataXMax': subsetXMax,
                        'dataYMin': subsetYMin,
                        'dataYMax': subsetYMax,

                        'histoH': histoHVisibleData,
                        'histoHMin': numpy.min(histoHVisibleData),
                        'histoHMax': numpy.max(histoHVisibleData),

                        'histoV': histoVVisibleData,
                        'histoVMin': numpy.min(histoVVisibleData),
                        'histoVMax': numpy.max(histoVVisibleData)
                    }

                    # Convert to histogram curve and update plots
                    # Taking into account origin and scale
                    coords = numpy.arange(2 * histoHVisibleData.size)
                    xCoords = (coords + 1) // 2 + subsetXMin
                    xCoords = origin[0] + scale[0] * xCoords
                    xData = numpy.take(histoHVisibleData, coords // 2)
                    self._histoHPlot.addCurve(xCoords, xData,
                                              xlabel='', ylabel='',
                                              replace=False,
                                              color=self.HISTOGRAMS_COLOR,
                                              linestyle='-',
                                              selectable=False)
                    vMin = self._cache['histoHMin']
                    vMax = self._cache['histoHMax']
                    vOffset = 0.1 * (vMax - vMin)
                    if vOffset == 0.:
                        vOffset = 1.
                    self._histoHPlot.getYAxis().setLimits(vMin - vOffset,
                                                          vMax + vOffset)

                    coords = numpy.arange(2 * histoVVisibleData.size)
                    yCoords = (coords + 1) // 2 + subsetYMin
                    yCoords = origin[1] + scale[1] * yCoords
                    yData = numpy.take(histoVVisibleData, coords // 2)
                    self._histoVPlot.addCurve(yData, yCoords,
                                              xlabel='', ylabel='',
                                              replace=False,
                                              color=self.HISTOGRAMS_COLOR,
                                              linestyle='-',
                                              selectable=False)
                    vMin = self._cache['histoVMin']
                    vMax = self._cache['histoVMax']
                    vOffset = 0.1 * (vMax - vMin)
                    if vOffset == 0.:
                        vOffset = 1.
                    self._histoVPlot.getXAxis().setLimits(vMin - vOffset,
                                                          vMax + vOffset)
            else:
                self._dirtyCache()
                self._histoHPlot.remove(kind='curve')
                self._histoVPlot.remove(kind='curve')

            self._updatingLimits = wasUpdatingLimits

    def _updateRadarView(self):
        """Update radar view visible area.

        Takes care of y coordinate conversion.
        """
        xMin, xMax = self.getXAxis().getLimits()
        yMin, yMax = self.getYAxis().getLimits()
        self._radarView.setVisibleRect(xMin, yMin, xMax - xMin, yMax - yMin)

    # Plots event listeners

    def _imagePlotCB(self, eventDict):
        """Callback for imageView plot events."""
        if eventDict['event'] == 'mouseMoved':
            activeImage = self.getActiveImage()
            if activeImage is not None:
                data = activeImage.getData(copy=False)
                height, width = data.shape

                # Get corresponding coordinate in image
                origin = activeImage.getOrigin()
                scale = activeImage.getScale()
                if (eventDict['x'] >= origin[0] and
                        eventDict['y'] >= origin[1]):
                    x = int((eventDict['x'] - origin[0]) / scale[0])
                    y = int((eventDict['y'] - origin[1]) / scale[1])

                    if x >= 0 and x < width and y >= 0 and y < height:
                        self.valueChanged.emit(float(x), float(y),
                                               data[y][x])

        elif eventDict['event'] == 'limitsChanged':
            self._updateHistogramsLimits()

    def _updateHistogramsLimits(self):
            # Do not handle histograms limitsChanged while
            # updating their limits from here.
            self._updatingLimits = True

            # Refresh histograms
            self._updateHistograms()

            xMin, xMax = self.getXAxis().getLimits()
            yMin, yMax = self.getYAxis().getLimits()

            # Set horizontal histo limits
            self._histoHPlot.getXAxis().setLimits(xMin, xMax)

            # Set vertical histo limits
            self._histoVPlot.getYAxis().setLimits(yMin, yMax)

            self._updateRadarView()

            self._updatingLimits = False

    def _histoHPlotCB(self, eventDict):
        """Callback for horizontal histogram plot events."""
        if eventDict['event'] == 'mouseMoved':
            if self._cache is not None:
                activeImage = self.getActiveImage()
                if activeImage is not None:
                    xOrigin = activeImage.getOrigin()[0]
                    xScale = activeImage.getScale()[0]

                    minValue = xOrigin + xScale * self._cache['dataXMin']

                    if eventDict['x'] >= minValue:
                        data = self._cache['histoH']
                        column = int((eventDict['x'] - minValue) / xScale)
                        if column >= 0 and column < data.shape[0]:
                            self.valueChanged.emit(
                                float('nan'),
                                float(column + self._cache['dataXMin']),
                                data[column])

        elif eventDict['event'] == 'limitsChanged':
            if (not self._updatingLimits and
                    eventDict['xdata'] != self.getXAxis().getLimits()):
                xMin, xMax = eventDict['xdata']
                self.getXAxis().setLimits(xMin, xMax)

    def _histoVPlotCB(self, eventDict):
        """Callback for vertical histogram plot events."""
        if eventDict['event'] == 'mouseMoved':
            if self._cache is not None:
                activeImage = self.getActiveImage()
                if activeImage is not None:
                    yOrigin = activeImage.getOrigin()[1]
                    yScale = activeImage.getScale()[1]

                    minValue = yOrigin + yScale * self._cache['dataYMin']

                    if eventDict['y'] >= minValue:
                        data = self._cache['histoV']
                        row = int((eventDict['y'] - minValue) / yScale)
                        if row >= 0 and row < data.shape[0]:
                            self.valueChanged.emit(
                                float(row + self._cache['dataYMin']),
                                float('nan'),
                                data[row])

        elif eventDict['event'] == 'limitsChanged':
            if (not self._updatingLimits and
                    eventDict['ydata'] != self.getYAxis().getLimits()):
                yMin, yMax = eventDict['ydata']
                self.getYAxis().setLimits(yMin, yMax)

    def _radarViewCB(self, left, top, width, height):
        """Slot for radar view visible rectangle changes."""
        if not self._updatingLimits:
            # Takes care of Y axis conversion
            self.setLimits(left, left + width, top, top + height)

    def _updateYAxisInverted(self, inverted=None):
        """Sync image, vertical histogram and radar view axis orientation."""
        if inverted is None:
            # Do not perform this when called from plot signal
            inverted = self.getYAxis().isInverted()

        self._histoVPlot.getYAxis().setInverted(inverted)

        # Use scale to invert radarView
        # RadarView default Y direction is from top to bottom
        # As opposed to Plot. So invert RadarView when Plot is NOT inverted.
        self._radarView.resetTransform()
        if not inverted:
            self._radarView.scale(1., -1.)
        self._updateRadarView()

        self._radarView.update()

    def _activeImageChangedSlot(self, previous, legend):
        """Handle Plot active image change.

        Resets side histograms cache
        """
        self._dirtyCache()
        self._updateHistograms()

    def getHistogram(self, axis):
        """Return the histogram and corresponding row or column extent.

        The returned value when an histogram is available is a dict with keys:

        - 'data': numpy array of the histogram values.
        - 'extent': (start, end) row or column index.
          end index is not included in the histogram.

        :param str axis: 'x' for horizontal, 'y' for vertical
        :return: The histogram and its extent as a dict or None.
        :rtype: dict
        """
        assert axis in ('x', 'y')
        if self._cache is None:
            return None
        else:
            if axis == 'x':
                return dict(
                    data=numpy.array(self._cache['histoH'], copy=True),
                    extent=(self._cache['dataXMin'], self._cache['dataXMax']))
            else:
                return dict(
                    data=numpy.array(self._cache['histoV'], copy=True),
                    extent=(self._cache['dataYMin'], self._cache['dataYMax']))

    def radarView(self):
        """Get the lower right radarView widget."""
        return self._radarView

    def setRadarView(self, radarView):
        """Change the lower right radarView widget.

        :param RadarView radarView: Widget subclassing RadarView to replace
                                    the lower right corner widget.
        """
        self._radarView.visibleRectDragged.disconnect(self._radarViewCB)
        self._radarView = radarView
        self._radarView.visibleRectDragged.connect(self._radarViewCB)
        self.centralWidget().layout().addWidget(self._radarView, 1, 1)

        self._updateYAxisInverted()

    # High-level API

    def getColormap(self):
        """Get the default colormap description.

        :return: A description of the current colormap.
                 See :meth:`setColormap` for details.
        :rtype: dict
        """
        return self.getDefaultColormap()

    def setColormap(self, colormap=None, normalization=None,
                    autoscale=None, vmin=None, vmax=None, colors=None):
        """Set the default colormap and update active image.

        Parameters that are not provided are taken from the current colormap.

        The colormap parameter can also be a dict with the following keys:

        - *name*: string. The colormap to use:
          'gray', 'reversed gray', 'temperature', 'red', 'green', 'blue'.
        - *normalization*: string. The mapping to use for the colormap:
          either 'linear' or 'log'.
        - *autoscale*: bool. Whether to use autoscale (True)
          or range provided by keys 'vmin' and 'vmax' (False).
        - *vmin*: float. The minimum value of the range to use if 'autoscale'
          is False.
        - *vmax*: float. The maximum value of the range to use if 'autoscale'
          is False.
        - *colors*: optional. Nx3 or Nx4 array of float in [0, 1] or uint8.
                    List of RGB or RGBA colors to use (only if name is None)

        :param colormap: Name of the colormap in
            'gray', 'reversed gray', 'temperature', 'red', 'green', 'blue'.
            Or the description of the colormap as a dict.
        :type colormap: dict or str.
        :param str normalization: Colormap mapping: 'linear' or 'log'.
        :param bool autoscale: Whether to use autoscale (True)
                               or [vmin, vmax] range (False).
        :param float vmin: The minimum value of the range to use if
                           'autoscale' is False.
        :param float vmax: The maximum value of the range to use if
                           'autoscale' is False.
        :param numpy.ndarray colors: Only used if name is None.
            Custom colormap colors as Nx3 or Nx4 RGB or RGBA arrays
        """
        cmap = self.getDefaultColormap()

        if isinstance(colormap, Colormap):
            # Replace colormap
            cmap = colormap

            self.setDefaultColormap(cmap)

            # Update active image colormap
            activeImage = self.getActiveImage()
            if isinstance(activeImage, items.ColormapMixIn):
                activeImage.setColormap(cmap)

        elif isinstance(colormap, dict):
            # Support colormap parameter as a dict
            assert normalization is None
            assert autoscale is None
            assert vmin is None
            assert vmax is None
            assert colors is None
            cmap._setFromDict(colormap)

        else:
            if colormap is not None:
                cmap.setName(colormap)
            if normalization is not None:
                cmap.setNormalization(normalization)
            if autoscale:
                cmap.setVRange(None, None)
            else:
                if vmin is not None:
                    cmap.setVMin(vmin)
                if vmax is not None:
                    cmap.setVMax(vmax)
            if colors is not None:
                cmap.setColormapLUT(colors)

        cursorColor = cursorColorForColormap(cmap.getName())
        self.setInteractiveMode('zoom', color=cursorColor)

    def setImage(self, image, origin=(0, 0), scale=(1., 1.),
                 copy=True, reset=True):
        """Set the image to display.

        :param image: A 2D array representing the image or None to empty plot.
        :type image: numpy.ndarray-like with 2 dimensions or None.
        :param origin: The (x, y) position of the origin of the image.
                       Default: (0, 0).
                       The origin is the lower left corner of the image when
                       the Y axis is not inverted.
        :type origin: Tuple of 2 floats: (origin x, origin y).
        :param scale: The scale factor to apply to the image on X and Y axes.
                      Default: (1, 1).
                      It is the size of a pixel in the coordinates of the axes.
                      Scales must be positive numbers.
        :type scale: Tuple of 2 floats: (scale x, scale y).
        :param bool copy: Whether to copy image data (default) or not.
        :param bool reset: Whether to reset zoom and ROI (default) or not.
        """
        self._dirtyCache()

        assert len(origin) == 2
        assert len(scale) == 2
        assert scale[0] > 0
        assert scale[1] > 0

        if image is None:
            self.remove(self._imageLegend, kind='image')
            return

        data = numpy.array(image, order='C', copy=copy)
        assert data.size != 0
        assert len(data.shape) == 2
        height, width = data.shape

        self.addImage(data,
                      legend=self._imageLegend,
                      origin=origin, scale=scale,
                      colormap=self.getColormap(),
                      resetzoom=False)
        self.setActiveImage(self._imageLegend)
        self._updateHistograms()

        self._radarView.setDataRect(origin[0],
                                    origin[1],
                                    width * scale[0],
                                    height * scale[1])

        if reset:
            self.resetZoom()
        else:
            self._updateHistogramsLimits()


# ImageViewMainWindow #########################################################

class ImageViewMainWindow(ImageView):
    """:class:`ImageView` with additional toolbars

    Adds extra toolbar and a status bar to :class:`ImageView`.
    """
    def __init__(self, parent=None, backend=None):
        self._dataInfo = None
        super(ImageViewMainWindow, self).__init__(parent, backend)
        self.setWindowFlags(qt.Qt.Window)

        self.getXAxis().setLabel('X')
        self.getYAxis().setLabel('Y')
        self.setGraphTitle('Image')

        # Add toolbars and status bar
        self.addToolBar(qt.Qt.BottomToolBarArea, LimitsToolBar(plot=self))

        self.statusBar()

        menu = self.menuBar().addMenu('File')
        menu.addAction(self.getOutputToolBar().getSaveAction())
        menu.addAction(self.getOutputToolBar().getPrintAction())
        menu.addSeparator()
        action = menu.addAction('Quit')
        action.triggered[bool].connect(qt.QApplication.instance().quit)

        menu = self.menuBar().addMenu('Edit')
        menu.addAction(self.getOutputToolBar().getCopyAction())
        menu.addSeparator()
        menu.addAction(self.getResetZoomAction())
        menu.addAction(self.getColormapAction())
        menu.addAction(actions.control.KeepAspectRatioAction(self, self))
        menu.addAction(actions.control.YAxisInvertedAction(self, self))

        menu = self.menuBar().addMenu('Profile')
        menu.addAction(self.profile.hLineAction)
        menu.addAction(self.profile.vLineAction)
        menu.addAction(self.profile.lineAction)
        menu.addAction(self.profile.clearAction)

        # Connect to ImageView's signal
        self.valueChanged.connect(self._statusBarSlot)

    def _statusBarSlot(self, row, column, value):
        """Update status bar with coordinates/value from plots."""
        if numpy.isnan(row):
            msg = 'Column: %d, Sum: %g' % (int(column), value)
        elif numpy.isnan(column):
            msg = 'Row: %d, Sum: %g' % (int(row), value)
        else:
            msg = 'Position: (%d, %d), Value: %g' % (int(row), int(column),
                                                     value)
        if self._dataInfo is not None:
            msg = self._dataInfo + ', ' + msg

        self.statusBar().showMessage(msg)

    def setImage(self, image, *args, **kwargs):
        """Set the displayed image.

        See :meth:`ImageView.setImage` for details.
        """
        if hasattr(image, 'dtype') and hasattr(image, 'shape'):
            assert len(image.shape) == 2
            height, width = image.shape
            self._dataInfo = 'Data: %dx%d (%s)' % (width, height,
                                                   str(image.dtype))
            self.statusBar().showMessage(self._dataInfo)
        else:
            self._dataInfo = None

        # Set the new image in ImageView widget
        super(ImageViewMainWindow, self).setImage(image, *args, **kwargs)
        self.setStatusBar(None)
