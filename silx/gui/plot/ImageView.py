# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
# ###########################################################################*/
__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
QWidget displaying a 2D image with histograms on its sides.

The :class:`ImageView` implements this widget, and
:class:`ImageViewMainWindow` provides a main window with additional toolbar
and status bar.

Basic usage of :class:`ImageView` is through the following methods:

- :meth:`ImageView.getColormap`, :meth:`ImageView.setColormap` to update the
  default colormap to use and update the currently displayed image.
- :meth:`ImageView.setImage` to update the displayed image.

The :class:`ImageView` uses :class:`PlotWindow` and also
exposes :class:`PyMca5.PyMcaGraph.Plot` API for further control
(plot title, axes labels, adding other images, ...).

For an example of use, see the implementation of :class:`ImageViewMainWindow`.

The ImageView module can also be used to open an EDF or TIFF file
from the shell command line.
To view an image file:
``python -m PyMca5.PyMcaGui.plotting.ImageView <file to open>``
To get help:
``python -m PyMca5.PyMcaGui.plotting.ImageView -h``
"""


# import ######################################################################

import numpy as np

try:
    from .. import PyMcaQt as qt
except ImportError:
    from PyMca5.PyMcaGui import PyMcaQt as qt

from .PlotWindow import PlotWindow
from .Toolbars import ProfileToolBar, LimitsToolBar

from PyMca5.PyMcaGraph import Plot


# utils #######################################################################

_COLORMAP_CURSOR_COLORS = {
    'gray': 'pink',
    'reversed gray': 'pink',
    'temperature': 'black',
    'red': 'gray',
    'green': 'gray',
    'blue': 'gray'}


def _cursorColorForColormap(colormapName):
    """Get a color suitable for overlay over a colormap.

    :param str colormapName: The name of the colormap.
    :return: Name of the color.
    :rtype: str
    """
    return _COLORMAP_CURSOR_COLORS.get(colormapName, 'black')


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

    visibleRectDragged = qt.pyqtSignal(float, float, float, float)
    """Signals that the visible rectangle has been dragged.

    It provides: left, top, width, height in data coordinates.
    """

    _DATA_PEN = qt.QPen(qt.QColor('white'))
    _DATA_BRUSH = qt.QBrush(qt.QColor('light gray'))
    _VISIBLE_PEN = qt.QPen(qt.QColor('red'))
    _VISIBLE_BRUSH = qt.QBrush(qt.QColor(0, 0, 0, 0))
    _TOOLTIP = 'Radar View:\nVisible area (in red)\nof the image (in gray).'

    _PIXMAP_SIZE = 256

    class _DraggableRectItem(qt.QGraphicsRectItem):
        """RectItem which signals its change through visibleRectDragged."""
        def __init__(self, *args, **kwargs):
            super(RadarView._DraggableRectItem, self).__init__(*args, **kwargs)
            self.setFlag(qt.QGraphicsItem.ItemIsMovable)
            self.setFlag(qt.QGraphicsItem.ItemSendsGeometryChanges)
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

class ImageView(qt.QWidget):
    """Display a single image with horizontal and vertical histograms.

    Use :meth:`setImage` to control the displayed image.
    This class also provides the :class:`PyMca5.PyMcaGraph.Plot` API.
    """

    HISTOGRAMS_COLOR = 'blue'
    """Color to use for the side histograms."""

    HISTOGRAMS_HEIGHT = 200
    """Height in pixels of the side histograms."""

    IMAGE_MIN_SIZE = 200
    """Minimum size in pixels of the image area."""

    # Qt signals
    valueChanged = qt.pyqtSignal(float, float, float)
    """Signals that the data value under the cursor has changed.

    It provides: row, column, data value.

    When the cursor is over an histogram, either row or column is Nan
    and the provided data value is the histogram value
    (i.e., the sum along the corresponding row/column).
    Row and columns are either Nan or integer values.
    """

    def __init__(self, parent=None, windowFlags=qt.Qt.Widget, backend=None):
        self._imageLegend = '__ImageView__image' + str(id(self))
        self._cache = None  # Store currently visible data information
        self._updatingLimits = False

        super(ImageView, self).__init__(parent, windowFlags)
        self.setStyleSheet('background-color: white;')
        self._initWidgets(backend)

        # Sync PlotBackend and ImageView
        self._updateYAxisInverted()

        # Set-up focus proxy to handle arrow key event
        self.setFocusProxy(self._imagePlot)

    def _initWidgets(self, backend):
        """Set-up layout and plots."""
        # Monkey-patch for histogram size
        # alternative: create a layout that does not use widget size hints
        def sizeHint():
            return qt.QSize(self.HISTOGRAMS_HEIGHT, self.HISTOGRAMS_HEIGHT)

        self._histoHPlot = Plot.Plot(backend=backend)
        self._histoHPlot.setZoomModeEnabled(True)
        self._histoHPlot.setCallback(self._histoHPlotCB)
        self._histoHPlot.getWidgetHandle().sizeHint = sizeHint
        self._histoHPlot.getWidgetHandle().minimumSizeHint = sizeHint

        self._imagePlot = PlotWindow(backend=backend, plugins=False,
                                     colormap=True, flip=True,
                                     grid=False, togglePoints=False,
                                     logx=False, logy=False,
                                     aspect=True)
        self._imagePlot.usePlotBackendColormap = True
        self._imagePlot.setPanWithArrowKeys(True)

        self._imagePlot.setZoomModeEnabled(True)  # Color is set in setColormap
        self._imagePlot.sigPlotSignal.connect(self._imagePlotCB)
        self._imagePlot.hFlipToolButton.clicked.connect(
            self._updateYAxisInverted)
        self._imagePlot.sigColormapChangedSignal.connect(self.setColormap)

        self._histoVPlot = Plot.Plot(backend=backend)
        self._histoVPlot.setZoomModeEnabled(True)
        self._histoVPlot.setCallback(self._histoVPlotCB)
        self._histoVPlot.getWidgetHandle().sizeHint = sizeHint
        self._histoVPlot.getWidgetHandle().minimumSizeHint = sizeHint

        self._radarView = RadarView()
        self._radarView.visibleRectDragged.connect(self._radarViewCB)

        self._layout = qt.QGridLayout()
        self._layout.addWidget(self._imagePlot, 0, 0)
        self._layout.addWidget(self._histoVPlot.getWidgetHandle(), 0, 1)
        self._layout.addWidget(self._histoHPlot.getWidgetHandle(), 1, 0)
        self._layout.addWidget(self._radarView, 1, 1)

        self._layout.setColumnMinimumWidth(0, self.IMAGE_MIN_SIZE)
        self._layout.setColumnStretch(0, 1)
        self._layout.setColumnMinimumWidth(1, self.HISTOGRAMS_HEIGHT)
        self._layout.setColumnStretch(1, 0)

        self._layout.setRowMinimumHeight(0, self.IMAGE_MIN_SIZE)
        self._layout.setRowStretch(0, 1)
        self._layout.setRowMinimumHeight(1, self.HISTOGRAMS_HEIGHT)
        self._layout.setRowStretch(1, 0)

        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(self._layout)

    def _dirtyCache(self):
        self._cache = None

    def _updateHistograms(self):
        """Update histograms content using current active image."""
        activeImage = self._imagePlot.getActiveImage()
        if activeImage is not None:
            wasUpdatingLimits = self._updatingLimits
            self._updatingLimits = True

            data, legend, info, pixmap = activeImage
            xScale, yScale = info['plot_xScale'], info['plot_yScale']
            height, width = data.shape

            xMin, xMax = self._imagePlot.getGraphXLimits()
            yMin, yMax = self._imagePlot.getGraphYLimits()

            # Convert plot area limits to image coordinates
            # and work in image coordinates (i.e., in pixels)
            xMin = int((xMin - xScale[0]) / xScale[1])
            xMax = int((xMax - xScale[0]) / xScale[1])
            yMin = int((yMin - yScale[0]) / yScale[1])
            yMax = int((yMax - yScale[0]) / yScale[1])

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
                    histoHVisibleData = np.sum(visibleData, axis=0)
                    histoVVisibleData = np.sum(visibleData, axis=1)

                    self._cache = {
                        'dataXMin': subsetXMin,
                        'dataXMax': subsetXMax,
                        'dataYMin': subsetYMin,
                        'dataYMax': subsetYMax,

                        'histoH': histoHVisibleData,
                        'histoHMin': np.min(histoHVisibleData),
                        'histoHMax': np.max(histoHVisibleData),

                        'histoV': histoVVisibleData,
                        'histoVMin': np.min(histoVVisibleData),
                        'histoVMax': np.max(histoVVisibleData)
                    }

                    # Convert to histogram curve and update plots
                    # Taking into account origin and scale
                    coords = np.arange(2 * histoHVisibleData.size)
                    xCoords = (coords + 1) // 2 + subsetXMin
                    xCoords = xScale[0] + xScale[1] * xCoords
                    xData = np.take(histoHVisibleData, coords // 2)
                    self._histoHPlot.addCurve(xCoords, xData,
                                              xlabel='', ylabel='',
                                              replace=False, replot=False,
                                              color=self.HISTOGRAMS_COLOR,
                                              linestyle='-',
                                              selectable=False)
                    vMin = self._cache['histoHMin']
                    vMax = self._cache['histoHMax']
                    vOffset = 0.1 * (vMax - vMin)
                    if vOffset == 0.:
                        vOffset = 1.
                    self._histoHPlot.setGraphYLimits(vMin - vOffset,
                                                     vMax + vOffset)

                    coords = np.arange(2 * histoVVisibleData.size)
                    yCoords = (coords + 1) // 2 + subsetYMin
                    yCoords = yScale[0] + yScale[1] * yCoords
                    yData = np.take(histoVVisibleData, coords // 2)
                    self._histoVPlot.addCurve(yData, yCoords,
                                              xlabel='', ylabel='',
                                              replace=False, replot=False,
                                              color=self.HISTOGRAMS_COLOR,
                                              linestyle='-',
                                              selectable=False)
                    vMin = self._cache['histoVMin']
                    vMax = self._cache['histoVMax']
                    vOffset = 0.1 * (vMax - vMin)
                    if vOffset == 0.:
                        vOffset = 1.
                    self._histoVPlot.setGraphXLimits(vMin - vOffset,
                                                     vMax + vOffset)
            else:
                self._dirtyCache()
                self._histoHPlot.clearCurves()
                self._histoVPlot.clearCurves()

            self._updatingLimits = wasUpdatingLimits

    def _updateRadarView(self):
        """Update radar view visible area.

        Takes care of y coordinate conversion.
        """
        xMin, xMax = self._imagePlot.getGraphXLimits()
        yMin, yMax = self._imagePlot.getGraphYLimits()
        self._radarView.setVisibleRect(xMin, yMin, xMax - xMin, yMax - yMin)

    # Plots event listeners

    def _imagePlotCB(self, eventDict):
        """Callback for imageView plot events."""
        if eventDict['event'] == 'mouseMoved':
            activeImage = self._imagePlot.getActiveImage()
            if activeImage is not None:
                data = activeImage[0]
                height, width = data.shape
                x, y = int(eventDict['x']), int(eventDict['y'])
                if x >= 0 and x < width and y >= 0 and y < height:
                    self.valueChanged.emit(float(x), float(y),
                                           data[y][x])
        elif eventDict['event'] == 'limitsChanged':
            # Do not handle histograms limitsChanged while
            # updating their limits from here.
            self._updatingLimits = True

            # Refresh histograms
            self._updateHistograms()

            # could use eventDict['xdata'], eventDict['ydata'] instead
            xMin, xMax = self._imagePlot.getGraphXLimits()
            yMin, yMax = self._imagePlot.getGraphYLimits()

            # Set horizontal histo limits
            self._histoHPlot.setGraphXLimits(xMin, xMax)
            self._histoHPlot.replot()

            # Set vertical histo limits
            self._histoVPlot.setGraphYLimits(yMin, yMax)
            self._histoVPlot.replot()

            self._updateRadarView()

            self._updatingLimits = False

            # Replot in case limitsChanged due to set*Limits
            # called from console.
            # This results in an extra replot call in other cases.
            self._imagePlot.replot()

    def _histoHPlotCB(self, eventDict):
        """Callback for horizontal histogram plot events."""
        if eventDict['event'] == 'mouseMoved':
            if self._cache is not None:
                activeImage = self._imagePlot.getActiveImage()
                if activeImage is not None:
                    xOrigin, xScaleFactor = activeImage[2]['plot_xScale']

                    minValue = xOrigin + xScaleFactor * self._cache['dataXMin']
                    data = self._cache['histoH']
                    width = data.shape[0]
                    x = int(eventDict['x'])
                    if x >= minValue and x < minValue + width:
                        self.valueChanged.emit(float('nan'), float(x),
                                               data[x - minValue])
        elif eventDict['event'] == 'limitsChanged':
            if (not self._updatingLimits and
                    eventDict['xdata'] != self._imagePlot.getGraphXLimits()):
                xMin, xMax = eventDict['xdata']
                self._imagePlot.setGraphXLimits(xMin, xMax)
                self._imagePlot.replot()

    def _histoVPlotCB(self, eventDict):
        """Callback for vertical histogram plot events."""
        if eventDict['event'] == 'mouseMoved':
            if self._cache is not None:
                activeImage = self._imagePlot.getActiveImage()
                if activeImage is not None:
                    yOrigin, yScaleFactor = activeImage[2]['plot_yScale']

                    minValue = yOrigin + yScaleFactor * self._cache['dataYMin']
                    data = self._cache['histoV']
                    height = data.shape[0]
                    y = int(eventDict['y'])
                    if y >= minValue and y < minValue + height:
                        self.valueChanged.emit(float(y), float('nan'),
                                               data[y - minValue])
        elif eventDict['event'] == 'limitsChanged':
            if (not self._updatingLimits and
                    eventDict['ydata'] != self._imagePlot.getGraphYLimits()):
                yMin, yMax = eventDict['ydata']
                self._imagePlot.setGraphYLimits(yMin, yMax)
                self._imagePlot.replot()

    def _radarViewCB(self, left, top, width, height):
        """Slot for radar view visible rectangle changes."""
        if not self._updatingLimits:
            # Takes care of Y axis conversion
            self._imagePlot.setLimits(left, left + width, top, top + height)
            self._imagePlot.replot()

    def _updateYAxisInverted(self):
        """Sync image, vertical histogram and radar view axis orientation."""
        inverted = self._imagePlot.isYAxisInverted()

        self._imagePlot.invertYAxis(inverted)
        self._histoVPlot.invertYAxis(inverted)

        # Use scale to invert radarView
        # RadarView default Y direction is from top to bottom
        # As opposed to Plot. So invert RadarView when Plot is NOT inverted.
        self._radarView.resetTransform()
        if not inverted:
            self._radarView.scale(1., -1.)
        self._updateRadarView()

        self._imagePlot.replot()
        self._histoVPlot.replot()
        self._radarView.update()

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
                    data=np.array(self._cache['histoH'], copy=True),
                    extent=(self._cache['dataXMin'], self._cache['dataXMax']))
            else:
                return dict(
                    data=np.array(self._cache['histoV'], copy=True),
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
        self._layout.addWidget(self._radarView, 1, 1)

        self._updateYAxisInverted()

    # PlotWindow toolbar

    def toolBar(self):
        """Returns the tool bar associated with the image plot.

        This is the toolBar provided by :class:`PlotWindow`.

        :return: The toolBar associated to the image plot.
        :rtype: QToolBar
        """
        return self._imagePlot.toolBar

    # High-level API

    def getColormap(self):
        """Get the default colormap description.

        :return: A description of the current colormap.
                 See :meth:`setColormap` for details.
        :rtype: dict
        """
        return self._imagePlot.getDefaultColormap()

    def setColormap(self, colormap=None, normalization=None,
                    autoscale=None, vmin=None, vmax=None, colors=256):
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
        """
        cmapDict = self._imagePlot.getDefaultColormap()

        if isinstance(colormap, dict):
            # Support colormap parameter as a dict
            assert normalization is None
            assert autoscale is None
            assert vmin is None
            assert vmax is None
            assert colors == 256
            for key, value in colormap.items():
                cmapDict[key] = value

        else:
            if colormap is not None:
                cmapDict['name'] = colormap
            if normalization is not None:
                cmapDict['normalization'] = normalization
            if autoscale is not None:
                cmapDict['autoscale'] = autoscale
            if vmin is not None:
                cmapDict['vmin'] = vmin
            if vmax is not None:
                cmapDict['vmax'] = vmax

        if 'colors' not in cmapDict:
            cmapDict['colors'] = 256

        cursorColor = _cursorColorForColormap(cmapDict['name'])
        self._imagePlot.setZoomModeEnabled(True, color=cursorColor)

        self._imagePlot.setDefaultColormap(cmapDict)

        activeImage = self._imagePlot.getActiveImage()
        if activeImage is not None:  # Refresh image with new colormap
            data, legend, info, pixmap = activeImage

            self._imagePlot.addImage(data, legend=legend, info=info,
                                     colormap=self.getColormap(),
                                     replace=False, replot=False)
            self._imagePlot.replot()

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
            self._imagePlot.removeImage(self._imageLegend, replot=False)
            return

        data = np.array(image, order='C', copy=copy)
        assert data.size != 0
        assert len(data.shape) == 2
        height, width = data.shape

        self._imagePlot.addImage(data,
                                 legend=self._imageLegend,
                                 xScale=(origin[0], scale[0]),
                                 yScale=(origin[1], scale[1]),
                                 colormap=self.getColormap(),
                                 replace=False,
                                 replot=False)
        self._imagePlot.setActiveImage(self._imageLegend)
        self._updateHistograms()

        self._radarView.setDataRect(origin[0],
                                    origin[1],
                                    width * scale[0],
                                    height * scale[1])

        if reset:
            self.resetZoom()
        else:
            self._histoHPlot.replot()
            self._histoVPlot.replot()
            self._imagePlot.replot()

    ####################
    # Plot API proxies #
    ####################

    # Rebuild side histograms if active image gets changed through the Plot API

    def addImage(self, data, legend=None, info=None,
                 replace=True, replot=True,
                 xScale=None, yScale=None, z=0,
                 selectable=False, draggable=False,
                 colormap=None, **kw):
        if legend == self._imagePlot.getActiveImage(just_legend=True):
            # Updating active image, resets side histograms cache
            self._dirtyCache()

        result = self._imagePlot.addImage(data, legend, info, replace, replot,
                                          xScale, yScale, z,
                                          selectable, draggable,
                                          colormap, **kw)
        self._updateHistograms()

        if replot:
            self._histoHPlot.replot()
            self._histoVPlot.replot()

        return result

    def clear(self):
        self._dirtyCache()
        return self._imagePlot.clear()

    def clearImages(self):
        self._dirtyCache()
        return self._imagePlot.clearImages()

    def removeImage(self, legend, replot=True):
        if legend == self._imagePlot.getActiveImage(just_legend=True):
            # Removing active image, resets side histograms cache
            self._dirtyCache()

        result = self._imageView.removeImage(legend, replot)
        self._updateHistograms()

        if replot:
            self._histoHPlot.replot()
            self._histoVPlot.replot()

        return result

    def setActiveImage(self, legend, replot=True):
        # Active image changes, resets side histogram cache
        self._dirtyCache()

        result = self._imagePlot.setActiveImage(legend, replot)
        self._updateHistograms()

        if replot:
            self._histoHPlot.replot()
            self._histoVPlot.replot()
        return result

    # Invert axes

    def invertYAxis(self, flag=True):
        result = self._imagePlot.invertYAxis(flag)
        self._updateYAxisInverted()  # To sync vert. histo and radar view
        return result

    # Ugly yet simple proxy for the Plot API

    def __getattr__(self, name):
        """Proxy to expose image plot API."""
        return getattr(self._imagePlot, name)


# ImageViewMainWindow #########################################################

class ImageViewMainWindow(qt.QMainWindow):
    """QMainWindow embedding an ImageView.

    Surrounds the ImageView with an associated toolbar and status bar.
    """

    def __init__(self, parent=None, windowFlags=qt.Qt.Widget, backend=None):
        self._dataInfo = None
        super(ImageViewMainWindow, self).__init__(parent, windowFlags)

        # Create the ImageView widget and add it to the QMainWindow
        self.imageView = ImageView(backend=backend)
        self.imageView.setGraphXLabel('X')
        self.imageView.setGraphYLabel('Y')
        self.imageView.setGraphTitle('Image')
        self.imageView._imagePlot.sigColormapChangedSignal.connect(
            self._colormapUpdated)
        self.setCentralWidget(self.imageView)

        # Using PlotWindow's toolbar
        self.addToolBar(self.imageView.toolBar())
        self.profileToolBar = ProfileToolBar(self.imageView._imagePlot)
        self.addToolBar(self.profileToolBar)
        self.addToolBar(qt.Qt.BottomToolBarArea, LimitsToolBar(self.imageView))

        self.statusBar()

        # Connect to ImageView's signal
        self.imageView.valueChanged.connect(self._statusBarSlot)

    def _colormapUpdated(self, colormap):
        """Sync ROI color with current colormap"""
        self.profileToolBar.overlayColor = _cursorColorForColormap(
            colormap['name'])

    def _statusBarSlot(self, row, column, value):
        """Update status bar with coordinates/value from plots."""
        if np.isnan(row):
            msg = 'Column: %d, Sum: %g' % (int(column), value)
        elif np.isnan(column):
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
        self.imageView.setImage(image, *args, **kwargs)
        self.profileToolBar.updateProfile()
        self.setStatusBar(None)


# main ########################################################################

if __name__ == "__main__":
    import argparse
    import os.path
    import sys

    from PyMca5.PyMcaIO.EdfFile import EdfFile

    # Command-line arguments
    parser = argparse.ArgumentParser(
        description='Browse the images of an EDF file.')
    parser.add_argument(
        '-b', '--backend',
        choices=('mpl', 'opengl', 'osmesa'),
        help="""The plot backend to use: Matplotlib (mpl, the default),
        OpenGL 2.1 (opengl, requires appropriate OpenGL drivers) or
        Off-screen Mesa OpenGL software pipeline (osmesa,
        requires appropriate OSMesa library).""")
    parser.add_argument(
        '-o', '--origin', nargs=2,
        type=float, default=(0., 0.),
        help="""Coordinates of the origin of the image: (x, y).
        Default: 0., 0.""")
    parser.add_argument(
        '-s', '--scale', nargs=2,
        type=float, default=(1., 1.),
        help="""Scale factors applied to the image: (sx, sy).
        Default: 1., 1.""")
    parser.add_argument('filename', help='EDF filename of the image to open')
    args = parser.parse_args()

    # Open the input file
    if not os.path.isfile(args.filename):
        raise RuntimeError('No input file: %s' % args.filename)

    edfFile = EdfFile(args.filename)
    nbFrames = edfFile.GetNumImages()
    if nbFrames == 0:
        raise RuntimeError(
            'Cannot read image(s) from file: %s' % args.filename)

    # Set-up Qt application and main window
    app = qt.QApplication([])

    mainWindow = ImageViewMainWindow(backend=args.backend)
    mainWindow.setImage(edfFile.GetData(0),
                        origin=args.origin,
                        scale=args.scale)

    if nbFrames > 1:
        # Add a toolbar for multi-frame EDF support
        multiFrameToolbar = qt.QToolBar('Multi-frame')
        multiFrameToolbar.addWidget(qt.QLabel(
            'Frame [0-%d]:' % (nbFrames - 1)))

        spinBox = qt.QSpinBox()
        spinBox.setRange(0, nbFrames-1)

        def updateImage(index):
            mainWindow.setImage(edfFile.GetData(index),
                                origin=args.origin,
                                scale=args.scale,
                                reset=False)
        spinBox.valueChanged[int].connect(updateImage)
        multiFrameToolbar.addWidget(spinBox)

        mainWindow.addToolBar(multiFrameToolbar)

    mainWindow.show()

    sys.exit(app.exec_())
