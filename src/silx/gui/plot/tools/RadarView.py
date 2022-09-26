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
"""QWidget displaying an overview of a 2D plot.

This shows the available range of the data, and the current location of the
plot view.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "22/02/2021"

import logging
import weakref
from ... import qt
from ...utils import LockReentrant

_logger = logging.getLogger(__name__)


class _DraggableRectItem(qt.QGraphicsRectItem):
    """RectItem which signals its change through visibleRectDragged."""
    def __init__(self, *args, **kwargs):
        super(_DraggableRectItem, self).__init__(
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
        super(_DraggableRectItem, self).setPos(*args, **kwargs)
        self._ignoreChange = False

    def moveBy(self, *args, **kwargs):
        """Overridden to ignore changes from API in itemChange."""
        self._ignoreChange = True
        super(_DraggableRectItem, self).moveBy(*args, **kwargs)
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

        return super(_DraggableRectItem, self).itemChange(
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
    _ACTIVEDATA_PEN = qt.QPen(qt.QColor('black'))
    _ACTIVEDATA_BRUSH = qt.QBrush(qt.QColor('transparent'))
    _ACTIVEDATA_PEN.setWidth(2)
    _ACTIVEDATA_PEN.setCosmetic(True)
    _VISIBLE_PEN = qt.QPen(qt.QColor('blue'))
    _VISIBLE_PEN.setWidth(2)
    _VISIBLE_PEN.setCosmetic(True)
    _VISIBLE_BRUSH = qt.QBrush(qt.QColor(0, 0, 0, 0))
    _TOOLTIP = 'Radar View:\nRed contour: Visible area\nGray area: The image'

    _PIXMAP_SIZE = 256

    def __init__(self, parent=None):
        self.__plotRef = None
        self._scene = qt.QGraphicsScene()
        self._dataRect = self._scene.addRect(0, 0, 1, 1,
                                             self._DATA_PEN,
                                             self._DATA_BRUSH)
        self._imageRect = self._scene.addRect(0, 0, 1, 1,
                                              self._ACTIVEDATA_PEN,
                                              self._ACTIVEDATA_BRUSH)
        self._imageRect.setVisible(False)
        self._scatterRect = self._scene.addRect(0, 0, 1, 1,
                                                self._ACTIVEDATA_PEN,
                                                self._ACTIVEDATA_BRUSH)
        self._scatterRect.setVisible(False)
        self._curveRect = self._scene.addRect(0, 0, 1, 1,
                                              self._ACTIVEDATA_PEN,
                                              self._ACTIVEDATA_BRUSH)
        self._curveRect.setVisible(False)

        self._visibleRect = _DraggableRectItem(0, 0, 1, 1)
        self._visibleRect.setPen(self._VISIBLE_PEN)
        self._visibleRect.setBrush(self._VISIBLE_BRUSH)
        self._scene.addItem(self._visibleRect)

        super(RadarView, self).__init__(self._scene, parent)
        self.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        self.setFocusPolicy(qt.Qt.NoFocus)
        self.setStyleSheet('border: 0px')
        self.setToolTip(self._TOOLTIP)

        self.__reentrant = LockReentrant()
        self.visibleRectDragged.connect(self._viewRectDragged)

        self.__timer = qt.QTimer(self)
        self.__timer.timeout.connect(self._updateDataContent)

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
        self.__visibleRect = left, top, width, height
        self._visibleRect.setRect(0, 0, width, height)
        self._visibleRect.setPos(left, top)
        self.fitInView(self._scene.itemsBoundingRect(), qt.Qt.KeepAspectRatio)

    def __setVisibleRectFromPlot(self, plot):
        """Update radar view visible area.

        Takes care of y coordinate conversion.
        """
        xMin, xMax = plot.getXAxis().getLimits()
        yMin, yMax = plot.getYAxis().getLimits()
        self.setVisibleRect(xMin, yMin, xMax - xMin, yMax - yMin)

    def getPlotWidget(self):
        """Returns the connected plot

        :rtype: Union[None,PlotWidget]
        """
        if self.__plotRef is None:
            return None
        plot = self.__plotRef()
        if plot is None:
            self.__plotRef = None
        return plot

    def setPlotWidget(self, plot):
        """Set the PlotWidget this radar view connects to.

        As result `setDataRect` and `setVisibleRect` will be called
        automatically.

        :param Union[None,PlotWidget] plot:
        """
        previousPlot = self.getPlotWidget()
        if previousPlot is not None:  # Disconnect previous plot
            plot.getXAxis().sigLimitsChanged.disconnect(self._xLimitChanged)
            plot.getYAxis().sigLimitsChanged.disconnect(self._yLimitChanged)
            plot.getYAxis().sigInvertedChanged.disconnect(self._updateYAxisInverted)

        # Reset plot and timer
        # FIXME: It would be good to clean up the display here
        self.__plotRef = None
        self.__timer.stop()

        if plot is not None:  # Connect new plot
            self.__plotRef = weakref.ref(plot)
            plot.getXAxis().sigLimitsChanged.connect(self._xLimitChanged)
            plot.getYAxis().sigLimitsChanged.connect(self._yLimitChanged)
            plot.getYAxis().sigInvertedChanged.connect(self._updateYAxisInverted)
            self.__setVisibleRectFromPlot(plot)
            self._updateYAxisInverted()
            self.__timer.start(500)

    def _xLimitChanged(self, vmin, vmax):
        plot = self.getPlotWidget()
        self.__setVisibleRectFromPlot(plot)

    def _yLimitChanged(self, vmin, vmax):
        plot = self.getPlotWidget()
        self.__setVisibleRectFromPlot(plot)

    def _updateYAxisInverted(self, inverted=None):
        """Sync radar view axis orientation."""
        plot = self.getPlotWidget()
        if inverted is None:
            # Do not perform this when called from plot signal
            inverted = plot.getYAxis().isInverted()
        # Use scale to invert radarView
        # RadarView default Y direction is from top to bottom
        # As opposed to Plot. So invert RadarView when Plot is NOT inverted.
        self.resetTransform()
        if not inverted:
            self.scale(1., -1.)
        self.update()

    def _viewRectDragged(self, left, top, width, height):
        """Slot for radar view visible rectangle changes."""
        plot = self.getPlotWidget()
        if plot is None:
            return

        if self.__reentrant.locked():
            return

        with self.__reentrant:
            plot.setLimits(left, left + width, top, top + height)

    def _updateDataContent(self):
        """Update the content to the current data content"""
        plot = self.getPlotWidget()
        if plot is None:
            return
        ranges = plot.getDataRange()
        xmin, xmax = ranges.x if ranges.x is not None else (0, 0)
        ymin, ymax = ranges.y if ranges.y is not None else (0, 0)
        self.setDataRect(xmin, ymin, xmax - xmin, ymax - ymin)

        self.__updateItem(self._imageRect, plot.getActiveImage())
        self.__updateItem(self._scatterRect, plot.getActiveScatter())
        self.__updateItem(self._curveRect, plot.getActiveCurve())

    def __updateItem(self, rect, item):
        """Sync rect with item bounds

        :param QGraphicsRectItem rect:
        :param Item item:
        """
        if item is None:
            rect.setVisible(False)
            return
        ranges = item._getBounds()
        if ranges is None:
            rect.setVisible(False)
            return
        xmin, xmax, ymin, ymax = ranges
        width = xmax - xmin
        height = ymax - ymin
        rect.setRect(xmin, ymin, width, height)
        rect.setVisible(True)
