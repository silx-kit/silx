# /*##########################################################################
#
# Copyright (c) 2015-2021 European Synchrotron Radiation Facility
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

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "26/04/2018"


import logging
import numpy
import collections
from typing import Union

import silx
from .. import qt
from .. import colors
from .. import icons

from . import items, PlotWindow, PlotWidget, actions
from ..colors import Colormap
from ..colors import cursorColorForColormap
from .tools import LimitsToolBar
from .Profile import ProfileToolBar
from ...utils.proxy import docstring
from ...utils.deprecation import deprecated
from ...utils.enum import Enum
from .tools.RadarView import RadarView
from .utils.axis import SyncAxes
from ..utils import blockSignals
from . import _utils
from .tools.profile import manager
from .tools.profile import rois
from .actions import PlotAction

_logger = logging.getLogger(__name__)


ProfileSumResult = collections.namedtuple("ProfileResult",
                                          ["dataXRange", "dataYRange",
                                           'histoH', 'histoHRange',
                                           'histoV', 'histoVRange',
                                           "xCoords", "xData",
                                           "yCoords", "yData"])


def computeProfileSumOnRange(imageItem, xRange, yRange, cache=None):
    """
    Compute a full vertical and horizontal profile on an image item using a
    a range in the plot referential.

    Optionally takes a previous computed result to be able to skip the
    computation.

    :rtype: ProfileSumResult
    """
    data = imageItem.getValueData(copy=False)
    origin = imageItem.getOrigin()
    scale = imageItem.getScale()
    height, width = data.shape

    xMin, xMax = xRange
    yMin, yMax = yRange

    # Convert plot area limits to image coordinates
    # and work in image coordinates (i.e., in pixels)
    xMin = int((xMin - origin[0]) / scale[0])
    xMax = int((xMax - origin[0]) / scale[0])
    yMin = int((yMin - origin[1]) / scale[1])
    yMax = int((yMax - origin[1]) / scale[1])

    if (xMin >= width or xMax < 0 or
            yMin >= height or yMax < 0):
        return None

    # The image is at least partly in the plot area
    # Get the visible bounds in image coords (i.e., in pixels)
    subsetXMin = 0 if xMin < 0 else xMin
    subsetXMax = (width if xMax >= width else xMax) + 1
    subsetYMin = 0 if yMin < 0 else yMin
    subsetYMax = (height if yMax >= height else yMax) + 1

    if cache is not None:
        if ((subsetXMin, subsetXMax) == cache.dataXRange and
                (subsetYMin, subsetYMax) == cache.dataYRange):
            # The visible area of data is the same
            return cache

    # Rebuild histograms for visible area
    visibleData = data[subsetYMin:subsetYMax,
                       subsetXMin:subsetXMax]
    histoHVisibleData = numpy.nansum(visibleData, axis=0)
    histoVVisibleData = numpy.nansum(visibleData, axis=1)
    histoHMin = numpy.nanmin(histoHVisibleData)
    histoHMax = numpy.nanmax(histoHVisibleData)
    histoVMin = numpy.nanmin(histoVVisibleData)
    histoVMax = numpy.nanmax(histoVVisibleData)

    # Convert to histogram curve and update plots
    # Taking into account origin and scale
    coords = numpy.arange(2 * histoHVisibleData.size)
    xCoords = (coords + 1) // 2 + subsetXMin
    xCoords = origin[0] + scale[0] * xCoords
    xData = numpy.take(histoHVisibleData, coords // 2)
    coords = numpy.arange(2 * histoVVisibleData.size)
    yCoords = (coords + 1) // 2 + subsetYMin
    yCoords = origin[1] + scale[1] * yCoords
    yData = numpy.take(histoVVisibleData, coords // 2)

    result = ProfileSumResult(
        dataXRange=(subsetXMin, subsetXMax),
        dataYRange=(subsetYMin, subsetYMax),
        histoH=histoHVisibleData,
        histoHRange=(histoHMin, histoHMax),
        histoV=histoVVisibleData,
        histoVRange=(histoVMin, histoVMax),
        xCoords=xCoords,
        xData=xData,
        yCoords=yCoords,
        yData=yData)

    return result


class _SideHistogram(PlotWidget):
    """
    Widget displaying one of the side profile of the ImageView.

    Implement ProfileWindow
    """

    sigClose = qt.Signal()

    sigMouseMoved = qt.Signal(float, float)

    def __init__(self, parent=None, backend=None, direction=qt.Qt.Horizontal):
        super(_SideHistogram, self).__init__(parent=parent, backend=backend)
        self._direction = direction
        self.sigPlotSignal.connect(self._plotEvents)
        self._color = "blue"
        self.__profile = None
        self.__profileSum = None

    def _plotEvents(self, eventDict):
        """Callback for horizontal histogram plot events."""
        if eventDict['event'] == 'mouseMoved':
            self.sigMouseMoved.emit(eventDict['x'], eventDict['y'])

    def setProfileColor(self, color):
        self._color = color

    def setProfileSum(self, result):
        self.__profileSum = result
        if self.__profile is None:
            self.__drawProfileSum()

    def prepareWidget(self, roi):
        """Implements `ProfileWindow`"""
        pass

    def setRoiProfile(self, roi):
        """Implements `ProfileWindow`"""
        if roi is None:
            return
        self._roiColor = colors.rgba(roi.getColor())

    def getProfile(self):
        """Implements `ProfileWindow`"""
        return self.__profile

    def setProfile(self, data):
        """Implements `ProfileWindow`"""
        self.__profile = data
        if data is None:
            self.__drawProfileSum()
        else:
            self.__drawProfile()

    def __drawProfileSum(self):
        """Only draw the profile sum on the plot.

        Other elements are removed
        """
        profileSum = self.__profileSum

        try:
            self.removeCurve('profile')
        except Exception:
            pass

        if profileSum is None:
            try:
                self.removeCurve('profilesum')
            except Exception:
                pass
            return

        if self._direction == qt.Qt.Horizontal:
            xx, yy = profileSum.xCoords, profileSum.xData
        elif self._direction == qt.Qt.Vertical:
            xx, yy = profileSum.yData, profileSum.yCoords
        else:
            assert False

        self.addCurve(xx, yy,
                      xlabel='', ylabel='',
                      legend="profilesum",
                      color=self._color,
                      linestyle='-',
                      selectable=False,
                      resetzoom=False)

        self.__updateLimits()

    def __drawProfile(self):
        """Only draw the profile on the plot.

        Other elements are removed
        """
        profile = self.__profile

        try:
            self.removeCurve('profilesum')
        except Exception:
            pass

        if profile is None:
            try:
                self.removeCurve('profile')
            except Exception:
                pass
            self.setProfileSum(self.__profileSum)
            return

        if self._direction == qt.Qt.Horizontal:
            xx, yy = profile.coords, profile.profile
        elif self._direction == qt.Qt.Vertical:
            xx, yy = profile.profile, profile.coords
        else:
            assert False

        self.addCurve(xx,
                      yy,
                      legend="profile",
                      color=self._roiColor,
                      resetzoom=False)

        self.__updateLimits()

    def __updateLimits(self):
        if self.__profile:
            data = self.__profile.profile
            vMin = numpy.nanmin(data)
            vMax = numpy.nanmax(data)
        elif self.__profileSum is not None:
            if self._direction == qt.Qt.Horizontal:
                vMin, vMax = self.__profileSum.histoHRange
            elif self._direction == qt.Qt.Vertical:
                vMin, vMax = self.__profileSum.histoVRange
            else:
                assert False
        else:
            vMin, vMax = 0, 0

        # Tune the result using the data margins
        margins = self.getDataMargins()
        if self._direction == qt.Qt.Horizontal:
            _, _, vMin, vMax = _utils.addMarginsToLimits(margins, False, False, 0, 0, vMin, vMax)
        elif self._direction == qt.Qt.Vertical:
            vMin, vMax, _, _ = _utils.addMarginsToLimits(margins, False, False, vMin, vMax, 0, 0)
        else:
            assert False

        if self._direction == qt.Qt.Horizontal:
            dataAxis = self.getYAxis()
        elif self._direction == qt.Qt.Vertical:
            dataAxis = self.getXAxis()
        else:
            assert False

        with blockSignals(dataAxis):
            dataAxis.setLimits(vMin, vMax)


class ShowSideHistogramsAction(PlotAction):
    """QAction to change visibility of side histogram of a :class:`.ImageView`.

    :param plot: :class:`.ImageView` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(ShowSideHistogramsAction, self).__init__(
            plot, icon='side-histograms', text='Show/hide side histograms',
            tooltip='Show/hide side histogram',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)

    def _actionTriggered(self, checked=False):
        if self.plot.isSideHistogramDisplayed() != checked:
            self.plot.setSideHistogramDisplayed(checked)


class AggregationModeAction(qt.QWidgetAction):
    """Action providing few filters to the image"""

    sigAggregationModeChanged = qt.Signal()

    def __init__(self, parent):
        qt.QWidgetAction.__init__(self, parent)

        toolButton = qt.QToolButton(parent)

        filterAction = qt.QAction(self)
        filterAction.setText("No filter")
        filterAction.setCheckable(True)
        filterAction.setChecked(True)
        filterAction.setProperty("aggregation", items.ImageDataAggregated.Aggregation.NONE)
        densityNoFilterAction = filterAction

        filterAction = qt.QAction(self)
        filterAction.setText("Max filter")
        filterAction.setCheckable(True)
        filterAction.setProperty("aggregation", items.ImageDataAggregated.Aggregation.MAX)
        densityMaxFilterAction = filterAction

        filterAction = qt.QAction(self)
        filterAction.setText("Mean filter")
        filterAction.setCheckable(True)
        filterAction.setProperty("aggregation", items.ImageDataAggregated.Aggregation.MEAN)
        densityMeanFilterAction = filterAction

        filterAction = qt.QAction(self)
        filterAction.setText("Min filter")
        filterAction.setCheckable(True)
        filterAction.setProperty("aggregation", items.ImageDataAggregated.Aggregation.MIN)
        densityMinFilterAction = filterAction

        densityGroup = qt.QActionGroup(self)
        densityGroup.setExclusive(True)
        densityGroup.addAction(densityNoFilterAction)
        densityGroup.addAction(densityMaxFilterAction)
        densityGroup.addAction(densityMeanFilterAction)
        densityGroup.addAction(densityMinFilterAction)
        densityGroup.triggered.connect(self._aggregationModeChanged)
        self.__densityGroup = densityGroup

        filterMenu = qt.QMenu(toolButton)
        filterMenu.addAction(densityNoFilterAction)
        filterMenu.addAction(densityMaxFilterAction)
        filterMenu.addAction(densityMeanFilterAction)
        filterMenu.addAction(densityMinFilterAction)

        toolButton.setPopupMode(qt.QToolButton.InstantPopup)
        toolButton.setMenu(filterMenu)
        toolButton.setText("Data filters")
        toolButton.setToolTip("Enable/disable filter on the image")
        icon = icons.getQIcon("aggregation-mode")
        toolButton.setIcon(icon)
        toolButton.setText("Pixel aggregation filter")

        self.setDefaultWidget(toolButton)

    def _aggregationModeChanged(self):
        self.sigAggregationModeChanged.emit()

    def setAggregationMode(self, mode):
        """Set an Aggregated enum from ImageDataAggregated"""
        for a in self.__densityGroup.actions():
            if a.property("aggregation") is mode:
                a.setChecked(True)

    def getAggregationMode(self):
        """Returns an Aggregated enum from ImageDataAggregated"""
        densityAction = self.__densityGroup.checkedAction()
        if densityAction is None:
            return items.ImageDataAggregated.Aggregation.NONE
        return densityAction.property("aggregation")


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
    valueChanged = qt.Signal(float, float, object)
    """Signals that the data value under the cursor has changed.

    It provides: row, column, data value.

    When the cursor is over an histogram, either row or column is Nan
    and the provided data value is the histogram value
    (i.e., the sum along the corresponding row/column).
    Row and columns are either Nan or integer values.
    """

    class ProfileWindowBehavior(Enum):
        """ImageView's profile window behavior options"""

        POPUP = 'popup'
        """All profiles are displayed in pop-up windows"""

        EMBEDDED = 'embedded'
        """Horizontal, vertical and cross profiles are displayed in
        sides widgets, others are displayed in pop-up windows.
        """

    def __init__(self, parent=None, backend=None):
        self._imageLegend = '__ImageView__image' + str(id(self))
        self._cache = None  # Store currently visible data information

        super(ImageView, self).__init__(parent=parent, backend=backend,
                                        resetzoom=True, autoScale=False,
                                        logScale=False, grid=False,
                                        curveStyle=False, colormap=True,
                                        aspectRatio=True, yInverted=True,
                                        copy=True, save=True, print_=True,
                                        control=False, position=False,
                                        roi=False, mask=True)

        # Enable mask synchronisation to use it in profiles
        maskToolsWidget = self.getMaskToolsDockWidget().widget()
        maskToolsWidget.setItemMaskUpdated(True)

        self.__showSideHistogramsAction = ShowSideHistogramsAction(self, self)
        self.__showSideHistogramsAction.setChecked(True)

        self.__aggregationModeAction = AggregationModeAction(self)
        self.__aggregationModeAction.sigAggregationModeChanged.connect(self._aggregationModeChanged)

        if parent is None:
            self.setWindowTitle('ImageView')

        if silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION == 'downward':
            self.getYAxis().setInverted(True)

        self._initWidgets(backend)

        toolBar = self.toolBar()
        toolBar.addAction(self.__showSideHistogramsAction)
        toolBar.addAction(self.__aggregationModeAction)

        self.__profileWindowBehavior = self.ProfileWindowBehavior.POPUP
        self.__profile = ProfileToolBar(plot=self)
        self.addToolBar(self.__profile)

    def _initWidgets(self, backend):
        """Set-up layout and plots."""
        self._histoHPlot = _SideHistogram(backend=backend, parent=self, direction=qt.Qt.Horizontal)
        widgetHandle = self._histoHPlot.getWidgetHandle()
        widgetHandle.setMinimumHeight(self.HISTOGRAMS_HEIGHT)
        widgetHandle.setMaximumHeight(self.HISTOGRAMS_HEIGHT)
        self._histoHPlot.setInteractiveMode('zoom')
        self._histoHPlot.setDataMargins(0., 0., 0.1, 0.1)
        self._histoHPlot.sigMouseMoved.connect(self._mouseMovedOnHistoH)
        self._histoHPlot.setProfileColor(self.HISTOGRAMS_COLOR)

        self._histoVPlot = _SideHistogram(backend=backend, parent=self, direction=qt.Qt.Vertical)
        widgetHandle = self._histoVPlot.getWidgetHandle()
        widgetHandle.setMinimumWidth(self.HISTOGRAMS_HEIGHT)
        widgetHandle.setMaximumWidth(self.HISTOGRAMS_HEIGHT)
        self._histoVPlot.setInteractiveMode('zoom')
        self._histoVPlot.setDataMargins(0.1, 0.1, 0., 0.)
        self._histoVPlot.sigMouseMoved.connect(self._mouseMovedOnHistoV)
        self._histoVPlot.setProfileColor(self.HISTOGRAMS_COLOR)

        self.setPanWithArrowKeys(True)
        self.setInteractiveMode('zoom')  # Color set in setColormap
        self.sigPlotSignal.connect(self._imagePlotCB)
        self.sigActiveImageChanged.connect(self._activeImageChangedSlot)

        self._radarView = RadarView(parent=self)
        self._radarView.setPlotWidget(self)

        self.__syncXAxis = SyncAxes([self.getXAxis(), self._histoHPlot.getXAxis()])
        self.__syncYAxis = SyncAxes([self.getYAxis(), self._histoVPlot.getYAxis()])

        self.__setCentralWidget()

    def __setCentralWidget(self):
        """Set central widget with all its content"""
        layout = qt.QGridLayout()
        layout.addWidget(self.getWidgetHandle(), 0, 0)
        layout.addWidget(self._histoVPlot, 0, 1)
        layout.addWidget(self._histoHPlot, 1, 0)
        layout.addWidget(self._radarView, 1, 1, 1, 2)
        layout.addWidget(self.getColorBarWidget(), 0, 2)

        self._radarView.setMinimumWidth(self.IMAGE_MIN_SIZE)
        self._radarView.setMinimumHeight(self.HISTOGRAMS_HEIGHT)
        self._histoHPlot.setMinimumWidth(self.IMAGE_MIN_SIZE)
        self._histoVPlot.setMinimumHeight(self.HISTOGRAMS_HEIGHT)

        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 0)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 0)

        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        centralWidget = qt.QWidget(self)
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    @docstring(PlotWidget)
    def setBackend(self, backend):
        # Use PlotWidget here since we override PlotWindow behavior
        PlotWidget.setBackend(self, backend)
        self.__setCentralWidget()

    def _dirtyCache(self):
        self._cache = None

    def getAggregationModeAction(self):
        return self.__aggregationModeAction

    def _aggregationModeChanged(self):
        item = self._getItem("image", self._imageLegend)
        if item is None:
            return
        aggregationMode = self.__aggregationModeAction.getAggregationMode()
        if aggregationMode is not None and isinstance(item, items.ImageDataAggregated):
            item.setAggregationMode(aggregationMode)
        else:
            # It means the item type have to be changed
            self.removeImage(self._imageLegend)
            image = item.getData(copy=False)
            if image is None:
                return
            origin = item.getOrigin()
            scale = item.getScale()
            self.setImage(image, origin, scale, copy=False, resetzoom=False)

    def getShowSideHistogramsAction(self):
        return self.__showSideHistogramsAction

    def setSideHistogramDisplayed(self, show):
        """Display or not the side histograms"""
        if self.isSideHistogramDisplayed() == show:
            return
        self._histoHPlot.setVisible(show)
        self._histoVPlot.setVisible(show)
        self._radarView.setVisible(show)
        self.__showSideHistogramsAction.setChecked(show)
        if show:
            # Probably have to be computed
            self._updateHistograms()

    def isSideHistogramDisplayed(self):
        """True if the side histograms are displayed"""
        return self._histoHPlot.isVisible()

    def _updateHistograms(self):
        """Update histograms content using current active image."""
        if not self.isSideHistogramDisplayed():
            # The histogram computation can be skipped
            return

        activeImage = self.getActiveImage()
        if activeImage is not None:
            xRange = self.getXAxis().getLimits()
            yRange = self.getYAxis().getLimits()
            result = computeProfileSumOnRange(activeImage, xRange, yRange, self._cache)
            self._cache = result
            self._histoHPlot.setProfileSum(result)
            self._histoVPlot.setProfileSum(result)

    # Plots event listeners

    def _imagePlotCB(self, eventDict):
        """Callback for imageView plot events."""
        if eventDict['event'] == 'mouseMoved':
            activeImage = self.getActiveImage()
            if activeImage is not None:
                data = activeImage.getData(copy=False)
                height, width = data.shape[0:2]

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
            self._updateHistograms()

    def _mouseMovedOnHistoH(self, x, y):
        if self._cache is None:
            return
        activeImage = self.getActiveImage()
        if activeImage is None:
            return

        xOrigin = activeImage.getOrigin()[0]
        xScale = activeImage.getScale()[0]

        minValue = xOrigin + xScale * self._cache.dataXRange[0]

        if x >= minValue:
            data = self._cache.histoH
            column = int((x - minValue) / xScale)
            if column >= 0 and column < data.shape[0]:
                self.valueChanged.emit(
                    float('nan'),
                    float(column + self._cache.dataXRange[0]),
                    data[column])

    def _mouseMovedOnHistoV(self, x, y):
        if self._cache is None:
            return
        activeImage = self.getActiveImage()
        if activeImage is None:
            return

        yOrigin = activeImage.getOrigin()[1]
        yScale = activeImage.getScale()[1]

        minValue = yOrigin + yScale * self._cache.dataYRange[0]

        if y >= minValue:
            data = self._cache.histoV
            row = int((y - minValue) / yScale)
            if row >= 0 and row < data.shape[0]:
                self.valueChanged.emit(
                    float(row + self._cache.dataYRange[0]),
                    float('nan'),
                    data[row])

    def _activeImageChangedSlot(self, previous, legend):
        """Handle Plot active image change.

        Resets side histograms cache
        """
        self._dirtyCache()
        self._updateHistograms()

    def setProfileWindowBehavior(self, behavior: Union[str, ProfileWindowBehavior]):
        """Set where profile widgets are displayed.

        :param ProfileWindowBehavior behavior:
        - 'popup': All profiles are displayed in pop-up windows
        - 'embedded': Horizontal, vertical and cross profiles are displayed in
          sides widgets, others are displayed in pop-up windows.
        """
        behavior = self.ProfileWindowBehavior.from_value(behavior)
        if behavior is not self.getProfileWindowBehavior():
            manager = self.__profile.getProfileManager()
            manager.clearProfile()
            manager.requestUpdateAllProfile()

            if behavior is self.ProfileWindowBehavior.EMBEDDED:
                horizontalProfileWindow = self._histoHPlot
                verticalProfileWindow = self._histoVPlot
            else:
                horizontalProfileWindow = None
                verticalProfileWindow = None

            manager.setSpecializedProfileWindow(
                rois.ProfileImageHorizontalLineROI, horizontalProfileWindow
            )
            manager.setSpecializedProfileWindow(
                rois.ProfileImageVerticalLineROI, verticalProfileWindow
            )
            self.__profileWindowBehavior = behavior

    def getProfileWindowBehavior(self) -> ProfileWindowBehavior:
        """Returns current profile display behavior.

        See :meth:`setProfileWindowBehavior` and :class:`ProfileWindowBehavior`
        """
        return self.__profileWindowBehavior

    def getProfileToolBar(self):
        """"Returns profile tools attached to this plot.

        :rtype: silx.gui.plot.PlotTools.ProfileToolBar
        """
        return self.__profile

    @property
    @deprecated(replacement="getProfileToolBar()")
    def profile(self):
        return self.getProfileToolBar()

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
                    data=numpy.array(self._cache.histoH, copy=True),
                    extent=self._cache.dataXRange)
            else:
                return dict(
                    data=numpy.array(self._cache.histoV, copy=True),
                    extent=(self._cache.dataYRange))

    def radarView(self):
        """Get the lower right radarView widget."""
        return self._radarView

    def setRadarView(self, radarView):
        """Change the lower right radarView widget.

        :param RadarView radarView: Widget subclassing RadarView to replace
                                    the lower right corner widget.
        """
        self._radarView = radarView
        self._radarView.setPlotWidget(self)
        self.centralWidget().layout().addWidget(self._radarView, 1, 1)

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
                 copy=True, reset=None, resetzoom=True):
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
        :param bool reset: Deprecated. Alias for `resetzoom`.
        :param bool resetzoom: Whether to reset zoom and ROI (default) or not.
        """
        self._dirtyCache()

        if reset is not None:
            resetzoom = reset

        assert len(origin) == 2
        assert len(scale) == 2
        assert scale[0] > 0
        assert scale[1] > 0

        if image is None:
            self.remove(self._imageLegend, kind='image')
            return

        data = numpy.array(image, order='C', copy=copy)
        if data.size == 0:
            self.remove(self._imageLegend, kind='image')
            return

        assert data.ndim == 2 or (data.ndim == 3 and data.shape[2] in (3, 4))

        aggregation = self.getAggregationModeAction().getAggregationMode()
        if data.ndim != 2 and aggregation is not None:
            # RGB/A with aggregation is not supported
            aggregation = items.ImageDataAggregated.Aggregation.NONE

        if aggregation is items.ImageDataAggregated.Aggregation.NONE:
            self.addImage(data,
                          legend=self._imageLegend,
                          origin=origin, scale=scale,
                          colormap=self.getColormap(),
                          resetzoom=False)
        else:
            item = self._getItem("image", self._imageLegend)
            if isinstance(item, items.ImageDataAggregated):
                item.setData(data)
                item.setOrigin(origin)
                item.setScale(scale)
            else:
                if isinstance(item, items.ImageDataAggregated):
                    imageItem = item
                    wasCreated = False
                else:
                    if item is not None:
                        self.removeImage(self._imageLegend)
                    imageItem = items.ImageDataAggregated()
                    imageItem.setName(self._imageLegend)
                    imageItem.setColormap(self.getColormap())
                    wasCreated = True
                imageItem.setData(data)
                imageItem.setOrigin(origin)
                imageItem.setScale(scale)
                imageItem.setAggregationMode(aggregation)
                if wasCreated:
                    self.addItem(imageItem)

        self.setActiveImage(self._imageLegend)
        self._updateHistograms()
        if resetzoom:
            self.resetZoom()


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
        menu.addAction(self.getShowSideHistogramsAction())

        self.__profileMenu = self.menuBar().addMenu('Profile')
        self.__updateProfileMenu()

        # Connect to ImageView's signal
        self.valueChanged.connect(self._statusBarSlot)

    def __updateProfileMenu(self):
        """Update actions available in 'Profile' menu"""
        profile = self.getProfileToolBar()
        self.__profileMenu.clear()
        self.__profileMenu.addAction(profile.hLineAction)
        self.__profileMenu.addAction(profile.vLineAction)
        self.__profileMenu.addAction(profile.crossAction)
        self.__profileMenu.addAction(profile.lineAction)
        self.__profileMenu.addAction(profile.clearAction)

    def _formatValueToString(self, value):
        try:
            if isinstance(value, numpy.ndarray):
                if len(value) == 4:
                    return "RGBA: %.3g, %.3g, %.3g, %.3g" % (value[0], value[1], value[2], value[3])
                elif len(value) == 3:
                    return "RGB: %.3g, %.3g, %.3g" % (value[0], value[1], value[2])
            else:
                return "Value: %g" % value
        except Exception:
            _logger.error("Error while formatting pixel value", exc_info=True)
            pass
        return "Value: %s" % value

    def _statusBarSlot(self, row, column, value):
        """Update status bar with coordinates/value from plots."""
        if numpy.isnan(row):
            msg = 'Column: %d, Sum: %g' % (int(column), value)
        elif numpy.isnan(column):
            msg = 'Row: %d, Sum: %g' % (int(row), value)
        else:
            msg_value = self._formatValueToString(value)
            msg = 'Position: (%d, %d), %s' % (int(row), int(column), msg_value)
        if self._dataInfo is not None:
            msg = self._dataInfo + ', ' + msg

        self.statusBar().showMessage(msg)

    @docstring(ImageView)
    def setProfileWindowBehavior(self, behavior: str):
        super().setProfileWindowBehavior(behavior)
        self.__updateProfileMenu()

    @docstring(ImageView)
    def setImage(self, image, *args, **kwargs):
        if hasattr(image, 'dtype') and hasattr(image, 'shape'):
            assert image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (3, 4))
            height, width = image.shape[0:2]
            dataInfo = 'Data: %dx%d (%s)' % (width, height, str(image.dtype))
        else:
            dataInfo = None

        if self._dataInfo != dataInfo:
            self._dataInfo = dataInfo
            self.statusBar().showMessage(self._dataInfo)

        # Set the new image in ImageView widget
        super(ImageViewMainWindow, self).setImage(image, *args, **kwargs)
