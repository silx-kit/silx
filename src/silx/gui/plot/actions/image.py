# /*##########################################################################
#
# Copyright (c) 2024 European Synchrotron Radiation Facility
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
"""
:mod:`silx.gui.plot.actions.image` provides a set of QAction relative to data processing
and outputs for a :class:`.PlotWidget`.

The following QAction are available:

- :class:`AggregationModeAction`
- :class:`DynamicColormapAction`
"""

from __future__ import annotations

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "19/09/2024"

import logging
from typing import Any

from silx.gui import qt
from silx.gui import icons
from silx.gui.colors import cursorColorForColormap
from .. import items
from .._utils.image import imageIndexSlices
from ..items.image_aggregated import ImageDataAggregated
from .PlotAction import PlotAction

_logger = logging.getLogger(__name__)


class DynamicColormapAction(PlotAction):
    """Checkable QAction adjusting the colormap range to the mouse position.

    When checked, each time the mouse moves over a colormapped image,
    the range of the image's colormap is updated from the data in a
    rectangular region centered on the mouse cursor.
    The range is computed with the colormap autoscale mode.

    The region is at least :meth:`getScreenRegionSize` pixels wide on screen
    and at least :meth:`getImageRegionSize` image pixels wide.
    It is displayed as a dashed rectangle.

    When unchecked, the colormap keeps the last range.

    This action is not part of the default toolbars.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent: qt.QObject | None = None):
        super().__init__(
            plot,
            icon="image-select-box",
            text="Dynamic colormap",
            tooltip="Adjust the colormap range to the data around the mouse cursor",
            triggered=self.__triggered,
            checkable=True,
            parent=parent,
        )
        self.__screenRegionSize = 20
        self.__imageRegionSize = 20
        self.__outline: items.Shape | None = None

    def getScreenRegionSize(self) -> int:
        """Returns the minimum size of the region in screen pixels."""
        return self.__screenRegionSize

    def setScreenRegionSize(self, size: int):
        """Set the minimum size of the region in screen pixels."""
        self.__screenRegionSize = max(1, int(size))

    def getImageRegionSize(self) -> int:
        """Returns the minimum size of the region in image pixels."""
        return self.__imageRegionSize

    def setImageRegionSize(self, size: int):
        """Set the minimum size of the region in image pixels."""
        self.__imageRegionSize = max(1, int(size))

    def __triggered(self, checked: bool):
        plot = self.plot
        if plot is None:
            return
        if checked:
            plot.sigPlotSignal.connect(self.__plotSignalReceived)
        else:
            plot.sigPlotSignal.disconnect(self.__plotSignalReceived)
            self.__removeOutline()

    @staticmethod
    def __isColormappedImage(item: items.Item) -> bool:
        return isinstance(item, items.ImageBase) and isinstance(
            item, items.ColormapMixIn
        )

    def __plotSignalReceived(self, event: dict[str, Any]):
        if event.get("event") != "mouseMoved":
            return
        if not self.__updateColormap(event):
            self.__hideOutline()

    def __updateColormap(self, event: dict[str, Any]) -> bool:
        """Update the colormap of the image under the mouse.

        :returns: True if a colormap was updated, False otherwise
        """
        plot = self.plot
        if plot is None:
            return False

        xPixel, yPixel = event["xpixel"], event["ypixel"]
        result = next(
            plot.pickItems(xPixel, yPixel, condition=self.__isColormappedImage),
            None,
        )
        if result is None:
            return False

        image = result.getItem()
        colormap = image.getColormap()
        if colormap is None or not colormap.isEditable():
            return False
        data = image.getColormappedData(copy=False)
        if data is None or data.ndim != 2:
            return False

        bounds = self.__regionBounds(image, event)
        slices = imageIndexSlices(
            image.getOrigin(), image.getScale(), data.shape, bounds
        )
        if slices is None:
            return False

        vmin, vmax = colormap._computeAutoscaleRange(data[slices])
        if vmin is None or vmax is None:
            return False
        colormap.setVRange(vmin, vmax)

        self.__showOutline(image, slices)
        return True

    def __regionBounds(
        self, image: items.ImageBase, event: dict[str, Any]
    ) -> tuple[float, float, float, float]:
        """Returns the region in data coordinates as (xmin, xmax, ymin, ymax)"""
        plot = self.plot
        xPixel, yPixel = event["xpixel"], event["ypixel"]
        halfScreen = self.getScreenRegionSize() / 2
        x0, y0 = plot.pixelToData(xPixel - halfScreen, yPixel - halfScreen)
        x1, y1 = plot.pixelToData(xPixel + halfScreen, yPixel + halfScreen)

        # Make sure the region covers enough image pixels
        sx, sy = image.getScale()
        halfImage = self.getImageRegionSize() / 2
        x, y = event["x"], event["y"]
        xmin = min(x0, x1, x - halfImage * abs(sx))
        xmax = max(x0, x1, x + halfImage * abs(sx))
        ymin = min(y0, y1, y - halfImage * abs(sy))
        ymax = max(y0, y1, y + halfImage * abs(sy))
        return xmin, xmax, ymin, ymax

    def __showOutline(self, image: items.ImageBase, slices: tuple[slice, slice]):
        """Display the outline of the image pixels used for the range"""
        plot = self.plot
        if self.__outline is None or self.__outline.getPlot() is not plot:
            # Create it or re-add it if it was removed (e.g., plot cleared)
            self.__outline = items.Shape("rectangle")
            self.__outline.setName("__DynamicColormapAction_outline__")
            self.__outline.setOverlay(True)
            self.__outline.setFill(False)
            self.__outline.setLineStyle("--")
            plot.addItem(self.__outline)

        rows, columns = slices
        ox, oy = image.getOrigin()
        sx, sy = image.getScale()
        xs = ox + columns.start * sx, ox + columns.stop * sx
        ys = oy + rows.start * sy, oy + rows.stop * sy
        self.__outline.setPoints(((min(xs), min(ys)), (max(xs), max(ys))))
        self.__outline.setColor(cursorColorForColormap(image.getColormap().getName()))
        self.__outline.setVisible(True)

    def __hideOutline(self):
        if self.__outline is not None:
            self.__outline.setVisible(False)

    def __removeOutline(self):
        if self.__outline is not None:
            plot = self.__outline.getPlot()
            if plot is not None:
                plot.removeItem(self.__outline)
            self.__outline = None


class AggregationModeAction(qt.QWidgetAction):
    """Action providing filters for an aggregated image"""

    sigAggregationModeChanged = qt.Signal()
    """Signal emitted when the aggregation mode has changed"""

    def __init__(self, parent):
        qt.QWidgetAction.__init__(self, parent)

        toolButton = qt.QToolButton(parent)

        filterAction = qt.QAction(self)
        filterAction.setText("No filter")
        filterAction.setCheckable(True)
        filterAction.setChecked(True)
        filterAction.setProperty("aggregation", ImageDataAggregated.Aggregation.NONE)
        densityNoFilterAction = filterAction

        filterAction = qt.QAction(self)
        filterAction.setText("Max filter")
        filterAction.setCheckable(True)
        filterAction.setProperty("aggregation", ImageDataAggregated.Aggregation.MAX)
        densityMaxFilterAction = filterAction

        filterAction = qt.QAction(self)
        filterAction.setText("Mean filter")
        filterAction.setCheckable(True)
        filterAction.setProperty("aggregation", ImageDataAggregated.Aggregation.MEAN)
        densityMeanFilterAction = filterAction

        filterAction = qt.QAction(self)
        filterAction.setText("Min filter")
        filterAction.setCheckable(True)
        filterAction.setProperty("aggregation", ImageDataAggregated.Aggregation.MIN)
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
            return ImageDataAggregated.Aggregation.NONE
        return densityAction.property("aggregation")
