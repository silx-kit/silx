"""Compute a colormap range from the data around the mouse cursor"""

from __future__ import annotations

import weakref
from typing import Any

from .. import qt
from ..colors import cursorColorForColormap
from ..qt import inspect as qtinspect
from ..plot import items
from ..plot import PlotWidget
from ..plot.items.core import LineStyleType
from ..plot._utils.image import imageIndexSlices


class ScaleToCursorTool(qt.QObject):
    """Follow the mouse over an image and compute a colormap range from it.

    The range is computed with the colormap autoscale mode from the data in a
    rectangular region centered on the mouse cursor.
    This region is displayed as a dashed rectangle.
    """

    MIN_SCREEN_REGION_SIZE: int = 20
    """Minimum size of the region in screen pixels"""

    MIN_IMAGE_REGION_SIZE: int = 20
    """Minimum size of the region in image pixels"""

    OUTLINE_LINE_STYLE: LineStyleType = "--"
    """Line style of the region outline."""
    
    OUTLINE_LINE_WIDTH: float = 1.
    """Line width of the region outline."""

    sigRangeChanged = qt.Signal(object, object)
    """Signal emitted with the new (vmin, vmax) range"""

    sigStopped = qt.Signal()
    """Signal emitted when the tool stopped because the image was removed"""

    def __init__(self, parent: qt.QObject | None = None):
        super().__init__(parent)
        self.__item: weakref.ref[items.ImageBase] | None = None
        self.__plot: weakref.ref[PlotWidget] | None = None
        self.__outline: items.Shape | None = None

    def getItem(self) -> items.ImageBase | None:
        """Returns the image the tool is attached to, if any."""
        return None if self.__item is None else self.__item()

    def start(self, item: items.ImageBase) -> bool:
        """Start following the mouse over the given image.

        :returns: True if started, False if the image is not in a plot
        """
        self.stop()
        plot = item.getPlot()
        if plot is None:
            return False
        self.__item = weakref.ref(item)
        self.__plot = weakref.ref(plot)
        plot.sigPlotSignal.connect(self.__plotSignalReceived)
        plot.sigItemRemoved.connect(self.__itemRemoved)
        return True

    def stop(self):
        """Stop following the mouse and remove the region outline."""
        plot = None if self.__plot is None else self.__plot()
        if plot is not None and qtinspect.isValid(plot):
            plot.sigPlotSignal.disconnect(self.__plotSignalReceived)
            plot.sigItemRemoved.disconnect(self.__itemRemoved)
            if self.__outline is not None and self.__outline.getPlot() is plot:
                plot.removeItem(self.__outline)
        self.__outline = None
        self.__item = None
        self.__plot = None

    def __itemRemoved(self, item: items.Item):
        if item is self.getItem():
            self.stop()
            self.sigStopped.emit()

    def __plotSignalReceived(self, event: dict[str, Any]):
        if event.get("event") != "mouseMoved":
            return
        if not self.__updateRange(event):
            if self.__outline is not None:
                self.__outline.setVisible(False)

    def __updateRange(self, event: dict[str, Any]) -> bool:
        """Compute the range from the region around the mouse.

        :returns: True if the range was updated, False otherwise
        """
        image = self.getItem()
        if image is None:
            return False
        # Only handle the mouse when it is over the image
        if image.pick(event["xpixel"], event["ypixel"]) is None:
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

        vmin, vmax = image.getColormap()._computeAutoscaleRange(data[slices])
        if vmin is None or vmax is None:
            return False

        self.__showOutline(image, slices)
        self.sigRangeChanged.emit(vmin, vmax)
        return True

    def __regionBounds(
        self, image: items.ImageBase, event: dict[str, Any]
    ) -> tuple[float, float, float, float]:
        """Returns the region in data coordinates as (xmin, xmax, ymin, ymax)"""
        plot = self.__plot()
        xPixel, yPixel = event["xpixel"], event["ypixel"]
        halfScreen = self.MIN_SCREEN_REGION_SIZE / 2
        x0, y0 = plot.pixelToData(xPixel - halfScreen, yPixel - halfScreen)
        x1, y1 = plot.pixelToData(xPixel + halfScreen, yPixel + halfScreen)

        # Make sure the region covers enough image pixels
        sx, sy = image.getScale()
        halfImage = self.MIN_IMAGE_REGION_SIZE / 2
        x, y = event["x"], event["y"]
        xmin = min(x0, x1, x - halfImage * abs(sx))
        xmax = max(x0, x1, x + halfImage * abs(sx))
        ymin = min(y0, y1, y - halfImage * abs(sy))
        ymax = max(y0, y1, y + halfImage * abs(sy))
        return xmin, xmax, ymin, ymax

    def __createOutline(self) -> items.Shape:
        """Create the shape displaying the region used for the range."""
        outline = items.Shape("rectangle")
        outline.setName("__ScaleToCursorTool_outline__")
        outline.setOverlay(True)
        outline.setFill(False)
        outline.setLineStyle(self.OUTLINE_LINE_STYLE)
        outline.setLineWidth(self.OUTLINE_LINE_WIDTH)
        return outline
    
    def __showOutline(self, image: items.ImageBase, slices: tuple[slice, slice]):
        """Display the outline of the image pixels used for the range"""
        plot = self.__plot()
        if self.__outline is None or self.__outline.getPlot() is not plot:
            # Create it or re-add it if it was removed (e.g., plot cleared)
            self.__outline = self.__createOutline()
            plot.addItem(self.__outline)

        rows, columns = slices
        ox, oy = image.getOrigin()
        sx, sy = image.getScale()
        xs = ox + columns.start * sx, ox + columns.stop * sx
        ys = oy + rows.start * sy, oy + rows.stop * sy
        self.__outline.setPoints(((min(xs), min(ys)), (max(xs), max(ys))))
        self.__outline.setColor(cursorColorForColormap(image.getColormap().getName()))
        self.__outline.setVisible(True)
