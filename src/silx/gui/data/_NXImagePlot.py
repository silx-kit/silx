from __future__ import annotations

import h5py
import numpy

from silx.io.commonh5 import Dataset
from silx.io.nxdata._utils import AxisDataset

from ..colors import Colormap
from ..plot.actions.image import AggregationModeAction
from ..plot.items.image_aggregated import ImageDataAggregated
from ..plot.items.types import AxisScaleType
from ._BaseImagePlot import BaseImagePlot
from ._utils import getAxesCalib, setImageCoords


class NXImagePlot(BaseImagePlot):
    """
    Widget for plotting an image from a multi-dimensional signal array
    and two 1D axes array.

    Sliders are provided to select indices on the first (n - 2) dimensions of
    the signal array, and the plot is updated to show the image corresponding
    to the selection.

    The dimensions can be changed when the signal array has more than 2 dimensions.

    If one or both of the axes does not have regularly spaced values, the
    the image is plotted as a coloured scatter plot.
    """

    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super().__init__(parent)
        self._axesScales = None

        self._plot.setDefaultColormap(
            Colormap(
                name="viridis", vmin=None, vmax=None, normalization=Colormap.LINEAR
            )
        )
        self._plot.getIntensityHistogramAction().setVisible(True)
        self._plot.setKeepDataAspectRatio(True)

        self.__aggregationModeAction = AggregationModeAction(parent=self)
        self.getPlot().toolBar().addAction(self.__aggregationModeAction)
        self.__aggregationModeAction.sigAggregationModeChanged.connect(
            self._aggregationModeChanged
        )

    def getAggregationModeAction(self) -> AggregationModeAction:
        """Action toggling the aggregation mode action"""
        return self.__aggregationModeAction

    def _aggregationModeChanged(self):
        item = self.getPlot()._getItem("image")

        if item is None:
            return

        if isinstance(item, ImageDataAggregated):
            item.setAggregationMode(
                self.getAggregationModeAction().getAggregationMode()
            )

    def setImageData(
        self,
        signals: list[h5py.Dataset | Dataset],
        axes: list[AxisDataset] | None = None,
        signals_names: list[str] | None = None,
        axes_names: list[str] | None = None,
        axes_scales: list[AxisScaleType | None] | None = None,
        title: str | None = None,
        selection: tuple[int | slice, ...] | None = None,
    ):
        """
        Sets signals, axes and axes metadata that will be used to set the displayed image.

        :param signals: list of n-D datasets or list of 3D datasets interpreted as RGBA image.
        :param axes: list of 1D datasets to be used as axes
        :param signals_names: Names for each image, used as subtitle and legend.
        :param axes_names: Names for each axis, used as graph label.
        :param axes_scales: Scale of axes
        :param title: Graph title
        :param selection: Initial selection of the dimensions.
        """
        self._axesScales = axes_scales
        super().setImageData(
            signals,
            axes=axes,
            signalsNames=signals_names,
            axesNames=axes_names,
            title=title,
            selection=selection,
        )

    def _addItemToPlot(self, xAxis, yAxis):
        """Updates the image axes. Called when the user selects a different axis than the displayed one."""

        self._plot.remove(
            kind=(
                "scatter",
                "image",
            )
        )
        image = self._getImageToDisplay()
        xCalib, yCalib = getAxesCalib(image.shape[:2], xAxis, yAxis)
        if xCalib.is_affine() and yCalib.is_affine():
            if image.ndim != 2:
                raise ValueError(f"image dims should be 2. Got {image.ndim}")
            imageItem = ImageDataAggregated()
            imageItem.setColormap(self._plot.getDefaultColormap())
            imageItem.setAggregationMode(
                self.getAggregationModeAction().getAggregationMode()
            )
            setImageCoords(imageItem, xCalib, yCalib)
            imageItem.setName(self._signalSelector.getCurrentSignalName())
            imageItem.setData(image)

            self._plot.addItem(imageItem)
            self._plot.setActiveImage(imageItem)
            self._plot.getXAxis().setScale("linear")
            self._plot.getYAxis().setScale("linear")
        else:
            if self._axesScales:
                xAxisIndex, yAxisIndex = self._getXYIndices()
                xAxisScale = self._axesScales[xAxisIndex]
                if xAxisScale is not None:
                    self._plot.getXAxis().setScale(xAxisScale)
                yAxisScale = self._axesScales[yAxisIndex]
                if yAxisScale is not None:
                    self._plot.getYAxis().setScale(yAxisScale)

            xScatter, yScatter = numpy.meshgrid(xAxis, yAxis)
            self._plot.addScatter(
                numpy.ravel(xScatter),
                numpy.ravel(yScatter),
                numpy.ravel(image),
                legend=self._signalSelector.getCurrentSignalName(),
            )
