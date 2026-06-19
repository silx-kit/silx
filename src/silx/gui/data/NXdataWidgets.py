# /*##########################################################################
#
# Copyright (c) 2017-2023 European Synchrotron Radiation Facility
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
"""This module defines widgets used by _NXdataView."""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "12/11/2018"

import logging
from typing import Literal

import h5py
import numpy

from silx.gui import qt
from silx.gui.colors import Colormap
from silx.gui.data._RgbaImagePlot import BaseImagePlot
from silx.gui.data._SignalSelector import SignalSelector
from silx.gui.data.NumpyAxesSelector import NumpyAxesSelector
from silx.gui.plot import ScatterView, StackView
from silx.gui.plot.actions.image import AggregationModeAction
from silx.gui.plot.ComplexImageView import ComplexImageView
from silx.gui.plot.items.image_aggregated import ImageDataAggregated
from silx.io.commonh5 import Dataset
from silx.math.calibration import ArrayCalibration, LinearCalibration, NoCalibration

from ._utils import getAxesCalib, setImageCoords
from ..utils import blockSignals
from ...utils.deprecation import deprecated, deprecated_warning
from .ArrayCurvePlot import ArrayCurvePlot as _ArrayCurvePlot

_logger = logging.getLogger(__name__)


class ArrayCurvePlot(_ArrayCurvePlot):
    def __init__(self, *args, **kwargs):
        deprecated_warning(
            type_="Class",
            name="ArrayCurvePlot",
            reason="ArrayCurvePlot was moved to silx.gui.data.ArrayCurvePlot. Please import it from here.",
            replacement="from silx.gui.data.ArrayCurvePlot import ArrayCurvePlot",
            since_version="3.0.2",
        )
        super().__init__(*args, **kwargs)


class XYVScatterPlot(qt.QWidget):
    """
    Widget for plotting one or more scatters
    (with identical x, y coordinates).
    """

    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super().__init__(parent)

        self.__y_axis = None
        """1D array"""
        self.__y_axis_name = None
        self.__values = None
        """List of 1D arrays (for multiple scatters with identical
        x, y coordinates)"""

        self.__x_axis = None
        self.__x_axis_name = None
        self.__x_axis_errors = None
        self.__y_axis = None
        self.__y_axis_name = None
        self.__y_axis_errors = None

        self._plot = ScatterView(self)
        self._plot.setColormap(
            Colormap(
                name="viridis", vmin=None, vmax=None, normalization=Colormap.LINEAR
            )
        )

        self._signalSelector = SignalSelector(parent=self)
        self._signalSelector.selectionChanged.connect(self._signalChanges)
        self._signalSelector.setToolTip("Select signal")

        layout = qt.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot, 0, 0)
        layout.addWidget(self._signalSelector, 1, 0)

        self.setLayout(layout)

    def _signalChanges(self, value):
        self._updateScatter()

    def getScatterView(self):
        """Returns the :class:`ScatterView` used for the display

        :rtype: ScatterView
        """
        return self._plot

    def getPlot(self):
        """Returns the plot used for the display

        :rtype: PlotWidget
        """
        return self._plot.getPlotWidget()

    def setScattersData(
        self,
        y,
        x,
        values,
        yerror=None,
        xerror=None,
        ylabel=None,
        xlabel=None,
        title="",
        scatter_titles=None,
        xscale=None,
        yscale=None,
    ):
        """

        :param ndarray y: 1D array for y (vertical) coordinates.
        :param ndarray x: 1D array for x coordinates.
        :param List[ndarray] values: List of 1D arrays of values.
            This will be used to compute the color map and assign colors
            to the points. There should be as many arrays in the list as
            scatters to be represented.
        :param ndarray yerror: 1D array of errors for y (same shape), or None.
        :param ndarray xerror: 1D array of errors for x, or None
        :param str ylabel: Label for Y axis
        :param str xlabel: Label for X axis
        :param str title: Main graph title
        :param List[str] scatter_titles:  Subtitles (one per scatter)
        :param str xscale: Scale of X axis in (None, 'linear', 'log')
        :param str yscale: Scale of Y axis in (None, 'linear', 'log')
        """
        self.__y_axis = y
        self.__x_axis = x
        self.__x_axis_name = xlabel or "X"
        self.__y_axis_name = ylabel or "Y"
        self.__x_axis_errors = xerror
        self.__y_axis_errors = yerror
        self.__values = values

        self.__graph_title = title or ""
        self.__scatter_titles = scatter_titles

        self._signalSelector.selectionChanged.disconnect(self._signalChanges)
        self._signalSelector.setSignalNames(scatter_titles)
        if len(scatter_titles) > 1:
            self._signalSelector.show()
        else:
            self._signalSelector.hide()
        self._signalSelector.setSignalIndex(0)
        self._signalSelector.selectionChanged.connect(self._signalChanges)

        if xscale is not None:
            self._plot.getXAxis().setScale("log" if xscale == "log" else "linear")
        if yscale is not None:
            self._plot.getYAxis().setScale("log" if yscale == "log" else "linear")

        self._updateScatter()

    def _updateScatter(self):
        x = self.__x_axis
        y = self.__y_axis

        idx = self._signalSelector.getSignalIndex()

        if self.__graph_title:
            title = self.__graph_title  # main NXdata @title
            if len(self.__scatter_titles) > 1:
                # Append dataset name only when there is many datasets
                title += "\n" + self.__scatter_titles[idx]
        else:
            title = self.__scatter_titles[idx]  # scatter dataset name

        self._plot.setGraphTitle(title)
        self._plot.setData(
            x,
            y,
            self.__values[idx],
            xerror=self.__x_axis_errors,
            yerror=self.__y_axis_errors,
        )
        self._plot.resetZoom()
        self._plot.getXAxis().setLabel(self.__x_axis_name)
        self._plot.getYAxis().setLabel(self.__y_axis_name)

    def clear(self):
        self._plot.getPlotWidget().clear()


class ArrayImagePlot(BaseImagePlot):
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
        axes: list[h5py.Dataset | Dataset] | None = None,
        signals_names: list[str] | None = None,
        axes_names: list[str] | None = None,
        axes_scales: list[Literal["linear", "log"] | None] | None = None,
        title: str | None = None,
    ):
        """
        Sets signals, axes and axes metadata that will be used to set the displayed image.

        :param signals: list of n-D datasets or list of 3D datasets interpreted as RGBA image.
        :param axes: list of 1D datasets to be used as axes
        :param signals_names: Names for each image, used as subtitle and legend.
        :param axes_names: Names for each axis, used as graph label.
        :param axes_scales: Scale of axes in (None, 'linear', 'log')
        :param title: Graph title
        :param isRgba: True if data is a 3D RGBA image
        """
        if len(signals) == 0:
            raise ValueError("Cannot set image data from empty signals")
        self._signals = signals
        self._axes = axes
        self._axesNames = axes_names
        self._axesScales = axes_scales
        self._title = title

        with blockSignals(self._axesSelector, self._signalSelector):
            self._axesSelector.clear()
            self._axesSelector.setAxisNames(["Y", "X"])
            self._axesSelector.setNamedAxesSelectorVisibility(True)

            # Labels need to be set before the data
            if self._axesNames:
                self._axesSelector.setLabels(self._axesNames)
            self._axesSelector.setData(signals[0])

            if len(signals[0].shape) <= 2:
                self._axesSelector.hide()
            else:
                self._axesSelector.show()

            self._signalSelector.setSignalNames(signals_names)
            if len(signals) > 1:
                self._signalSelector.show()
            else:
                self._signalSelector.hide()
            self._signalSelector.setSignalIndex(0)

        self._updateImageAxes()
        self._plot.resetZoom()

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
                yAxisScale = self._axesScales[yAxisIndex]
            else:
                xAxisScale = None
                yAxisScale = None

            self._plot.setXAxisLogarithmic(xAxisScale == "log")
            self._plot.setYAxisLogarithmic(yAxisScale == "log")

            xScatter, yScatter = numpy.meshgrid(xAxis, yAxis)
            self._plot.addScatter(
                numpy.ravel(xScatter),
                numpy.ravel(yScatter),
                numpy.ravel(image),
                legend=self._signalSelector.getCurrentSignalName(),
            )


class ArrayComplexImagePlot(qt.QWidget):
    """
    Widget for plotting an image of complex from a multi-dimensional signal array
    and two 1D axes array.

    The signal array can have an arbitrary number of dimensions, the only
    limitation being that the last two dimensions must have the same length as
    the axes arrays.

    Sliders are provided to select indices on the first (n - 2) dimensions of
    the signal array, and the plot is updated to show the image corresponding
    to the selection.

    If one or both of the axes does not have regularly spaced values, the
    the image is plotted as a coloured scatter plot.
    """

    def __init__(self, parent=None, colormap=None):
        """

        :param parent: Parent QWidget
        """
        super().__init__(parent)

        self.__signals = None
        self.__signals_names = None
        self.__x_axis = None
        self.__x_axis_name = None
        self.__y_axis = None
        self.__y_axis_name = None

        self._plot = ComplexImageView(self)
        if colormap is not None:
            for mode in (
                ComplexImageView.ComplexMode.ABSOLUTE,
                ComplexImageView.ComplexMode.SQUARE_AMPLITUDE,
                ComplexImageView.ComplexMode.REAL,
                ComplexImageView.ComplexMode.IMAGINARY,
            ):
                self._plot.setColormap(colormap, mode)

        self._plot.getPlot().getIntensityHistogramAction().setVisible(True)
        self._plot.setKeepDataAspectRatio(True)
        maskToolWidget = self._plot.getPlot().getMaskToolsDockWidget().widget()
        maskToolWidget.setItemMaskUpdated(True)

        # not closable
        self._axesSelector = NumpyAxesSelector(self)
        self._axesSelector.setNamedAxesSelectorVisibility(False)
        self._axesSelector.selectionChanged.connect(self._updateImage)

        self._signalSelector = SignalSelector(parent=self)
        self._signalSelector.selectionChanged.connect(self._signalChanges)
        self._signalSelector.setToolTip("Select signal")

        layout = qt.QVBoxLayout()
        layout.addWidget(self._plot)
        layout.addWidget(self._axesSelector)
        layout.addWidget(self._signalSelector)

        self.setLayout(layout)

    def _signalChanges(self, value):
        self._updateImage()

    def getPlot(self):
        """Returns the plot used for the display

        :rtype: PlotWidget
        """
        return self._plot.getPlot()

    def setImageData(
        self,
        signals,
        x_axis=None,
        y_axis=None,
        signals_names=None,
        xlabel=None,
        ylabel=None,
        title=None,
        keep_ratio: bool = True,
    ):
        """

        :param signals: list of n-D datasets, whose last 2 dimensions are used as the
            image's values, or list of 3D datasets interpreted as RGBA image.
        :param x_axis: 1-D dataset used as the image's x coordinates. If
            provided, its lengths must be equal to the length of the last
            dimension of ``signal``.
        :param y_axis: 1-D dataset used as the image's y. If provided,
            its lengths must be equal to the length of the 2nd to last
            dimension of ``signal``.
        :param signals_names: Names for each image, used as subtitle and legend.
        :param xlabel: Label for X axis
        :param ylabel: Label for Y axis
        :param title: Graph title
        :param keep_ratio: Toggle plot keep aspect ratio
        """
        self._axesSelector.selectionChanged.disconnect(self._updateImage)
        self._signalSelector.selectionChanged.disconnect(self._signalChanges)

        self.__signals = signals
        self.__signals_names = signals_names
        self.__x_axis = x_axis
        self.__x_axis_name = xlabel
        self.__y_axis = y_axis
        self.__y_axis_name = ylabel
        self.__title = title

        self._axesSelector.clear()
        self._axesSelector.setAxisNames(["Y", "X"])
        self._axesSelector.setData(signals[0])

        if len(signals[0].shape) <= 2:
            self._axesSelector.hide()
        else:
            self._axesSelector.show()

        self._signalSelector.setSignalNames(signals_names)
        if len(signals) > 1:
            self._signalSelector.show()
        else:
            self._signalSelector.hide()
        self._signalSelector.setSignalIndex(0)

        self._updateImage()
        self._plot.setKeepDataAspectRatio(keep_ratio)
        self._plot.getPlot().resetZoom()

        self._axesSelector.selectionChanged.connect(self._updateImage)
        self._signalSelector.selectionChanged.connect(self._signalChanges)

    def _updateImage(self):
        axes_selection = self._axesSelector.selection()
        signal_index = self._signalSelector.getSignalIndex()

        images = [img[axes_selection] for img in self.__signals]
        image = images[signal_index]

        x_axis = self.__x_axis
        y_axis = self.__y_axis

        xcalib, ycalib = getAxesCalib(image.shape[0:2], x_axis, y_axis)
        self._plot.setData(image)
        if xcalib.is_affine():
            xorigin, xscale = xcalib(0), xcalib.get_slope()
        else:
            _logger.warning("Unsupported complex image X axis calibration")
            xorigin, xscale = 0.0, 1.0

        if ycalib.is_affine():
            yorigin, yscale = ycalib(0), ycalib.get_slope()
        else:
            _logger.warning("Unsupported complex image Y axis calibration")
            yorigin, yscale = 0.0, 1.0

        self._plot.setOrigin((xorigin, yorigin))
        self._plot.setScale((xscale, yscale))

        if self.__title:
            title = self.__title
            if len(self.__signals_names) > 1:
                # Append dataset name only when there is many datasets
                title += "\n" + self.__signals_names[signal_index]
        else:
            title = self.__signals_names[signal_index]
        self._plot.setGraphTitle(title)
        self._plot.getXAxis().setLabel(self.__x_axis_name)
        self._plot.getYAxis().setLabel(self.__y_axis_name)

    def clear(self):
        old = self._axesSelector.blockSignals(True)
        self._axesSelector.clear()
        self._axesSelector.blockSignals(old)
        self._plot.setData(None)


@deprecated(since_version="3.0.0")
class ArrayStackPlot(qt.QWidget):
    """
    Widget for plotting a n-D array (n >= 3) as a stack of images.
    Three axis arrays can be provided to calibrate the axes.

    The signal array can have an arbitrary number of dimensions, the only
    limitation being that the last 3 dimensions must have the same length as
    the axes arrays.

    Sliders are provided to select indices on the first (n - 3) dimensions of
    the signal array, and the plot is updated to load the stack corresponding
    to the selection.
    """

    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super().__init__(parent)

        self.__signal = None
        self.__signal_name = None
        # the Z, Y, X axes apply to the last three dimensions of the signal
        # (in that order)
        self.__z_axis = None
        self.__z_axis_name = None
        self.__y_axis = None
        self.__y_axis_name = None
        self.__x_axis = None
        self.__x_axis_name = None

        self._stack_view = StackView(self)
        maskToolWidget = (
            self._stack_view.getPlotWidget().getMaskToolsDockWidget().widget()
        )
        maskToolWidget.setItemMaskUpdated(True)

        self._hline = qt.QFrame(self)
        self._hline.setFrameStyle(qt.QFrame.HLine)
        self._hline.setFrameShadow(qt.QFrame.Sunken)
        self._legend = qt.QLabel(self)
        self._axesSelector = NumpyAxesSelector(self)
        self._axesSelector.setNamedAxesSelectorVisibility(False)
        self.__axes_selector_is_connected = False

        layout = qt.QVBoxLayout()
        layout.addWidget(self._stack_view)
        layout.addWidget(self._hline)
        layout.addWidget(self._legend)
        layout.addWidget(self._axesSelector)

        self.setLayout(layout)

    def getStackView(self):
        """Returns the plot used for the display

        :rtype: StackView
        """
        return self._stack_view

    def setStackData(
        self,
        signal,
        x_axis=None,
        y_axis=None,
        z_axis=None,
        signal_name=None,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        title=None,
    ):
        """

        :param signal: n-D dataset, whose last 3 dimensions are used as the
            3D stack values.
        :param x_axis: 1-D dataset used as the image's x coordinates. If
            provided, its lengths must be equal to the length of the last
            dimension of ``signal``.
        :param y_axis: 1-D dataset used as the image's y. If provided,
            its lengths must be equal to the length of the 2nd to last
            dimension of ``signal``.
        :param z_axis: 1-D dataset used as the image's z. If provided,
            its lengths must be equal to the length of the 3rd to last
            dimension of ``signal``.
        :param signal_name: Label used in the legend
        :param xlabel: Label for X axis
        :param ylabel: Label for Y axis
        :param zlabel: Label for Z axis
        :param title: Graph title
        """
        if self.__axes_selector_is_connected:
            self._axesSelector.selectionChanged.disconnect(self._updateStack)
            self.__axes_selector_is_connected = False

        self.__signal = signal
        self.__signal_name = signal_name or ""
        self.__x_axis = x_axis
        self.__x_axis_name = xlabel
        self.__y_axis = y_axis
        self.__y_axis_name = ylabel
        self.__z_axis = z_axis
        self.__z_axis_name = zlabel

        self._axesSelector.setData(signal)
        self._axesSelector.setAxisNames(["Y", "X", "Z"])

        self._stack_view.setGraphTitle(title or "")
        # by default, the z axis is the image position (dimension not plotted)
        self._stack_view.getPlotWidget().getXAxis().setLabel(self.__x_axis_name or "X")
        self._stack_view.getPlotWidget().getYAxis().setLabel(self.__y_axis_name or "Y")

        self._updateStack()

        ndims = len(signal.shape)
        self._stack_view.setFirstStackDimension(ndims - 3)

        # the legend label shows the selection slice producing the volume
        # (only interesting for ndim > 3)
        if ndims > 3:
            self._axesSelector.setVisible(True)
            self._legend.setVisible(True)
            self._hline.setVisible(True)
        else:
            self._axesSelector.setVisible(False)
            self._legend.setVisible(False)
            self._hline.setVisible(False)

        if not self.__axes_selector_is_connected:
            self._axesSelector.selectionChanged.connect(self._updateStack)
            self.__axes_selector_is_connected = True

    @staticmethod
    def _get_origin_scale(axis):
        """Assuming axis is a regularly spaced 1D array,
        return a tuple (origin, scale) where:
            - origin = axis[0]
            - scale = (axis[n-1] - axis[0]) / (n -1)
        :param axis: 1D numpy array
        :return: Tuple (axis[0], (axis[-1] - axis[0]) / (len(axis) - 1))
        """
        return axis[0], (axis[-1] - axis[0]) / (len(axis) - 1)

    def _updateStack(self):
        """Update displayed stack according to the current axes selector
        data."""
        stack = self._axesSelector.selectedData()
        x_axis = self.__x_axis
        y_axis = self.__y_axis
        z_axis = self.__z_axis

        calibrations = []
        for axis in [z_axis, y_axis, x_axis]:
            if axis is None:
                calibrations.append(NoCalibration())
            elif len(axis) == 2:
                calibrations.append(
                    LinearCalibration(y_intercept=axis[0], slope=axis[1])
                )
            else:
                calibrations.append(ArrayCalibration(axis))

        legend = self.__signal_name + "["
        for sl in self._axesSelector.selection():
            if sl == slice(None):
                legend += ":, "
            else:
                legend += str(sl) + ", "
        legend = legend[:-2] + "]"
        self._legend.setText("Displayed data: " + legend)

        self._stack_view.setStack(stack, calibrations=calibrations)
        self._stack_view.setStackName(self.__signal_name)
        self._stack_view.setLabels(
            labels=[self.__z_axis_name, self.__y_axis_name, self.__x_axis_name]
        )

    def clear(self):
        old = self._axesSelector.blockSignals(True)
        self._axesSelector.clear()
        self._axesSelector.blockSignals(old)
        self._stack_view.clear()


class ArrayVolumePlot(qt.QWidget):
    """
    Widget for plotting a n-D array (n >= 3) as a 3D scalar field.
    Three axis arrays can be provided to calibrate the axes.

    The signal array can have an arbitrary number of dimensions, the only
    limitation being that the last 3 dimensions must have the same length as
    the axes arrays.

    Sliders are provided to select indices on the first (n - 3) dimensions of
    the signal array, and the plot is updated to load the stack corresponding
    to the selection.
    """

    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super().__init__(parent)

        self.__signal = None
        self.__signal_name = None
        # the Z, Y, X axes apply to the last three dimensions of the signal
        # (in that order)
        self.__z_axis = None
        self.__z_axis_name = None
        self.__y_axis = None
        self.__y_axis_name = None
        self.__x_axis = None
        self.__x_axis_name = None

        from ._VolumeWindow import VolumeWindow

        self._view = VolumeWindow(self)

        self._hline = qt.QFrame(self)
        self._hline.setFrameStyle(qt.QFrame.HLine)
        self._hline.setFrameShadow(qt.QFrame.Sunken)
        self._legend = qt.QLabel(self)
        self._axesSelector = NumpyAxesSelector(self)
        self._axesSelector.setNamedAxesSelectorVisibility(False)
        self.__axes_selector_is_connected = False

        layout = qt.QVBoxLayout()
        layout.addWidget(self._view)
        layout.addWidget(self._hline)
        layout.addWidget(self._legend)
        layout.addWidget(self._axesSelector)

        self.setLayout(layout)

    def getVolumeView(self):
        """Returns the plot used for the display

        :rtype: SceneWindow
        """
        return self._view

    def setData(
        self,
        signal,
        x_axis=None,
        y_axis=None,
        z_axis=None,
        signal_name=None,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        title=None,
    ):
        """

        :param signal: n-D dataset, whose last 3 dimensions are used as the
            3D stack values.
        :param x_axis: 1-D dataset used as the image's x coordinates. If
            provided, its lengths must be equal to the length of the last
            dimension of ``signal``.
        :param y_axis: 1-D dataset used as the image's y. If provided,
            its lengths must be equal to the length of the 2nd to last
            dimension of ``signal``.
        :param z_axis: 1-D dataset used as the image's z. If provided,
            its lengths must be equal to the length of the 3rd to last
            dimension of ``signal``.
        :param signal_name: Label used in the legend
        :param xlabel: Label for X axis
        :param ylabel: Label for Y axis
        :param zlabel: Label for Z axis
        :param title: Graph title
        """
        if self.__axes_selector_is_connected:
            self._axesSelector.selectionChanged.disconnect(self._updateVolume)
            self.__axes_selector_is_connected = False

        self.__signal = signal
        self.__signal_name = signal_name or ""
        self.__x_axis = x_axis
        self.__x_axis_name = xlabel
        self.__y_axis = y_axis
        self.__y_axis_name = ylabel
        self.__z_axis = z_axis
        self.__z_axis_name = zlabel

        self._axesSelector.setData(signal)
        self._axesSelector.setAxisNames(["Y", "X", "Z"])

        self._updateVolume()

        # the legend label shows the selection slice producing the volume
        # (only interesting for ndim > 3)
        if signal.ndim > 3:
            self._axesSelector.setVisible(True)
            self._legend.setVisible(True)
            self._hline.setVisible(True)
        else:
            self._axesSelector.setVisible(False)
            self._legend.setVisible(False)
            self._hline.setVisible(False)

        if not self.__axes_selector_is_connected:
            self._axesSelector.selectionChanged.connect(self._updateVolume)
            self.__axes_selector_is_connected = True

    def _updateVolume(self):
        """Update displayed stack according to the current axes selector
        data."""
        x_axis = self.__x_axis
        y_axis = self.__y_axis
        z_axis = self.__z_axis

        offset = []
        scale = []
        for axis in [x_axis, y_axis, z_axis]:
            if axis is None:
                calibration = NoCalibration()
            elif len(axis) == 2:
                calibration = LinearCalibration(y_intercept=axis[0], slope=axis[1])
            else:
                calibration = ArrayCalibration(axis)
            if not calibration.is_affine():
                _logger.warning("Axis has not linear values, ignored")
                offset.append(0.0)
                scale.append(1.0)
            else:
                offset.append(calibration(0))
                scale.append(calibration.get_slope())

        legend = self.__signal_name + "["
        for sl in self._axesSelector.selection():
            if sl == slice(None):
                legend += ":, "
            else:
                legend += str(sl) + ", "
        legend = legend[:-2] + "]"
        self._legend.setText("Displayed data: " + legend)

        # Update SceneWidget
        data = self._axesSelector.selectedData()

        volumeView = self.getVolumeView()
        volumeView.setData(data, offset=offset, scale=scale)
        volumeView.setAxesLabels(
            self.__x_axis_name, self.__y_axis_name, self.__z_axis_name
        )

    def clear(self):
        old = self._axesSelector.blockSignals(True)
        self._axesSelector.clear()
        self._axesSelector.blockSignals(old)
        self.getVolumeView().clear()
