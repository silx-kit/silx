import h5py
import numpy

from ...io.commonh5 import Dataset
from ...io.nxdata import get_attr_as_unicode
from .. import qt
from ..plot import Plot2D
from ..plot.items import ImageBase
from ..plot.MaskToolsWidget import MaskToolsWidget
from ..utils import blockSignals
from ._SignalSelector import SignalSelector
from ._utils import ImageAxis
from .NumpyAxesSelector import NumpyAxesSelector


class BaseImagePlot(qt.QWidget):
    """
    Widget for plotting images with a multi-dimensional signal array
    and two 1D axes array.
    """

    _DISPLAY_AXES: tuple[str, ...] = ("Y", "X")
    """Names of the axes used to display the image"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._signals: list[h5py.Dataset | Dataset] = []
        self._axes: list[h5py.Dataset | Dataset] = []
        self._axesNames: list[str] = []
        self._title = ""

        self._plot = Plot2D(self)
        self._plot.setKeepDataAspectRatio(True)
        self._plot.toolBar().addAction(self._plot.getShowAxisAction())
        maskToolWidget = self._plot.getMaskToolsDockWidget().widget()
        if isinstance(maskToolWidget, MaskToolsWidget):
            maskToolWidget.setItemMaskUpdated(True)

        self._axesSelector = NumpyAxesSelector(self)
        self._axesSelector.selectionChanged.connect(self._updateImage)
        self._axesSelector.selectedAxisChanged.connect(self._updateImageAxes)

        self._signalSelector = SignalSelector(parent=self)
        self._signalSelector.selectionChanged.connect(self._updateImageAxes)
        self._signalSelector.setToolTip("Select signal")

        layout = qt.QVBoxLayout()
        layout.addWidget(self._plot)
        layout.addWidget(self._axesSelector)
        layout.addWidget(self._signalSelector)

        self.setLayout(layout)

    def getPlot(self) -> Plot2D:
        return self._plot

    def setImageData(
        self,
        signals: list[h5py.Dataset | Dataset],
        axes: list[h5py.Dataset | Dataset] | None = None,
        signalsNames: list[str] | None = None,
        axesNames: list[str] | None = None,
        title: str | None = None,
        selection: tuple[int | slice, ...] | None = None,
    ):
        """
        Sets signals, axes and axes metadata that will be used to set the displayed image.

        :param signals: list of n-D datasets
        :param axes: list of 1D datasets to be used as axes
        :param signalsNames: Names for each image, used as subtitle and legend.
        :param axesNames: Names for each axis, used as graph label.
        :param title: Graph title
        :param selection: Initial selection of the dimensions.
        """
        if len(signals) == 0:
            raise ValueError("Cannot set image data from empty signals")

        self._signals = signals
        self._axes = axes or []
        self._axesNames = axesNames or []
        self._title = title

        with blockSignals(self._axesSelector, self._signalSelector):
            self._axesSelector.clear()
            self._axesSelector.setAxisNames(self._DISPLAY_AXES)
            self._axesSelector.setNamedAxesSelectorVisibility(True)

            # Labels need to be set before the data
            self._axesSelector.setLabels(self._axesNames)
            self._axesSelector.setData(signals[0])

            if selection is not None:
                self._axesSelector.setSelection(selection)

            if signals[0].ndim <= len(self._DISPLAY_AXES):
                self._axesSelector.hide()
            else:
                self._axesSelector.show()

            self._signalSelector.setSignalNames(signalsNames or [])
            if len(signals) > 1:
                self._signalSelector.show()
            else:
                self._signalSelector.hide()
            self._signalSelector.setSignalIndex(0)

        self._updateImageAxes(initialize_aspect_ratio=True)
        self._plot.resetZoom()

    def _getImageToDisplay(self):
        signalIndex = self._signalSelector.getSignalIndex()
        try:
            signal = self._signals[signalIndex]
        except IndexError as e:
            raise KeyError("No image found. Was an image loaded?") from e

        slice_selection = self._axesSelector.selection()
        if slice_selection is not None:
            signal = signal[slice_selection]

        axes_permutation = self._axesSelector.permutation()
        if axes_permutation is not None:
            signal = numpy.transpose(signal, axes_permutation)

        return signal

    def _getImageName(self):
        if len(self._signalSelector.getSignalNames()) > 0:
            return self._signalSelector.getCurrentSignalName()
        return ""

    def _updateImageAxes(self, initialize_aspect_ratio=False):
        """Updates the image axes. Called when the user selects a different axis."""
        xAxisIndex, yAxisIndex = self._getXYIndices()
        if self._axes:
            xAxis = self._axes[xAxisIndex]
            yAxis = self._axes[yAxisIndex]
            xUnits = get_attr_as_unicode(xAxis, "units") if xAxis else None
            yUnits = get_attr_as_unicode(yAxis, "units") if yAxis else None
        else:
            xAxis = None
            yAxis = None
            xUnits = None
            yUnits = None

        if initialize_aspect_ratio:
            keep_aspect_ratio = (xUnits is None and yUnits is None) or (
                xUnits is not None and xUnits == yUnits
            )
            self._plot.setKeepDataAspectRatio(keep_aspect_ratio)

        self._addItemToPlot(xAxis, yAxis)

        self._plot.setGraphTitle(self._graphTitle())
        self._plot.setGraphXLabel(self._axesNames[xAxisIndex])
        self._plot.setGraphYLabel(self._axesNames[yAxisIndex])
        self._plot.resetZoom()

    def _addItemToPlot(self, xAxis: ImageAxis, yAxis: ImageAxis):
        raise NotImplementedError()

    def clear(self):
        with blockSignals(self._axesSelector):
            self._axesSelector.clear()
        self._plot.clear()

    def _updateImage(self):
        """Updates the image itself. Called when the user slices through the image without changing the axes."""
        image = self._getImageToDisplay()
        activeImageItem = self._plot.getActiveImage()
        if isinstance(activeImageItem, ImageBase):
            activeImageItem.setData(image)

    def _graphTitle(self) -> str:
        title = self._title
        if not title:
            return self._getImageName()

        if len(self._signals) > 1:
            # Append dataset name only when there are many datasets
            return f"{title}\n{self._getImageName()}"
        return title

    def _getXYIndices(self) -> tuple[int, int]:
        axisIndices = self._axesSelector.getIndicesOfNamedAxes()
        try:
            return axisIndices["X"], axisIndices["Y"]
        except KeyError:
            raise KeyError("Axes X and Y not found. Was an image loaded?")
