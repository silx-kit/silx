import h5py

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

    def __init__(self, parent=None):
        super().__init__(parent)

        self._signals: list[h5py.Dataset | Dataset] = []
        self._axes: list[h5py.Dataset | Dataset] = []
        self._axesNames: list[str] = []
        self._title = ""

        self._plot = Plot2D(self)
        self._plot.setKeepDataAspectRatio(True)
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

    def _getImageToDisplay(self):
        signalIndex = self._signalSelector.getSignalIndex()
        try:
            signal = self._signals[signalIndex]
        except KeyError:
            raise KeyError("No image found. Was an image loaded?")
        return signal[self._axesSelector.selection()]

    def _getImageName(self):
        if len(self._signalSelector.getSignalNames()) > 0:
            return self._signalSelector.getCurrentSignalName()
        return ""

    def _updateImageAxes(self):
        """Updates the image axes. Called when the user selects a different axis than the displayed one."""

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
        self._plot.setKeepDataAspectRatio(xUnits == yUnits)

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
