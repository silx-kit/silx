import h5py

from ...io.commonh5 import Dataset
from ..plot import items
from ..utils import blockSignals
from ._BaseImagePlot import BaseImagePlot
from ._utils import ImageAxis, getAxesCalib, setImageCoords


class RgbaImagePlot(BaseImagePlot):
    """
    Widget for plotting a RGB(A) image with a multi-dimensional signal array
    and two 1D axes array.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Hide actions from the toolbar that are not relevant
        plot = self.getPlot()
        colormapAction = plot.getColormapAction()
        if colormapAction:
            colormapAction.setVisible(False)
        colorbarAction = plot.getColorBarAction()
        if colorbarAction:
            colorbarAction.setVisible(False)
        plot.getColorBarWidget().hide()

    def setImageData(
        self,
        signals: list[h5py.Dataset | Dataset],
        axes: list[h5py.Dataset | Dataset] | None = None,
        signalsNames: list[str] | None = None,
        axesNames: list[str] | None = None,
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
        """
        if len(signals) == 0:
            raise ValueError("Cannot set image data from empty signals")
        if axes is None:
            axes = []
        if signalsNames is None:
            signalsNames = []

        self._signals = signals
        self._axes = axes
        self._title = title

        with blockSignals(self._axesSelector, self._signalSelector):
            self._axesSelector.clear()
            self._axesSelector.setAxisNames(["Y", "X", "RGB(A) channel"])
            self._axesSelector.setNamedAxesSelectorVisibility(False)

            # Labels need to be set before the data
            if axesNames:
                self._axesNames = axesNames
                self._axesSelector.setLabels(axesNames)
            self._axesSelector.setData(signals[0])

            if len(signals[0].shape) <= 3:
                self._axesSelector.hide()
            else:
                self._axesSelector.show()
            if signalsNames:
                self._signalSelector.setSignalNames(signalsNames)
            if len(signals) > 1:
                self._signalSelector.show()
            else:
                self._signalSelector.hide()
            self._signalSelector.setSignalIndex(0)

        self._updateImageAxes()
        self._plot.resetZoom()

    def _addItemToPlot(self, xAxis: ImageAxis, yAxis: ImageAxis):
        self._plot.remove(kind="image")

        image = self._getImageToDisplay()

        if image.ndim != 3:
            raise ValueError(f"image dims should be 3. Got {image.ndim}")
        imageItem = items.ImageRgba()
        imageItem.setName(self._getImageName())
        imageItem.setData(image)
        xCalib, yCalib = getAxesCalib(image.shape[0:2], xAxis, yAxis)
        setImageCoords(imageItem, xCalib, yCalib)

        self._plot.addItem(imageItem)
        self._plot.setActiveImage(imageItem)
