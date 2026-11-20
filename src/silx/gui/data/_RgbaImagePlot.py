import h5py
from ...io.commonh5 import Dataset

from ...io.nxdata._utils import AxisDataset
from ..plot import items
from ._BaseImagePlot import BaseImagePlot
from ._utils import getAxesCalib, setImageCoords


class RgbaImagePlot(BaseImagePlot):
    """
    Widget for plotting a RGB(A) image with a multi-dimensional signal array
    and two 1D axes array.
    """

    _DISPLAY_AXES = ("Y", "X", "RGB(A) channel")

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
        axes: list[AxisDataset] | None = None,
        signalsNames: list[str] | None = None,
        axesNames: list[str] | None = None,
        title: str | None = None,
        selection: tuple[int | slice, ...] | None = None,
    ):
        super().setImageData(signals, axes, signalsNames, axesNames, title, selection)
        # Disable the RGBA channel option in all axes
        self._axesSelector.setNamedAxisEnabled("RGB(A) channel", False)
        # Disable the RGBA channel axis selected
        self._axesSelector.setNamedAxisSelectorEnabled(-1, False)

    def _addItemToPlot(self, xAxis: AxisDataset, yAxis: AxisDataset):
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
