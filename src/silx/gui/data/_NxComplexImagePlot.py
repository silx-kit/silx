import logging

import numpy

from ..colors import Colormap
from ..plot.ComplexImageView import _ComplexDataToolButton
from ..plot.items import ImageComplexData
from ..utils import blockSignals
from .. import qt
from ._BaseImagePlot import BaseImagePlot
from ._utils import getAxesCalib

_logger = logging.getLogger(__name__)


class NxComplexImagePlot(BaseImagePlot):
    """
    Widget for plotting a NXdata with a complex signal array as an image

    The signal array can have an arbitrary number of dimensions.
    """

    def __init__(self, parent: qt.QWidget | None = None, colormap: Colormap = None):
        super().__init__(parent)

        self._imageItem = ImageComplexData()
        self._imageItem.setName("__complex_image__")
        self._plot.addItem(self._imageItem)
        self._plot.setActiveImage(self._imageItem)

        if colormap is not None:
            for mode in (
                ImageComplexData.ComplexMode.ABSOLUTE,
                ImageComplexData.ComplexMode.SQUARE_AMPLITUDE,
                ImageComplexData.ComplexMode.REAL,
                ImageComplexData.ComplexMode.IMAGINARY,
            ):
                self._imageItem.setColormap(colormap, mode)

        toolBar = qt.QToolBar("Complex", self)
        toolBar.addWidget(_ComplexDataToolButton(parent=self, item=self._imageItem))
        self._plot.insertToolBar(self._plot.getProfileToolbar(), toolBar)

        self._plot.getIntensityHistogramAction().setVisible(True)

    def _addItemToPlot(self, xAxis, yAxis):
        image = self._getImageToDisplay()
        xCalib, yCalib = getAxesCalib(image.shape[0:2], xAxis, yAxis)
        self._imageItem.setData(image)

        if xCalib.is_affine():
            xOrigin, xScale = xCalib(0), xCalib.get_slope()
        else:
            _logger.warning("Unsupported complex image X axis calibration")
            xOrigin, xScale = 0.0, 1.0

        if yCalib.is_affine():
            yOrigin, yScale = yCalib(0), yCalib.get_slope()
        else:
            _logger.warning("Unsupported complex image Y axis calibration")
            yOrigin, yScale = 0.0, 1.0

        self._imageItem.setOrigin((xOrigin, yOrigin))
        self._imageItem.setScale((xScale, yScale))

    def clear(self):
        with blockSignals(self._axesSelector):
            self._axesSelector.clear()
        # Do not clear the plot since the item must stay in the plot
        self._imageItem.setData(numpy.zeros((0, 0), dtype=numpy.complex64))
