from __future__ import annotations

from silx.gui import qt
from silx.gui.utils import blockSignals


class ColormapPercentilesWidget(qt.QWidget):
    """
    Widget to define the percentiles to be used when computing the colormap in autoscale / percentile mode.

    A scalar value (that can be seen as saturation) is defined by the user and then converted to percentiles using the 'fromSaturationToPercentiles' function.
    """

    percentilesChanged = qt.Signal(tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(qt.QHBoxLayout())

        self._slider = qt.QSlider(qt.Qt.Horizontal, self)
        self.layout().addWidget(self._slider)

        self._spinBox = qt.QSpinBox(self)
        self.layout().addWidget(self._spinBox)

        self._setRange(0, 100)

        # connect signal / slot
        self._slider.valueChanged.connect(self.setSaturationValue)
        self._spinBox.valueChanged.connect(self.setSaturationValue)

    def setSaturationValue(self, value: int):
        with blockSignals(self._slider, self._spinBox):
            self._slider.setValue(value)
            self._spinBox.setValue(value)
        self.percentilesChanged.emit(self.getPercentilesRange())

    def getSaturationValue(self) -> int:
        return self._slider.value()

    def _setRange(self, min: int, max: int):
        """
        Set the slider / spin box range
        """
        self._slider.setRange(min, max)
        self._spinBox.setRange(min, max)

    def setPercentilesRange(self, percentiles: tuple[float, float]):
        self.setSaturationValue(self.fromPercentilesToSaturation(percentiles))

    def getPercentilesRange(self) -> tuple[float, float]:
        return self.fromSaturationToPercentiles(self.getSaturationValue())

    # expose API
    def setTickPosition(self, position):
        self._slider.setTickPosition(position)

    def setTracking(self, enable: bool):
        self._slider.setTracking(enable)

    @staticmethod
    def fromPercentilesToSaturation(
        percentiles: tuple[float, float],
    ) -> int:
        """
        Example: if we want to have saturation = 90% then the percentile we will return percentiles (5th, 95th)
        """
        return int(100 - (percentiles[0] + (100 - percentiles[1])))

    @staticmethod
    def fromSaturationToPercentiles(
        saturation: float | int,
    ) -> tuple[float, float]:
        """
        Example: if we use percentiles (1st, 99th) we use 98% of the percentiles. This is the saturation (can be seen also as the central percentile)
        """
        if not isinstance(saturation, (float, int)):
            raise TypeError(
                f"central_percentile is expected to be float. Got {type(saturation)}"
            )
        ignored_percentile = 100 - saturation
        return (ignored_percentile / 2.0, 100 - (ignored_percentile / 2.0))
