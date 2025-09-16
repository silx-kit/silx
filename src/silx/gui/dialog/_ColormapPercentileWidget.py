from __future__ import annotations

from silx.gui import qt
from silx.gui.utils import blockSignals


class ColormapPercentileWidget(qt.QWidget):
    """
    Widget with a slider and a spin box for a float
    """

    valueChanged = qt.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(qt.QHBoxLayout())

        self._slider = qt.QSlider(qt.Qt.Horizontal, self)
        self.layout().addWidget(self._slider)

        self._spinBox = qt.QSpinBox(self)
        self.layout().addWidget(self._spinBox)

        self.setRange(0, 100)

        # connect signal / slot
        self._slider.valueChanged.connect(self.setValue)
        self._spinBox.valueChanged.connect(self.setValue)

    def setValue(self, value: float):
        with blockSignals(self._slider, self._spinBox):
            self._slider.setValue(value)
            self._spinBox.setValue(value)
        self.valueChanged.emit(self.value())

    def value(self) -> int:
        return self._slider.value()

    def setRange(self, min: int, max: int):
        self._slider.setRange(min, max)
        self._spinBox.setRange(min, max)

    # expose API
    def setTickPosition(self, position):
        self._slider.setTickPosition(position)

    def setTracking(self, enable: bool):
        self._slider.setTracking(enable)

    @staticmethod
    def fromLateralPercentileRangeToCentralPercentile(
        lateral_percentile_range: tuple[float, float],
    ) -> float:
        """
        Example: if we use lateral percentile (1st, 99th) we use 98% of the percentiles. This is the central percentile
        """
        return 100 - (lateral_percentile_range[0] + (100 - lateral_percentile_range[1]))

    @staticmethod
    def fromCentralPercentileToLateralPercentileRange(
        central_percentile: float | int,
    ) -> tuple[float, float]:
        """
        Example: if we want to use 90% of the percentile we will return percentiles (5th, 95th)
        """
        if not isinstance(central_percentile, (float, int)):
            raise TypeError(
                f"central_percentile is expected to be float. Got {type(central_percentile)}"
            )
        ignored_percentile = 100 - central_percentile
        return (ignored_percentile / 2.0, 100 - (ignored_percentile / 2.0))
