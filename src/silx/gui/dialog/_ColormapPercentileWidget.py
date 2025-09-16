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
