import logging
from typing import Sequence
import numpy

from silx.gui import qt
from silx.math.calibration import ArrayCalibration, LinearCalibration, NoCalibration
from .NumpyAxesSelector import NumpyAxesSelector
from ..utils import blockSignals


_logger = logging.getLogger(__name__)


class NxVolumePlot(qt.QWidget):
    """
    Widget for plotting a NXdata with a nD signal (n >= 3) as a 3D scalar field.

    The signal array can have an arbitrary number of dimensions

    Sliders are provided to select indices and axis corresponding of the dimensions of
    the signal array, and the plot is updated to load the stack corresponding
    to the selection.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.__signal: numpy.ndarray | None = None
        self.__axes: list[numpy.ndarray | None] | None = None
        self.__axes_names: list[str | None] | None = None
        from ._VolumeWindow import VolumeWindow

        self._view = VolumeWindow(self)

        self._axesSelector = NumpyAxesSelector(self)

        layout = qt.QVBoxLayout()
        layout.addWidget(self._view)
        layout.addWidget(self._axesSelector)

        self.setLayout(layout)

    def setVolumeData(
        self,
        signal: numpy.ndarray,
        signal_name: str | None = None,
        axes: Sequence[numpy.ndarray | None] | None = None,
        axes_names: Sequence[str | None] | None = None,
    ):
        self.__signal = signal
        if axes:
            self.__axes = list(axes)
        else:
            self.__axes = [None] * signal.ndim
        if axes_names:
            if len(self.__axes) != len(axes_names):
                raise ValueError("Axis names must match the length of axes")
            self.__axes_names = list(axes_names)
        self._axesSelector.selectionChanged.connect(self._updateVolume)
        self._axesSelector.selectedAxisChanged.connect(self._updateVolume)

        with blockSignals(self._axesSelector):
            self._axesSelector.clear()
            self._axesSelector.setAxisNames(["Y", "X", "Z"])

            # Labels need to be set before the data
            if self.__axes_names:
                self._axesSelector.setLabels(self.__axes_names)
            self._axesSelector.setData(signal)
            self._axesSelector.setVisible(signal.ndim > 3)

        self.getVolumeView().setWindowTitle(signal_name)
        self._updateVolume()

    def _updateVolume(self):
        """Update displayed stack according to the current axes selector
        data."""
        if self.__signal is None or self.__axes is None:
            return

        axesIndices = self._axesSelector.getIndicesOfNamedAxes()
        xIndex = axesIndices["X"]
        x_axis = self.__axes[xIndex]
        yIndex = axesIndices["Y"]
        y_axis = self.__axes[yIndex]
        zIndex = axesIndices["Z"]
        z_axis = self.__axes[zIndex]

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

        self._view.setData(
            self._axesSelector.selectedData(), offset=offset, scale=scale
        )
        if self.__axes_names:
            self._view.setAxesLabels(
                self.__axes_names[xIndex],
                self.__axes_names[yIndex],
                self.__axes_names[zIndex],
            )

    def clear(self):
        with blockSignals(self._axesSelector):
            self._axesSelector.clear()
        self._view.clear()
