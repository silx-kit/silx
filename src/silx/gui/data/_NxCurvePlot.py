from typing import Sequence

import numpy

from silx.gui import qt
from silx.gui.plot.items.axis import AxisScaleType

from ..plot import Plot1D
from ..utils import blockSignals
from .NumpyAxesSelector import NumpyAxesSelector


class NxCurvePlot(qt.QWidget):
    """
    Widget for plotting NXdata signals as curves, with support of auxiliary signals errors and axes.

    The signal array can have an arbitrary number of dimensions.

    Only one will be plotted the user can change the indices of the other dimensions or the plotted dimension itself.
    """

    def __init__(self, parent: qt.QWidget | None = None):
        super().__init__(parent)

        self.__signals: list[numpy.ndarray] | None = None
        self.__signals_names: list[str] | None = None
        self.__signal_errors: list[numpy.ndarray] | None = None
        self.__signal_scale: AxisScaleType = "linear"
        self.__axes: list[numpy.ndarray] | None = None
        self.__axes_names: list[str] | None = None
        self.__axes_errors: list[numpy.ndarray | None] | None = None
        self.__axes_scales: list[AxisScaleType] | None = None

        self._plot = Plot1D(self)
        self._plot.setGraphGrid(True)

        self._axesSelector = NumpyAxesSelector(self)
        self._axesSelector.selectionChanged.connect(self._updateCurve)
        self._axesSelector.selectedAxisChanged.connect(self._updateCurve)

        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)
        layout.addWidget(self._axesSelector)

        self.setLayout(layout)

    def setCurvesData(
        self,
        signals: Sequence[numpy.ndarray],
        signal_names: Sequence[str],
        signal_errors: Sequence[numpy.ndarray] | None = None,
        signal_scale: AxisScaleType | None = None,
        axes: Sequence[numpy.ndarray] | None = None,
        axes_names: Sequence[str] | None = None,
        axes_errors: Sequence[numpy.ndarray | None] | None = None,
        axes_scales: Sequence[AxisScaleType | None] | None = None,
        title: str | None = None,
    ):
        self.__signals = list(signals)
        self.__signals_names = list(signal_names)
        self.__signal_errors = list(signal_errors) if signal_errors else None
        self.__signal_scale = signal_scale or "linear"
        self.__axes = list(axes) if axes else None
        self.__axes_names = list(axes_names) if axes_names else None
        self.__axes_errors = list(axes_errors) if axes_errors else None
        self.__axes_scales = (
            [scale or "linear" for scale in axes_scales] if axes_scales else None
        )

        with blockSignals(self._axesSelector):
            self._axesSelector.clear()
            self._axesSelector.setAxisNames(["X"])

            # Labels need to be set before the data
            if self.__axes_names:
                self._axesSelector.setLabels(self.__axes_names)
            self._axesSelector.setData(signals[0])

            if len(signals[0].shape) < 2:
                self._axesSelector.hide()
            else:
                self._axesSelector.show()

        self._plot.setGraphTitle(title or "")
        self._updateCurve()

    def _updateCurve(self):
        if self.__signals is None or self.__signals_names is None:
            return

        self._plot.clear()

        axes_selection = self._axesSelector.selection()
        xIndex = self._axesSelector.getIndicesOfNamedAxes()["X"]
        if self.__axes:
            x = self.__axes[xIndex]
        else:
            signalLength = len(self.__signals[0][axes_selection])
            x = numpy.arange(signalLength)
        if self.__axes_errors is not None:
            x_errors = self.__axes_errors[xIndex]
        else:
            x_errors = None

        # Main signal
        if self.__signal_errors is not None:
            y_errors = [errors[axes_selection] for errors in self.__signal_errors]
        else:
            y_errors = [None] * len(self.__signals)

        for signal, legend, y_error in zip(
            self.__signals, self.__signals_names, y_errors
        ):
            self._plot.addCurve(
                x,
                signal[axes_selection],
                legend=legend,
                xerror=x_errors,
                yerror=y_error,
            )
        self._plot.getYAxis().setScale(self.__signal_scale)
        self._plot.setActiveCurve(self.__signals_names[0])

        if self.__axes_names:
            self._plot.setGraphXLabel(self.__axes_names[xIndex])
        if self.__axes_scales:
            self._plot.getXAxis().setScale(self.__axes_scales[xIndex])
        self._plot.resetZoom()

    def clear(self):
        with blockSignals(self._axesSelector):
            self._axesSelector.clear()
        self._plot.clear()
