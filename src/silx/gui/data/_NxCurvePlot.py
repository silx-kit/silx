from typing import Sequence

import numpy

from silx.gui import qt

from ..plot import Plot1D
from ..plot.items import Curve
from ..plot.items.axis import AxisScaleType
from ..utils import blockSignals
from ._models import Axis, Signal
from .NumpyAxesSelector import NumpyAxesSelector


class NxCurvePlot(qt.QWidget):
    """
    Widget for plotting NXdata signals as curves, with support of auxiliary signals errors and axes.

    The signal array can have an arbitrary number of dimensions.

    Only one will be plotted the user can change the indices of the other dimensions or the plotted dimension itself.
    """

    def __init__(self, parent: qt.QWidget | None = None):
        super().__init__(parent)

        self.__signals: list[Signal] | None = None
        self.__axes: list[Axis] | None = None
        self.__signal_curves: list[Curve] = []

        self._plot = Plot1D(self)
        self._plot.setGraphGrid(True)

        self._axesSelector = NumpyAxesSelector(self)
        self._axesSelector.selectionChanged.connect(self._updateYData)
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
        signal_errors: Sequence[numpy.ndarray | None] | None = None,
        signal_scale: AxisScaleType | None = None,
        axes: Sequence[numpy.ndarray] | None = None,
        axes_names: Sequence[str] | None = None,
        axes_errors: Sequence[numpy.ndarray | None] | None = None,
        axes_scales: Sequence[AxisScaleType | None] | None = None,
        title: str | None = None,
    ) -> None:
        self.__signals = []
        for i in range(len(signals)):
            try:
                self.__signals.append(
                    Signal(
                        values=signals[i],
                        name=signal_names[i],
                        errors=signal_errors[i] if signal_errors else None,
                        scale=signal_scale,
                    )
                )
            except IndexError as e:
                raise ValueError(
                    "Length of signal lists should be equal to the number of signals"
                ) from e

        self.__axes = []
        for i in range(signals[0].ndim):
            try:
                self.__axes.append(
                    Axis(
                        values=axes[i] if axes else None,
                        name=axes_names[i] if axes_names else None,
                        errors=axes_errors[i] if axes_errors else None,
                        scale=axes_scales[i] if axes_scales else None,
                    )
                )
            except IndexError as e:
                raise ValueError(
                    "Length of axes lists should be equal to the number of dimensions of the signal"
                ) from e

        with blockSignals(self._axesSelector):
            self._axesSelector.clear()
            self._axesSelector.setAxisNames(["X"])

            # Labels need to be set before the data
            if axes_names:
                self._axesSelector.setLabels(axes_names)
            self._axesSelector.setData(signals[0])

            if len(signals[0].shape) < 2:
                self._axesSelector.hide()
            else:
                self._axesSelector.show()

        self._plot.setGraphTitle(title or "")
        self._updateCurve()

    def _updateCurve(self):
        """Updates X and Y data. Called when the plotted dimension changes"""
        if self.__signals is None or self.__axes is None:
            return

        axes_selection = self._axesSelector.selection()
        axis = self.__axes[self._axesSelector.getIndicesOfNamedAxes()["X"]]
        if axis.values is not None:
            x = axis.values
        else:
            signalLength = len(self.__signals[0].values[axes_selection])
            x = numpy.arange(signalLength)

        self._plot.clear()
        self.__signal_curves.clear()
        for signal in self.__signals:
            self.__signal_curves.append(
                self._plot.addCurve(
                    x,
                    signal.values[axes_selection],
                    legend=signal.name,
                    xerror=axis.errors,
                    yerror=signal.get_errors(axes_selection),
                )
            )
        mainSignal = self.__signals[0]
        self._plot.getYAxis().setScale(mainSignal.scale)
        self._plot.setActiveCurve(mainSignal.name)

        self._plot.getXAxis().setScale(axis.scale)
        self._plot.setGraphXLabel(axis.name)
        self._plot.resetZoom()

    def _updateYData(self):
        """Update Y data **only**. Called when the slicing indices change"""
        if self.__signals is None:
            return

        axes_selection = self._axesSelector.selection()
        for curve, signal in zip(self.__signal_curves, self.__signals):
            curve.setData(
                curve.getXData(False),
                signal.values[axes_selection],
                xerror=curve.getXErrorData(False),
                yerror=signal.get_errors(axes_selection),
            )

    def clear(self):
        with blockSignals(self._axesSelector):
            self._axesSelector.clear()
        self.__signal_curves.clear()
        self._plot.clear()
