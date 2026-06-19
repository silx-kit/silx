import numpy

from silx.gui import qt
from .NumpyAxesSelector import NumpyAxesSelector
from ..plot import Plot1D, items


class ArrayCurvePlot(qt.QWidget):
    """
    Widget for plotting a curve from a multi-dimensional signal array
    and a 1D axis array.

    The signal array can have an arbitrary number of dimensions, the only
    limitation being that the last dimension must have the same length as
    the axis array.

    The widget provides sliders to select indices on the first (n - 1)
    dimensions of the signal array, and buttons to add/replace selected
    curves to the plot.

    This widget also handles simple 2D or 3D scatter plots (third dimension
    displayed as colour of points).
    """

    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super().__init__(parent)

        self.__signals = None
        self.__signals_names = None
        self.__signal_errors = None
        self.__axis = None
        self.__axis_name = None
        self.__x_axis_errors = None

        self._plot = Plot1D(self)
        self._plot.setGraphGrid(True)

        self._axesSelector = NumpyAxesSelector(self)
        self.__axes_selector_is_connected = False

        self._plot.sigActiveCurveChanged.connect(self._setYLabelFromActiveLegend)

        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)
        layout.addWidget(self._axesSelector)

        self.setLayout(layout)

    def getPlot(self):
        """Returns the plot used for the display

        :rtype: Plot1D
        """
        return self._plot

    def setCurvesData(
        self,
        ys,
        x=None,
        yerror=None,
        xerror=None,
        ylabels=None,
        xlabel=None,
        title=None,
        xscale=None,
        yscale=None,
    ):
        """

        :param List[ndarray] ys: List of arrays to be represented by the y (vertical) axis.
            It can be multiple n-D array whose last dimension must
            have the same length as x (and values must be None)
        :param ndarray x: 1-D dataset used as the curve's x values. If provided,
            its lengths must be equal to the length of the last dimension of
            ``y`` (and equal to the length of ``value``, for a scatter plot).
        :param ndarray yerror: Single array of errors for y (same shape), or None.
            There can only be one array, and it applies to the first/main y
            (no y errors for auxiliary_signals curves).
        :param ndarray xerror: 1-D dataset of errors for x, or None
        :param str ylabels: Labels for each curve's Y axis
        :param str xlabel: Label for X axis
        :param str title: Graph title
        :param str xscale: Scale of X axis in (None, 'linear', 'log')
        :param str yscale: Scale of Y axis in (None, 'linear', 'log')
        """
        self.__signals = ys
        self.__signals_names = ylabels or (["Y"] * len(ys))
        self.__signal_errors = yerror
        self.__axis = x
        self.__axis_name = xlabel
        self.__x_axis_errors = xerror

        if self.__axes_selector_is_connected:
            self._axesSelector.selectionChanged.disconnect(self._updateCurve)
            self.__axes_selector_is_connected = False
        self._axesSelector.setData(ys[0])
        self._axesSelector.setAxisNames(["Y"])

        if len(ys[0].shape) < 2:
            self._axesSelector.hide()
        else:
            self._axesSelector.show()

        self._plot.setGraphTitle(title or "")
        if xscale is not None:
            self._plot.getXAxis().setScale("log" if xscale == "log" else "linear")
        if yscale is not None:
            self._plot.getYAxis().setScale("log" if yscale == "log" else "linear")
        self._updateCurve()

        if not self.__axes_selector_is_connected:
            self._axesSelector.selectionChanged.connect(self._updateCurve)
            self.__axes_selector_is_connected = True

    def _updateCurve(self):
        axes_selection = self._axesSelector.selection()
        ys = [sig[axes_selection] for sig in self.__signals]
        y0 = ys[0]
        len_y = len(y0)
        x = self.__axis
        if x is None:
            x = numpy.arange(len_y)
        elif numpy.isscalar(x) or len(x) == 1:
            # constant axis
            x = x * numpy.ones_like(y0)
        elif len(x) == 2 and len_y != 2:
            # linear calibration a + b * x
            x = x[0] + x[1] * numpy.arange(len_y)

        # Only remove curves that will no longer belong to the plot
        # So remaining curves keep their settings
        for item in self._plot.getItems():
            if (
                isinstance(item, items.Curve)
                and item.getName() not in self.__signals_names
            ):
                self._plot.remove(item)

        for i in range(len(self.__signals)):
            legend = self.__signals_names[i]

            # errors only supported for primary signal in NXdata
            y_errors = None
            if i == 0 and self.__signal_errors is not None:
                y_errors = self.__signal_errors[self._axesSelector.selection()]
            self._plot.addCurve(
                x, ys[i], legend=legend, xerror=self.__x_axis_errors, yerror=y_errors
            )
            if i == 0:
                self._plot.setActiveCurve(legend)

        self._plot.resetZoom()
        self._plot.getXAxis().setLabel(self.__axis_name)
        self._plot.getYAxis().setLabel(self.__signals_names[0])

    def _setYLabelFromActiveLegend(self, previous_legend, new_legend):
        for ylabel in self.__signals_names:
            if new_legend is not None and new_legend == ylabel:
                self._plot.getYAxis().setLabel(ylabel)
                break

    def clear(self):
        old = self._axesSelector.blockSignals(True)
        self._axesSelector.clear()
        self._axesSelector.blockSignals(old)
        self._plot.clear()
