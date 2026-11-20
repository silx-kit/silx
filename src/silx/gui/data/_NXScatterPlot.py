from typing import Sequence

import numpy

from .. import qt
from ..colors import Colormap
from ..plot import ScatterView
from ..plot.items.types import AxisScaleType
from ._SignalSelector import SignalSelector


class NXScatterPlot(qt.QWidget):
    """
    Widget for plotting one or more scatters
    (with identical x, y coordinates).
    """

    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super().__init__(parent)

        self.__y_axis = None
        """1D array"""
        self.__y_axis_name = None
        self.__values = None
        """List of 1D arrays (for multiple scatters with identical
        x, y coordinates)"""

        self.__x_axis = None
        self.__x_axis_name = None
        self.__x_axis_errors = None
        self.__y_axis = None
        self.__y_axis_name = None
        self.__y_axis_errors = None

        self._plot = ScatterView(self)
        self._plot.setColormap(
            Colormap(
                name="viridis", vmin=None, vmax=None, normalization=Colormap.LINEAR
            )
        )

        self._signalSelector = SignalSelector(parent=self)
        self._signalSelector.selectionChanged.connect(self._signalChanges)
        self._signalSelector.setToolTip("Select signal")

        layout = qt.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot, 0, 0)
        layout.addWidget(self._signalSelector, 1, 0)

        self.setLayout(layout)

    def _signalChanges(self, value):
        self._updateScatter()

    def getScatterView(self):
        """Returns the :class:`ScatterView` used for the display

        :rtype: ScatterView
        """
        return self._plot

    def getPlot(self):
        """Returns the plot used for the display

        :rtype: PlotWidget
        """
        return self._plot.getPlotWidget()

    def setScattersData(
        self,
        y: numpy.ndarray,
        x: numpy.ndarray,
        values: Sequence[numpy.ndarray],
        yerror: numpy.ndarray | None = None,
        xerror: numpy.ndarray | None = None,
        ylabel: numpy.ndarray | None = None,
        xlabel: numpy.ndarray | None = None,
        title: str = "",
        scatter_titles: Sequence[str] | None = None,
        xscale: AxisScaleType | None = None,
        yscale: AxisScaleType | None = None,
    ):
        """

        :param y: 1D array for y (vertical) coordinates.
        :param x: 1D array for x coordinates.
        :param values: List of 1D arrays of values.
            This will be used to compute the color map and assign colors
            to the points. There should be as many arrays in the list as
            scatters to be represented.
        :param yerror: 1D array of errors for y (same shape), or None.
        :param xerror: 1D array of errors for x, or None
        :param ylabel: Label for Y axis
        :param xlabel: Label for X axis
        :param title: Main graph title
        :param scatter_titles:  Subtitles (one per scatter)
        :param xscale: Scale of X axis
        :param yscale: Scale of Y axis
        """
        self.__y_axis = y
        self.__x_axis = x
        self.__x_axis_name = xlabel or "X"
        self.__y_axis_name = ylabel or "Y"
        self.__x_axis_errors = xerror
        self.__y_axis_errors = yerror
        self.__values = values

        self.__graph_title = title or ""
        self.__scatter_titles = scatter_titles

        self._signalSelector.selectionChanged.disconnect(self._signalChanges)
        self._signalSelector.setSignalNames(scatter_titles)
        if len(scatter_titles) > 1:
            self._signalSelector.show()
        else:
            self._signalSelector.hide()
        self._signalSelector.setSignalIndex(0)
        self._signalSelector.selectionChanged.connect(self._signalChanges)

        if xscale is not None:
            self._plot.getXAxis().setScale(xscale)
        if yscale is not None:
            self._plot.getYAxis().setScale(yscale)

        self._updateScatter()

    def _updateScatter(self):
        x = self.__x_axis
        y = self.__y_axis

        idx = self._signalSelector.getSignalIndex()

        if self.__graph_title:
            title = self.__graph_title  # main NXdata @title
            if len(self.__scatter_titles) > 1:
                # Append dataset name only when there is many datasets
                title += "\n" + self.__scatter_titles[idx]
        else:
            title = self.__scatter_titles[idx]  # scatter dataset name

        self._plot.setGraphTitle(title)
        self._plot.setData(
            x,
            y,
            self.__values[idx],
            xerror=self.__x_axis_errors,
            yerror=self.__y_axis_errors,
        )
        self._plot.resetZoom()
        self._plot.getXAxis().setLabel(self.__x_axis_name)
        self._plot.getYAxis().setLabel(self.__y_axis_name)

    def clear(self):
        self._plot.getPlotWidget().clear()
