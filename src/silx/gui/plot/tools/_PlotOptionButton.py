from __future__ import annotations

from typing import TYPE_CHECKING
from silx.gui import qt
import qtawesome
from .PlotToolButton import PlotToolButton

if TYPE_CHECKING:
    from ..PlotWindow import PlotWindow  # noqa: F401


class PlotOptionButton(PlotToolButton):
    """Button presented as a 'burger' menu to present user contextual actions such as 'Legend' or 'Region Of Interest'."""

    def __init__(self, parent: qt.QWidget | None = None):
        super().__init__(parent)

        self.setIcon(qtawesome.icon("fa6s.bars"))

        self._menu = qt.QMenu(self)

        self.clicked.connect(self._showPlotActionMenu)
        self._menu.aboutToShow.connect(self._customControlButtonMenu)

    def _customControlButtonMenu(self):
        plot = self.plot()
        if plot is None:
            raise RuntimeError(
                "Plot not found. Either the plot is not set (make sure 'setPlot' is called) or the plot has been deleted."
            )

        self._menu.clear()
        self._menu.addAction(plot.getLegendsDockWidget().toggleViewAction())
        self._menu.addAction(plot.getRoiAction())
        self._menu.addAction(plot.getStatsAction())
        self._menu.addAction(plot.getMaskAction())
        self._menu.addAction(plot.getConsoleAction())

        self._menu.addSeparator()
        self._menu.addAction(plot.getCrosshairAction())
        self._menu.addAction(plot.getPanWithArrowKeysAction())

    def _showPlotActionMenu(self):
        self._menu.exec(self.mapToGlobal(self.rect().bottomLeft()))

    def setPlot(self, plot: PlotWindow):
        from ..PlotWindow import PlotWindow  # noqq: F811 avoid cyclic import

        if not isinstance(plot, PlotWindow):
            raise TypeError(
                f"{plot!r} should be an instance of {PlotWindow}. Got {type(plot)}."
            )
        return super().setPlot(plot)
