from __future__ import annotations

from weakref import ref
from typing import TYPE_CHECKING
from silx.gui import qt
import qtawesome
if TYPE_CHECKING:
    from ..PlotWindow import PlotWindow


class PlotOptionButton(qt.QPushButton):
    """Button presented as a 'burger' menu to present user contextual actions such as 'Legend' or 'Region Of Interest'."""

    def __init__(self, parent: PlotWindow | None = None):
        super().__init__(parent)
        self._plot = None

        self.setFlat(True)
        # should be presented on the right of the toolbar and with the burger menu
        self.setIcon(qtawesome.icon("fa6s.bars"))

        self._menu = qt.QMenu(self)

        self.clicked.connect(self._showPlotActionMenu)
        self._menu.aboutToShow.connect(self._customControlButtonMenu)

    def setPlot(self, plot: PlotWindow):
        from ..PlotWindow import PlotWindow  # Avoid cyclic import
        if not isinstance(plot, PlotWindow):
            raise TypeError(f"plot should be an instanec of {PlotWindow}. Got {type(plot)}")
        self._plot = ref(plot)

    def getPlot(self) -> PlotWindow | None:
        if self._plot is None:
            return None
        return self._plot()

    def _customControlButtonMenu(self):
        plot = self.getPlot()
        if plot is None:
            raise RuntimeError("Plot not found. Either the plot is not set (make sure 'setPlot' is called) or the plot has been deleted.")

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
        self._menu.exec(
            self.mapToGlobal(
                self.rect().bottomLeft()
            )
        )
