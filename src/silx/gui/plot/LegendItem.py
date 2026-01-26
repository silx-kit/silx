import logging
import weakref
from typing import Optional

from .. import qt
from ..widgets.LegendIconWidget import LegendIconWidget
from . import items
from .items.core import HighlightedMixIn
from .PlotWidget import PlotWidget

_logger = logging.getLogger(__name__)


class LegendItemWidget(qt.QWidget):
    def __init__(self, parent: qt.QWidget, item: items.Item):
        super().__init__(parent)
        self._itemRef = weakref.ref(item)
        self.setLayout(qt.QHBoxLayout())
        self.layout().setSpacing(20)
        self.layout().setContentsMargins(10, 0, 10, 0)
        item.sigItemChanged.connect(self._itemChanged)

        self._icon = LegendIconWidget(parent=self)
        self.layout().addWidget(self._icon)

        self._label = qt.QLabel(item.getName())
        self._label.setAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
        self.layout().addWidget(self._label)
        self.layout().addStretch()

        self._label.setToolTip("Click to toggle visibility")
        self.setCursor(qt.Qt.PointingHandCursor)
        self._update()

    def getItem(self) -> Optional[items.Item]:
        return self._itemRef()

    def _itemChanged(self, event: items.ItemChangedType):
        """Handle update of curve/scatter item
        :param event: Kind of change
        """
        if event in (
            items.ItemChangedType.VISIBLE,
            items.ItemChangedType.HIGHLIGHTED,
            items.ItemChangedType.NAME,
            items.ItemChangedType.SYMBOL,
            items.ItemChangedType.SYMBOL_SIZE,
            items.ItemChangedType.LINE_WIDTH,
            items.ItemChangedType.LINE_STYLE,
            items.ItemChangedType.COLOR,
            items.ItemChangedType.ALPHA,
        ):
            self._update()

    def _update(self):
        item = self.getItem()
        if item is None:
            _logger.debug("Item no longer exists, disabling legend widget.")
            self.setEnabled(False)
            return

        self._icon.setEnabled(item.isVisible())
        if isinstance(item, items.SymbolMixIn):
            self._icon.setSymbol(item.getSymbol())
        if isinstance(item, items.LineMixIn):
            self._icon.setLineWidth(item.getLineWidth())
            self._icon.setLineStyle(item.getLineStyle())
        if isinstance(item, items.ColorMixIn):
            color = qt.QColor.fromRgbF(*item.getColor())
            self._icon.setLineColor(color)
            self._icon.setSymbolColor(color)
        if isinstance(item, items.ColormapMixIn):
            self._icon.setColormap(item.getColormap())

        self._label.setText(item.getName())

        palette = self.palette()
        if not item.isVisible():
            self._label.setStyleSheet(
                f"color: {palette.color(qt.QPalette.Disabled, qt.QPalette.WindowText).name()};"
            )
        else:
            self._label.setStyleSheet(
                f"color: {palette.color(qt.QPalette.Active, qt.QPalette.WindowText).name()};"
            )

        if isinstance(item, HighlightedMixIn) and item.isHighlighted():
            self.setAutoFillBackground(True)
            self.setBackgroundRole(qt.QPalette.Highlight)
        else:
            self.setAutoFillBackground(False)

    def mousePressEvent(self, event: qt.QMouseEvent):
        """Handle toggle visibility on click without monkey-patching."""
        if event.button() == qt.Qt.LeftButton:
            item = self.getItem()
            if item:
                item.setVisible(not item.isVisible())
        super().mousePressEvent(event)


class LegendItemList(qt.QWidget):
    def __init__(
        self,
        parent: Optional[qt.QWidget] = None,
        plotWidget: Optional[PlotWidget] = None,
    ):
        super().__init__(parent)
        self.setLayout(qt.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.layout().addStretch()
        self._plotRef = None
        self._itemWidgets = {}

        if plotWidget:
            self.setPlotWidget(plotWidget)

    def setPlotWidget(self, plot: PlotWidget):
        if self._plotRef is not None and self._plotRef() is not None:
            prev_plot = self._plotRef()
            prev_plot.sigItemAdded.disconnect(self._onItemAdded)
            prev_plot.sigItemRemoved.disconnect(self._onItemRemoved)

        self._plotRef = weakref.ref(plot)
        plot.sigItemAdded.connect(self._onItemAdded)
        plot.sigItemRemoved.connect(self._onItemRemoved)

        self._clear()
        for item in plot.getItems():
            self._onItemAdded(item)

    def _clear(self):
        """Helper to clear all widgets from a layout."""
        layout = self.layout()
        while layout.count() > 1:
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._itemWidgets.clear()

    def _onItemAdded(self, item: items.Item):
        if not isinstance(item, items.Item) or item in self._itemWidgets:
            return
        widget = LegendItemWidget(self, item)
        self.layout().insertWidget(self.layout().count() - 1, widget)
        self._itemWidgets[item] = widget

    def _onItemRemoved(self, item: items.Item):
        widget: qt.QWidget | None = self._itemWidgets.pop(item, None)
        if widget is not None:
            self.layout().removeWidget(widget)
            widget.deleteLater()
