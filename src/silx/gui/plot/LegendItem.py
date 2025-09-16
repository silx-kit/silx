import logging

from .. import qt
from ..widgets.LegendIconWidget import LegendIconWidget
from . import items
from .items.core import HighlightedMixIn
from .PlotWidget import PlotWidget

_logger = logging.getLogger(__name__)


class LegendItemWidget(qt.QWidget):
    def __init__(self, parent, item: items):
        super().__init__(parent)
        self._item = item
        self.setLayout(qt.QHBoxLayout())
        self.setSizePolicy(
            qt.QSizePolicy.Policy.Preferred, qt.QSizePolicy.Policy.Minimum
        )
        self.setMinimumWidth(150)
        self.setMaximumWidth(300)
        self.layout().setSpacing(20)
        self.layout().setContentsMargins(10, 0, 10, 0)
        item.sigItemChanged.connect(self._itemChanged)

        self._icon = LegendIconWidget(parent=self)
        self.layout().addWidget(self._icon)

        self._label = qt.QLabel(self._item.getName())
        self._label.setAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
        self.layout().addWidget(self._label)
        self.layout().addStretch()

        self._label.setToolTip("Click to toggle visibility")
        self._label.mousePressEvent = self._onLabelClicked
        self._label.setCursor(qt.Qt.PointingHandCursor)
        self._update()

    def _itemChanged(self, event: items.ItemChangedType):
        """Handle update of curve/scatter item
        :param event: Kind of change
        """
        if event in (
            items.ItemChangedType.VISIBLE,
            items.ItemChangedType.HIGHLIGHTED,
            items.ItemChangedType.NAME,
        ):
            self._update()

    def getItem(self) -> items.Item | None:
        return self._item

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
            color_rgba = item.getColor()
            color = qt.QColor.fromRgbF(
                color_rgba[0],
                color_rgba[1],
                color_rgba[2],
                color_rgba[3],
            )
            self._icon.setLineColor(color)
            self._icon.setSymbolColor(color)
        if isinstance(item, items.ColormapMixIn):
            self._icon.setColormap(item.getColormap())

        self._label.setText(item.getName())

        style = ""
        palette = self.palette()
        text_color = palette.color(qt.QPalette.ColorRole.Text).name()
        disabled_text_color = (
            palette.color(qt.QPalette.ColorRole.Text).darker(150).name()
        )

        if isinstance(item, HighlightedMixIn) and item.isHighlighted():
            style += "border: 1px solid black; "

        if not item.isVisible():
            style += f"color: {disabled_text_color};"
        else:
            style += f"color: {text_color};"

        self._label.setStyleSheet(style)

    def _onLabelClicked(self, event):
        if event.button() == qt.Qt.LeftButton:
            if self.getItem():
                self.getItem().setVisible(not self._item.isVisible())


class LegendItemList(qt.QWidget):
    def __init__(
        self,
        parent: qt.QWidget | None = None,
        plotWidget: PlotWidget | None = None,
    ):
        super().__init__(parent)
        self.setLayout(qt.QVBoxLayout(self))
        self.setSizePolicy(
            qt.QSizePolicy.Policy.Preferred, qt.QSizePolicy.Policy.Expanding
        )
        self.setMinimumWidth(150)
        self.setMaximumWidth(300)

        self._plot: PlotWidget | None = plotWidget
        self._binding(plotWidget=plotWidget)
        self._itemWidgets = {}

    def _binding(self, plotWidget: PlotWidget):
        """Binds this widget to the signals of a parent PlotWidget."""
        self._plot = plotWidget
        self._plot.sigItemAdded.connect(self._updateAllItemsList)
        self._plot.sigItemRemoved.connect(self._onItemRemoved)
        self._updateAllItemsList()
        self.show()

    def _clearLayout(self, layout):
        """Helper to clear all widgets from a layout."""
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

    def _updateAllItemsList(self):

        if self._plot is None:
            return
        self._clearLayout(self.layout())
        _activeItem: tuple = self._plot.getItems()

        for item in _activeItem:
            self._onItemAdded(item)

    def _onItemRemoved(self, item: items.Item):
        if item in self._itemWidgets:
            self.layout().removeWidget(self._itemWidgets[item])
            del self._itemWidgets[item]

    def _onItemAdded(self, item: items.Item):
        _legendIcon = LegendItemWidget(None, item)
        self.layout().addWidget(_legendIcon)
        self._itemWidgets[item] = _legendIcon
