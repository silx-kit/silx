import logging
import weakref

from .. import qt
from ..widgets.LegendIconWidget import LegendIconWidget
from . import items
from .PlotWidget import PlotWidget

_logger = logging.getLogger(__name__)


class LegendItemIcon(LegendIconWidget):

    def __init__(self, parent=None, item=None):
        super().__init__(parent)
        self._itemRef = None
        self.setItem(item)

    def getItem(self):
        """Returns item associated to this widget

        :rtype: Union[~silx.gui.plot.items,...]
        """
        return None if self._itemRef is None else self._itemRef()

    def setItem(self, item):
        """Set the item with which to synchronize this widget.

        :param curve: Union[~silx.gui.plot.items,...]
        """
        assert (
            item is None
            or isinstance(item, items.ColormapMixIn)
            or isinstance(item, items.SymbolMixIn)
            or isinstance(item, items.ColorMixIn)
            or isinstance(item, items.LineMixIn)
        )

        previousItem = self.getItem()
        if item == previousItem:
            return

        if previousItem is not None:
            previousItem.sigItemChanged.disconnect(self._itemChanged)

        self._itemRef = None if item is None else weakref.ref(item)

        if item is not None:
            item.sigItemChanged.connect(self._itemChanged)

        self._update()

    def _update(self):
        item = self.getItem()
        if item is None:
            _logger.debug("Item no longer exists, disabling legend widget.")
            self.setEnabled(False)
            return

        self.setEnabled(item.isVisible())
        if isinstance(item, items.SymbolMixIn):
            self.setSymbol(item.getSymbol())
        if isinstance(item, items.LineMixIn):
            self.setLineWidth(item.getLineWidth())
            self.setLineStyle(item.getLineStyle())
        if isinstance(item, items.ColorMixIn):
            color_rgba = item.getColor()
            color = qt.QColor.fromRgbF(
                color_rgba[0],
                color_rgba[1],
                color_rgba[2],
                color_rgba[3],
            )
            self.setLineColor(color)
            self.setSymbolColor(color)
        if isinstance(item, items.ColormapMixIn):
            self.setColormap(item.getColormap())

        self.update()

    def _itemChanged(self, event):
        if event in (
            items.ItemChangedType.VISIBLE,
            items.ItemChangedType.SYMBOL,
            items.ItemChangedType.SYMBOL_SIZE,
            items.ItemChangedType.LINE_WIDTH,
            items.ItemChangedType.LINE_STYLE,
            items.ItemChangedType.COLOR,
            items.ItemChangedType.ALPHA,
            items.ItemChangedType.HIGHLIGHTED,
            items.ItemChangedType.HIGHLIGHTED_STYLE,
        ):
            self._update()


class LegendItemWidget(qt.QWidget):
    def __init__(self, parent, item: items):
        super().__init__(parent)
        self._item = item
        self._itemRef = weakref.ref(item)
        self.setLayout(qt.QHBoxLayout())
        self.layout().setContentsMargins(10, 0, 10, 0)
        item.sigItemChanged.connect(self._itemChanged)

        self.icon = LegendItemIcon(parent=self, item=item)
        self.layout().addWidget(self.icon)

        self.label = qt.QLabel(self._item.getName())
        self.label.setAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
        self.layout().addWidget(self.label)
        self.layout().addStretch()

        self.label.setToolTip("Click to toggle visibility")
        self.label.mousePressEvent = self._onLabelClicked
        self.label.setCursor(qt.Qt.PointingHandCursor)
        self._update()

    def _itemChanged(self, event: items.ItemChangedType):
        """Handle update of curve/scatter item
        :param event: Kind of change
        """
        if event in (
            items.ItemChangedType.VISIBLE,
            items.ItemChangedType.SYMBOL,
            items.ItemChangedType.SYMBOL_SIZE,
            items.ItemChangedType.LINE_WIDTH,
            items.ItemChangedType.LINE_STYLE,
            items.ItemChangedType.COLOR,
            items.ItemChangedType.ALPHA,
            items.ItemChangedType.HIGHLIGHTED,
            items.ItemChangedType.HIGHLIGHTED_STYLE,
        ):
            self._update()

    def item(self) -> items.Item | None:
        return self._itemRef

    def _update(self):
        _item = self._item
        if _item is None:
            _logger.debug("Item no longer exists, disabling legend widget.")
            self.setEnabled(False)
            return

        self.label.setText(_item.getName())

        style = ""
        if isinstance(_item, items.Curve) and _item.isHighlighted():
            style += "border: 1px solid black; "

        if not _item.isVisible():
            style += "color: gray;"
        else:
            style += "color: black;"

        self.label.setStyleSheet(style)

    def _onLabelClicked(self, event):
        if event.button() == qt.Qt.LeftButton:
            if self._item:
                self._item.setVisible(not self._item.isVisible())


class LegendItemList(qt.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent, qt.Qt.Window)
        self.setWindowTitle("Plot Info")
        self.resize(300, 150)
        self.central_widget = qt.QWidget()
        self.setCentralWidget(self.central_widget)
        self._layout = qt.QVBoxLayout(self.central_widget)
        self._plot: PlotWidget | None = None

    def binding(self, plotWidget: PlotWidget):
        """Binds this widget to the signals of a PlotWidget."""
        self._plot = plotWidget
        self._plot.sigActiveCurveChanged.connect(self._onActiveItemChanged)
        self._plot.sigActiveImageChanged.connect(self._onActiveItemChanged)
        self._plot.sigActiveScatterChanged.connect(self._onActiveItemChanged)
        self._plot.sigItemAdded.connect(self._onContentChanged)
        self._plot.sigItemRemoved.connect(self._onContentChanged)
        self._updateAllItemsList()

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
        self._clearLayout(self._layout)
        _activeItem: tuple = self._plot.getItems()

        for item in _activeItem:
            _legendIcon = LegendItemWidget(None, item)
            self._layout.addWidget(_legendIcon)

    def _onActiveItemChanged(self, previous, current):
        self._updateAllItemsList()

    def _onContentChanged(self, item):
        self._updateAllItemsList()
