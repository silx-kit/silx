import weakref

import numpy

from .. import qt
from ..widgets.LegendIconWidget import LegendIconWidget
from . import items
from .PlotWidget import PlotWidget


class _LegendItemWidget(qt.QWidget):
    def __init__(self, parent: qt.QWidget, item: items.Item):
        super().__init__(parent)
        self._item = item
        self.setLayout(qt.QHBoxLayout())
        self.layout().setSpacing(20)
        self.layout().setContentsMargins(10, 0, 10, 0)
        item.sigItemChanged.connect(self._itemChanged)

        self._icon = LegendIconWidget(parent=self)
        self.layout().addWidget(self._icon)

        self._label = qt.QLabel(item.getName())
        self._label.setAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
        self._label.setToolTip(
            f"{self._item.__class__.__name__}:\nClick to toggle visibility"
        )
        self.layout().addWidget(self._label)
        self.layout().addStretch()

        self.setCursor(qt.Qt.PointingHandCursor)
        self._update()

    def _itemChanged(self, event: items.ItemChangedType):
        if event in (
            items.ItemChangedType.VISIBLE,
            items.ItemChangedType.HIGHLIGHTED,
            items.ItemChangedType.HIGHLIGHTED_STYLE,
            items.ItemChangedType.NAME,
            items.ItemChangedType.SYMBOL,
            items.ItemChangedType.SYMBOL_SIZE,
            items.ItemChangedType.LINE_WIDTH,
            items.ItemChangedType.LINE_STYLE,
            items.ItemChangedType.COLOR,
            items.ItemChangedType.ALPHA,
            items.ItemChangedType.COLORMAP,
        ):
            self._update()

    def _update(self):
        enabled = self._item.isVisible()
        self._icon.setEnabled(enabled)
        self._label.setEnabled(enabled)

        self._label.setText(self._item.getName())

        if isinstance(self._item, items.AlphaMixIn):
            alpha = self._item.getAlpha()
        else:
            alpha = 1.0

        if isinstance(self._item, items.Curve):
            curveStyle = self._item.getCurrentStyle()
            self._icon.setSymbol(curveStyle.getSymbol())
            self._icon.setLineWidth(curveStyle.getLineWidth())
            self._icon.setLineStyle(curveStyle.getLineStyle())
            self._setColor(curveStyle.getColor(), alpha)
        else:
            if isinstance(self._item, items.SymbolMixIn):
                self._icon.setSymbol(self._item.getSymbol())
            if isinstance(self._item, items.LineMixIn):
                self._icon.setLineWidth(self._item.getLineWidth())
                self._icon.setLineStyle(self._item.getLineStyle())
            if isinstance(self._item, items.ColorMixIn):
                self._setColor(self._item.getColor(), alpha)

        if isinstance(self._item, items.ColormapMixIn):
            self._icon.setColormap(self._item.getColormap())

    def _setColor(
        self,
        color: tuple[float, float, float, float] | numpy.ndarray,
        alpha: float = 1.0,
    ):
        if numpy.asarray(color).ndim == 1:
            color = qt.QColor.fromRgbF(color[0], color[1], color[2], color[3] * alpha)
            self._icon.setLineColor(color)
            self._icon.setSymbolColor(color)
        else:
            # array of colors: use first color for line and default for symbol
            if len(color) == 0:
                self._icon.setLineColor(None)
            else:
                firstColor = color[0]
                color = qt.QColor.fromRgbF(
                    firstColor[0],
                    firstColor[1],
                    firstColor[2],
                    (firstColor[3] if len(firstColor) >= 4 else 1.0) * alpha,
                )
                self._icon.setLineColor(color)
            self._icon.setSymbolColor(None)

    def mousePressEvent(self, event: qt.QMouseEvent):
        """Toggle visibility on click"""
        if event.button() == qt.Qt.LeftButton:
            self._item.setVisible(not self._item.isVisible())
        super().mousePressEvent(event)


class LegendsWidget(qt.QWidget):
    def __init__(
        self,
        parent: qt.QWidget | None = None,
        plotWidget: PlotWidget | None = None,
    ):
        super().__init__(parent)
        self.setLayout(qt.QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.layout().addStretch()
        self._plotRef = None
        self._itemWidgets = {}

        self.setPlotWidget(plotWidget)

    def setPlotWidget(self, plot: PlotWidget | None):
        previousPlot = None if self._plotRef is None else self._plotRef()
        if previousPlot is not None:
            previousPlot.sigItemAdded.disconnect(self._onItemAdded)
            previousPlot.sigItemRemoved.disconnect(self._onItemRemoved)

        self._clear()

        if plot is None:
            self._plotRef = None
        else:
            plot.sigItemAdded.connect(self._onItemAdded)
            plot.sigItemRemoved.connect(self._onItemRemoved)
            self._plotRef = weakref.ref(plot)

            for item in plot.getItems():
                self._onItemAdded(item)

    def _clear(self):
        layout = self.layout()
        while layout.count() > 1:
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._itemWidgets.clear()

    def _onItemAdded(self, item: items.Item):
        if item in self._itemWidgets:
            return
        widget = _LegendItemWidget(self, item)
        self.layout().insertWidget(self.layout().count() - 1, widget)
        self._itemWidgets[item] = widget

    def _onItemRemoved(self, item: items.Item):
        widget: qt.QWidget | None = self._itemWidgets.pop(item, None)
        if widget is not None:
            self.layout().removeWidget(widget)
            widget.deleteLater()
