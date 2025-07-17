from silx.gui import qt
from silx.gui.qt import inspect as qt_inspect
from silx.gui.plot import PlotWidget


class OverlayMixIn:
    "MixIn class for overlay widget"

    def __init__(self, parent):
        self._registerParent(parent=parent)

    def _listenedWidget(self, parent: qt.QWidget) -> qt.QWidget:
        """Returns widget to register event filter to according to parent"""
        if isinstance(parent, PlotWidget):
            return parent.getWidgetHandle()
        return parent

    def _backendChanged(self):
        self._listenedWidget(self.parent()).installEventFilter(self)
        self._resizeLater()

    def _registerParent(self, parent: qt.QWidget | None):
        if parent is None:
            return
        self._listenedWidget(parent).installEventFilter(self)
        if isinstance(parent, PlotWidget):
            parent.sigBackendChanged.connect(self._backendChanged)
        self._resize()

    def _unregisterParent(self, parent: qt.QWidget | None):
        if parent is None:
            return
        if isinstance(parent, PlotWidget):
            parent.sigBackendChanged.disconnect(self._backendChanged)
        self._listenedWidget(parent).removeEventFilter(self)

    def setParent(self, parent: qt.QWidget):
        self._unregisterParent(self.parent())
        super().setParent(parent)
        self._registerParent(parent)

    def _resize(self):
        if not qt_inspect.isValid(self):
            return  # For _resizeLater in case the widget has been deleted

        parent = self.parent()
        if parent is None:
            return

        size = self.sizeHint()
        if isinstance(parent, PlotWidget):
            offset = parent.getWidgetHandle().mapTo(parent, qt.QPoint(0, 0))
            left, top, width, height = parent.getPlotBoundsInPixels()
            rect = qt.QRect(
                qt.QPoint(
                    int(offset.x() + left + width / 2 - size.width() / 2),
                    int(offset.y() + top + height / 2 - size.height() / 2),
                ),
                size,
            )
        else:
            position = parent.size()
            position = (position - size) / 2
            rect = qt.QRect(qt.QPoint(position.width(), position.height()), size)
        self.setGeometry(rect)
        self.raise_()

    def _resizeLater(self):
        qt.QTimer.singleShot(0, self._resize)

    def eventFilter(self, watched: qt.QWidget, event: qt.QEvent):
        if event.type() == qt.QEvent.Resize:
            self._resize()
            self._resizeLater()  # Defer resize for the receiver to have handled it
        return super().eventFilter(watched, event)
