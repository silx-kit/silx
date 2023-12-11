import weakref
from typing import Optional
from silx.gui.widgets.WaitingPushButton import WaitingPushButton
from silx.gui import qt
from silx.gui.qt import inspect as qt_inspect
from silx.gui.plot import PlotWidget


class WaitingOverlay(qt.QWidget):
    """Widget overlaying another widget with a processing wheel icon.

    :param parent: widget on top of which to display the "processing/waiting wheel"
    """

    def __init__(self, parent: qt.QWidget) -> None:
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

        self._waitingButton = WaitingPushButton(self)
        self._waitingButton.setDown(True)
        self._waitingButton.setWaiting(True)
        self._waitingButton.setStyleSheet(
            "QPushButton { background-color: rgba(150, 150, 150, 40); border: 0px; border-radius: 10px; }"
        )
        self._registerParent(parent)

    def text(self) -> str:
        """Returns displayed text"""
        return self._waitingButton.text()

    def setText(self, text: str):
        """Set displayed text"""
        self._waitingButton.setText(text)
        self._resize()

    def _listenedWidget(self, parent: qt.QWidget) -> qt.QWidget:
        """Returns widget to register event filter to according to parent"""
        if isinstance(parent, PlotWidget):
            return parent.getWidgetHandle()
        return parent

    def _backendChanged(self):
        self._listenedWidget(self.parent()).installEventFilter(self)
        self._resizeLater()

    def _registerParent(self, parent: Optional[qt.QWidget]):
        if parent is None:
            return
        self._listenedWidget(parent).installEventFilter(self)
        if isinstance(parent, PlotWidget):
            parent.sigBackendChanged.connect(self._backendChanged)
        self._resize()

    def _unregisterParent(self, parent: Optional[qt.QWidget]):
        if parent is None:
            return
        if isinstance(parent, PlotWidget):
            parent.sigBackendChanged.disconnect(self._backendChanged)
        self._listenedWidget(parent).removeEventFilter(self)

    def setParent(self, parent: qt.QWidget):
        self._unregisterParent(self.parent())
        super().setParent(parent)
        self._registerParent(parent)

    def showEvent(self, event: qt.QShowEvent):
        super().showEvent(event)
        self._waitingButton.setVisible(True)

    def hideEvent(self, event: qt.QHideEvent):
        super().hideEvent(event)
        self._waitingButton.setVisible(False)

    def _resize(self):
        if not qt_inspect.isValid(self):
            return  # For _resizeLater in case the widget has been deleted

        parent = self.parent()
        if parent is None:
            return

        size = self._waitingButton.sizeHint()
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

    # expose Waiting push button API
    def setIconSize(self, size):
        self._waitingButton.setIconSize(size)
