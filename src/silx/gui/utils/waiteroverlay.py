import weakref
from typing import Optional
from silx.gui.widgets.WaitingPushButton import WaitingPushButton
from silx.gui import qt
from silx.gui.plot import PlotWidget


class WaiterOverlay(qt.QObject):
    def __init__(self, underlying_widget: qt.QWidget) -> None:
        """
        :param qt.QWidget underlying_widget: widget on top of which we want to displat the "processing/waiting wheel"
        :param str waiting_text: text to apply near the processing wheel
        """
        super().__init__()

        # TO be checked with thomas
        if isinstance(underlying_widget, PlotWidget):
            underlying_widget = underlying_widget.getWidgetHandle()

        if not isinstance(underlying_widget, qt.QWidget):
            raise TypeError(f"underlying_widget is expected to be an instance of QWidget. {type(underlying_widget)} provided.")
        self._baseWidget = weakref.ref(underlying_widget)
        self._waitingButton = WaitingPushButton(
            parent=underlying_widget,
        )
        self._waitingButton.setDown(True)
        self._waitingButton.setVisible(False)
        self._waitingButton.setStyleSheet("QPushButton { background-color: rgba(150, 150, 150, 40); border: 0px; border-radius: 10px; }")
        self._resize()
        # register to resize event
        underlying_widget.installEventFilter(self)

    def setText(self, text: str):
        self._waitingButton.setText(text)
    
    def close(self):
        self._waitingButton.setWaiting(False)
        super().close()

    def getBaseWidget(self) -> Optional[qt.QWidget]:
        return self._baseWidget()
    
    def setWaiting(self, activate=True):
        self._waitingButton.setWaiting(activate)
        self._waitingButton.setVisible(activate)
    
    def _resize(self):
        parent = self.getBaseWidget()
        if parent is None:
            return
        
        position = parent.size()
        size = self._waitingButton.sizeHint()
        position = (position - size) / 2
        rect = qt.QRect(qt.QPoint(position.width(), position.height()), size)
        self._waitingButton.setGeometry(rect)

    def eventFilter(self, watched: qt.QWidget, event: qt.QEvent):
        if event.type() == qt.QEvent.Resize:
            self._resize()
        return super().eventFilter(watched, event)
