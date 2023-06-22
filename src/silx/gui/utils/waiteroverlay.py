import weakref
from typing import Optional
from silx.gui.widgets.WaitingPushButton import WaitingPushButton
from silx.gui import qt


class WaiterOverlay:
    def __init__(self, underlying_widget: qt.QWidget, text: Optional[str]=None) -> None:
        """
        :param qt.QWidget underlying_widget: widget on top of which we want to displat the "processing/waiting wheel"
        :param str waiting_text: text to apply near the processing wheel
        """
        if not isinstance(underlying_widget, qt.QWidget):
            raise TypeError
        self._baseWidget = weakref.ref(underlying_widget)
        self._waitingButton = WaitingPushButton(
            parent=underlying_widget,
            text=text,
        )
        self._waitingButton.setDown(True)
        self._waitingButton.setVisible(False)
        self._waitingButton.setStyleSheet("QPushButton { background-color: transparent; border: 0px }")
        self.resize()
    
    def getBaseWidget(self) -> Optional[qt.QWidget]:
        return self._baseWidget()
    
    def setWaiting(self, activate=True):
        self._waitingButton.setWaiting(activate)
        self._waitingButton.setVisible(activate)
    
    def resize(self):
        # should be called by the widget when resized to keep displaying it in the middle
        parent = self.getBaseWidget()
        if parent is None:
            return
        
        position = parent.size()
        size = self._waitingButton.sizeHint()
        position = (position - size) / 2
        rect = qt.QRect(qt.QPoint(position.width(), position.height()), size)
        self._waitingButton.setGeometry(rect)
