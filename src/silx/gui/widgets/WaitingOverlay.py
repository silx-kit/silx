from silx.gui.widgets.WaitingPushButton import WaitingPushButton
from silx.gui import qt
from .OverlayMixIn import _OverlayMixIn


class WaitingOverlay(_OverlayMixIn, qt.QWidget):
    """Widget overlaying another widget with a processing wheel icon.

    :param parent: widget on top of which to display the "processing/waiting wheel"
    """

    def __init__(self, parent: qt.QWidget) -> None:
        qt.QWidget.__init__(self, parent)
        _OverlayMixIn.__init__(self, parent)
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

    def showEvent(self, event: qt.QShowEvent):
        super().showEvent(event)
        self._waitingButton.setVisible(True)

    def hideEvent(self, event: qt.QHideEvent):
        super().hideEvent(event)
        self._waitingButton.setVisible(False)

    # expose Waiting push button API
    def setIconSize(self, size):
        self._waitingButton.setIconSize(size)

    def sizeHin(self):
        return self._waitingButton.sizeHint()
