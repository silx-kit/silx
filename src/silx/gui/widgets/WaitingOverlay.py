from silx.gui.widgets.WaitingPushButton import WaitingPushButton
from silx.gui import qt
from .OverlayMixIn import OverlayMixIn as _OverlayMixIn


class WaitingOverlay(_OverlayMixIn, WaitingPushButton):
    """Widget overlaying another widget with a processing wheel icon.

    :param parent: widget on top of which to display the "processing/waiting wheel"
    """

    def __init__(self, parent: qt.QWidget) -> None:
        WaitingPushButton.__init__(self, parent)
        _OverlayMixIn.__init__(self, parent)
        self.setContentsMargins(0, 0, 0, 0)
        self.setDown(True)
        self.setWaiting(True)
        self.setStyleSheet(
            "QPushButton { background-color: rgba(150, 150, 150, 40); border: 0px; border-radius: 10px; }"
        )
        self._registerParent(parent)

    def setText(self, text: str):
        """Set displayed text"""
        super().setText(text)
        self._resize()
