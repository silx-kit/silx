from silx.gui import qt

from ._OverlayMixIn import OverlayMixIn as _OverlayMixIn


class ButtonOverlay(_OverlayMixIn, qt.QPushButton):
    """Display a Button on top of a QWidget"""

    def __init__(self, parent: qt.QWidget | None = None) -> None:
        qt.QPushButton.__init__(self, parent)
        _OverlayMixIn.__init__(self, parent)
