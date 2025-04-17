from silx.gui import qt

from .OverlayMixIn import _OverlayMixIn


class ButtonOverlay(_OverlayMixIn, qt.QPushButton):
    """Display a Label on top of a PlotWidget"""

    def __init__(self, parent: qt.QWidget | None = None) -> None:
        qt.QPushButton.__init__(self, parent)
        _OverlayMixIn.__init__(self, parent)
