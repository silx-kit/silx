from silx.gui import qt

from .OverlayMixIn import _OverlayMixIn


class LabelOverlay(_OverlayMixIn, qt.QLabel):
    """Display a Label on top of a PlotWidget"""

    def __init__(self, parent: qt.QWidget | None = None) -> None:
        qt.QLabel.__init__(self, parent)
        _OverlayMixIn.__init__(self, parent)
