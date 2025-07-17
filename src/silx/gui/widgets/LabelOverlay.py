from silx.gui import qt

from .OverlayMixIn import OverlayMixIn as _OverlayMixIn


class LabelOverlay(_OverlayMixIn, qt.QLabel):
    """Display a Label on top of a widget"""

    def __init__(self, parent: qt.QWidget | None = None) -> None:
        qt.QLabel.__init__(self, parent)
        _OverlayMixIn.__init__(self, parent)
