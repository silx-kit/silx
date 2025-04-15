from silx.gui import qt

from .OverlayMixIn import _OverlayMixIn


class LabelOverlay(_OverlayMixIn, qt.QLabel):
    """Display a Label on top of a PlotWidget"""

    def __init__(self, parent: qt.QWidget, *args, **kwargs) -> None:
        qt.QLabel.__init__(self, parent, *args, **kwargs)
        _OverlayMixIn.__init__(self, parent)
