import pytest
from silx.gui import qt
from silx.gui.utils.waiteroverlay import WaiterOverlay
from silx.gui.plot import Plot2D
from silx.gui.plot.PlotWidget import PlotWidget


@pytest.mark.parametrize("widget_parent", (Plot2D, qt.QFrame))
def test_show(widget_parent):
    # simple test of the WaiterOverlay component
    widget = widget_parent()
    waitingOverlay = WaiterOverlay(widget)
    waitingOverlay.setWaiting(True)
    widget.show()
    waitingOverlay.setWaiting(False)

    assert waitingOverlay.parent() is widget
    if isinstance(widget, PlotWidget):
        assert waitingOverlay._waitingButton.parent() is widget.getWidgetHandle()
    else:
        assert waitingOverlay._waitingButton.parent() is widget

    widget.close()
    waitingOverlay.close()
