import pytest
from silx.gui import qt
from silx.gui.widgets.WaitingOverlay import WaitingOverlay
from silx.gui.plot import Plot2D
from silx.gui.plot.PlotWidget import PlotWidget


@pytest.mark.parametrize("widget_parent", (Plot2D, qt.QFrame))
def test_show(qapp, qapp_utils, widget_parent):
    """Simple test of the WaitingOverlay component"""
    widget = widget_parent()
    widget.setAttribute(qt.Qt.WA_DeleteOnClose)

    waitingOverlay = WaitingOverlay(widget)
    waitingOverlay.setAttribute(qt.Qt.WA_DeleteOnClose)

    widget.show()
    qapp_utils.qWaitForWindowExposed(widget)
    assert waitingOverlay._waitingButton.isWaiting()

    waitingOverlay.setText("test")
    qapp.processEvents()
    assert waitingOverlay.text() == "test"
    qapp_utils.qWait(1000)

    waitingOverlay.hide()
    qapp.processEvents()

    widget.close()
    waitingOverlay.close()
    qapp.processEvents()
