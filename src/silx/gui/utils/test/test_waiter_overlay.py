from silx.gui.utils.waiteroverlay import WaiterOverlay
from silx.gui.plot import Plot2D
from silx.gui import qt


def test_show(qapp_utils):
    # simple test of the WaiterOverlay component
    plot = Plot2D()
    waitingOverlay = WaiterOverlay(plot)
    waitingOverlay.setWaiting(True)
    plot.show()
    waitingOverlay.setWaiting(False)

    assert waitingOverlay.getBaseWidget() is plot

    plot.close()
    waitingOverlay.close()
