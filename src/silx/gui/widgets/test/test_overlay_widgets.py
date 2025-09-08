import pytest
from silx.gui import qt
from silx.gui.widgets.WaitingOverlay import WaitingOverlay
from silx.gui.widgets.LabelOverlay import LabelOverlay
from silx.gui.widgets.ButtonOverlay import ButtonOverlay
from silx.gui.plot import Plot2D


@pytest.mark.parametrize("widget_parent", (Plot2D, qt.QFrame))
def test_show(qapp, qapp_utils, widget_parent):
    """Simple test of the WaitingOverlay component"""
    widget = widget_parent()
    widget.setAttribute(qt.Qt.WA_DeleteOnClose)

    waitingOverlay = WaitingOverlay(widget)
    waitingOverlay.setAttribute(qt.Qt.WA_DeleteOnClose)

    widget.show()
    qapp_utils.qWaitForWindowExposed(widget)
    assert waitingOverlay.isWaiting()

    waitingOverlay.setText("test")
    qapp.processEvents()
    assert waitingOverlay.text() == "test"
    qapp_utils.qWait(1000)

    waitingOverlay.hide()
    qapp.processEvents()

    widget.close()
    waitingOverlay.close()
    qapp.processEvents()


@pytest.mark.parametrize(
    "alignment",
    (
        qt.Qt.AlignLeft,
        qt.Qt.AlignRight | qt.Qt.AlignTop,
        qt.Qt.AlignCenter,
        qt.Qt.AlignLeft | qt.Qt.AlignBottom,
    ),
)
@pytest.mark.parametrize("widget_parent", (Plot2D, qt.QFrame))
@pytest.mark.parametrize("constructor", (LabelOverlay, ButtonOverlay))
def test_overlay_widgets(qapp, qapp_utils, widget_parent, constructor, alignment):
    widget = widget_parent()
    widget.setAttribute(qt.Qt.WA_DeleteOnClose)

    overlayWidget = constructor(widget)
    overlayWidget.setAttribute(qt.Qt.WA_DeleteOnClose)
    overlayWidget.setAlignment(alignment)
    assert overlayWidget.getAlignment() == alignment

    widget.show()
    qapp_utils.qWaitForWindowExposed(widget)
    widget.close()
    overlayWidget.close()
    qapp.processEvents()
