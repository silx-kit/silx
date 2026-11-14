"""Notifications and controls usable by applications hosting data views."""

import numpy

from silx.gui import qt
from silx.gui.data import DataViews
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.data.NumpyAxesSelector import NumpyAxesSelector
from silx.gui.utils.testutils import SignalListener


def test_displayed_reports_new_data_with_an_unchanged_selection(
    qWidgetFactory, qapp_utils
):
    frame = qWidgetFactory(DataViewerFrame)
    displayed, selected = SignalListener(), SignalListener()
    frame.dataDisplayed.connect(displayed)
    frame.selectionChanged.connect(selected)
    frame.setData(numpy.arange(12).reshape(3, 4))
    frame.setDisplayMode(DataViews.RAW_MODE)
    qapp_utils.qWait(30)
    previous_updates, previous_selections = displayed.callCount(), selected.callCount()
    assert previous_updates > 0

    frame.setData(numpy.full((3, 4), 99))
    qapp_utils.qWait(30)

    assert displayed.callCount() > previous_updates
    assert selected.callCount() == previous_selections
    numpy.testing.assert_array_equal(frame.data(), numpy.full((3, 4), 99))


def test_axes_controls_keep_driving_the_view_after_reparenting(
    qWidgetFactory, qapp_utils
):
    frame = qWidgetFactory(DataViewerFrame)
    frame.setData(numpy.arange(60).reshape(3, 4, 5))
    frame.setDisplayMode(DataViews.RAW_MODE)
    qapp_utils.qWait(30)
    controls = frame.getAxesSelectorWidget()
    selector = controls.findChild(NumpyAxesSelector)
    host = qWidgetFactory(qt.QWidget)
    host.setLayout(qt.QVBoxLayout())
    original_parent = controls.parentWidget()
    original_layout = original_parent.layout()
    try:
        host.layout().addWidget(controls)
        assert controls.parentWidget() is host
        updates = SignalListener()
        frame.selectionChanged.connect(updates)
        selector.setSelection((1, slice(None), slice(None)))
        qapp_utils.qWait(30)
        assert updates.callCount() > 0
        assert frame.getAxesSelectorWidget() is controls
    finally:
        original_layout.addWidget(controls)
