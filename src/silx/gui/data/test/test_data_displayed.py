"""Notifications and controls usable by applications hosting data views."""

import numpy

from silx.gui.data import DataViews
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.utils.testutils import SignalListener


def test_data_displayed_reports_updates_that_selection_changes_miss(
    qWidgetFactory, qapp_utils
):
    """`dataDisplayed` tells when the view actually shows new data.

    The view is updated from a deferred call, so `dataChanged` comes too
    early, and `selectionChanged` is not emitted at all when the new data has
    the same shape as the previous one. An application mirroring the displayed
    data needs a signal covering that case.
    """
    frame = qWidgetFactory(DataViewerFrame)
    displayed, selected = SignalListener(), SignalListener()
    frame.dataDisplayed.connect(displayed)
    frame.selectionChanged.connect(selected)
    frame.setData(numpy.arange(12).reshape(3, 4))
    frame.setDisplayMode(DataViews.RAW_MODE)
    qapp_utils.qWait(30)
    displayed.clear()
    selected.clear()

    # Same shape: the displayed data changes, the selection does not
    frame.setData(numpy.full((3, 4), 99))
    qapp_utils.qWait(30)

    assert displayed.callCount() == 1
    assert selected.callCount() == 0
    numpy.testing.assert_array_equal(frame.data(), numpy.full((3, 4), 99))

    displayed.clear()

    # A different shape changes the selection as well
    frame.setData(numpy.arange(60).reshape(3, 4, 5))
    qapp_utils.qWait(30)

    assert displayed.callCount() == 1
    assert selected.callCount() == 1
