"""Tests of the ColormapDialog scale to cursor mode"""

import numpy
import pytest

from silx.gui.plot import Plot2D, items


@pytest.fixture
def plot(qWidgetFactory, qapp_utils):
    # Left half in [0, 1), right half in [100, 101)
    data = numpy.zeros((100, 100), dtype=numpy.float32)
    data[:, :50] = numpy.random.random((100, 50))
    data[:, 50:] = 100 + numpy.random.random((100, 50))

    plot = qWidgetFactory(Plot2D)
    plot.resize(600, 500)
    plot.addImage(data, legend="image", origin=(-50, 10), scale=(0.5, 2))
    # Image covers x in [-50, 0] and y in [10, 210]
    plot.setLimits(-100, 50, 0, 300)
    qapp_utils.qWait(100)
    yield plot
    # Do not leave the dialog open/attached after the test
    dialog = plot.getColormapAction().getColormapDialog()
    dialog.getScaleToCursorAction().setChecked(False)
    dialog.close()
    qapp_utils.qWait(100)


def _moveMouse(qapp_utils, plot, x, y):
    # Make sure the plot is rendered, picking requires it
    plot.replot()
    qapp_utils.qWait(50)
    xPixel, yPixel = plot.dataToPixel(x, y)
    plot.onMouseMove(xPixel, yPixel)


def _outlines(plot):
    return [item for item in plot.getItems() if isinstance(item, items.Shape)]


def testScaleToCursor(plot, qapp_utils):
    image = plot.getImage("image")
    colormap = image.getColormap()
    initialRange = colormap.getVRange()

    # Dialog is created but not shown: it already knows the image
    dialog = plot.getColormapAction().getColormapDialog()
    assert not dialog.isVisible()
    action = dialog.getScaleToCursorAction()
    assert action.isEnabled()

    # Not checked: no update
    _moveMouse(qapp_utils, plot, -40, 110)
    assert colormap.getVRange() == initialRange

    action.setChecked(True)
    _moveMouse(qapp_utils, plot, -40, 110)  # Left half of the image
    vmin, vmax = colormap.getVRange()
    assert 0 <= vmin < vmax < 1
    assert len(_outlines(plot)) == 1

    _moveMouse(qapp_utils, plot, -10, 110)  # Right half of the image
    vmin, vmax = colormap.getVRange()
    assert 100 <= vmin < vmax < 101

    # Mouse outside of the image: no update, outline hidden
    _moveMouse(qapp_utils, plot, -60, 110)
    assert colormap.getVRange() == (vmin, vmax)
    assert not _outlines(plot)[0].isVisible()

    # Unchecked: range is kept and outline is removed
    action.setChecked(False)
    assert _outlines(plot) == []
    _moveMouse(qapp_utils, plot, -40, 110)
    assert colormap.getVRange() == (vmin, vmax)


def testScaleToCursorDisabled(qWidgetFactory):
    plot = qWidgetFactory(Plot2D)
    dialog = plot.getColormapAction().getColormapDialog()
    action = dialog.getScaleToCursorAction()
    assert not action.isEnabled()  # No item

    # Not supported for scatters
    plot.addScatter((0, 1), (0, 1), (0, 1), legend="scatter")
    plot.setActiveScatter("scatter")
    assert not action.isEnabled()


def _changeImage(plot, dialog):
    plot.addImage(numpy.ones((10, 10)), legend="other")
    plot.setActiveImage("other")


_STOPPERS = {
    "image-changed": _changeImage,
    "image-removed": lambda plot, dialog: plot.remove("image", kind="image"),
    "visible-area": lambda plot, dialog: dialog._visibleAreaButton.click(),
    "selection": lambda plot, dialog: dialog._selectedAreaButton.setChecked(True),
    "autoscale": lambda plot, dialog: dialog._autoButtons.click(),
    "min-autoscale": lambda plot, dialog: dialog._minValue._autoToggleAction.trigger(),
    "max-autoscale": lambda plot, dialog: dialog._maxValue._autoToggleAction.trigger(),
}


@pytest.mark.parametrize("stop", _STOPPERS.values(), ids=_STOPPERS.keys())
def testScaleToCursorStopped(plot, qapp_utils, stop):
    dialog = plot.getColormapAction().getColormapDialog()
    action = dialog.getScaleToCursorAction()
    plot.getColormapAction().trigger()  # Show the dialog
    qapp_utils.qWait(100)

    action.setChecked(True)
    # Set a fixed range, so that autoscale controls switch autoscale on
    _moveMouse(qapp_utils, plot, -40, 110)
    assert None not in plot.getImage("image").getColormap().getVRange()

    stop(plot, dialog)
    qapp_utils.qWait(100)
    assert not action.isChecked()


def testScaleToCursorButton(plot, qapp_utils):
    dialog = plot.getColormapAction().getColormapDialog()
    action = dialog.getScaleToCursorAction()

    plot.getColormapAction().trigger()  # Show the dialog
    qapp_utils.qWait(100)
    assert dialog.isVisible()

    # Button and action are synchronized
    dialog._cursorButton.click()
    assert action.isChecked()
    action.setChecked(False)
    assert not dialog._cursorButton.isChecked()

    # Cursor mode stops selection mode
    dialog._selectedAreaButton.setChecked(True)
    qapp_utils.qWait(100)  # Selection button handler is a queued connection
    action.setChecked(True)
    qapp_utils.qWait(100)
    assert not dialog._selectedAreaButton.isChecked()

    # Mode is kept when the dialog is hidden
    dialog.hide()
    qapp_utils.qWait(100)
    assert action.isChecked()


def testVisibleAreaStopsSelection(plot, qapp_utils):
    dialog = plot.getColormapAction().getColormapDialog()
    plot.getColormapAction().trigger()
    qapp_utils.qWait(100)
    dialog._selectedAreaButton.setChecked(True)
    dialog._visibleAreaButton.click()
    qapp_utils.qWait(100)
    assert not dialog._selectedAreaButton.isChecked()
