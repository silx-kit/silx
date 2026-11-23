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

    # Mode is kept when the dialog is hidden
    action.setChecked(True)
    dialog.hide()
    qapp_utils.qWait(100)
    assert action.isChecked()

    # Selection and cursor modes are exclusive
    dialog.show()
    qapp_utils.qWait(100)
    dialog._selectedAreaButton.setChecked(True)
    qapp_utils.qWait(100)
    assert not action.isChecked()
    action.setChecked(True)
    qapp_utils.qWait(100)
    assert not dialog._selectedAreaButton.isChecked()


def testScaleToCursorStopped(plot, qapp_utils):
    dialog = plot.getColormapAction().getColormapDialog()
    action = dialog.getScaleToCursorAction()

    # Stopped when the dialog item changes
    action.setChecked(True)
    plot.addImage(numpy.ones((10, 10)), legend="other")
    plot.setActiveImage("other")
    qapp_utils.qWait(100)
    assert not action.isChecked()

    # Stopped when the image is removed
    action.setChecked(True)
    assert action.isChecked()
    plot.remove("other", kind="image")
    qapp_utils.qWait(100)
    assert not action.isChecked()


def testScaleToCursorDisabled(qWidgetFactory):
    plot = qWidgetFactory(Plot2D)
    dialog = plot.getColormapAction().getColormapDialog()
    action = dialog.getScaleToCursorAction()
    assert not action.isEnabled()  # No item

    # Not supported for scatters
    plot.addScatter((0, 1), (0, 1), (0, 1), legend="scatter")
    plot.setActiveScatter("scatter")
    assert not action.isEnabled()
    action.setChecked(True)
    assert not action.isChecked() or not action.isEnabled()


def testScaleToModesExclusive(plot, qapp_utils):
    dialog = plot.getColormapAction().getColormapDialog()
    action = dialog.getScaleToCursorAction()
    plot.getColormapAction().trigger()  # Show the dialog
    qapp_utils.qWait(100)

    # Visible area stops cursor mode
    action.setChecked(True)
    dialog._visibleAreaButton.click()
    qapp_utils.qWait(100)
    assert not action.isChecked()

    # Visible area stops selection mode
    dialog._selectedAreaButton.setChecked(True)
    qapp_utils.qWait(100)
    dialog._visibleAreaButton.click()
    qapp_utils.qWait(100)
    assert not dialog._selectedAreaButton.isChecked()


@pytest.mark.parametrize("control", ["autoscale", "min", "max"])
def testAutoscaleStopsScaleToCursor(plot, qapp_utils, control):
    dialog = plot.getColormapAction().getColormapDialog()
    action = dialog.getScaleToCursorAction()
    colormap = plot.getImage("image").getColormap()
    plot.getColormapAction().trigger()  # Show the dialog
    qapp_utils.qWait(100)

    action.setChecked(True)
    _moveMouse(qapp_utils, plot, -40, 110)
    assert None not in colormap.getVRange()

    if control == "autoscale":
        dialog._autoButtons.click()
    elif control == "min":
        dialog._minValue._autoToggleAction.trigger()
    else:
        dialog._maxValue._autoToggleAction.trigger()
    qapp_utils.qWait(100)
    assert not action.isChecked()
    assert None in colormap.getVRange()

    # Mouse moves no longer change the range
    vrange = colormap.getVRange()
    _moveMouse(qapp_utils, plot, -10, 110)
    assert colormap.getVRange() == vrange
