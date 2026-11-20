"""Tests of the matplotlib backend with a layout engine"""

import numpy
import pytest

import silx
from silx.gui.plot import PlotWidget
from silx.gui.plot.PlotInteraction import Pan


def _getAspectRatio(plot):
    width, height = plot.getPlotBoundsInPixels()[2:]
    xmin, xmax = plot.getXAxis().getLimits()
    ymin, ymax = plot.getYAxis().getLimits()
    return ((xmax - xmin) / width) / ((ymax - ymin) / height)


@pytest.fixture
def tightPlot(qWidgetFactory, qapp, monkeypatch):
    monkeypatch.setattr(silx.config, "_MPL_TIGHT_LAYOUT", True)
    plot = qWidgetFactory(PlotWidget, backend="mpl")
    plot.resize(600, 400)
    plot.setKeepDataAspectRatio(True)
    plot.setGraphXLabel("x")
    plot.setGraphYLabel("a long\ny label")
    plot.addImage(numpy.random.rand(100, 300) * 1e5)
    qapp.processEvents()
    plot.getWidgetHandle().draw()
    return plot


def testKeepAspectRatioWithTightLayout(tightPlot):
    """Aspect ratio is kept once the layout engine has resized the axes"""
    backend = tightPlot._backend
    assert backend.fig.get_layout_engine() is not None
    assert _getAspectRatio(tightPlot) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize(
    "size", [(400, 300), (300, 600), (900, 300), (800, 500), (200, 200)]
)
def testKeepAspectRatioWithTightLayoutAfterResize(tightPlot, qapp, size):
    """Aspect ratio is kept after resizing the widget"""
    tightPlot.resize(*size)
    qapp.processEvents()
    tightPlot.getWidgetHandle().draw()
    assert _getAspectRatio(tightPlot) == pytest.approx(1.0, abs=1e-3)


def testKeepAspectRatioWithTightLayoutAfterPan(tightPlot, qapp):
    """Aspect ratio is kept after panning"""
    pan = Pan(tightPlot)
    pan.beginDrag(300, 200, "left")
    for i in range(1, 30):
        pan.drag(300 + i * 5, 200 + i * 3, "left")
    pan.endDrag((300, 200), (450, 290), "left")
    qapp.processEvents()
    tightPlot.getWidgetHandle().draw()

    assert _getAspectRatio(tightPlot) == pytest.approx(1.0, abs=1e-3)
