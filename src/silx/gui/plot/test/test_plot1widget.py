import numpy
import pytest

from silx.gui.plot import Plot1D


@pytest.mark.parametrize("yaxis", ["left", "right"])
def test_plot1d_resetzoom_xaxis_noautoscale(qapp, qWidgetFactory, yaxis):
    """
    Test that resetting the zoom results in the Y-axis limits to be
    the Y data range inside the zoomed X-range.
    """
    plot = qWidgetFactory(Plot1D)

    x = numpy.linspace(0, 100, 5000)

    y_base = numpy.sin(x * 0.2)
    y_spike50 = y_base + 2.0 * numpy.exp(-0.5 * ((x - 50) / 0.3) ** 2)
    y_spike85 = y_base + 20.0 * numpy.exp(-0.5 * ((x - 85) / 0.2) ** 2)

    plot.addCurve(x, y_spike50, legend="y_spike50", yaxis=yaxis)
    plot.addCurve(x, y_spike85, legend="y_spike85", yaxis=yaxis)

    qapp.processEvents()

    xaxis = plot.getXAxis()
    yaxis = plot.getYAxis()

    xaxis.setLimits(40, 60)
    xaxis.setAutoScale(False)

    qapp.processEvents()

    plot.resetZoom()
    qapp.processEvents()

    ymin, ymax = yaxis.getLimits()

    mask = (x >= 40) & (x <= 60)
    expected_ymin = numpy.min(y_spike50[mask])
    expected_ymax = numpy.max(y_spike50[mask])

    assert ymin == pytest.approx(expected_ymin, rel=1e-6, abs=1e-12)
    assert ymax == pytest.approx(expected_ymax, rel=1e-6, abs=1e-12)
