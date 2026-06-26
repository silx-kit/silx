"""Compare all three backends: matplotlib, opengl, pygfx.

Displays the same data in three side-by-side PlotWidgets.
"""

import numpy
from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.utils.axis import SyncAxes


def populate(plot):
    """Add curves, image, scatter, and markers to a plot."""
    x = numpy.linspace(0, 10, 200)

    # Curves
    plot.addCurve(x, numpy.sin(x), legend="sin", color="blue", linewidth=2)
    plot.addCurve(
        x,
        numpy.cos(x),
        legend="cos",
        color="red",
        linewidth=1.5,
        linestyle="--",
        symbol="o",
    )

    # Markers
    plot.addMarker(5, 0, legend="center", text="center", color="green", symbol="d")
    plot.addXMarker(numpy.pi, legend="pi", text="pi", color="orange")

    plot.setActiveCurveHandling(False)
    plot.resetZoom()


def main():
    app = qt.QApplication([])

    window = qt.QWidget()
    window.setWindowTitle("Backend Comparison: mpl vs opengl vs pygfx")
    layout = qt.QHBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)

    backends = ["mpl", "opengl", "pygfx"]
    plots = []

    for backend in backends:
        try:
            p = PlotWidget(backend=backend)
            p.setGraphTitle(backend)
            populate(p)
            plots.append(p)
            layout.addWidget(p)
        except Exception as e:
            label = qt.QLabel(f"{backend}: {e}")
            layout.addWidget(label)

    # Sync axes across all plots
    if len(plots) > 1:
        SyncAxes([p.getXAxis() for p in plots])
        SyncAxes([p.getYAxis() for p in plots])

    window.resize(1500, 500)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
