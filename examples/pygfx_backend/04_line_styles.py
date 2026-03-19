"""Line styles and symbols with pygfx backend.

Demonstrates: all line styles (solid, dashed, dash-dot, dotted),
various symbols, line widths, gap colors.
"""

import numpy
from silx.gui import qt
from silx.gui.plot import PlotWidget


def main():
    app = qt.QApplication([])

    plot = PlotWidget(backend="pygfx")
    plot.setWindowTitle("pygfx - Line Styles & Symbols")
    plot.setGraphTitle("Line Styles and Symbols")
    plot.setGraphXLabel("X")
    plot.setGraphYLabel("Y")

    x = numpy.linspace(0, 10, 100)

    # Line styles
    styles = [
        ("-", "solid"),
        ("--", "dashed"),
        ("-.", "dash-dot"),
        (":", "dotted"),
    ]
    for i, (style, name) in enumerate(styles):
        y = numpy.sin(x) + i * 2.5
        plot.addCurve(x, y, legend=name, linestyle=style, linewidth=2, symbol="")

    # Gap color example
    y = numpy.sin(x) + 10
    curve = plot.addCurve(
        x,
        y,
        legend="dashed+gapcolor",
        linestyle="--",
        linewidth=2,
        symbol="",
        color="blue",
    )

    # Symbols (only those supported by silx SymbolMixIn)
    symbols = ["o", ".", "+", "x", "d", "s", ",", "|", "_"]
    x_sym = numpy.linspace(0, 10, 30)
    for i, sym in enumerate(symbols):
        y_sym = numpy.cos(x_sym) + 15 + i * 1.5
        plot.addCurve(
            x_sym,
            y_sym,
            legend=f"sym '{sym}'",
            symbol=sym,
            linestyle=" ",
            color=f"C{i % 10}",
        )

    plot.setActiveCurveHandling(False)
    plot.resetZoom()
    plot.show()
    app.exec()


if __name__ == "__main__":
    main()
