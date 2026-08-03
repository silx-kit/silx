"""Markers and shapes with pygfx backend.

Demonstrates: point markers with text, x/y markers, shapes (rectangle, polyline).
"""

import numpy
from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot import items


def main():
    app = qt.QApplication([])

    plot = PlotWidget(backend="pygfx")
    plot.setWindowTitle("pygfx - Markers & Shapes")
    plot.setGraphTitle("Markers, Text Labels and Shapes")

    # Background image for visual reference
    size = 100
    xx, yy = numpy.meshgrid(numpy.linspace(0, 1, size), numpy.linspace(0, 1, size))
    image = numpy.sin(10 * xx) * numpy.cos(10 * yy)
    plot.addImage(image, origin=(0, 0), scale=(100 / size, 100 / size))
    plot.getDefaultColormap().setName("gray")

    # Point markers with text
    plot.addMarker(20, 80, legend="marker1", text="Point A", color="red", symbol="o")
    plot.addMarker(50, 60, legend="marker2", text="Point B", color="blue", symbol="d")
    plot.addMarker(80, 80, legend="marker3", text="Point C", color="green", symbol="s")

    # Horizontal and vertical markers
    plot.addXMarker(30, legend="x_marker", text="X=30", color="yellow")
    plot.addYMarker(40, legend="y_marker", text="Y=40", color="cyan")

    # Rectangle shape
    rect = items.Shape("rectangle")
    rect.setPoints(numpy.array([(10, 10), (45, 45)]))
    rect.setColor("red")
    rect.setLineWidth(2)
    plot.addItem(rect)

    # Polyline shape
    poly = items.Shape("polylines")
    poly.setPoints(numpy.array([(55, 10), (70, 40), (85, 15), (95, 35)]))
    poly.setColor("green")
    poly.setLineWidth(2)
    plot.addItem(poly)

    plot.setGraphXLimits(-5, 105)
    plot.setGraphYLimits(-5, 105)
    plot.resetZoom()
    plot.show()
    app.exec()


if __name__ == "__main__":
    main()
