"""3D surface (height map) visualization.

Demonstrates: 2D scatter as solid surface with height map,
wireframe, and points modes side by side.
"""

import numpy
from silx.gui import qt
from silx.gui.plot3d.SceneWindow import SceneWindow


def main():
    app = qt.QApplication([])

    window = SceneWindow()
    window.setWindowTitle("3D Surface - Height Map")

    scene = window.getSceneWidget()
    scene.setBackgroundColor((0.12, 0.12, 0.18, 1.0))
    scene.setForegroundColor((0.9, 0.9, 0.9, 1.0))
    scene.setTextColor((0.9, 0.9, 0.9, 1.0))

    # Generate surface data on a grid
    n = 50
    x = numpy.linspace(-3, 3, n)
    y = numpy.linspace(-3, 3, n)
    xx, yy = numpy.meshgrid(x, y)
    xx = xx.ravel()
    yy = yy.ravel()
    values = numpy.sin(xx) * numpy.cos(yy) * numpy.exp(-(xx**2 + yy**2) / 8)

    modes = ["solid", "lines", "points"]
    for i, mode in enumerate(modes):
        scatter2d = scene.add2DScatter(xx, yy, values)
        scatter2d.setTranslation(i * 8.0, 0.0, 0.0)
        scatter2d.setHeightMap(True)
        scatter2d.setVisualization(mode)
        scatter2d.getColormap().setName("coolwarm")
        scatter2d.setLabel(f"Surface ({mode})")
        if mode == "points":
            scatter2d.setSymbolSize(4)
        if mode == "lines":
            scatter2d.setLineWidth(1.5)

    window.resize(1000, 600)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
