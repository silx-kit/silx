"""3D scatter plot with SceneWidget.

Demonstrates: 3D scatter points with colormap, symbol customization,
and group transforms.
"""

import numpy
from silx.gui import qt
from silx.gui.plot3d.SceneWindow import SceneWindow, items


def main():
    app = qt.QApplication([])

    window = SceneWindow()
    window.setWindowTitle("3D Scatter Plot")

    scene = window.getSceneWidget()
    scene.setBackgroundColor((0.15, 0.15, 0.2, 1.0))
    scene.setForegroundColor((0.9, 0.9, 0.9, 1.0))
    scene.setTextColor((0.9, 0.9, 0.9, 1.0))

    # Generate clustered 3D scatter data
    n_per_cluster = 500
    clusters = []
    centers = [(-2, -2, -2), (2, 2, 2), (-2, 2, 0), (2, -2, 0)]
    for cx, cy, cz in centers:
        x = numpy.random.normal(cx, 0.8, n_per_cluster)
        y = numpy.random.normal(cy, 0.8, n_per_cluster)
        z = numpy.random.normal(cz, 0.8, n_per_cluster)
        clusters.append((x, y, z))

    x = numpy.concatenate([c[0] for c in clusters])
    y = numpy.concatenate([c[1] for c in clusters])
    z = numpy.concatenate([c[2] for c in clusters])
    values = numpy.sqrt(x**2 + y**2 + z**2)  # distance from origin

    scatter = items.Scatter3D()
    scatter.setData(x, y, z, values)
    scatter.getColormap().setName("viridis")
    scatter.setSymbol("o")
    scatter.setSymbolSize(6)
    scatter.setLabel("Clustered scatter")

    scene.addItem(scatter)

    window.resize(800, 600)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
