"""3D scalar field volume with isosurfaces and cut plane.

Demonstrates: ScalarField3D, isosurfaces at multiple levels,
interactive cut plane with colormap.
"""

import numpy
from silx.gui import qt
from silx.gui.plot3d.SceneWindow import SceneWindow, items


def main():
    app = qt.QApplication([])

    window = SceneWindow()
    window.setWindowTitle("3D Volume - Isosurfaces & Cut Plane")

    scene = window.getSceneWidget()
    scene.setBackgroundColor((0.1, 0.1, 0.15, 1.0))
    scene.setForegroundColor((0.9, 0.9, 0.9, 1.0))
    scene.setTextColor((0.9, 0.9, 0.9, 1.0))

    # Generate 3D Gaussian + noise volume
    size = 64
    lin = numpy.linspace(-3, 3, size)
    x, y, z = numpy.meshgrid(lin, lin, lin)
    data = (
        numpy.exp(-(x**2 + y**2 + z**2))
        + 0.5 * numpy.exp(-((x - 1.5) ** 2 + (y - 1) ** 2 + (z + 1) ** 2) / 0.5)
        + 0.02 * numpy.random.random((size, size, size))
    )

    volume = items.ScalarField3D()
    volume.setData(data.astype(numpy.float32))
    volume.setLabel("Gaussian blobs")

    # Add isosurfaces at different levels
    volume.addIsosurface(0.3, "#FF660080")  # orange, semi-transparent
    volume.addIsosurface(0.6, "#3399FF80")  # blue, semi-transparent
    volume.addIsosurface(0.9, "#FF3366CC")  # red, more opaque

    # Set up cut plane
    cutPlane = volume.getCutPlanes()[0]
    cutPlane.setVisible(True)
    cutPlane.getColormap().setName("magma")
    cutPlane.setNormal((0.0, 0.0, 1.0))
    cutPlane.moveToCenter()

    scene.addItem(volume)

    window.resize(800, 600)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
