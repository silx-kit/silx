"""3D mesh and primitive shapes.

Demonstrates: custom Mesh with triangle data, Box, Cylinder, Hexagon
primitives with transforms and grouping.
"""

import numpy
from silx.gui import qt
from silx.gui.plot3d.SceneWindow import SceneWindow, items


def make_sphere_mesh(radius=1.0, n_lat=20, n_lon=20):
    """Generate sphere triangle mesh vertices and per-vertex colors."""
    positions = []
    colorvals = []

    for i in range(n_lat):
        theta0 = numpy.pi * i / n_lat
        theta1 = numpy.pi * (i + 1) / n_lat
        for j in range(n_lon):
            phi0 = 2 * numpy.pi * j / n_lon
            phi1 = 2 * numpy.pi * (j + 1) / n_lon

            # Two triangles per quad
            p00 = [
                radius * numpy.sin(theta0) * numpy.cos(phi0),
                radius * numpy.sin(theta0) * numpy.sin(phi0),
                radius * numpy.cos(theta0),
            ]
            p10 = [
                radius * numpy.sin(theta1) * numpy.cos(phi0),
                radius * numpy.sin(theta1) * numpy.sin(phi0),
                radius * numpy.cos(theta1),
            ]
            p01 = [
                radius * numpy.sin(theta0) * numpy.cos(phi1),
                radius * numpy.sin(theta0) * numpy.sin(phi1),
                radius * numpy.cos(theta0),
            ]
            p11 = [
                radius * numpy.sin(theta1) * numpy.cos(phi1),
                radius * numpy.sin(theta1) * numpy.sin(phi1),
                radius * numpy.cos(theta1),
            ]

            positions.extend([p00, p10, p11, p00, p11, p01])
            # Color based on latitude
            c = float(i) / n_lat
            color = [c, 0.3, 1.0 - c, 1.0]
            colorvals.extend([color] * 6)

    return (
        numpy.array(positions, dtype=numpy.float32),
        numpy.array(colorvals, dtype=numpy.float32),
    )


def main():
    app = qt.QApplication([])

    window = SceneWindow()
    window.setWindowTitle("3D Mesh & Primitives")

    scene = window.getSceneWidget()
    scene.setBackgroundColor((0.1, 0.1, 0.15, 1.0))
    scene.setForegroundColor((0.9, 0.9, 0.9, 1.0))
    scene.setTextColor((0.9, 0.9, 0.9, 1.0))

    # Custom sphere mesh
    positions, vertex_colors = make_sphere_mesh(radius=2.0)
    normals = positions / numpy.linalg.norm(positions, axis=1, keepdims=True)

    mesh = items.Mesh()
    mesh.setData(
        position=positions,
        color=vertex_colors,
        normal=normals,
        mode="triangles",
    )
    mesh.setLabel("Sphere mesh")
    scene.addItem(mesh)

    # Box primitive
    box = items.Box()
    box.setData(size=(2, 2, 2))
    box.color = (0.2, 0.8, 0.3, 0.8)
    box.setTranslation(5, 0, 0)
    box.setLabel("Box")
    scene.addItem(box)

    # Cylinder primitive
    cylinder = items.Cylinder()
    cylinder.setData(radius=1.0, height=3.0)
    cylinder.color = (0.8, 0.3, 0.2, 0.8)
    cylinder.setTranslation(10, 0, 0)
    cylinder.setLabel("Cylinder")
    scene.addItem(cylinder)

    # Hexagon primitive
    hexagon = items.Hexagon()
    hexagon.setData(radius=1.5, height=2.0)
    hexagon.color = (0.3, 0.3, 0.9, 0.8)
    hexagon.setTranslation(15, 0, 0)
    hexagon.setLabel("Hexagon")
    scene.addItem(hexagon)

    window.resize(900, 600)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
