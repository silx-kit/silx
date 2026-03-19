"""3D clipping plane and group transforms.

Demonstrates: ClipPlane to slice through 3D objects,
GroupItem for shared transforms, multiple items in a scene.
"""

import numpy
from silx.gui import qt
from silx.gui.plot3d.SceneWindow import SceneWindow, items


def main():
    app = qt.QApplication([])

    window = SceneWindow()
    window.setWindowTitle("3D Clipping & Groups")

    scene = window.getSceneWidget()
    scene.setBackgroundColor((0.1, 0.1, 0.15, 1.0))
    scene.setForegroundColor((0.9, 0.9, 0.9, 1.0))
    scene.setTextColor((0.9, 0.9, 0.9, 1.0))

    # Create a group for clipped items
    group = items.GroupItem()
    group.setLabel("Clipped group")

    # Add a clipping plane
    clip = items.ClipPlane()
    clip.setNormal((1.0, 0.3, 0.0))
    clip.setPoint((32, 32, 32))
    group.addItem(clip)

    # Add a 3D volume to the group
    size = 64
    lin = numpy.linspace(-3, 3, size)
    x, y, z = numpy.meshgrid(lin, lin, lin)
    data = numpy.exp(-(x**2 + y**2 + z**2)).astype(numpy.float32)

    volume = items.ScalarField3D()
    volume.setData(data)
    volume.setLabel("Volume")
    volume.addIsosurface(0.3, "#FF8800AA")
    volume.addIsosurface(0.7, "#0088FFCC")
    group.addItem(volume)

    # Add a 3D scatter to the same group (also gets clipped)
    n = 2000
    sx = numpy.random.normal(32, 15, n).astype(numpy.float32)
    sy = numpy.random.normal(32, 15, n).astype(numpy.float32)
    sz = numpy.random.normal(32, 15, n).astype(numpy.float32)
    sv = numpy.sqrt((sx - 32) ** 2 + (sy - 32) ** 2 + (sz - 32) ** 2)

    scatter = items.Scatter3D()
    scatter.setData(sx, sy, sz, sv)
    scatter.getColormap().setName("plasma")
    scatter.setSymbol("o")
    scatter.setSymbolSize(4)
    scatter.setLabel("Scatter (clipped)")
    group.addItem(scatter)

    scene.addItem(group)

    # Add an unclipped reference box outside the group
    box = items.Box()
    box.setData(size=(10, 10, 10))
    box.color = (0.5, 0.9, 0.5, 0.5)
    box.setTranslation(80, 32, 32)
    box.setLabel("Unclipped box")
    scene.addItem(box)

    window.resize(900, 600)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
