"""3D image and height map display.

Demonstrates: ImageData with colormap, ImageRgba, and HeightMapData
displayed in a 3D scene with transforms.
"""

import numpy
from silx.gui import qt
from silx.gui.plot3d.SceneWindow import SceneWindow, items


def main():
    app = qt.QApplication([])

    window = SceneWindow()
    window.setWindowTitle("3D Images & Height Maps")

    scene = window.getSceneWidget()
    scene.setBackgroundColor((0.12, 0.12, 0.18, 1.0))
    scene.setForegroundColor((0.9, 0.9, 0.9, 1.0))
    scene.setTextColor((0.9, 0.9, 0.9, 1.0))

    size = 256

    # 1. Grayscale image with colormap
    xx, yy = numpy.meshgrid(numpy.linspace(-5, 5, size), numpy.linspace(-5, 5, size))
    data = numpy.sin(xx) * numpy.cos(yy)

    imageData = scene.addImage(data.astype(numpy.float32))
    imageData.setLabel("Grayscale (magma)")
    imageData.getColormap().setName("magma")
    imageData.setInterpolation("linear")

    # 2. RGBA image
    rgba = numpy.zeros((size, size, 3), dtype=numpy.float32)
    rgba[:, :, 0] = numpy.clip((xx + 5) / 10, 0, 1)  # R: left-right gradient
    rgba[:, :, 1] = numpy.clip(numpy.exp(-(xx**2 + yy**2) / 8), 0, 1)  # G: center blob
    rgba[:, :, 2] = numpy.clip((yy + 5) / 10, 0, 1)  # B: bottom-top gradient

    imageRgba = scene.addImage(rgba)
    imageRgba.setLabel("RGB image")
    imageRgba.setTranslation(size + 20, 0, 0)

    # 3. Height map
    heightData = numpy.exp(-(xx**2 + yy**2) / 4).astype(numpy.float32)

    heightMap = items.HeightMapData()
    heightMap.setData(heightData)
    heightMap.getColormap().setName("viridis")
    heightMap.setTranslation(0, size + 20, 0)
    heightMap.setScale(1, 1, 50)  # exaggerate height
    heightMap.setLabel("Height map")
    scene.addItem(heightMap)

    window.resize(900, 600)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
