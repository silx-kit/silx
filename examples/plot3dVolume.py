# /*##########################################################################
#
# Copyright (c) 2024 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/

"""
This script displays a 3D scalar field volume with isosurfaces and a cut plane.

It demonstrates ScalarField3D with multiple isosurface levels and an
interactive cut plane with colormap.
"""

from __future__ import annotations

__license__ = "MIT"

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
