#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
"""Find contours examples

.. note:: This module has an optional dependancy with sci-kit image library.
    You might need to install it if you don't already have it.
"""

import logging
import sys
import numpy

logging.basicConfig()
_logger = logging.getLogger("find_contours")

from silx.gui import qt
import silx.gui.plot
from silx.gui.plot.Colormap import Colormap
import silx.image.bilinear


try:
    import skimage
except ImportError:
    skimage = None

if skimage is not None:
    try:
        from silx.image.marchingsquare._skimage import MarchingSquareSciKitImage
    except ImportError:
        MarchingSquareSciKitImage = None
else:
    MarchingSquareSciKitImage = None


def rescale_image(image, shape):
    y, x = numpy.ogrid[:shape[0], :shape[1]]
    y, x = y * 1.0 * image.shape[0] / shape[0], x * 1.0 * image.shape[1] / shape[1]
    b = silx.image.bilinear.BilinearImage(image)
    # TODO: could be optimized using strides
    x2d = numpy.zeros_like(y) + x
    y2d = numpy.zeros_like(x) + y
    result = b.map_coordinates((y2d, x2d))
    return result


def create_spiral(size, nb=1, freq=100):
    half = size // 2
    y, x = numpy.ogrid[-half:half, -half:half]
    coef = 1 / half
    y, x = y * coef, x * coef + 0.0001
    distance = numpy.sqrt(x * x + y * y)
    angle = numpy.arctan(y / x)
    data = numpy.sin(angle * nb * 2 + distance * freq * half / 100, dtype=numpy.float32)
    return data


def create_magnetic_field(size, x1=0.0, y1=0.0, x2=0.0, y2=0.0):
    half = size // 2
    yy, xx = numpy.ogrid[-half:half, -half:half]
    coef = 1 / half
    yy1, xx1 = (yy + half * y1) * coef, (xx + half * x1) * coef
    distance1 = numpy.sqrt(xx1 * xx1 + yy1 * yy1)
    yy2, xx2 = (yy + half * y2) * coef, (xx + half * x2) * coef
    distance2 = numpy.sqrt(xx2 * xx2 + yy2 * yy2)
    return (numpy.arctan2(distance1, distance2) - numpy.pi * 0.25) * 1000


def create_gravity_field(size, objects):
    half = size // 2
    yy, xx = numpy.ogrid[-half:half, -half:half]
    coef = 1 / half

    def distance(x, y):
        yy1, xx1 = (yy + half * y) * coef, (xx + half * x) * coef
        return numpy.sqrt(xx1 ** 2 + yy1 ** 2)
    result = numpy.zeros((size, size), dtype=numpy.float32)
    for x, y, m in objects:
        result += m / distance(x, y)
    return numpy.log(result) * 1000


def create_rings(size, dx=0, dy=0, freq=100, sx=1.0, sy=1.0):
    half = size // 2
    yy, xx = numpy.ogrid[-half:half, -half:half]
    coef = 1 / half
    yy, xx = (yy - (dy * half)) * coef, (xx - (dx * half)) * coef + 0.0001
    distance = numpy.sqrt(numpy.sqrt(xx * xx * sx + yy * yy * sy))
    data = numpy.fmod(distance * freq * half / 100, 1, dtype=numpy.float32)
    return data


def create_gradient(size, dx=0, dy=0, sx=1.0, sy=1.0):
    half = size // 2
    yy, xx = numpy.ogrid[-half:half, -half:half]
    coef = 1 / half
    yy, xx = (yy - (dy * half)) * coef, (xx - (dx * half)) * coef + 0.0001
    distance = numpy.sqrt(xx * xx * sx + yy * yy * sy)
    return distance


def create_value_noise(shape, octaves=8, weights=None, first_array=None):
    data = numpy.zeros(shape, dtype=numpy.float32)
    t = 2
    for i in range(octaves):
        if t > shape[0] and t > shape[1]:
            break
        if i == 0 and first_array is not None:
            d = first_array
        else:
            if weights is None:
                w = (256 >> i) - 1
            else:
                w = weights[i]
            d = numpy.random.randint(w, size=(t, t), dtype=numpy.uint8)
        d = rescale_image(d, shape)
        data = data + d
        t = t << 1
    return data


def create_island(shape, summit, under_water):
    # Force a centric shape
    first_array = numpy.zeros((4, 4), dtype=numpy.uint8)
    first_array[1:3, 1:3] = 255
    weights = [255] + [(256 >> (i)) - 1 for i in range(8)]
    data = create_value_noise(shape, octaves=7, first_array=first_array, weights=weights)
    # more slops
    data *= data
    # normalize the height
    data -= data.min()
    data = data * ((summit + under_water) / data.max()) - under_water
    return data


def createRgbaMaskImage(mask, color):
    """Generate an RGBA image where a custom color is apply to the location of
    the mask. Non masked part of the image is transparent."""
    image = numpy.zeros((mask.shape[0], mask.shape[1], 4), dtype=numpy.uint8)
    color = numpy.array(color)
    image[mask == True] = color
    return image


class FindContours(qt.QMainWindow):
    """
    This window show an example of use of a Hdf5TreeView.

    The tree is initialized with a list of filenames. A panel allow to play
    with internal property configuration of the widget, and a text screen
    allow to display events.
    """

    def __init__(self, filenames=None):
        """
        :param files_: List of HDF5 or Spec files (pathes or
            :class:`silx.io.spech5.SpecH5` or :class:`h5py.File`
            instances)
        """
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Silx HDF5 widget example")

        self.__plot = silx.gui.plot.Plot2D(parent=self)
        dummy = numpy.array([[0]])
        self.__plot.addImage(dummy, legend="image", z=-10, replace=False)
        dummy = numpy.array([[[0, 0, 0, 0]]])
        self.__plot.addImage(dummy, legend="mask", z=-5, replace=False)
        self.__plot.addImage(dummy, legend="iso-pixels", z=0, replace=False)

        self.__algo = None
        self.__polygons = []
        self.__customPolygons = []
        self.__image = None
        self.__mask = None

        mainPanel = qt.QWidget(self)
        layout = qt.QHBoxLayout()
        layout.addWidget(self.__createConfigurationPanel(self))
        layout.addWidget(self.__plot)
        mainPanel.setLayout(layout)

        self.setCentralWidget(mainPanel)

    def __createConfigurationPanel(self, parent):
        panel = qt.QWidget(parent=parent)
        layout = qt.QVBoxLayout()
        panel.setLayout(layout)

        self.__kind = qt.QButtonGroup(self)
        self.__kind.setExclusive(True)

        button = qt.QPushButton(parent=panel)
        button.setText("Island")
        button.clicked.connect(self.generateIsland)
        button.setCheckable(True)
        button.setChecked(True)
        layout.addWidget(button)
        self.__kind.addButton(button)

        button = qt.QPushButton(parent=panel)
        button.setText("Gravity")
        button.clicked.connect(self.generateGravityField)
        button.setCheckable(True)
        layout.addWidget(button)
        self.__kind.addButton(button)

        button = qt.QPushButton(parent=panel)
        button.setText("Magnetic")
        button.clicked.connect(self.generateMagneticField)
        button.setCheckable(True)
        layout.addWidget(button)
        self.__kind.addButton(button)

        button = qt.QPushButton(parent=panel)
        button.setText("Spiral")
        button.clicked.connect(self.generateSpiral)
        button.setCheckable(True)
        layout.addWidget(button)
        self.__kind.addButton(button)

        button = qt.QPushButton(parent=panel)
        button.setText("Rings")
        button.clicked.connect(self.generateRings)
        button.setCheckable(True)
        layout.addWidget(button)
        self.__kind.addButton(button)

        button = qt.QPushButton(parent=panel)
        button.setText("Gradient")
        button.clicked.connect(self.generateGradient)
        button.setCheckable(True)
        layout.addWidget(button)
        self.__kind.addButton(button)

        layout.addSpacing(5)

        button = qt.QPushButton(parent=panel)
        button.setText("Generate")
        button.clicked.connect(self.generate)
        layout.addWidget(button)
        self.__kind.addButton(button)

        layout.addSpacing(5)

        label = qt.QLabel(parent=panel)
        label.setText("Custom level:")
        self.__value = qt.QSlider(panel)
        self.__value.sliderMoved.connect(self.updateValue)
        self.__value.valueChanged.connect(self.updateValue)
        layout.addWidget(label)
        layout.addWidget(self.__value)

        return panel

    def __cleanCustomValue(self):
        for name in self.__customPolygons:
            self.__plot.removeCurve(name)
        self.__customPolygons = []
        dummy = numpy.array([[[0, 0, 0, 0]]])
        item = self.__plot.getImage(legend="iso-pixels")
        item.setData([[[0, 0, 0, 0]]])

    def clean(self):
        self.__cleanCustomValue()
        for name in self.__polygons:
            self.__plot.removeCurve(name)
        self.__polygons = []
        self.__image = None
        self.__mask = None
        item = self.__plot.getImage(legend="image")
        item.setData([[0]])
        item = self.__plot.getImage(legend="mask")
        item.setData([[[0, 0, 0, 0]]])

    def updateValue(self, value):
        if self.__algo is None:
            return

        self.__cleanCustomValue()

        # iso pixels
        iso_pixels = self.__algo.find_pixels(value)
        if len(iso_pixels) != 0:
            mask = numpy.zeros(self.__image.shape, dtype=numpy.int8)
            indexes = iso_pixels[:, 0] * self.__image.shape[1] + iso_pixels[:, 1]
            mask = mask.ravel()
            mask[indexes] = 1
            mask.shape = self.__image.shape
            mask = createRgbaMaskImage(mask, color=numpy.array([255, 0, 0, 128]))
            item = self.__plot.getImage(legend="iso-pixels")
            item.setData(mask)

        # iso contours
        polygons = self.__algo.find_contours(value)
        for ipolygon, polygon in enumerate(polygons):
            if len(polygon) == 0:
                continue
            isClosed = numpy.allclose(polygon[0], polygon[-1])
            x = polygon[:, 1] + 0.5
            y = polygon[:, 0] + 0.5
            legend = "custom-polygon-%d" % ipolygon
            self.__customPolygons.append(legend)
            self.__plot.addCurve(x=x, y=y, linestyle="--", color="red", linewidth=2.0, legend=legend, resetzoom=False)

    def setData(self, image, mask=None, value=0.0):
        self.clean()
        if MarchingSquareSciKitImage is not None:
            self.__algo = MarchingSquareSciKitImage(image, mask)
        self.__image = image
        self.__mask = mask

        # image
        item = self.__plot.getImage(legend="image")
        item.setData(image)
        item.setColormap(self.__colormap)

        # mask
        if mask is not None:
            mask = createRgbaMaskImage(mask, color=numpy.array([255, 0, 255, 128]))
            item = self.__plot.getImage(legend="mask")
            item.setData(mask)

        self.__plot.resetZoom()

    def __drawRings(self, values, lineStyleCallback=None):
        if self.__algo is None:
            return

        # iso contours
        values = range(-800, 5000, 200)

        # iso contours
        ipolygon = 0
        for ivalue, value in enumerate(values):
            polygons = self.__algo.find_contours(value)
            for polygon in polygons:
                if len(polygon) == 0:
                    continue
                isClosed = numpy.allclose(polygon[0], polygon[-1])
                x = polygon[:, 1] + 0.5
                y = polygon[:, 0] + 0.5
                legend = "polygon-%d" % ipolygon
                if lineStyleCallback is not None:
                    extraStyle = lineStyleCallback(value, ivalue, ipolygon)
                else:
                    extraStyle = {"linestyle": "-", "linewidth": 1.0, "color": "black"}
                self.__polygons.append(legend)
                self.__plot.addCurve(x=x, y=y, legend=legend, resetzoom=False, **extraStyle)
                ipolygon += 1

    def __defineDefaultValues(self, value=None):
        # Do not use min and max to avoid to create iso contours on small
        # and many artefacts
        if value is None:
            value = self.__image.mean()
        div = 12
        delta = (self.__image.max() - self.__image.min()) / div
        self.__value.setValue(value)
        self.__value.setRange(self.__image.min() + delta,
                              self.__image.min() + delta * (div - 1))
        self.updateValue(value)

    def generate(self):
        self.__kind.checkedButton().click()

    def generateSpiral(self):
        shape = 512
        nb_spiral = numpy.random.randint(1, 8)
        freq = numpy.random.randint(2, 50)
        image = create_spiral(shape, nb_spiral, freq)
        image *= 1000.0
        self.__colormap = Colormap("viridis")
        self.setData(image=image, mask=None)
        self.__defineDefaultValues()

    def generateIsland(self):
        shape = (512, 512)
        image = create_island(shape, summit=4808.72, under_water=1500)
        self.__colormap = Colormap("terrain")
        self.setData(image=image, mask=None)

        values = range(-800, 5000, 200)

        def styleCallback(value, ivalue, ipolygon):
            if value == 0:
                style = {"linestyle": "-", "linewidth": 1.0, "color": "black"}
            elif value % 1000 == 0:
                style = {"linestyle": "--", "linewidth": 0.5, "color": "black"}
            else:
                style = {"linestyle": "--", "linewidth": 0.1, "color": "black"}
            return style

        self.__drawRings(values, styleCallback)

        self.__value.setValue(0)
        self.__value.setRange(0, 5000)
        self.updateValue(0)

    def generateMagneticField(self):
        shape = 512
        x1 = numpy.random.random() * 2 - 1
        y1 = numpy.random.random() * 2 - 1
        x2 = numpy.random.random() * 2 - 1
        y2 = numpy.random.random() * 2 - 1
        image = create_magnetic_field(shape, x1, y1, x2, y2)
        self.__colormap = Colormap("coolwarm")
        self.setData(image=image, mask=None)

        maximum = abs(image.max())
        m = abs(image.min())
        if m > maximum:
            maximum = m
        maximum = int(maximum)
        values = range(-maximum, maximum, maximum // 20)

        def styleCallback(value, ivalue, ipolygon):
            if (ivalue % 2) == 0:
                style = {"linestyle": "-", "linewidth": 0.5, "color": "black"}
            else:
                style = {"linestyle": "-", "linewidth": 0.5, "color": "white"}
            return style

        self.__drawRings(values, styleCallback)
        self.__defineDefaultValues(value=0)

    def generateGravityField(self):
        shape = 512
        nb = numpy.random.randint(2, 10)
        objects = []
        for _ in range(nb):
            x = numpy.random.random() * 2 - 1
            y = numpy.random.random() * 2 - 1
            m = numpy.random.random() * 10 + 1.0
            objects.append((x, y, m))
        image = create_gravity_field(shape, objects)
        self.__colormap = Colormap("inferno")
        self.setData(image=image, mask=None)

        delta = int((image.max() - image.min()) // 30)
        values = range(int(image.min()), int(image.max()), delta)

        def styleCallback(value, ivalue, ipolygon):
            return {"linestyle": "-", "linewidth": 0.1, "color": "white"}

        self.__drawRings(values, styleCallback)
        self.__defineDefaultValues()

    def generateRings(self):
        shape = 512
        dx = numpy.random.random() * 2 - 1
        dy = numpy.random.random() * 2 - 1
        freq = numpy.random.randint(1, 100) / 100.0
        sx = numpy.random.randint(10, 5000) / 10.0
        sy = numpy.random.randint(10, 5000) / 10.0
        image = create_rings(shape, dx=dx, dy=dy, freq=freq, sx=sx, sy=sy)
        image *= 1000.0
        self.__colormap = Colormap("viridis")
        self.setData(image=image, mask=None)
        self.__defineDefaultValues()

    def generateGradient(self):
        shape = 512
        dx = numpy.random.random() * 2 - 1
        dy = numpy.random.random() * 2 - 1
        sx = numpy.random.randint(10, 5000) / 10.0
        sy = numpy.random.randint(10, 5000) / 10.0
        image = create_gradient(shape, dx=dx, dy=dy, sx=sx, sy=sy)
        image *= 1000.0
        self.__colormap = Colormap("viridis")
        self.setData(image=image, mask=None)
        self.__defineDefaultValues()


def main():
    app = qt.QApplication([])
    sys.excepthook = qt.exceptionHandler
    window = FindContours()
    window.generateIsland()
    window.show()
    result = app.exec_()
    # remove ending warnings relative to QTimer
    app.deleteLater()
    return result


if __name__ == "__main__":
    result = main()
    sys.exit(result)
