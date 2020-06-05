#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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

.. note:: This module has an optional dependency with sci-kit image library.
    You might need to install it if you don't already have it.
"""

import logging
import sys
import numpy
import time

logging.basicConfig()
_logger = logging.getLogger("find_contours")

from silx.gui import qt
import silx.gui.plot
from silx.gui.colors import Colormap
import silx.image.bilinear


try:
    import skimage
except ImportError:
    _logger.debug("Error while importing skimage", exc_info=True)
    skimage = None

if skimage is not None:
    try:
        from silx.image.marchingsquares._skimage import MarchingSquaresSciKitImage
    except ImportError:
        _logger.debug("Error while importing MarchingSquaresSciKitImage", exc_info=True)
        MarchingSquaresSciKitImage = None
else:
    MarchingSquaresSciKitImage = None


def rescale_image(image, shape):
    y, x = numpy.ogrid[:shape[0], :shape[1]]
    y, x = y * 1.0 * (image.shape[0] - 1) / (shape[0] - 1), x * 1.0 * (image.shape[1] - 1) / (shape[1] - 1)
    b = silx.image.bilinear.BilinearImage(image)
    # TODO: could be optimized using strides
    x2d = numpy.zeros_like(y) + x
    y2d = numpy.zeros_like(x) + y
    result = b.map_coordinates((y2d, x2d))
    return result


def create_spiral(size, nb=1, freq=100):
    half = size // 2
    y, x = numpy.ogrid[-half:half, -half:half]
    coef = 1.0 / half
    y, x = y * coef, x * coef + 0.0001
    distance = numpy.sqrt(x * x + y * y)
    angle = numpy.arctan(y / x)
    data = numpy.sin(angle * nb * 2 + distance * freq * half / 100, dtype=numpy.float32)
    return data


def create_magnetic_field(size, x1=0.0, y1=0.0, x2=0.0, y2=0.0):
    half = size // 2
    yy, xx = numpy.ogrid[-half:half, -half:half]
    coef = 1.0 / half
    yy1, xx1 = (yy + half * y1) * coef, (xx + half * x1) * coef
    distance1 = numpy.sqrt(xx1 * xx1 + yy1 * yy1)
    yy2, xx2 = (yy + half * y2) * coef, (xx + half * x2) * coef
    distance2 = numpy.sqrt(xx2 * xx2 + yy2 * yy2)
    return (numpy.arctan2(distance1, distance2) - numpy.pi * 0.25) * 1000


def create_gravity_field(size, objects):
    half = size // 2
    yy, xx = numpy.ogrid[-half:half, -half:half]
    coef = 1.0 / half

    def distance(x, y):
        yy1, xx1 = (yy + half * y) * coef, (xx + half * x) * coef
        return numpy.sqrt(xx1 ** 2 + yy1 ** 2)
    result = numpy.zeros((size, size), dtype=numpy.float32)
    for x, y, m in objects:
        result += m / distance(x, y)
    return numpy.log(result) * 1000


def create_gradient(size, dx=0, dy=0, sx=1.0, sy=1.0):
    half = size // 2
    yy, xx = numpy.ogrid[-half:half, -half:half]
    coef = 1.0 / half
    yy, xx = (yy - (dy * half)) * coef, (xx - (dx * half)) * coef + 0.0001
    distance = numpy.sqrt(xx * xx * sx + yy * yy * sy)
    return distance


def create_composite_gradient(size, dx=0, dy=0, sx=1.0, sy=1.0):
    hole = (size - 4) // 4
    gap = 10
    base = create_gradient(size + hole + gap * 4, dx, dy, sx, sy)
    result = numpy.zeros((size, size))
    width = (size - 2) // 2
    half_hole = hole // 2

    def copy_module(x1, y1, x2, y2, width, height):
        result[y1:y1 + height, x1:x1 + width] = base[y2:y2 + height, x2:x2 + width]

    y1 = 0
    y2 = 0
    copy_module(0, y1, half_hole, y2, width, hole)
    copy_module(width + 1, y1, half_hole + width, y2, width, hole)

    y1 += hole + 1
    y2 += hole + gap
    copy_module(0, y1, 0, y2, width, hole)
    copy_module(width + 1, y1, width + hole, y2, width, hole)

    y1 += hole + 1
    y2 += hole + gap
    copy_module(0, y1, half_hole, y2, width, hole)
    copy_module(width + 1, y1, half_hole + width, y2, width, hole)

    y1 += hole + 1
    y2 += hole + gap
    copy_module(0, y1, half_hole, y2, width, hole)
    copy_module(width + 1, y1, half_hole + width, y2, width, hole)

    return result


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
            d = numpy.random.randint(w, size=(t, t)).astype(dtype=numpy.uint8)
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
        self.__plot.addImage(dummy, legend="iso-pixels", z=0, replace=False)

        self.__algo = None
        self.__polygons = []
        self.__customPolygons = []
        self.__image = None
        self.__mask = None
        self.__customValue = None

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

        group = qt.QGroupBox(self)
        group.setTitle("Image")
        layout.addWidget(group)
        groupLayout = qt.QVBoxLayout(group)

        button = qt.QRadioButton(parent=panel)
        button.setText("Island")
        button.clicked.connect(self.generateIsland)
        button.setCheckable(True)
        button.setChecked(True)
        groupLayout.addWidget(button)
        self.__kind.addButton(button)

        button = qt.QRadioButton(parent=panel)
        button.setText("Gravity")
        button.clicked.connect(self.generateGravityField)
        button.setCheckable(True)
        groupLayout.addWidget(button)
        self.__kind.addButton(button)

        button = qt.QRadioButton(parent=panel)
        button.setText("Magnetic")
        button.clicked.connect(self.generateMagneticField)
        button.setCheckable(True)
        groupLayout.addWidget(button)
        self.__kind.addButton(button)

        button = qt.QRadioButton(parent=panel)
        button.setText("Spiral")
        button.clicked.connect(self.generateSpiral)
        button.setCheckable(True)
        groupLayout.addWidget(button)
        self.__kind.addButton(button)

        button = qt.QRadioButton(parent=panel)
        button.setText("Gradient")
        button.clicked.connect(self.generateGradient)
        button.setCheckable(True)
        groupLayout.addWidget(button)
        self.__kind.addButton(button)

        button = qt.QRadioButton(parent=panel)
        button.setText("Composite gradient")
        button.clicked.connect(self.generateCompositeGradient)
        button.setCheckable(True)
        groupLayout.addWidget(button)
        self.__kind.addButton(button)

        button = qt.QPushButton(parent=panel)
        button.setText("Generate a new image")
        button.clicked.connect(self.generate)
        groupLayout.addWidget(button)

        # Contours

        group = qt.QGroupBox(self)
        group.setTitle("Contours")
        layout.addWidget(group)
        groupLayout = qt.QVBoxLayout(group)

        button = qt.QCheckBox(parent=panel)
        button.setText("Use the plot's mask")
        button.setCheckable(True)
        button.setChecked(True)
        button.clicked.connect(self.updateContours)
        groupLayout.addWidget(button)
        self.__useMaskButton = button

        button = qt.QPushButton(parent=panel)
        button.setText("Update contours")
        button.clicked.connect(self.updateContours)
        groupLayout.addWidget(button)

        # Implementations

        group = qt.QGroupBox(self)
        group.setTitle("Implementation")
        layout.addWidget(group)
        groupLayout = qt.QVBoxLayout(group)

        self.__impl = qt.QButtonGroup(self)
        self.__impl.setExclusive(True)

        button = qt.QRadioButton(parent=panel)
        button.setText("silx")
        button.clicked.connect(self.updateContours)
        button.setCheckable(True)
        button.setChecked(True)
        groupLayout.addWidget(button)
        self.__implMerge = button
        self.__impl.addButton(button)

        button = qt.QRadioButton(parent=panel)
        button.setText("silx with cache")
        button.clicked.connect(self.updateContours)
        button.setCheckable(True)
        groupLayout.addWidget(button)
        self.__implMergeCache = button
        self.__impl.addButton(button)

        button = qt.QRadioButton(parent=panel)
        button.setText("skimage")
        button.clicked.connect(self.updateContours)
        button.setCheckable(True)
        groupLayout.addWidget(button)
        self.__implSkimage = button
        self.__impl.addButton(button)
        if MarchingSquaresSciKitImage is None:
            button.setEnabled(False)
            button.setToolTip("skimage is not installed or not compatible")

        # Processing

        group = qt.QGroupBox(self)
        group.setTitle("Processing")
        layout.addWidget(group)
        group.setLayout(self.__createInfoLayout(group))

        # Processing

        group = qt.QGroupBox(self)
        group.setTitle("Custom level")
        layout.addWidget(group)
        groupLayout = qt.QVBoxLayout(group)

        label = qt.QLabel(parent=panel)
        self.__value = qt.QSlider(panel)
        self.__value.setOrientation(qt.Qt.Horizontal)
        self.__value.sliderMoved.connect(self.__updateCustomContours)
        self.__value.valueChanged.connect(self.__updateCustomContours)
        groupLayout.addWidget(self.__value)

        return panel

    def __createInfoLayout(self, parent):
        layout = qt.QGridLayout()

        header = qt.QLabel(parent=parent)
        header.setText("Time: ")
        label = qt.QLabel(parent=parent)
        label.setText("")
        layout.addWidget(header, 0, 0)
        layout.addWidget(label, 0, 1)
        self.__timeLabel = label

        header = qt.QLabel(parent=parent)
        header.setText("Nb polygons: ")
        label = qt.QLabel(parent=parent)
        label.setText("")
        layout.addWidget(header, 2, 0)
        layout.addWidget(label, 2, 1)
        self.__polygonsLabel = label

        header = qt.QLabel(parent=parent)
        header.setText("Nb points: ")
        label = qt.QLabel(parent=parent)
        label.setText("")
        layout.addWidget(header, 1, 0)
        layout.addWidget(label, 1, 1)
        self.__pointsLabel = label

        return layout

    def __cleanCustomContour(self):
        for name in self.__customPolygons:
            self.__plot.removeCurve(name)
        self.__customPolygons = []
        dummy = numpy.array([[[0, 0, 0, 0]]])
        item = self.__plot.getImage(legend="iso-pixels")
        item.setData([[[0, 0, 0, 0]]])

    def __cleanPolygons(self):
        for name in self.__polygons:
            self.__plot.removeCurve(name)

    def clean(self):
        self.__cleanCustomContour()
        self.__cleanPolygons()
        self.__polygons = []
        self.__image = None
        self.__mask = None

    def updateContours(self):
        self.__redrawContours()
        self.updateCustomContours()

    def __updateCustomContours(self, value):
        self.__customValue = value
        self.updateCustomContours()

    def updateCustomContours(self):
        if self.__algo is None:
            return
        value = self.__customValue
        self.__cleanCustomContour()
        if value is None:
            return

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

    def __updateAlgo(self, image, mask=None):
        if mask is None:
            if self.__useMaskButton.isChecked():
                mask = self.__plot.getMaskToolsDockWidget().getSelectionMask()

        self.__image = image
        self.__mask = mask

        implButton = self.__impl.checkedButton()
        if implButton == self.__implMerge:
            from silx.image.marchingsquares import MarchingSquaresMergeImpl
            self.__algo = MarchingSquaresMergeImpl(self.__image, self.__mask)
        elif implButton == self.__implMergeCache:
            from silx.image.marchingsquares import MarchingSquaresMergeImpl
            self.__algo = MarchingSquaresMergeImpl(self.__image, self.__mask, use_minmax_cache=True)
        elif implButton == self.__implSkimage and MarchingSquaresSciKitImage is not None:
            self.__algo = MarchingSquaresSciKitImage(self.__image, self.__mask)
        else:
            _logger.error("No algorithm available")
            self.__algo = None

    def setData(self, image, mask=None, value=0.0):
        self.clean()

        self.__updateAlgo(image, mask=None)

        # image
        item = self.__plot.getImage(legend="image")
        item.setData(image)
        item.setColormap(self.__colormap)

        self.__plot.resetZoom()

    def __redrawContours(self):
        self.__updateAlgo(self.__image)
        if self.__algo is None:
            return
        self.__cleanPolygons()
        self.__drawContours(self.__values, self.__lineStyleCallback)

    def __drawContours(self, values, lineStyleCallback=None):
        if self.__algo is None:
            return

        self.__values = values
        self.__lineStyleCallback = lineStyleCallback
        if self.__values is None:
            return

        nbTime = 0
        nbPolygons = 0
        nbPoints = 0

        # iso contours
        ipolygon = 0
        for ivalue, value in enumerate(values):
            startTime = time.time()
            polygons = self.__algo.find_contours(value)
            nbTime += (time.time() - startTime)
            nbPolygons += len(polygons)
            for polygon in polygons:
                if len(polygon) == 0:
                    continue
                nbPoints += len(polygon)
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

        self.__timeLabel.setText("%0.3fs" % nbTime)
        self.__polygonsLabel.setText("%d" % nbPolygons)
        self.__pointsLabel.setText("%d" % nbPoints)

    def __defineDefaultValues(self, value=None):
        # Do not use min and max to avoid to create iso contours on small
        # and many artefacts
        if value is None:
            value = self.__image.mean()
        self.__customValue = value
        div = 12
        delta = (self.__image.max() - self.__image.min()) / div
        self.__value.setValue(int(numpy.round(value)))
        minv = self.__image.min() + delta
        maxv = self.__image.min() + delta * (div - 1)
        self.__value.setRange(int(numpy.floor(minv)), int(numpy.ceil(maxv)))
        self.updateCustomContours()

    def generate(self):
        self.__kind.checkedButton().click()

    def generateSpiral(self):
        shape = 512
        nb_spiral = numpy.random.randint(1, 8)
        freq = numpy.random.randint(2, 50)
        image = create_spiral(shape, nb_spiral, freq)
        image *= 1000.0
        self.__colormap = Colormap("cool")
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

        self.__drawContours(values, styleCallback)

        self.__value.setValue(0)
        self.__value.setRange(0, 5000)
        self.__updateCustomContours(0)

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

        self.__drawContours(values, styleCallback)
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

        delta = (image.max() - image.min()) / 30.0
        values = numpy.arange(image.min(), image.max(), delta)

        def styleCallback(value, ivalue, ipolygon):
            return {"linestyle": "-", "linewidth": 0.1, "color": "white"}

        self.__drawContours(values, styleCallback)
        self.__defineDefaultValues()

    def generateGradient(self):
        shape = 512
        dx = numpy.random.random() * 2 - 1
        dy = numpy.random.random() * 2 - 1
        sx = numpy.random.randint(10, 5000) / 10.0
        sy = numpy.random.randint(10, 5000) / 10.0
        image = create_gradient(shape, dx=dx, dy=dy, sx=sx, sy=sy)
        image *= 1000.0

        def styleCallback(value, ivalue, ipolygon):
            colors = ["#9400D3", "#4B0082", "#0000FF", "#00FF00", "#FFFF00", "#FF7F00", "#FF0000"]
            color = colors[ivalue % len(colors)]
            style = {"linestyle": "-", "linewidth": 2.0, "color": color}
            return style
        delta = (image.max() - image.min()) / 9.0
        values = numpy.arange(image.min(), image.max(), delta)
        values = values[1:8]

        self.__colormap = Colormap("Greys")
        self.setData(image=image, mask=None)
        self.__drawContours(values, styleCallback)
        self.__defineDefaultValues()

    def generateCompositeGradient(self):
        shape = 512
        hole = 1 / 4.0
        dx = numpy.random.random() * hole - hole / 2.0
        dy = numpy.random.random() * hole - hole * 2
        sx = numpy.random.random() * 10.0 + 1
        sy = numpy.random.random() * 10.0 + 1
        image = create_composite_gradient(shape, dx, dy, sx, sy)
        image *= 1000.0

        def styleCallback(value, ivalue, ipolygon):
            colors = ["#9400D3", "#4B0082", "#0000FF", "#00FF00", "#FFFF00", "#FF7F00", "#FF0000"]
            color = colors[ivalue % len(colors)]
            style = {"linestyle": "-", "linewidth": 2.0, "color": color}
            return style
        delta = (image.max() - image.min()) / 9.0
        values = numpy.arange(image.min(), image.max(), delta)
        values = values[1:8]

        self.__colormap = Colormap("Greys")
        self.setData(image=image, mask=None)
        self.__drawContours(values, styleCallback)
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
