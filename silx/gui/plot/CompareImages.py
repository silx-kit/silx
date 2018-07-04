# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""A widget dedicated to compare 2 images.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "04/07/2018"


from silx.gui import qt
from silx.gui import plot
import logging
import numpy
from silx.gui.colors import Colormap


_logger = logging.getLogger(__name__)


class CompareImages(qt.QMainWindow):

    def __init__(self):
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Plot with synchronized axes")
        widget = qt.QWidget(self)
        self.setCentralWidget(widget)

        layout = qt.QVBoxLayout()
        widget.setLayout(layout)

        backend = "matplotlib"

        self.__data1 = None
        self.__data2 = None
        self.__previousSeparatorPosition = None

        self.__plot2d = plot.Plot2D(parent=widget, backend=backend)
        self.__plot2d.setKeepDataAspectRatio(True)
        # self.__plot2d.setInteractiveMode('pan')
        self.__plot2d.sigPlotSignal.connect(self.__plotSlot)

        layout.addWidget(self.__plot2d)

        self.__plot2d.addXMarker(
            0,
            legend='separator',
            text='separator',
            draggable=True,
            color='blue',
            constraint=self.__separatorConstraint)
        self.__separator = self.__plot2d._getMarker('separator')

        self.__toolBar = self._createToolBar()
        layout.addWidget(self.__toolBar)

    def _createToolBar(self):
        toolbar = qt.QToolBar(self)

        icon = qt.QIcon("compare-ab-vline.svg")
        action = qt.QAction(icon, "Vertical compare mode", self)
        toolbar.addAction(action)

        icon = qt.QIcon("compare-b-align.svg")
        action = qt.QAction(icon, "Auto-alignment of the second image", self)
        action.setCheckable(True)
        action.triggered.connect(self.__invalidateData)
        toolbar.addAction(action)
        self.__autoAlignAction = action

        icon = qt.QIcon("compare-keypoints.svg")
        action = qt.QAction(icon, "Display/hide alignment keypoints", self)
        action.setCheckable(True)
        action.setChecked(True)
        action.triggered.connect(self.__invalidateScatter)
        toolbar.addAction(action)
        self.__displayKeypoints = action

        return toolbar

    def __plotSlot(self, event):
        """Handle events from the plot"""
        if event['event'] in ('markerMoving', 'markerMoved'):
            if event['label'] == 'separator':
                value = int(float(str(event['xdata'])))
                if self.__previousSeparatorPosition != value:
                    self.__separatorMoved(value)
                    self.__previousSeparatorPosition = value

    def __separatorConstraint(self, x, y):
        if self.__data1 is None:
            return 0, 0
        x = int(x)
        if x < 0:
            x = 0
        elif x > self.__data1.shape[1]:
            x = self.__data1.shape[1]
        return x, y

    def __separatorMoved(self, pos):
        if self.__data1 is None:
            return

        if pos <= 0:
            pos = 0
        elif pos >= self.__data1.shape[1]:
            pos = self.__data1.shape[1]
        data1 = self.__data1[:, 0:pos]
        data2 = self.__data2[:, pos:]
        self.__image1.setData(data1, copy=False)
        self.__image2.setData(data2, copy=False)
        self.__image2.setOrigin((pos, 0))

    def setData(self, image1, image2):
        self.__raw1 = image1
        self.__raw2 = image2
        self.__invalidateData()
        self.__plot2d.resetZoom()

    def __invalidateScatter(self):
        if self.__displayKeypoints.isChecked():
            data = self.__matching_keypoints
        else:
            data = [], [], []
        self.__plot2d.addScatter(x=data[0],
                                 y=data[1],
                                 z=1,
                                 value=data[2],
                                 legend="keypoints",
                                 colormap=Colormap("spring"))

    def __invalidateData(self):
        raw1, raw2 = self.__raw1, self.__raw2
        if not self.__autoAlignAction.isChecked():
            data1, data2 = self.normalizeImageShape(raw1, raw2, mode="transparent_margin")
            self.__matching_keypoints = [0.0], [0.0], [1.0]
        else:
            # TODO: sift implementation do not support RGBA images
            data1, data2 = self.normalizeImageShape(raw1, raw2, mode="margin")
            self.__matching_keypoints = [0.0], [0.0], [1.0]
            try:
                data1, data2 = self.__createSiftData(data1, data2)
                if data2 is None:
                    raise ValueError("Unexpected None value")
            except Exception as e:
                # TODO: Display it on the GUI
                print(e)
                self.__autoAlignAction.setChecked(False)
                self.__invalidateData()
                return

        self.__data1, self.__data2 = data1, data2
        self.__plot2d.addImage(data1, z=0, legend="image1", resetzoom=False)
        self.__plot2d.addImage(data2, z=0, legend="image2", resetzoom=False)
        self.__image1 = self.__plot2d.getImage("image1")
        self.__image2 = self.__plot2d.getImage("image2")
        self.__invalidateScatter()

        # Set the separator into the middle
        if self.__previousSeparatorPosition is None:
            value = self.__data1.shape[1] // 2
            self.__separator.setPosition(value, 0)
        else:
            value = self.__previousSeparatorPosition
        self.__separatorMoved(value)
        self.__previousSeparatorPosition = value

        # Avoid to change the colormap range when the separator is moving
        # TODO: The colormap histogram will still be wrong
        mode1 = self.__getImageMode(data1)
        mode2 = self.__getImageMode(data2)
        if mode1 == "intensity" and mode1 == mode2:
            vmin = min(self.__data1.min(), self.__data2.min())
            vmax = max(self.__data1.max(), self.__data2.max())
            colormap = Colormap(vmin=vmin, vmax=vmax)
            self.__image1.setColormap(colormap)
            self.__image2.setColormap(colormap)

    def __getImageMode(self, image):
        if len(image.shape) == 2:
            return "intensity"
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                return "rgb"
            elif image.shape[2] == 4:
                return "rgba"
        raise TypeError("'image' argument is not an image.")

    def __createMarginImage(self, image, size, transparent=False):
        """Returns a new image with margin to respect the requested size.
        """
        mode = self.__getImageMode(image)
        if mode == "intensity":
            data = numpy.zeros(size, dtype=image.dtype)
            data[0:image.shape[0], 0:image.shape[1]] = image
            # TODO: It is maybe possible to put NaN on the margin
        else:
            if transparent:
                data = numpy.zeros((size[0], size[1], 4), dtype=numpy.uint8)
            else:
                data = numpy.zeros((size[0], size[1], 3), dtype=numpy.uint8)
            depth = min(data.shape[2], image.shape[2])
            data[0:image.shape[0], 0:image.shape[1], 0:depth] = image[:, :, 0:depth]
            if transparent:
                data[0:image.shape[0], 0:image.shape[1], 3] = 255
        return data

    def normalizeImageShape(self, image, image2, mode="crop"):
        """
        Returns 2 images with the same shape.
        """
        if image.shape == image2.shape:
            return image, image2
        if mode == "crop":
            yy = min(image.shape[0], image2.shape[0])
            xx = min(image.shape[1], image2.shape[1])
            return image[0:yy, 0:xx], image2[0:yy, 0:xx]
        elif mode == "margin":
            yy = max(image.shape[0], image2.shape[0])
            xx = max(image.shape[1], image2.shape[1])
            size = yy, xx
            image = self.__createMarginImage(image, size)
            image2 = self.__createMarginImage(image2, size)
            return image, image2
        elif mode == "transparent_margin":
            yy = max(image.shape[0], image2.shape[0])
            xx = max(image.shape[1], image2.shape[1])
            size = yy, xx
            image = self.__createMarginImage(image, size, transparent=True)
            image2 = self.__createMarginImage(image2, size, transparent=True)
            return image, image2

    def __createSiftData(self, image, second_image):
        devicetype = "GPU"

        # Compute base image
        from silx.image import sift
        sift_ocl = sift.SiftPlan(template=image, devicetype=devicetype)
        keypoints = sift_ocl(image)

        # Check image compatibility
        second_keypoints = sift_ocl(second_image)
        mp = sift.MatchPlan()
        match = mp(keypoints, second_keypoints)
        print("Number of Keypoints within image 1: %i" % keypoints.size)
        print("                    within image 2: %i" % second_keypoints.size)

        matching_keypoints = match.shape[0]
        self.__matching_keypoints = (match[:].x[:, 0],
                                     match[:].y[:, 0],
                                     match[:].scale[:, 0])
        print("Matching keypoints: %i" % matching_keypoints)
        if matching_keypoints == 0:
            return image, second_image

        # TODO: Problem here is we have to compute 2 time sift
        # The first time to extract matching keypoints, second time
        # to extract the aligned image.

        # Normalize the second image
        sa = sift.LinearAlign(image, devicetype=devicetype)
        data1 = image
        # TODO: Create a sift issue: if data1 is RGB and data2 intensity
        # it returns None, while extracting manually keypoints (above) works
        data2 = sa.align(second_image)
        return data1, data2
