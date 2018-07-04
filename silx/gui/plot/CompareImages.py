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


import logging
import numpy

import silx.image.bilinear
from silx.gui import qt
from silx.gui import plot
from silx.gui import icons
from silx.gui.colors import Colormap

_logger = logging.getLogger(__name__)

try:
    from silx.image import sift
except ImportError as e:
    _logger.warning("Error while importing sift: %s", str(e))
    _logger.debug("Backtrace", exc_info=True)
    sift = None


class CompareImages(qt.QMainWindow):

    def __init__(self):
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Plot with synchronized axes")
        widget = qt.QWidget(self)
        self.setCentralWidget(widget)

        layout = qt.QVBoxLayout()
        widget.setLayout(layout)

        backend = "matplotlib"
        # backend = "opengl"

        self.__raw1 = None
        self.__raw2 = None
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
            legend='vline',
            text='separator',
            draggable=True,
            color='blue',
            constraint=self.__separatorConstraint)
        self.__vline = self.__plot2d._getMarker('vline')

        self.__plot2d.addYMarker(
            0,
            legend='hline',
            text='separator',
            draggable=True,
            color='blue',
            constraint=self.__separatorConstraint)
        self.__hline = self.__plot2d._getMarker('hline')

        self.__toolBar = self._createToolBar()
        layout.addWidget(self.__toolBar)

        # default values
        self.__vlineModeAction.trigger()
        self.__originAlignAction.trigger()
        self.__displayKeypoints.setChecked(True)
        self.__previousSeparatorPosition = None

    def getPlot(self):
        """Returns the plot which is used to display the images.

        :rtype: silx.gui.plot.Plot2D
        """
        return self.__plot2d

    def _createToolBar(self):
        toolbar = qt.QToolBar(self)

        self.__interactionGroup = qt.QActionGroup(self)
        self.__interactionGroup.setExclusive(True)
        self.__interactionGroup.triggered.connect(self.__interactionChanged)

        icon = icons.getQIcon("compare-mode-vline")
        action = qt.QAction(icon, "Vertical compare mode", self)
        action.setCheckable(True)
        toolbar.addAction(action)
        self.__vlineModeAction = action
        self.__interactionGroup.addAction(action)

        icon = icons.getQIcon("compare-mode-hline")
        action = qt.QAction(icon, "Horizontal compare mode", self)
        action.setCheckable(True)
        toolbar.addAction(action)
        self.__hlineModeAction = action
        self.__interactionGroup.addAction(action)

        icon = icons.getQIcon("compare-mode-channel")
        action = qt.QAction(icon, "Blue/red compare mode", self)
        action.setCheckable(True)
        toolbar.addAction(action)
        self.__channelModeAction = action
        self.__interactionGroup.addAction(action)

        toolbar.addSeparator()

        menu = qt.QMenu(self)
        self.__alignmentAction = qt.QAction(self)
        self.__alignmentAction.setMenu(menu)
        toolbar.addAction(self.__alignmentAction)
        self.__alignmentGroup = qt.QActionGroup(self)
        self.__alignmentGroup.setExclusive(True)
        self.__alignmentGroup.triggered.connect(self.__alignmentChanged)

        icon = icons.getQIcon("compare-align-origin")
        action = qt.QAction(icon, "Align images on there upper-left pixel", self)
        action.setCheckable(True)
        self.__originAlignAction = action
        menu.addAction(action)
        self.__alignmentGroup.addAction(action)

        icon = icons.getQIcon("compare-align-center")
        action = qt.QAction(icon, "Center images", self)
        action.setCheckable(True)
        self.__centerAlignAction = action
        menu.addAction(action)
        self.__alignmentGroup.addAction(action)

        icon = icons.getQIcon("compare-align-stretch")
        action = qt.QAction(icon, "Stretch the second image on the first one", self)
        action.setCheckable(True)
        self.__stretchAlignAction = action
        menu.addAction(action)
        self.__alignmentGroup.addAction(action)

        icon = icons.getQIcon("compare-align-auto")
        action = qt.QAction(icon, "Auto-alignment of the second image", self)
        action.setCheckable(True)
        self.__autoAlignAction = action
        menu.addAction(action)
        if sift is None:
            action.setEnabled(False)
            action.setToolTip("Sift module is not available")
        self.__alignmentGroup.addAction(action)

        icon = icons.getQIcon("compare-keypoints")
        action = qt.QAction(icon, "Display/hide alignment keypoints", self)
        action.setCheckable(True)
        action.triggered.connect(self.__invalidateScatter)
        toolbar.addAction(action)
        self.__displayKeypoints = action

        return toolbar

    def __interactionChanged(self, selectedAction):
        mode = self.__getInteractionMode()
        self.__vline.setVisible(mode == "vline")
        self.__hline.setVisible(mode == "hline")
        self.__invalidateData()

    def __getInteractionMode(self):
        if self.__vlineModeAction.isChecked():
            return "vline"
        elif self.__hlineModeAction.isChecked():
            return "hline"
        elif self.__channelModeAction.isChecked():
            return "channel"
        else:
            raise ValueError("Unknown interaction mode")

    def __alignmentChanged(self, selectedAction):
        self.__alignmentAction.setText(selectedAction.text())
        self.__alignmentAction.setIcon(selectedAction.icon())
        self.__alignmentAction.setToolTip(selectedAction.toolTip())
        self.__invalidateData()

    def __getAlignmentMode(self):
        action = self.__alignmentGroup.checkedAction()
        if action is self.__originAlignAction:
            return "origin"
        if action is self.__centerAlignAction:
            return "center"
        if action is self.__stretchAlignAction:
            return "stretch"
        if action is self.__autoAlignAction:
            return "auto"
        raise ValueError("Unknown alignment mode")

    def __setDefaultAlignmentMode(self):
        self.__originAlignAction.trigger()

    def __plotSlot(self, event):
        """Handle events from the plot"""
        if event['event'] in ('markerMoving', 'markerMoved'):
            mode = self.__getInteractionMode()
            if event['label'] == mode:
                if mode == "vline":
                    value = int(float(str(event['xdata'])))
                elif mode == "hline":
                    value = int(float(str(event['ydata'])))
                else:
                    assert(False)
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
        y = int(y)
        if y < 0:
            y = 0
        elif y > self.__data1.shape[0]:
            y = self.__data1.shape[0]
        return x, y

    def __invalidateSeparator(self):
        mode = self.__getInteractionMode()
        if mode == "vline":
            pos = self.__vline.getXPosition()
        elif mode == "hline":
            pos = self.__hline.getYPosition()
        elif mode == "channel":
            return
        else:
            assert(False)
        self.__separatorMoved(pos)
        self.__previousSeparatorPosition = pos

    def __separatorMoved(self, pos):
        if self.__data1 is None:
            return

        mode = self.__getInteractionMode()
        if mode == "vline":
            pos = int(pos)
            if pos <= 0:
                pos = 0
            elif pos >= self.__data1.shape[1]:
                pos = self.__data1.shape[1]
            data1 = self.__data1[:, 0:pos]
            data2 = self.__data2[:, pos:]
            self.__image1.setData(data1, copy=False)
            self.__image2.setData(data2, copy=False)
            self.__image2.setOrigin((pos, 0))
        elif mode == "hline":
            pos = int(pos)
            if pos <= 0:
                pos = 0
            elif pos >= self.__data1.shape[0]:
                pos = self.__data1.shape[0]
            data1 = self.__data1[0:pos, :]
            data2 = self.__data2[pos:, :]
            self.__image1.setData(data1, copy=False)
            self.__image2.setData(data2, copy=False)
            self.__image2.setOrigin((0, pos))
        else:
            assert(False)

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
        if raw1 is None or raw2 is None:
            return

        alignmentMode = self.__getAlignmentMode()

        if alignmentMode == "origin":
            yy = max(raw1.shape[0], raw2.shape[0])
            xx = max(raw1.shape[1], raw2.shape[1])
            size = yy, xx
            data1 = self.__createMarginImage(raw1, size, transparent=True)
            data2 = self.__createMarginImage(raw2, size, transparent=True)
            self.__matching_keypoints = [0.0], [0.0], [1.0]
        elif alignmentMode == "center":
            yy = max(raw1.shape[0], raw2.shape[0])
            xx = max(raw1.shape[1], raw2.shape[1])
            size = yy, xx
            data1 = self.__createMarginImage(raw1, size, transparent=True, center=True)
            data2 = self.__createMarginImage(raw2, size, transparent=True, center=True)
            self.__matching_keypoints = ([data1.shape[1] // 2],
                                         [data1.shape[0] // 2],
                                         [1.0])
        elif alignmentMode == "stretch":
            data1 = raw1
            data2 = self.__rescaleImage(raw2, data1.shape)
            self.__matching_keypoints = ([0, data1.shape[1], data1.shape[1], 0],
                                         [0, 0, data1.shape[0], data1.shape[0]],
                                         [1.0, 1.0, 1.0, 1.0])
        elif alignmentMode == "auto":
            # TODO: sift implementation do not support RGBA images
            yy = max(raw1.shape[0], raw2.shape[0])
            xx = max(raw1.shape[1], raw2.shape[1])
            size = yy, xx
            data1 = self.__createMarginImage(raw1, size)
            data2 = self.__createMarginImage(raw2, size)
            self.__matching_keypoints = [0.0], [0.0], [1.0]
            try:
                data1, data2 = self.__createSiftData(data1, data2)
                if data2 is None:
                    raise ValueError("Unexpected None value")
            except Exception as e:
                # TODO: Display it on the GUI
                _logger.error(e)
                self.__setDefaultAlignmentMode()
                return
        else:
            assert(False)

        mode = self.__getInteractionMode()
        if mode == "channel":
            intensity1 = self.__intensityImage(data1)
            intensity2 = self.__intensityImage(data2)
            shape = data1.shape
            data1 = numpy.empty((shape[0], shape[1], 3), dtype=numpy.uint8)
            data1[:, :, 0] = intensity2 * 255
            data1[:, :, 1] = 0
            data1[:, :, 2] = intensity1 * 255
            data2 = numpy.empty((0, 0))

        self.__data1, self.__data2 = data1, data2
        self.__plot2d.addImage(data1, z=0, legend="image1", resetzoom=False)
        self.__plot2d.addImage(data2, z=0, legend="image2", resetzoom=False)
        self.__image1 = self.__plot2d.getImage("image1")
        self.__image2 = self.__plot2d.getImage("image2")
        self.__invalidateScatter()

        # Set the separator into the middle
        if self.__previousSeparatorPosition is None:
            value = self.__data1.shape[1] // 2
            self.__vline.setPosition(value, 0)
            value = self.__data1.shape[0] // 2
            self.__hline.setPosition(0, value)
        self.__invalidateSeparator()

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

    def __rescaleImage(self, image, shape):
        mode = self.__getImageMode(image)
        if mode == "intensity":
            data = self.__rescaleChannel(image, shape)
        elif mode == "rgb":
            data = numpy.empty((shape[0], shape[1], 3), dtype=image.dtype)
            for c in range(3):
                data[:, :, c] = self.__rescaleChannel(image[:, :, c], shape)
        elif mode == "rgba":
            data = numpy.empty((shape[0], shape[1], 4), dtype=image.dtype)
            for c in range(4):
                data[:, :, c] = self.__rescaleChannel(image[:, :, c], shape)
        return data

    def __intensityImage(self, image):
        mode = self.__getImageMode(image)
        if mode == "intensity":
            return image
        elif mode in ["rgb", "rgba"]:
            is_uint8 = image.dtype.type == numpy.uint8
            # luminosity
            image = 0.21 * image[..., 0] + 0.72 * image[..., 1] + 0.07 * image[..., 2]
            if is_uint8:
                image = image / 256.0
            return image
        return image

    def __rescaleChannel(self, image, shape):
        y, x = numpy.ogrid[:shape[0], :shape[1]]
        y, x = y * 1.0 * (image.shape[0] - 1) / (shape[0] - 1), x * 1.0 * (image.shape[1] - 1) / (shape[1] - 1)
        b = silx.image.bilinear.BilinearImage(image)
        # TODO: could be optimized using strides
        x2d = numpy.zeros_like(y) + x
        y2d = numpy.zeros_like(x) + y
        result = b.map_coordinates((y2d, x2d))
        return result

    def __createMarginImage(self, image, size, transparent=False, center=False):
        """Returns a new image with margin to respect the requested size.
        """
        assert(image.shape[0] <= size[0])
        assert(image.shape[1] <= size[1])
        if image.shape == size:
            return image
        mode = self.__getImageMode(image)

        if center:
            pos0 = size[0] // 2 - image.shape[0] // 2
            pos1 = size[1] // 2 - image.shape[1] // 2
        else:
            pos0, pos1 = 0, 0

        if mode == "intensity":
            data = numpy.zeros(size, dtype=image.dtype)
            data[pos0:pos0 + image.shape[0], pos1:pos1 + image.shape[1]] = image
            # TODO: It is maybe possible to put NaN on the margin
        else:
            if transparent:
                data = numpy.zeros((size[0], size[1], 4), dtype=numpy.uint8)
            else:
                data = numpy.zeros((size[0], size[1], 3), dtype=numpy.uint8)
            depth = min(data.shape[2], image.shape[2])
            data[pos0:pos0 + image.shape[0], pos1:pos1 + image.shape[1], 0:depth] = image[:, :, 0:depth]
            if transparent and depth == 3:
                data[pos0:pos0 + image.shape[0], pos1:pos1 + image.shape[1], 3] = 255
        return data

    def __createSiftData(self, image, second_image):
        devicetype = "GPU"

        # Compute base image
        sift_ocl = sift.SiftPlan(template=image, devicetype=devicetype)
        keypoints = sift_ocl(image)

        # Check image compatibility
        second_keypoints = sift_ocl(second_image)
        mp = sift.MatchPlan()
        match = mp(keypoints, second_keypoints)
        _logger.info("Number of Keypoints within image 1: %i" % keypoints.size)
        _logger.info("                    within image 2: %i" % second_keypoints.size)

        matching_keypoints = match.shape[0]
        self.__matching_keypoints = (match[:].x[:, 0],
                                     match[:].y[:, 0],
                                     match[:].scale[:, 0])
        _logger.info("Matching keypoints: %i" % matching_keypoints)
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
