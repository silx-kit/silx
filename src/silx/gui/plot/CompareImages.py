# /*##########################################################################
#
# Copyright (c) 2018-2021 European Synchrotron Radiation Facility
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
__date__ = "23/07/2018"


import logging
import numpy
import math

import silx.image.bilinear
from silx.gui import qt
from silx.gui import plot
from silx.gui.colors import Colormap
from silx.gui.plot import tools
from silx.utils.deprecation import deprecated_warning
from silx.utils.weakref import WeakMethodProxy
from silx.gui.plot.items import Scatter
from silx.math.colormap import normalize

from .tools.compare.core import sift
from .tools.compare.core import VisualizationMode
from .tools.compare.core import AlignmentMode
from .tools.compare.core import AffineTransformation
from .tools.compare.toolbar import CompareImagesToolBar
from .tools.compare.statusbar import CompareImagesStatusBar
from .tools.compare.core import _CompareImageItem


_logger = logging.getLogger(__name__)


class CompareImages(qt.QMainWindow):
    """Widget providing tools to compare 2 images.

    .. image:: img/CompareImages.png

    :param Union[qt.QWidget,None] parent: Parent of this widget.
    :param backend: The backend to use, in:
                    'matplotlib' (default), 'mpl', 'opengl', 'gl', 'none'
                    or a :class:`BackendBase.BackendBase` class
    :type backend: str or :class:`BackendBase.BackendBase`
    """

    VisualizationMode = VisualizationMode
    """Available visualization modes"""

    AlignmentMode = AlignmentMode
    """Available alignment modes"""

    sigConfigurationChanged = qt.Signal()
    """Emitted when the configuration of the widget (visualization mode,
    alignment mode...) have changed."""

    def __init__(self, parent=None, backend=None):
        qt.QMainWindow.__init__(self, parent)
        self._resetZoomActive = True
        self._colormap = Colormap()
        """Colormap shared by all modes, except the compose images (rgb image)"""
        self._colormapKeyPoints = Colormap("spring")
        """Colormap used for sift keypoints"""

        self._colormap.sigChanged.connect(self.__colormapChanged)

        if parent is None:
            self.setWindowTitle("Compare images")
        else:
            self.setWindowFlags(qt.Qt.Widget)

        self.__transformation = None
        self.__item = _CompareImageItem()
        self.__item.setName("_virtual")
        self.__item.setColormap(self._colormap)

        self.__raw1 = None
        self.__raw2 = None
        self.__data1 = None
        self.__data2 = None
        self.__previousSeparatorPosition = None

        self.__plot = plot.PlotWidget(parent=self, backend=backend)
        self.__plot.setDefaultColormap(self._colormap)
        self.__plot.getXAxis().setLabel("Columns")
        self.__plot.getYAxis().setLabel("Rows")
        if silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION == "downward":
            self.__plot.getYAxis().setInverted(True)
        self.__plot.addItem(self.__item)
        self.__plot.setActiveImage(self.__item)

        self.__plot.setKeepDataAspectRatio(True)
        self.__plot.sigPlotSignal.connect(self.__plotSlot)
        self.__plot.setAxesDisplayed(False)

        self.__scatter = Scatter()
        self.__scatter.setZValue(1)
        self.__scatter.setColormap(self._colormapKeyPoints)
        self.__plot.addItem(self.__scatter)

        self.setCentralWidget(self.__plot)

        legend = VisualizationMode.VERTICAL_LINE.name
        self.__plot.addXMarker(
            0,
            legend=legend,
            text="",
            draggable=True,
            color="blue",
            constraint=WeakMethodProxy(self.__separatorConstraint),
        )
        self.__vline = self.__plot._getMarker(legend)

        legend = VisualizationMode.HORIZONTAL_LINE.name
        self.__plot.addYMarker(
            0,
            legend=legend,
            text="",
            draggable=True,
            color="blue",
            constraint=WeakMethodProxy(self.__separatorConstraint),
        )
        self.__hline = self.__plot._getMarker(legend)

        # default values
        self.__visualizationMode = ""
        self.__alignmentMode = ""
        self.__keypointsVisible = True

        self.setAlignmentMode(AlignmentMode.ORIGIN)
        self.setVisualizationMode(VisualizationMode.VERTICAL_LINE)
        self.setKeypointsVisible(False)

        # Toolbars

        self._createToolBars(self.__plot)
        if self._interactiveModeToolBar is not None:
            self.addToolBar(self._interactiveModeToolBar)
        if self._imageToolBar is not None:
            self.addToolBar(self._imageToolBar)
        if self._compareToolBar is not None:
            self.addToolBar(self._compareToolBar)

        # Statusbar

        self._createStatusBar(self.__plot)
        if self._statusBar is not None:
            self.setStatusBar(self._statusBar)

    def __getSealedColormap(self):
        vrange = self._colormap.getColormapRange(
            self.__item.getColormappedData(copy=False)
        )
        sealed = self._colormap.copy()
        sealed.setVRange(*vrange)
        return sealed

    def __colormapChanged(self):
        sealed = self.__getSealedColormap()
        if self.__image1 is not None:
            if self.__getImageMode(self.__image1.getData(copy=False)) == "intensity":
                self.__image1.setColormap(sealed)
        if self.__image2 is not None:
            if self.__getImageMode(self.__image2.getData(copy=False)) == "intensity":
                self.__image2.setColormap(sealed)

        if "COMPOSITE" in self.__visualizationMode.name:
            self.__updateData()

    def _createStatusBar(self, plot):
        self._statusBar = CompareImagesStatusBar(self)
        self._statusBar.setCompareWidget(self)

    def _createToolBars(self, plot):
        """Create tool bars displayed by the widget"""
        toolBar = tools.InteractiveModeToolBar(parent=self, plot=plot)
        self._interactiveModeToolBar = toolBar
        toolBar = tools.ImageToolBar(parent=self, plot=plot)
        self._imageToolBar = toolBar
        toolBar = CompareImagesToolBar(self)
        toolBar.setCompareWidget(self)
        self._compareToolBar = toolBar

    def _getVirtualPlotItem(self):
        return self.__item

    def getPlot(self):
        """Returns the plot which is used to display the images.

        :rtype: silx.gui.plot.PlotWidget
        """
        return self.__plot

    def getColormap(self):
        """

        :return: colormap used for compare image
        :rtype: silx.gui.colors.Colormap
        """
        return self._colormap

    def getRawPixelData(self, x, y):
        """Return the raw pixel of each image data from axes positions.

        If the coordinate is outside of the image it returns None element in
        the tuple.

        The pixel is reach from the raw data image without filter or
        transformation. But the coordinate x and y are in the reference of the
        current displayed mode.

        :param float x: X-coordinate of the pixel in the current displayed plot
        :param float y: Y-coordinate of the pixel in the current displayed plot
        :return: A tuple of for each images containing pixel information. It
            could be a scalar value or an array in case of RGB/RGBA informations.
            It also could be a string containing information is some cases.
        :rtype: Tuple(Union[int,float,numpy.ndarray,str],Union[int,float,numpy.ndarray,str])
        """
        alignmentMode = self.__alignmentMode
        raw1, raw2 = self.__raw1, self.__raw2

        if raw1 is None or raw2 is None:
            x1 = x
            y1 = y
            x2 = x
            y2 = y
        elif alignmentMode == AlignmentMode.ORIGIN:
            x1 = x
            y1 = y
            x2 = x
            y2 = y
        elif alignmentMode == AlignmentMode.CENTER:
            yy = max(raw1.shape[0], raw2.shape[0])
            xx = max(raw1.shape[1], raw2.shape[1])
            x1 = x - (xx - raw1.shape[1]) * 0.5
            x2 = x - (xx - raw2.shape[1]) * 0.5
            y1 = y - (yy - raw1.shape[0]) * 0.5
            y2 = y - (yy - raw2.shape[0]) * 0.5
        elif alignmentMode == AlignmentMode.STRETCH:
            x1 = x
            y1 = y
            x2 = x * raw2.shape[1] / raw1.shape[1]
            y2 = x * raw2.shape[1] / raw1.shape[1]
        elif alignmentMode == AlignmentMode.AUTO:
            x1 = x
            y1 = y
            # Not implemented
            x2 = -1
            y2 = -1
        else:
            assert False

        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        if raw1 is None:
            data1 = "No image A"
        elif y1 < 0 or y1 >= raw1.shape[0] or x1 < 0 or x1 >= raw1.shape[1]:
            data1 = ""
        else:
            data1 = raw1[y1, x1]

        if raw2 is None:
            data2 = "No image B"
        elif alignmentMode == AlignmentMode.AUTO:
            data2 = "Not implemented with sift"
        elif y2 < 0 or y2 >= raw2.shape[0] or x2 < 0 or x2 >= raw2.shape[1]:
            data2 = None
        else:
            data2 = raw2[y2, x2]

        return data1, data2

    def setVisualizationMode(self, mode):
        """Set the visualization mode.

        :param str mode: New visualization to display the image comparison
        """
        if self.__visualizationMode == mode:
            return
        self.__visualizationMode = mode
        self.__item.setVizualisationMode(mode)
        self.__vline.setVisible(mode == VisualizationMode.VERTICAL_LINE)
        self.__hline.setVisible(mode == VisualizationMode.HORIZONTAL_LINE)
        self.__updateData()
        self.sigConfigurationChanged.emit()

    def centerLines(self):
        """Center the line used to compare the 2 images."""
        if self.__image1 is None:
            return
        data_range = self.__plot.getDataRange()

        if data_range[0] is not None:
            cx = (data_range[0][0] + data_range[0][1]) * 0.5
        else:
            cx = 0
        if data_range[1] is not None:
            cy = (data_range[1][0] + data_range[1][1]) * 0.5
        else:
            cy = 0
        self.__vline.setPosition(cx, cy)
        self.__hline.setPosition(cx, cy)
        self.__updateSeparators()

    def getVisualizationMode(self):
        """Returns the current interaction mode."""
        return self.__visualizationMode

    def setAlignmentMode(self, mode):
        """Set the alignment mode.

        :param str mode: New alignement to apply to images
        """
        if self.__alignmentMode == mode:
            return
        self.__alignmentMode = mode
        self.__updateData()
        self.sigConfigurationChanged.emit()

    def getAlignmentMode(self):
        """Returns the current selected alignemnt mode."""
        return self.__alignmentMode

    def getKeypointsVisible(self):
        """Returns true if the keypoints are displayed"""
        return self.__keypointsVisible

    def setKeypointsVisible(self, isVisible):
        """Set keypoints visibility.

        :param bool isVisible: If True, keypoints are displayed (if some)
        """
        if self.__keypointsVisible == isVisible:
            return
        self.__keypointsVisible = isVisible
        self.__updateKeyPoints()
        self.sigConfigurationChanged.emit()

    def __setDefaultAlignmentMode(self):
        """Reset the alignemnt mode to the default value"""
        self.setAlignmentMode(AlignmentMode.ORIGIN)

    def __plotSlot(self, event):
        """Handle events from the plot"""
        if event["event"] in ("markerMoving", "markerMoved"):
            mode = self.getVisualizationMode()
            legend = mode.name
            if event["label"] == legend:
                if mode == VisualizationMode.VERTICAL_LINE:
                    value = int(float(str(event["xdata"])))
                elif mode == VisualizationMode.HORIZONTAL_LINE:
                    value = int(float(str(event["ydata"])))
                else:
                    assert False
                if self.__previousSeparatorPosition != value:
                    self.__separatorMoved(value)
                    self.__previousSeparatorPosition = value

    def __separatorConstraint(self, x, y):
        """Manage contains on the separators to clamp them inside the images."""
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

    def __updateSeparators(self):
        """Redraw images according to the current state of the separators."""
        mode = self.getVisualizationMode()
        if mode == VisualizationMode.VERTICAL_LINE:
            pos = self.__vline.getXPosition()
            self.__separatorMoved(pos)
            self.__previousSeparatorPosition = pos
        elif mode == VisualizationMode.HORIZONTAL_LINE:
            pos = self.__hline.getYPosition()
            self.__separatorMoved(pos)
            self.__previousSeparatorPosition = pos
        else:
            self.__image1.setOrigin((0, 0))
            if self.__image2 is not None:
                self.__image2.setOrigin((0, 0))

    def __separatorMoved(self, pos):
        """Called when vertical or horizontal separators have moved.

        Update the displayed images.
        """
        if self.__data1 is None:
            return

        mode = self.getVisualizationMode()
        if mode == VisualizationMode.VERTICAL_LINE:
            pos = int(pos)
            if pos <= 0:
                pos = 0
            elif pos >= self.__data1.shape[1]:
                pos = self.__data1.shape[1]
            data1 = self.__data1[:, 0:pos]
            data2 = self.__data2[:, pos:]
            self.__image1.setData(data1, copy=False)
            if self.__image2 is not None:
                self.__image2.setData(data2, copy=False)
                self.__image2.setOrigin((pos, 0))
        elif mode == VisualizationMode.HORIZONTAL_LINE:
            pos = int(pos)
            if pos <= 0:
                pos = 0
            elif pos >= self.__data1.shape[0]:
                pos = self.__data1.shape[0]
            data1 = self.__data1[0:pos, :]
            data2 = self.__data2[pos:, :]
            self.__image1.setData(data1, copy=False)
            if self.__image2 is not None:
                self.__image2.setData(data2, copy=False)
                self.__image2.setOrigin((0, pos))
        else:
            assert False

    def clear(self):
        self.setData(None, None)

    def setData(self, image1, image2, updateColormap="deprecated"):
        """Set images to compare.

        Images can contains floating-point or integer values, or RGB and RGBA
        values, but should have comparable intensities.

        RGB and RGBA images are provided as an array as `[width,height,channels]`
        of unsigned integer 8-bits or floating-points between 0.0 to 1.0.

        :param numpy.ndarray image1: The first image
        :param numpy.ndarray image2: The second image
        """
        if updateColormap != "deprecated":
            deprecated_warning(
                "Argument", "setData's updateColormap argument", since_version="2.0.0"
            )

        self.__raw1 = image1
        self.__raw2 = image2
        self.__updateData()
        if self.isAutoResetZoom():
            self.__plot.resetZoom()

    def setImage1(self, image1, updateColormap="deprecated"):
        """Set image1 to be compared.

        Images can contains floating-point or integer values, or RGB and RGBA
        values, but should have comparable intensities.

        RGB and RGBA images are provided as an array as `[width,height,channels]`
        of unsigned integer 8-bits or floating-points between 0.0 to 1.0.

        :param numpy.ndarray image1: The first image
        """
        if updateColormap != "deprecated":
            deprecated_warning(
                "Argument", "setImage1's updateColormap argument", since_version="2.0.0"
            )

        self.__raw1 = image1
        self.__updateData()
        if self.isAutoResetZoom():
            self.__plot.resetZoom()

    def setImage2(self, image2, updateColormap="deprecated"):
        """Set image2 to be compared.

        Images can contains floating-point or integer values, or RGB and RGBA
        values, but should have comparable intensities.

        RGB and RGBA images are provided as an array as `[width,height,channels]`
        of unsigned integer 8-bits or floating-points between 0.0 to 1.0.

        :param numpy.ndarray image2: The second image
        """
        if updateColormap != "deprecated":
            deprecated_warning(
                "Argument", "setImage2's updateColormap argument", since_version="2.0.0"
            )

        self.__raw2 = image2
        self.__updateData()
        if self.isAutoResetZoom():
            self.__plot.resetZoom()

    def __updateKeyPoints(self):
        """Update the displayed keypoints using cached keypoints."""
        if self.__keypointsVisible and self.__matching_keypoints:
            data = self.__matching_keypoints
        else:
            data = [], [], []
        self.__scatter.setData(x=data[0], y=data[1], value=data[2])

    def __updateData(self):
        """Compute aligned image when the alignment mode changes.

        This function cache input images which are used when
        vertical/horizontal separators moves.
        """
        raw1, raw2 = self.__raw1, self.__raw2

        alignmentMode = self.getAlignmentMode()
        self.__transformation = None

        if raw1 is None or raw2 is None:
            # No need to realign the 2 images
            # But create a dummy image when there is None for simplification
            if raw1 is None:
                data1 = numpy.empty((0, 0))
            else:
                data1 = raw1
            if raw2 is None:
                data2 = numpy.empty((0, 0))
            else:
                data2 = raw2
            self.__matching_keypoints = None
        else:
            if alignmentMode == AlignmentMode.ORIGIN:
                yy = max(raw1.shape[0], raw2.shape[0])
                xx = max(raw1.shape[1], raw2.shape[1])
                size = yy, xx
                data1 = self.__createMarginImage(raw1, size, transparent=True)
                data2 = self.__createMarginImage(raw2, size, transparent=True)
                self.__matching_keypoints = [0.0], [0.0], [1.0]
            elif alignmentMode == AlignmentMode.CENTER:
                yy = max(raw1.shape[0], raw2.shape[0])
                xx = max(raw1.shape[1], raw2.shape[1])
                size = yy, xx
                data1 = self.__createMarginImage(
                    raw1, size, transparent=True, center=True
                )
                data2 = self.__createMarginImage(
                    raw2, size, transparent=True, center=True
                )
                self.__matching_keypoints = (
                    [data1.shape[1] // 2],
                    [data1.shape[0] // 2],
                    [1.0],
                )
            elif alignmentMode == AlignmentMode.STRETCH:
                data1 = raw1
                data2 = self.__rescaleImage(raw2, data1.shape)
                self.__matching_keypoints = (
                    [0, data1.shape[1], data1.shape[1], 0],
                    [0, 0, data1.shape[0], data1.shape[0]],
                    [1.0, 1.0, 1.0, 1.0],
                )
            elif alignmentMode == AlignmentMode.AUTO:
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
                assert False

        self.__item.setImageData1(data1)
        self.__item.setImageData2(data2)

        mode = self.getVisualizationMode()
        if mode == VisualizationMode.COMPOSITE_RED_BLUE_GRAY_NEG:
            data1 = self.__composeRgbImage(data1, data2, mode)
            data2 = None
        elif mode == VisualizationMode.COMPOSITE_RED_BLUE_GRAY:
            data1 = self.__composeRgbImage(data1, data2, mode)
            data2 = None
        elif mode == VisualizationMode.COMPOSITE_A_MINUS_B:
            data1 = self.__composeAMinusBImage(data1, data2)
            data2 = None
        elif mode == VisualizationMode.ONLY_A:
            data2 = None
        elif mode == VisualizationMode.ONLY_B:
            data1 = numpy.empty((0, 0))

        self.__data1, self.__data2 = data1, data2

        colormap = self.__getSealedColormap()
        mode1 = self.__getImageMode(self.__data1)
        if mode1 == "intensity":
            colormap1 = colormap
        else:
            colormap1 = None
        self.__plot.addImage(
            data1, z=0, legend="image1", resetzoom=False, colormap=colormap1
        )
        self.__image1 = self.__plot.getImage("image1")

        if data2 is not None:
            mode2 = self.__getImageMode(data2)
            if mode2 == "intensity":
                colormap2 = colormap
            else:
                colormap2 = None
            self.__plot.addImage(
                data2, z=0, legend="image2", resetzoom=False, colormap=colormap2
            )
            self.__image2 = self.__plot.getImage("image2")
            self.__image2.setVisible(True)
        else:
            if self.__image2 is not None:
                self.__image2.setVisible(False)
            self.__image2 = None
            self.__data2 = numpy.empty((0, 0))
        self.__updateKeyPoints()

        # Set the separator into the middle
        if self.__previousSeparatorPosition is None:
            value = self.__data1.shape[1] // 2
            self.__vline.setPosition(value, 0)
            value = self.__data1.shape[0] // 2
            self.__hline.setPosition(0, value)
        self.__updateSeparators()

    def __getImageMode(self, image):
        """Returns a value identifying the way the image is stored in the
        array.

        :param numpy.ndarray image: Image to check
        :rtype: str
        """
        if len(image.shape) == 2:
            return "intensity"
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                return "rgb"
            elif image.shape[2] == 4:
                return "rgba"
        raise TypeError("'image' argument is not an image.")

    def __rescaleImage(self, image, shape):
        """Rescale an image to the requested shape.

        :rtype: numpy.ndarray
        """
        mode = self.__getImageMode(image)
        if mode == "intensity":
            data = self.__rescaleArray(image, shape)
        elif mode == "rgb":
            data = numpy.empty((shape[0], shape[1], 3), dtype=image.dtype)
            for c in range(3):
                data[:, :, c] = self.__rescaleArray(image[:, :, c], shape)
        elif mode == "rgba":
            data = numpy.empty((shape[0], shape[1], 4), dtype=image.dtype)
            for c in range(4):
                data[:, :, c] = self.__rescaleArray(image[:, :, c], shape)
        return data

    def __composeRgbImage(self, data1, data2, mode):
        """Returns an RBG image containing composition of data1 and data2 in 2
        different channels

        A data image of a size of 0 is considered as missing. This does not
        interrupt the processing.

        :param numpy.ndarray data1: First image
        :param numpy.ndarray data1: Second image
        :param VisualizationMode mode: Composition mode.
        :rtype: numpy.ndarray
        """
        if data1.size != 0 and data2.size != 0:
            assert data1.shape[0:2] == data2.shape[0:2]

        sealed = self.__getSealedColormap()
        vmin, vmax = sealed.getVRange()

        if data1.size == 0:
            intensity1 = numpy.zeros(data2.shape[0:2])
        else:
            mode1 = self.__getImageMode(data1)
            if mode1 in ["rgb", "rgba"]:
                intensity1 = self.__luminosityImage(data1)
            else:
                intensity1 = data1

        if data2.size == 0:
            intensity2 = numpy.zeros(data1.shape[0:2])
        else:
            mode2 = self.__getImageMode(data2)
            if mode2 in ["rgb", "rgba"]:
                intensity2 = self.__luminosityImage(data2)
            else:
                intensity2 = data2

        shape = intensity1.shape
        result = numpy.empty((shape[0], shape[1], 3), dtype=numpy.uint8)
        a, _, _ = normalize(
            intensity1,
            norm=sealed.getNormalization(),
            autoscale=sealed.getAutoscaleMode(),
            vmin=sealed.getVMin(),
            vmax=sealed.getVMax(),
            gamma=sealed.getGammaNormalizationParameter(),
        )
        b, _, _ = normalize(
            intensity2,
            norm=sealed.getNormalization(),
            autoscale=sealed.getAutoscaleMode(),
            vmin=sealed.getVMin(),
            vmax=sealed.getVMax(),
            gamma=sealed.getGammaNormalizationParameter(),
        )
        if mode == VisualizationMode.COMPOSITE_RED_BLUE_GRAY:
            result[:, :, 0] = a
            result[:, :, 1] = a // 2 + b // 2
            result[:, :, 2] = b
        elif mode == VisualizationMode.COMPOSITE_RED_BLUE_GRAY_NEG:
            result[:, :, 0] = 255 - b
            result[:, :, 1] = 255 - (a // 2 + b // 2)
            result[:, :, 2] = 255 - a
        return result

    def __composeAMinusBImage(self, data1, data2):
        """Returns an intensity image containing the composition of `A-B`.

        A data image of a size of 0 is considered as missing. This does not
        interrupt the processing.

        :param numpy.ndarray data1: First image
        :param numpy.ndarray data1: Second image
        :rtype: numpy.ndarray
        """
        if data1.size != 0 and data2.size != 0:
            assert data1.shape[0:2] == data2.shape[0:2]

        data1 = self.__asIntensityImage(data1)
        data2 = self.__asIntensityImage(data2)
        if data1.size == 0:
            result = data2
        elif data2.size == 0:
            result = data1
        else:
            result = data1.astype(numpy.float32) - data2.astype(numpy.float32)
        return result

    def __asIntensityImage(self, image: numpy.ndarray):
        """Returns an intensity image.

        If the image use a single channel, it will be returned as it is.

        If the image is an RBG(A) image, the luminosity (0..1) is extracted and
        returned. The alpha channel is ignored.

        :rtype: numpy.ndarray
        """
        mode = self.__getImageMode(image)
        if mode in ["rgb", "rgba"]:
            return self.__luminosityImage(image)
        return image

    def __luminosityImage(self, image: numpy.ndarray):
        """Returns the luminosity channel from an RBG(A) image.

        The alpha channel is ignored.

        :rtype: numpy.ndarray
        """
        mode = self.__getImageMode(image)
        assert mode in ["rgb", "rgba"]
        is_uint8 = image.dtype.type == numpy.uint8
        # luminosity
        image = 0.21 * image[..., 0] + 0.72 * image[..., 1] + 0.07 * image[..., 2]
        if is_uint8:
            image = image / 255.0
        return image

    def __rescaleArray(self, image, shape):
        """Rescale a 2D array to the requested shape.

        :rtype: numpy.ndarray
        """
        y, x = numpy.ogrid[: shape[0], : shape[1]]
        y, x = y * 1.0 * (image.shape[0] - 1) / (shape[0] - 1), x * 1.0 * (
            image.shape[1] - 1
        ) / (shape[1] - 1)
        b = silx.image.bilinear.BilinearImage(image)
        # TODO: could be optimized using strides
        x2d = numpy.zeros_like(y) + x
        y2d = numpy.zeros_like(x) + y
        result = b.map_coordinates((y2d, x2d))
        return result

    def __createMarginImage(self, image, size, transparent=False, center=False):
        """Returns a new image with margin to respect the requested size.

        :rtype: numpy.ndarray
        """
        assert image.shape[0] <= size[0]
        assert image.shape[1] <= size[1]
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
            data[pos0 : pos0 + image.shape[0], pos1 : pos1 + image.shape[1]] = image
            # TODO: It is maybe possible to put NaN on the margin
        else:
            if transparent:
                data = numpy.zeros((size[0], size[1], 4), dtype=numpy.uint8)
            else:
                data = numpy.zeros((size[0], size[1], 3), dtype=numpy.uint8)
            depth = min(data.shape[2], image.shape[2])
            data[
                pos0 : pos0 + image.shape[0], pos1 : pos1 + image.shape[1], 0:depth
            ] = image[:, :, 0:depth]
            if transparent and depth == 3:
                data[
                    pos0 : pos0 + image.shape[0], pos1 : pos1 + image.shape[1], 3
                ] = 255
        return data

    def __toAffineTransformation(self, sift_result):
        """Returns an affine transformation from the sift result.

        :param dict sift_result: Result of sift when using `all_result=True`
        :rtype: AffineTransformation
        """
        offset = sift_result["offset"]
        matrix = sift_result["matrix"]

        tx = offset[0]
        ty = offset[1]
        a = matrix[0, 0]
        b = matrix[0, 1]
        c = matrix[1, 0]
        d = matrix[1, 1]
        rot = math.atan2(-b, a)
        sx = (-1.0 if a < 0 else 1.0) * math.sqrt(a**2 + b**2)
        sy = (-1.0 if d < 0 else 1.0) * math.sqrt(c**2 + d**2)
        return AffineTransformation(tx, ty, sx, sy, rot)

    def getTransformation(self):
        """Returns the affine transformation applied to the second image to align
        it to the first image.

        This result is only valid for sift alignment.

        :rtype: Union[None,AffineTransformation]
        """
        return self.__transformation

    def __createSiftData(self, image, second_image):
        """Generate key points and aligned images from 2 images.

        If no keypoints matches, unaligned data are anyway returns.

        :rtype: Tuple(numpy.ndarray,numpy.ndarray)
        """
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

        self.__matching_keypoints = (
            match[:].x[:, 0],
            match[:].y[:, 0],
            match[:].scale[:, 0],
        )
        matching_keypoints = match.shape[0]
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
        result = sa.align(second_image, return_all=True)
        data2 = result["result"]
        self.__transformation = self.__toAffineTransformation(result)
        return data1, data2

    def resetZoom(self, dataMargins=None):
        """Reset the plot limits to the bounds of the data and redraw the plot."""
        self.__plot.resetZoom(dataMargins)

    def setAutoResetZoom(self, activate=True):
        """

        :param bool activate: True if we want to activate the automatic
                              plot reset zoom when setting images.
        """
        self._resetZoomActive = activate

    def isAutoResetZoom(self):
        """

        :return: True if the automatic call to resetzoom is activated
        :rtype: bool
        """
        return self._resetZoomActive
