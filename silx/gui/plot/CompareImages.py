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
__date__ = "06/07/2018"


import logging
import numpy
import weakref

import silx.image.bilinear
from silx.gui import qt
from silx.gui import plot
from silx.gui import icons
from silx.gui.colors import Colormap
from silx.gui.plot import tools

_logger = logging.getLogger(__name__)

try:
    from silx.image import sift
except ImportError as e:
    _logger.warning("Error while importing sift: %s", str(e))
    _logger.debug("Backtrace", exc_info=True)
    sift = None


class CompareImagesToolBar(qt.QToolBar):
    """ToolBar containing specific tools to custom the configuration of a
    :class:`CompareImages` widget

    Use :meth:`setCompareWidget` to connect this toolbar to a specific
    :class:`CompareImages` widget.

    :param Union[qt.QWidget,None] parent: Parent of this widget.
    """
    def __init__(self, parent=None):
        qt.QToolBar.__init__(self, parent)

        self.__compareWidget = None

        menu = qt.QMenu(self)
        self.__visualizationAction = qt.QAction(self)
        self.__visualizationAction.setMenu(menu)
        self.__visualizationAction.setCheckable(False)
        self.addAction(self.__visualizationAction)
        self.__visualizationGroup = qt.QActionGroup(self)
        self.__visualizationGroup.setExclusive(True)
        self.__visualizationGroup.triggered.connect(self.__visualizationModeChanged)

        icon = icons.getQIcon("compare-mode-a")
        action = qt.QAction(icon, "Display the first image only", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_A))
        menu.addAction(action)
        self.__aModeAction = action
        self.__visualizationGroup.addAction(action)

        icon = icons.getQIcon("compare-mode-b")
        action = qt.QAction(icon, "Display the second image only", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_B))
        menu.addAction(action)
        self.__bModeAction = action
        self.__visualizationGroup.addAction(action)

        icon = icons.getQIcon("compare-mode-vline")
        action = qt.QAction(icon, "Vertical compare mode", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_V))
        menu.addAction(action)
        self.__vlineModeAction = action
        self.__visualizationGroup.addAction(action)

        icon = icons.getQIcon("compare-mode-hline")
        action = qt.QAction(icon, "Horizontal compare mode", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_H))
        menu.addAction(action)
        self.__hlineModeAction = action
        self.__visualizationGroup.addAction(action)

        icon = icons.getQIcon("compare-mode-br-channel")
        action = qt.QAction(icon, "Blue/red compare mode (additive mode)", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_C))
        menu.addAction(action)
        self.__brChannelModeAction = action
        self.__visualizationGroup.addAction(action)

        icon = icons.getQIcon("compare-mode-yc-channel")
        action = qt.QAction(icon, "Yellow/cyan compare mode (substractive mode)", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_W))
        menu.addAction(action)
        self.__ycChannelModeAction = action
        self.__visualizationGroup.addAction(action)

        menu = qt.QMenu(self)
        self.__alignmentAction = qt.QAction(self)
        self.__alignmentAction.setMenu(menu)
        self.__alignmentAction.setIconVisibleInMenu(True)
        self.addAction(self.__alignmentAction)
        self.__alignmentGroup = qt.QActionGroup(self)
        self.__alignmentGroup.setExclusive(True)
        self.__alignmentGroup.triggered.connect(self.__alignmentModeChanged)

        icon = icons.getQIcon("compare-align-origin")
        action = qt.QAction(icon, "Align images on there upper-left pixel", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        self.__originAlignAction = action
        menu.addAction(action)
        self.__alignmentGroup.addAction(action)

        icon = icons.getQIcon("compare-align-center")
        action = qt.QAction(icon, "Center images", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        self.__centerAlignAction = action
        menu.addAction(action)
        self.__alignmentGroup.addAction(action)

        icon = icons.getQIcon("compare-align-stretch")
        action = qt.QAction(icon, "Stretch the second image on the first one", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        self.__stretchAlignAction = action
        menu.addAction(action)
        self.__alignmentGroup.addAction(action)

        icon = icons.getQIcon("compare-align-auto")
        action = qt.QAction(icon, "Auto-alignment of the second image", self)
        action.setIconVisibleInMenu(True)
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
        action.triggered.connect(self.__keypointVisibilityChanged)
        self.addAction(action)
        self.__displayKeypoints = action

    def setCompareWidget(self, widget):
        """
        Connect this tool bar to a specific :class:`CompareImages` widget.

        :param Union[None,CompareImages] widget: The widget to connect with.
        """
        compareWidget = self.getCompareWidget()
        if compareWidget is not None:
            compareWidget.sigConfigurationChanged.disconnect(self.__updateSelectedActions)
        compareWidget = widget
        if compareWidget is None:
            self.__compareWidget = None
        else:
            self.__compareWidget = weakref.ref(compareWidget)
        if compareWidget is not None:
            widget.sigConfigurationChanged.connect(self.__updateSelectedActions)
        self.__updateSelectedActions()

    def getCompareWidget(self):
        """Returns the connected widget.

        :rtype: CompareImages
        """
        if self.__compareWidget is None:
            return None
        else:
            return self.__compareWidget()

    def __updateSelectedActions(self):
        """
        Update the state of this tool bar according to the state of the
        connected :class:`CompareImages` widget.
        """
        widget = self.getCompareWidget()
        if widget is None:
            return

        mode = widget.getVisualizationMode()
        if mode == "a":
            action = self.__aModeAction
        elif mode == "b":
            action = self.__bModeAction
        elif mode == "vline":
            action = self.__vlineModeAction
        elif mode == "hline":
            action = self.__hlineModeAction
        elif mode == "brchannel":
            action = self.__brChannelModeAction
        elif mode == "ycchannel":
            action = self.__ycChannelModeAction
        else:
            action = None
        old = self.__visualizationGroup.blockSignals(True)
        if action is not None:
            # Check this action
            action.setChecked(True)
        else:
            action = self.__visualizationGroup.checkedAction()
            if action is not None:
                # Uncheck this action
                action.setChecked(False)
        self.__updateVisualizationMenu()
        self.__visualizationGroup.blockSignals(old)

        mode = widget.getAlignmentMode()
        if mode == "origin":
            action = self.__originAlignAction
        elif mode == "center":
            action = self.__centerAlignAction
        elif mode == "stretch":
            action = self.__stretchAlignAction
        elif mode == "auto":
            action = self.__autoAlignAction
        else:
            action = None
        old = self.__alignmentGroup.blockSignals(True)
        if action is not None:
            # Check this action
            action.setChecked(True)
        else:
            action = self.__alignmentGroup.checkedAction()
            if action is not None:
                # Uncheck this action
                action.setChecked(False)
        self.__updateAlignmentMenu()
        self.__alignmentGroup.blockSignals(old)

    def __visualizationModeChanged(self, selectedAction):
        """Called when user requesting changes of the visualization mode.
        """
        self.__updateVisualizationMenu()
        widget = self.getCompareWidget()
        if widget is not None:
            mode = self.__getVisualizationMode()
            widget.setVisualizationMode(mode)

    def __updateVisualizationMenu(self):
        """Update the state of the action containing visualization menu.
        """
        selectedAction = self.__visualizationGroup.checkedAction()
        if selectedAction is not None:
            self.__visualizationAction.setText(selectedAction.text())
            self.__visualizationAction.setIcon(selectedAction.icon())
            self.__visualizationAction.setToolTip(selectedAction.toolTip())
        else:
            self.__visualizationAction.setText("")
            self.__visualizationAction.setIcon(qt.QIcon())
            self.__visualizationAction.setToolTip("")

    def __getVisualizationMode(self):
        """Returns the current visualization mode."""
        if self.__aModeAction.isChecked():
            return "a"
        elif self.__bModeAction.isChecked():
            return "b"
        elif self.__vlineModeAction.isChecked():
            return "vline"
        elif self.__hlineModeAction.isChecked():
            return "hline"
        elif self.__ycChannelModeAction.isChecked():
            return "ycchannel"
        elif self.__brChannelModeAction.isChecked():
            return "brchannel"
        else:
            raise ValueError("Unknown interaction mode")

    def __alignmentModeChanged(self, selectedAction):
        """Called when user requesting changes of the alignment mode.
        """
        self.__updateAlignmentMenu()
        widget = self.getCompareWidget()
        if widget is not None:
            mode = self.__getAlignmentMode()
            widget.setAlignmentMode(mode)

    def __updateAlignmentMenu(self):
        """Update the state of the action containing alignment menu.
        """
        selectedAction = self.__alignmentGroup.checkedAction()
        if selectedAction is not None:
            self.__alignmentAction.setText(selectedAction.text())
            self.__alignmentAction.setIcon(selectedAction.icon())
            self.__alignmentAction.setToolTip(selectedAction.toolTip())
        else:
            self.__alignmentAction.setText("")
            self.__alignmentAction.setIcon(qt.QIcon())
            self.__alignmentAction.setToolTip("")

    def __getAlignmentMode(self):
        """Returns the current selected alignemnt mode."""
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

    def __keypointVisibilityChanged(self):
        """Called when action managing keypoints visibility changes"""
        widget = self.getCompareWidget()
        if widget is not None:
            keypointsVisible = self.__displayKeypoints.isChecked()
            widget.setKeypointsVisible(keypointsVisible)


class CompareImages(qt.QWidget):
    """Widget providing tools to compare 2 images.

    :param Union[qt.QWidget,None] parent: Parent of this widget.
    :param backend: The backend to use, in:
                    'matplotlib' (default), 'mpl', 'opengl', 'gl', 'none'
                    or a :class:`BackendBase.BackendBase` class
    :type backend: str or :class:`BackendBase.BackendBase`
    """

    sigConfigurationChanged = qt.Signal()
    """Emitted when the configuration of the widget (visualization mode,
    alignement mode...) have changed."""

    def __init__(self, parent=None, backend=None):
        qt.QMainWindow.__init__(self, parent)
        self.setWindowTitle("Plot with synchronized axes")

        if parent is None:
            self.setWindowTitle('Compare images')

        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.__raw1 = None
        self.__raw2 = None
        self.__data1 = None
        self.__data2 = None
        self.__previousSeparatorPosition = None

        self.__plot = plot.PlotWidget(parent=self, backend=backend)
        self.__plot.getXAxis().setLabel('Columns')
        self.__plot.getYAxis().setLabel('Rows')
        if silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION == 'downward':
            self.__plot.getYAxis().setInverted(True)

        self.__plot.setKeepDataAspectRatio(True)
        self.__plot.sigPlotSignal.connect(self.__plotSlot)

        layout.addWidget(self.__plot)

        self.__plot.addXMarker(
            0,
            legend='vline',
            text='',
            draggable=True,
            color='blue',
            constraint=self.__separatorConstraint)
        self.__vline = self.__plot._getMarker('vline')

        self.__plot.addYMarker(
            0,
            legend='hline',
            text='',
            draggable=True,
            color='blue',
            constraint=self.__separatorConstraint)
        self.__hline = self.__plot._getMarker('hline')

        # default values
        self.__visualizationMode = ""
        self.__alignmentMode = ""
        self.__keypointsVisible = True

        self.setAlignmentMode("origin")
        self.setVisualizationMode("vline")
        self.setKeypointsVisible(False)

        # Toolbars

        self._createToolBars(self.__plot)
        if self._interactiveModeToolBar is not None:
            self.__plot.addToolBar(self._interactiveModeToolBar)
        if self._imageToolBar is not None:
            self.__plot.addToolBar(self._imageToolBar)
        if self._compareToolBar is not None:
            self.__plot.addToolBar(self._compareToolBar)

    def _createToolBars(self, plot):
        """Create tool bars displayed by the widget"""
        toolBar = tools.InteractiveModeToolBar(parent=self, plot=plot)
        self._interactiveModeToolBar = toolBar
        toolBar = tools.ImageToolBar(parent=self, plot=plot)
        self._imageToolBar = toolBar
        toolBar = CompareImagesToolBar(self)
        toolBar.setCompareWidget(self)
        self._compareToolBar = toolBar

    def getPlot(self):
        """Returns the plot which is used to display the images.

        :rtype: silx.gui.plot.PlotWidget
        """
        return self.__plot

    def setVisualizationMode(self, mode):
        """Set the visualization mode.

        :param str mode: New visualization to display the image comparison
        """
        if self.__visualizationMode == mode:
            return
        self.__visualizationMode = mode
        mode = self.getVisualizationMode()
        self.__vline.setVisible(mode == "vline")
        self.__hline.setVisible(mode == "hline")
        self.__invalidateData()
        self.sigConfigurationChanged.emit()

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
        self.__invalidateData()
        self.sigConfigurationChanged.emit()

    def getAlignmentMode(self):
        """Returns the current selected alignemnt mode."""
        return self.__alignmentMode

    def setKeypointsVisible(self, isVisible):
        """Set keyboard visibility.

        :param bool isVisible: If True, keypoints are displayed (if some)
        """
        if self.__keypointsVisible == isVisible:
            return
        self.__keypointsVisible = isVisible
        self.__invalidateScatter()
        self.sigConfigurationChanged.emit()

    def __setDefaultAlignmentMode(self):
        """Reset the alignemnt mode to the default value"""
        self.setAlignmentMode("origin")

    def __plotSlot(self, event):
        """Handle events from the plot"""
        if event['event'] in ('markerMoving', 'markerMoved'):
            mode = self.getVisualizationMode()
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

    def __invalidateSeparator(self):
        """Redraw images according to the current state of the separators.
        """
        mode = self.getVisualizationMode()
        if mode == "vline":
            pos = self.__vline.getXPosition()
        elif mode == "hline":
            pos = self.__hline.getYPosition()
        else:
            self.__image1.setOrigin((0, 0))
            self.__image2.setOrigin((0, 0))
            return
        self.__separatorMoved(pos)
        self.__previousSeparatorPosition = pos

    def __separatorMoved(self, pos):
        """Called when vertical or horizontal separators have moved.

        Update the displayed images.
        """
        if self.__data1 is None:
            return

        mode = self.getVisualizationMode()
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
        """Set images to compare.

        Images can contains floating-point or integer values, or RGB and RGBA
        values, but should have comparable intensities.

        RGB and RGBA images are provided as an array as `[width,height,channels]`
        of usigned integer 8-bits or floating-points between 0.0 to 1.0.

        :param numpy.ndarray image1: The first image
        :param numpy.ndarray image2: The second image
        """
        self.__raw1 = image1
        self.__raw2 = image2
        self.__invalidateData()
        self.__plot.resetZoom()

    def __invalidateScatter(self):
        """Update the displayed keypoints using cached keypoints.
        """
        if self.__keypointsVisible:
            data = self.__matching_keypoints
        else:
            data = [], [], []
        self.__plot.addScatter(x=data[0],
                               y=data[1],
                               z=1,
                               value=data[2],
                               legend="keypoints",
                               colormap=Colormap("spring"))

    def __invalidateData(self):
        """Compute aligned image when the alignement mode changes.

        This function cache input images which are used when
        vertical/horizontal separators moves.
        """
        raw1, raw2 = self.__raw1, self.__raw2
        if raw1 is None or raw2 is None:
            return

        alignmentMode = self.getAlignmentMode()

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

        mode = self.getVisualizationMode()
        if mode == "ycchannel":
            data1 = self.__composeImage(data1, data2, "yc")
            data2 = numpy.empty((0, 0))
        elif mode == "brchannel":
            data1 = self.__composeImage(data1, data2, "br")
            data2 = numpy.empty((0, 0))
        elif mode == "a":
            data2 = numpy.empty((0, 0))
        elif mode == "b":
            data1 = numpy.empty((0, 0))

        self.__data1, self.__data2 = data1, data2
        self.__plot.addImage(data1, z=0, legend="image1", resetzoom=False)
        self.__plot.addImage(data2, z=0, legend="image2", resetzoom=False)
        self.__image1 = self.__plot.getImage("image1")
        self.__image2 = self.__plot.getImage("image2")
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
            if self.__data1.size == 0:
                vmin = self.__data2.min()
                vmax = self.__data2.max()
            elif self.__data2.size == 0:
                vmin = self.__data1.min()
                vmax = self.__data1.max()
            else:
                vmin = min(self.__data1.min(), self.__data2.min())
                vmax = max(self.__data1.max(), self.__data2.max())
            colormap = Colormap(vmin=vmin, vmax=vmax)
            self.__image1.setColormap(colormap)
            self.__image2.setColormap(colormap)

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

    def __composeImage(self, data1, data2, mode):
        """Returns an RBG image containing composition of data1 and data2 in 2
        different channels

        :param numpy.ndarray data1: First image
        :param numpy.ndarray data1: Second image
        :param str mode: Composition mode. Supporting "yc" (yellow/cyan)
            and "br" (blue/red).
        :rtype: numpy.ndarray
        """
        assert(data1.shape[0:2] == data2.shape[0:2])
        mode1 = self.__getImageMode(data1)
        if mode1 in ["rgb", "rgba"]:
            intensity1 = self.__luminosityImage(data1)
            vmin1, vmax1 = 0.0, 1.0
        else:
            intensity1 = data1
            vmin1, vmax1 = data1.min(), data1.max()

        mode2 = self.__getImageMode(data2)
        if mode2 in ["rgb", "rgba"]:
            intensity2 = self.__luminosityImage(data2)
            vmin2, vmax2 = 0.0, 1.0
        else:
            intensity2 = data2
            vmin2, vmax2 = data2.min(), data2.max()

        vmin, vmax = min(vmin1, vmin2) * 1.0, max(vmax1, vmax2) * 1.0
        shape = data1.shape
        result = numpy.empty((shape[0], shape[1], 3), dtype=numpy.uint8)
        if mode == "br":
            result[:, :, 0] = (intensity2 - vmin) * (1.0 / (vmax - vmin)) * 255.0
            result[:, :, 1] = 0
            result[:, :, 2] = (intensity1 - vmin) * (1.0 / (vmax - vmin)) * 255.0
        elif mode == "yc":
            result[:, :, 0] = 255 - (intensity2 - vmin) * (1.0 / (vmax - vmin)) * 255.0
            result[:, :, 1] = 255
            result[:, :, 2] = 255 - (intensity1 - vmin) * (1.0 / (vmax - vmin)) * 255.0
        return result

    def __luminosityImage(self, image):
        """Returns the lominosity channel from an RBG(A) image.
        The alpha channel is ignored.

        :rtype: numpy.ndarray
        """
        mode = self.__getImageMode(image)
        assert(mode in ["rgb", "rgba"])
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

        :rtype: numpy.ndarray
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

        self.__matching_keypoints = (match[:].x[:, 0],
                                     match[:].y[:, 0],
                                     match[:].scale[:, 0])
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
        data2 = sa.align(second_image)
        return data1, data2
