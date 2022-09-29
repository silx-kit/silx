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


import enum
import logging
import numpy
import weakref
import collections
import math

import silx.image.bilinear
from silx.gui import qt
from silx.gui import plot
from silx.gui import icons
from silx.gui.colors import Colormap
from silx.gui.plot import tools
from silx.utils.weakref import WeakMethodProxy

_logger = logging.getLogger(__name__)

from silx.opencl import ocl
if ocl is not None:
    try:
        from silx.opencl import sift
    except ImportError:
        # sift module is not available (e.g., in official Debian packages)
        sift = None
else:  # No OpenCL device or no pyopencl
    sift = None


@enum.unique
class VisualizationMode(enum.Enum):
    """Enum for each visualization mode available."""
    ONLY_A = 'a'
    ONLY_B = 'b'
    VERTICAL_LINE = 'vline'
    HORIZONTAL_LINE = 'hline'
    COMPOSITE_RED_BLUE_GRAY = "rbgchannel"
    COMPOSITE_RED_BLUE_GRAY_NEG = "rbgnegchannel"
    COMPOSITE_A_MINUS_B = "aminusb"


@enum.unique
class AlignmentMode(enum.Enum):
    """Enum for each alignment mode available."""
    ORIGIN = 'origin'
    CENTER = 'center'
    STRETCH = 'stretch'
    AUTO = 'auto'


AffineTransformation = collections.namedtuple("AffineTransformation",
                                              ["tx", "ty", "sx", "sy", "rot"])
"""Contains a 2D affine transformation: translation, scale and rotation"""


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
        self.__visualizationToolButton = qt.QToolButton(self)
        self.__visualizationToolButton.setMenu(menu)
        self.__visualizationToolButton.setPopupMode(qt.QToolButton.InstantPopup)
        self.addWidget(self.__visualizationToolButton)
        self.__visualizationGroup = qt.QActionGroup(self)
        self.__visualizationGroup.setExclusive(True)
        self.__visualizationGroup.triggered.connect(self.__visualizationModeChanged)

        icon = icons.getQIcon("compare-mode-a")
        action = qt.QAction(icon, "Display the first image only", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_A))
        action.setProperty("mode", VisualizationMode.ONLY_A)
        menu.addAction(action)
        self.__aModeAction = action
        self.__visualizationGroup.addAction(action)

        icon = icons.getQIcon("compare-mode-b")
        action = qt.QAction(icon, "Display the second image only", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_B))
        action.setProperty("mode", VisualizationMode.ONLY_B)
        menu.addAction(action)
        self.__bModeAction = action
        self.__visualizationGroup.addAction(action)

        icon = icons.getQIcon("compare-mode-vline")
        action = qt.QAction(icon, "Vertical compare mode", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_V))
        action.setProperty("mode", VisualizationMode.VERTICAL_LINE)
        menu.addAction(action)
        self.__vlineModeAction = action
        self.__visualizationGroup.addAction(action)

        icon = icons.getQIcon("compare-mode-hline")
        action = qt.QAction(icon, "Horizontal compare mode", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_H))
        action.setProperty("mode", VisualizationMode.HORIZONTAL_LINE)
        menu.addAction(action)
        self.__hlineModeAction = action
        self.__visualizationGroup.addAction(action)

        icon = icons.getQIcon("compare-mode-rb-channel")
        action = qt.QAction(icon, "Blue/red compare mode (additive mode)", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_C))
        action.setProperty("mode", VisualizationMode.COMPOSITE_RED_BLUE_GRAY)
        menu.addAction(action)
        self.__brChannelModeAction = action
        self.__visualizationGroup.addAction(action)

        icon = icons.getQIcon("compare-mode-rbneg-channel")
        action = qt.QAction(icon, "Yellow/cyan compare mode (subtractive mode)", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_W))
        action.setProperty("mode", VisualizationMode.COMPOSITE_RED_BLUE_GRAY_NEG)
        menu.addAction(action)
        self.__ycChannelModeAction = action
        self.__visualizationGroup.addAction(action)

        icon = icons.getQIcon("compare-mode-a-minus-b")
        action = qt.QAction(icon, "Raw A minus B compare mode", self)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        action.setShortcut(qt.QKeySequence(qt.Qt.Key_W))
        action.setProperty("mode", VisualizationMode.COMPOSITE_A_MINUS_B)
        menu.addAction(action)
        self.__ycChannelModeAction = action
        self.__visualizationGroup.addAction(action)

        menu = qt.QMenu(self)
        self.__alignmentToolButton = qt.QToolButton(self)
        self.__alignmentToolButton.setMenu(menu)
        self.__alignmentToolButton.setPopupMode(qt.QToolButton.InstantPopup)
        self.addWidget(self.__alignmentToolButton)
        self.__alignmentGroup = qt.QActionGroup(self)
        self.__alignmentGroup.setExclusive(True)
        self.__alignmentGroup.triggered.connect(self.__alignmentModeChanged)

        icon = icons.getQIcon("compare-align-origin")
        action = qt.QAction(icon, "Align images on their upper-left pixel", self)
        action.setProperty("mode", AlignmentMode.ORIGIN)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        self.__originAlignAction = action
        menu.addAction(action)
        self.__alignmentGroup.addAction(action)

        icon = icons.getQIcon("compare-align-center")
        action = qt.QAction(icon, "Center images", self)
        action.setProperty("mode", AlignmentMode.CENTER)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        self.__centerAlignAction = action
        menu.addAction(action)
        self.__alignmentGroup.addAction(action)

        icon = icons.getQIcon("compare-align-stretch")
        action = qt.QAction(icon, "Stretch the second image on the first one", self)
        action.setProperty("mode", AlignmentMode.STRETCH)
        action.setIconVisibleInMenu(True)
        action.setCheckable(True)
        self.__stretchAlignAction = action
        menu.addAction(action)
        self.__alignmentGroup.addAction(action)

        icon = icons.getQIcon("compare-align-auto")
        action = qt.QAction(icon, "Auto-alignment of the second image", self)
        action.setProperty("mode", AlignmentMode.AUTO)
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
        action = None
        for a in self.__visualizationGroup.actions():
            actionMode = a.property("mode")
            if mode == actionMode:
                action = a
                break
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
        action = None
        for a in self.__alignmentGroup.actions():
            actionMode = a.property("mode")
            if mode == actionMode:
                action = a
                break
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
            mode = selectedAction.property("mode")
            widget.setVisualizationMode(mode)

    def __updateVisualizationMenu(self):
        """Update the state of the action containing visualization menu.
        """
        selectedAction = self.__visualizationGroup.checkedAction()
        if selectedAction is not None:
            self.__visualizationToolButton.setText(selectedAction.text())
            self.__visualizationToolButton.setIcon(selectedAction.icon())
            self.__visualizationToolButton.setToolTip(selectedAction.toolTip())
        else:
            self.__visualizationToolButton.setText("")
            self.__visualizationToolButton.setIcon(qt.QIcon())
            self.__visualizationToolButton.setToolTip("")

    def __alignmentModeChanged(self, selectedAction):
        """Called when user requesting changes of the alignment mode.
        """
        self.__updateAlignmentMenu()
        widget = self.getCompareWidget()
        if widget is not None:
            mode = selectedAction.property("mode")
            widget.setAlignmentMode(mode)

    def __updateAlignmentMenu(self):
        """Update the state of the action containing alignment menu.
        """
        selectedAction = self.__alignmentGroup.checkedAction()
        if selectedAction is not None:
            self.__alignmentToolButton.setText(selectedAction.text())
            self.__alignmentToolButton.setIcon(selectedAction.icon())
            self.__alignmentToolButton.setToolTip(selectedAction.toolTip())
        else:
            self.__alignmentToolButton.setText("")
            self.__alignmentToolButton.setIcon(qt.QIcon())
            self.__alignmentToolButton.setToolTip("")

    def __keypointVisibilityChanged(self):
        """Called when action managing keypoints visibility changes"""
        widget = self.getCompareWidget()
        if widget is not None:
            keypointsVisible = self.__displayKeypoints.isChecked()
            widget.setKeypointsVisible(keypointsVisible)


class CompareImagesStatusBar(qt.QStatusBar):
    """StatusBar containing specific information contained in a
    :class:`CompareImages` widget

    Use :meth:`setCompareWidget` to connect this toolbar to a specific
    :class:`CompareImages` widget.

    :param Union[qt.QWidget,None] parent: Parent of this widget.
    """
    def __init__(self, parent=None):
        qt.QStatusBar.__init__(self, parent)
        self.setSizeGripEnabled(False)
        self.layout().setSpacing(0)
        self.__compareWidget = None
        self._label1 = qt.QLabel(self)
        self._label1.setFrameShape(qt.QFrame.WinPanel)
        self._label1.setFrameShadow(qt.QFrame.Sunken)
        self._label2 = qt.QLabel(self)
        self._label2.setFrameShape(qt.QFrame.WinPanel)
        self._label2.setFrameShadow(qt.QFrame.Sunken)
        self._transform = qt.QLabel(self)
        self._transform.setFrameShape(qt.QFrame.WinPanel)
        self._transform.setFrameShadow(qt.QFrame.Sunken)
        self.addWidget(self._label1)
        self.addWidget(self._label2)
        self.addWidget(self._transform)
        self._pos = None
        self._updateStatusBar()

    def setCompareWidget(self, widget):
        """
        Connect this tool bar to a specific :class:`CompareImages` widget.

        :param Union[None,CompareImages] widget: The widget to connect with.
        """
        compareWidget = self.getCompareWidget()
        if compareWidget is not None:
            compareWidget.getPlot().sigPlotSignal.disconnect(self.__plotSignalReceived)
            compareWidget.sigConfigurationChanged.disconnect(self.__dataChanged)
        compareWidget = widget
        if compareWidget is None:
            self.__compareWidget = None
        else:
            self.__compareWidget = weakref.ref(compareWidget)
        if compareWidget is not None:
            compareWidget.getPlot().sigPlotSignal.connect(self.__plotSignalReceived)
            compareWidget.sigConfigurationChanged.connect(self.__dataChanged)

    def getCompareWidget(self):
        """Returns the connected widget.

        :rtype: CompareImages
        """
        if self.__compareWidget is None:
            return None
        else:
            return self.__compareWidget()

    def __plotSignalReceived(self, event):
        """Called when old style signals at emmited from the plot."""
        if event["event"] == "mouseMoved":
            x, y = event["x"], event["y"]
            self.__mouseMoved(x, y)

    def __mouseMoved(self, x, y):
        """Called when mouse move over the plot."""
        self._pos = x, y
        self._updateStatusBar()

    def __dataChanged(self):
        """Called when internal data from the connected widget changes."""
        self._updateStatusBar()

    def _formatData(self, data):
        """Format pixel of an image.

        It supports intensity, RGB, and RGBA.

        :param Union[int,float,numpy.ndarray,str]: Value of a pixel
        :rtype: str
        """
        if data is None:
            return "No data"
        if isinstance(data, (int, numpy.integer)):
            return "%d" % data
        if isinstance(data, (float, numpy.floating)):
            return "%f" % data
        if isinstance(data, numpy.ndarray):
            # RGBA value
            if data.shape == (3,):
                return "R:%d G:%d B:%d" % (data[0], data[1], data[2])
            elif data.shape == (4,):
                return "R:%d G:%d B:%d A:%d" % (data[0], data[1], data[2], data[3])
        _logger.debug("Unsupported data format %s. Cast it to string.", type(data))
        return str(data)

    def _updateStatusBar(self):
        """Update the content of the status bar"""
        widget = self.getCompareWidget()
        if widget is None:
            self._label1.setText("Image1: NA")
            self._label2.setText("Image2: NA")
            self._transform.setVisible(False)
        else:
            transform = widget.getTransformation()
            self._transform.setVisible(transform is not None)
            if transform is not None:
                has_notable_translation = not numpy.isclose(transform.tx, 0.0, atol=0.01) \
                    or not numpy.isclose(transform.ty, 0.0, atol=0.01)
                has_notable_scale = not numpy.isclose(transform.sx, 1.0, atol=0.01) \
                    or not numpy.isclose(transform.sy, 1.0, atol=0.01)
                has_notable_rotation = not numpy.isclose(transform.rot, 0.0, atol=0.01)

                strings = []
                if has_notable_translation:
                    strings.append("Translation")
                if has_notable_scale:
                    strings.append("Scale")
                if has_notable_rotation:
                    strings.append("Rotation")
                if strings == []:
                    has_translation = not numpy.isclose(transform.tx, 0.0) \
                        or not numpy.isclose(transform.ty, 0.0)
                    has_scale = not numpy.isclose(transform.sx, 1.0) \
                        or not numpy.isclose(transform.sy, 1.0)
                    has_rotation = not numpy.isclose(transform.rot, 0.0)
                    if has_translation or has_scale or has_rotation:
                        text = "No big changes"
                    else:
                        text = "No changes"
                else:
                    text = "+".join(strings)
                self._transform.setText("Align: " + text)

                strings = []
                if not numpy.isclose(transform.ty, 0.0):
                    strings.append("Translation x: %0.3fpx" % transform.tx)
                if not numpy.isclose(transform.ty, 0.0):
                    strings.append("Translation y: %0.3fpx" % transform.ty)
                if not numpy.isclose(transform.sx, 1.0):
                    strings.append("Scale x: %0.3f" % transform.sx)
                if not numpy.isclose(transform.sy, 1.0):
                    strings.append("Scale y: %0.3f" % transform.sy)
                if not numpy.isclose(transform.rot, 0.0):
                    strings.append("Rotation: %0.3fdeg" % (transform.rot * 180 / numpy.pi))
                if strings == []:
                    text = "No transformation"
                else:
                    text = "\n".join(strings)
                self._transform.setToolTip(text)

            if self._pos is None:
                self._label1.setText("Image1: NA")
                self._label2.setText("Image2: NA")
            else:
                data1, data2 = widget.getRawPixelData(self._pos[0], self._pos[1])
                if isinstance(data1, str):
                    self._label1.setToolTip(data1)
                    text1 = "NA"
                else:
                    self._label1.setToolTip("")
                    text1 = self._formatData(data1)
                if isinstance(data2, str):
                    self._label2.setToolTip(data2)
                    text2 = "NA"
                else:
                    self._label2.setToolTip("")
                    text2 = self._formatData(data2)
                self._label1.setText("Image1: %s" % text1)
                self._label2.setText("Image2: %s" % text2)


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
    alignement mode...) have changed."""

    def __init__(self, parent=None, backend=None):
        qt.QMainWindow.__init__(self, parent)
        self._resetZoomActive = True
        self._colormap = Colormap()
        """Colormap shared by all modes, except the compose images (rgb image)"""
        self._colormapKeyPoints = Colormap('spring')
        """Colormap used for sift keypoints"""

        if parent is None:
            self.setWindowTitle('Compare images')
        else:
            self.setWindowFlags(qt.Qt.Widget)

        self.__transformation = None
        self.__raw1 = None
        self.__raw2 = None
        self.__data1 = None
        self.__data2 = None
        self.__previousSeparatorPosition = None

        self.__plot = plot.PlotWidget(parent=self, backend=backend)
        self.__plot.setDefaultColormap(self._colormap)
        self.__plot.getXAxis().setLabel('Columns')
        self.__plot.getYAxis().setLabel('Rows')
        if silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION == 'downward':
            self.__plot.getYAxis().setInverted(True)

        self.__plot.setKeepDataAspectRatio(True)
        self.__plot.sigPlotSignal.connect(self.__plotSlot)
        self.__plot.setAxesDisplayed(False)

        self.setCentralWidget(self.__plot)

        legend = VisualizationMode.VERTICAL_LINE.name
        self.__plot.addXMarker(
                0,
                legend=legend,
                text='',
                draggable=True,
                color='blue',
                constraint=WeakMethodProxy(self.__separatorConstraint))
        self.__vline = self.__plot._getMarker(legend)

        legend = VisualizationMode.HORIZONTAL_LINE.name
        self.__plot.addYMarker(
                0,
                legend=legend,
                text='',
                draggable=True,
                color='blue',
                constraint=WeakMethodProxy(self.__separatorConstraint))
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
        data2 = None
        alignmentMode = self.__alignmentMode
        raw1, raw2 = self.__raw1, self.__raw2
        if alignmentMode == AlignmentMode.ORIGIN:
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
            data2 = "Not implemented with sift"
        else:
            assert(False)

        x1, y1 = int(x1), int(y1)
        if raw1 is None or y1 < 0 or y1 >= raw1.shape[0] or x1 < 0 or x1 >= raw1.shape[1]:
            data1 = None
        else:
            data1 = raw1[y1, x1]

        if data2 is None:
            x2, y2 = int(x2), int(y2)
            if raw2 is None or y2 < 0 or y2 >= raw2.shape[0] or x2 < 0 or x2 >= raw2.shape[1]:
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
        previousMode = self.getVisualizationMode()
        self.__visualizationMode = mode
        mode = self.getVisualizationMode()
        self.__vline.setVisible(mode == VisualizationMode.VERTICAL_LINE)
        self.__hline.setVisible(mode == VisualizationMode.HORIZONTAL_LINE)
        visModeRawDisplay = (VisualizationMode.ONLY_A,
                             VisualizationMode.ONLY_B,
                             VisualizationMode.VERTICAL_LINE,
                             VisualizationMode.HORIZONTAL_LINE)
        updateColormap = not(previousMode in visModeRawDisplay and
                             mode in visModeRawDisplay)
        self.__updateData(updateColormap=updateColormap)
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
        self.__updateData(updateColormap=False)
        self.sigConfigurationChanged.emit()

    def getAlignmentMode(self):
        """Returns the current selected alignemnt mode."""
        return self.__alignmentMode

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
        if event['event'] in ('markerMoving', 'markerMoved'):
            mode = self.getVisualizationMode()
            legend = mode.name
            if event['label'] == legend:
                if mode == VisualizationMode.VERTICAL_LINE:
                    value = int(float(str(event['xdata'])))
                elif mode == VisualizationMode.HORIZONTAL_LINE:
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

    def __updateSeparators(self):
        """Redraw images according to the current state of the separators.
        """
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
            self.__image2.setData(data2, copy=False)
            self.__image2.setOrigin((0, pos))
        else:
            assert(False)

    def setData(self, image1, image2, updateColormap=True):
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
        self.__updateData(updateColormap=updateColormap)
        if self.isAutoResetZoom():
            self.__plot.resetZoom()

    def setImage1(self, image1, updateColormap=True):
        """Set image1 to be compared.

        Images can contains floating-point or integer values, or RGB and RGBA
        values, but should have comparable intensities.

        RGB and RGBA images are provided as an array as `[width,height,channels]`
        of usigned integer 8-bits or floating-points between 0.0 to 1.0.

        :param numpy.ndarray image1: The first image
        """
        self.__raw1 = image1
        self.__updateData(updateColormap=updateColormap)
        if self.isAutoResetZoom():
            self.__plot.resetZoom()

    def setImage2(self, image2, updateColormap=True):
        """Set image2 to be compared.

        Images can contains floating-point or integer values, or RGB and RGBA
        values, but should have comparable intensities.

        RGB and RGBA images are provided as an array as `[width,height,channels]`
        of usigned integer 8-bits or floating-points between 0.0 to 1.0.

        :param numpy.ndarray image2: The second image
        """
        self.__raw2 = image2
        self.__updateData(updateColormap=updateColormap)
        if self.isAutoResetZoom():
            self.__plot.resetZoom()

    def __updateKeyPoints(self):
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
                               colormap=self._colormapKeyPoints,
                               legend="keypoints")

    def __updateData(self, updateColormap):
        """Compute aligned image when the alignment mode changes.

        This function cache input images which are used when
        vertical/horizontal separators moves.
        """
        raw1, raw2 = self.__raw1, self.__raw2
        if raw1 is None or raw2 is None:
            return

        alignmentMode = self.getAlignmentMode()
        self.__transformation = None

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
            data1 = self.__createMarginImage(raw1, size, transparent=True, center=True)
            data2 = self.__createMarginImage(raw2, size, transparent=True, center=True)
            self.__matching_keypoints = ([data1.shape[1] // 2],
                                         [data1.shape[0] // 2],
                                         [1.0])
        elif alignmentMode == AlignmentMode.STRETCH:
            data1 = raw1
            data2 = self.__rescaleImage(raw2, data1.shape)
            self.__matching_keypoints = ([0, data1.shape[1], data1.shape[1], 0],
                                         [0, 0, data1.shape[0], data1.shape[0]],
                                         [1.0, 1.0, 1.0, 1.0])
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
            assert(False)

        mode = self.getVisualizationMode()
        if mode == VisualizationMode.COMPOSITE_RED_BLUE_GRAY_NEG:
            data1 = self.__composeImage(data1, data2, mode)
            data2 = numpy.empty((0, 0))
        elif mode == VisualizationMode.COMPOSITE_RED_BLUE_GRAY:
            data1 = self.__composeImage(data1, data2, mode)
            data2 = numpy.empty((0, 0))
        elif mode == VisualizationMode.COMPOSITE_A_MINUS_B:
            data1 = self.__composeImage(data1, data2, mode)
            data2 = numpy.empty((0, 0))
        elif mode == VisualizationMode.ONLY_A:
            data2 = numpy.empty((0, 0))
        elif mode == VisualizationMode.ONLY_B:
            data1 = numpy.empty((0, 0))

        self.__data1, self.__data2 = data1, data2
        self.__plot.addImage(data1, z=0, legend="image1", resetzoom=False)
        self.__plot.addImage(data2, z=0, legend="image2", resetzoom=False)
        self.__image1 = self.__plot.getImage("image1")
        self.__image2 = self.__plot.getImage("image2")
        self.__updateKeyPoints()

        # Set the separator into the middle
        if self.__previousSeparatorPosition is None:
            value = self.__data1.shape[1] // 2
            self.__vline.setPosition(value, 0)
            value = self.__data1.shape[0] // 2
            self.__hline.setPosition(0, value)
        self.__updateSeparators()
        if updateColormap:
            self.__updateColormap()

    def __updateColormap(self):
        # TODO: The colormap histogram will still be wrong
        mode1 = self.__getImageMode(self.__data1)
        mode2 = self.__getImageMode(self.__data2)
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
            colormap = self.getColormap()
            colormap.setVRange(vmin=vmin, vmax=vmax)
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
        :param VisualizationMode mode: Composition mode.
        :rtype: numpy.ndarray
        """
        assert(data1.shape[0:2] == data2.shape[0:2])
        if mode == VisualizationMode.COMPOSITE_A_MINUS_B:
            # TODO: this calculation has no interest of generating a 'composed'
            # rgb image, this could be moved in an other function or doc
            # should be modified
            _type = data1.dtype
            result = data1.astype(numpy.float64) - data2.astype(numpy.float64)
            return result
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
        a = (intensity1 - vmin) * (1.0 / (vmax - vmin)) * 255.0
        b = (intensity2 - vmin) * (1.0 / (vmax - vmin)) * 255.0
        if mode == VisualizationMode.COMPOSITE_RED_BLUE_GRAY:
            result[:, :, 0] = a
            result[:, :, 1] = (a + b) / 2
            result[:, :, 2] = b
        elif mode == VisualizationMode.COMPOSITE_RED_BLUE_GRAY_NEG:
            result[:, :, 0] = 255 - b
            result[:, :, 1] = 255 - (a + b) / 2
            result[:, :, 2] = 255 - a
        return result

    def __luminosityImage(self, image):
        """Returns the luminosity channel from an RBG(A) image.
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
        """Retuns the affine transformation applied to the second image to align
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
        result = sa.align(second_image, return_all=True)
        data2 = result["result"]
        self.__transformation = self.__toAffineTransformation(result)
        return data1, data2

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
