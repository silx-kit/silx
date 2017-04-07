# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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

from __future__ import division

import math
import logging
import os
import numpy

from .. import qt
from .. import icons
from ...image import shapes

from .MaskToolsWidget import BaseMask
from .Colors import cursorColorForColormap, rgba


_logger = logging.getLogger(__name__)


class ScatterMask(BaseMask):
    """A 1D mask for scatter data.
    """
    def __init__(self, scatter=None):
        """

        :param scatter: :class:`silx.gui.plot.items.Scatter` instance
        """
        BaseMask.__init__(self)
        if scatter is not None:
            self.setScatter(scatter)

    def setScatter(self, scatter):
        self._x = scatter.getXData(copy=False)
        self._y = scatter.getYData(copy=False)
        self._value = scatter.getValueData(copy=False)

        self._mask = numpy.zeros_like(self._x, dtype=numpy.uint8)

    def save(self, filename, kind):
        # TODO
        pass

    def updateStencil(self, level, stencil, mask=True):
        """Mask/Unmask points from boolean mask: all elements that are True
        in the boolean mask are set to ``level`` (if ``mask=True``) or 0
        (if ``mask=False``)

        :param int level: Mask level to update.
        :param stencil: Boolean mask.
        :type stencil: numpy.array of same dimension as the mask
        :param bool mask: True to mask (default), False to unmask.
        """
        if mask:
            self._mask[stencil] = level
        else:
            self._mask[stencil][self._mask[stencil] == level] = 0
        self._notify()

    def updatePoints(self, level, indices, mask=True):
        """Mask/Unmask points with given indices.

        :param int level: Mask level to update.
        :param indices: Sequence or 1D array of indices of points to be
            updated
        :param bool mask: True to mask (default), False to unmask.
        """
        if mask:
            self._mask[indices] = level
        else:
            # unmask only where mask level is the specified value
            indices_stencil = numpy.zeros_like(self._mask, dtype=numpy.bool)
            indices_stencil[indices] = True
            self._mask[numpy.logical_and(self._mask == level,
                                         indices_stencil)    ] = 0   # noqa
        self._notify()

    # update shapes
    def updatePolygon(self, level, vertices, mask=True):
        """Mask/Unmask a polygon of the given mask level.

        :param int level: Mask level to update.
        :param vertices: Nx2 array of polygon corners as (y, x) or (row, col)
        :param bool mask: True to mask (default), False to unmask.
        """
        polygon = shapes.Polygon(vertices)

        # TODO: this could be optimized if necessary
        indices_in_polygon = [idx for idx in range(len(self._x)) if
                              polygon.is_inside(self._y[idx], self._x[idx])]

        self.updatePoints(level, indices_in_polygon, mask)

    def updateRectangle(self, level, y, x, height, width, mask=True):
        """Mask/Unmask data inside a rectangle

        :param int level: Mask level to update.
        :param float y: Y coordinate of bottom left corner of the rectangle
        :param float x: X coordinate of bottom left corner of the rectangle
        :param float height:
        :param float width:
        :param bool mask: True to mask (default), False to unmask.
        """
        vertices = [(y, x),
                    (y + height, x),
                    (y + height, x + width),
                    (y, x + width)]
        self.updatePolygon(level, vertices, mask)

    def updateDisk(self, level, cy, cx, radius, mask=True):
        """Mask/Unmask a disk of the given mask level.

        :param int level: Mask level to update.
        :param float cy: Disk center (y).
        :param float cx: Disk center (x).
        :param float radius: Radius of the disk in mask array unit
        :param bool mask: True to mask (default), False to unmask.
        """
        stencil = (self._y - cy)**2 + (self._x - cx)**2 < radius**2
        self.updateStencil(level, stencil, mask)

    def updateLine(self, level, y0, x0, y1, x1, width, mask=True):
        """Mask/Unmask points inside a rectangle defined by a line (two
        end points) and a width.

        :param int level: Mask level to update.
        :param float y0: Row of the starting point.
        :param float x0: Column of the starting point.
        :param float row1: Row of the end point.
        :param float col1: Column of the end point.
        :param float width: Width of the line.
        :param bool mask: True to mask (default), False to unmask.
        """
        # theta is the angle between the horizontal and the line
        theta = math.atan((y1 - y0) / (x1 - x0)) if x1 - x0 else 0
        w_over_2_sin_theta = width / 2. * math.sin(theta)
        w_over_2_cos_theta = width / 2. * math.cos(theta)

        vertices = [(y0 - w_over_2_cos_theta, x0 + w_over_2_sin_theta),
                    (y0 + w_over_2_cos_theta, x0 - w_over_2_sin_theta),
                    (y1 + w_over_2_cos_theta, x1 - w_over_2_sin_theta),
                    (y1 - w_over_2_cos_theta, x1 + w_over_2_sin_theta)]

        self.updatePolygon(level, vertices, mask)

    # update thresholds
    def updateBelowThreshold(self, level, threshold, mask=True):
        """Mask/unmask all points whose values are below a threshold.

        :param int level:
        :param float threshold: Threshold
        :param bool mask: True to mask (default), False to unmask.
        """
        self.updateStencil(level,
                           self._value < threshold,
                           mask)

    def updateBetweenThresholds(self, level, min_, max_, mask=True):
        """Mask/unmask all points whose values are in a range.

        :param int level:
        :param float min_: Lower threshold
        :param float max_: Upper threshold
        :param bool mask: True to mask (default), False to unmask.
        """
        stencil = numpy.logical_and(min_ <= self._value,
                                    self._value <= max_)
        self.updateStencil(level, stencil, mask)

    def updateAboveThreshold(self, level, threshold, mask=True):
        """Mask/unmask all points whose values are above a threshold.

        :param int level: Mask level to update.
        :param float threshold: Threshold.
        :param bool mask: True to mask (default), False to unmask.
        """
        self.updateStencil(level,
                           self._value > threshold,
                           mask)


class ScatterMaskToolsWidget(qt.QWidget):
    """Widget with tools for drawing mask on an image in a MaskScatterWidget
    (a PlotWidget with an additional sigActiveScatterChanged signal)."""

    _maxLevelNumber = 255

    def __init__(self, parent=None, plot=None):
        # register if the user as force a color for the corresponding mask level
        self._defaultColors = numpy.ones((self._maxLevelNumber + 1), dtype=numpy.bool)
        # overlays colors set by the user
        self._overlayColors = numpy.zeros((self._maxLevelNumber + 1, 3), dtype=numpy.float32)

        self._plot = plot
        self._maskName = '__MASK_TOOLS_%d' % id(self)  # Legend of the mask

        self._colormap = {
            'name': None,
            'normalization': 'linear',
            'autoscale': False,
            'vmin': 0, 'vmax': self._maxLevelNumber,
            'colors': None}
        self._defaultOverlayColor = rgba('gray')  # Color of the mask
        self._setMaskColors(1, 0.5)

        self._z = 2  # Mask layer in plot
        self._data_scatter = None
        """plot Scatter item for data"""
        self._mask_scatter = None
        """plot Scatter item for representing the mask"""

        self._mask = ScatterMask()
        self._mask.sigChanged.connect(self._updatePlotMask)

        self._drawingMode = None  # Store current drawing mode
        self._lastPencilPos = None

        self._multipleMasks = 'exclusive'

        super(ScatterMaskToolsWidget, self).__init__(parent)

        self._initWidgets()

        self._maskFileDir = qt.QDir.home().absolutePath()

        self.plot.sigInteractiveModeChanged.connect(
            self._interactiveModeChanged)

    def getSelectionMask(self, copy=True):
        """Get the current mask as an array.

        :param bool copy: True (default) to get a copy of the mask.
                          If False, the returned array MUST not be modified.
        :return: The array of the mask with dimension of the 'active' image.
                 If there is no active image, an empty array is returned.
        :rtype: numpy.ndarray of uint8
        """
        return self._mask.getMask(copy=copy)

    def setSelectionMask(self, mask, copy=True):
        """Set the mask to a new array.

        :param numpy.ndarray mask: The array to use for the mask.
        :type mask: numpy.ndarray of uint8, C-contiguous.
                    Array of other types are converted.
        :param bool copy: True (the default) to copy the array,
                          False to use it as is if possible.
        :return: None if failed, shape of mask as 1-tuple if successful.
                 The mask can be cropped or padded to fit active image,
                 the returned shape is that of the scatter data.
        """
        mask = numpy.array(mask, copy=False, dtype=numpy.uint8)

        if self._data_scatter.getXData(copy=False).shape == (0,) \
                or mask.shape == self._data_scatter.getXData(copy=False).shape:
            self._mask.setMask(mask, copy=copy)
            self._mask.commit()
            return mask.shape
        else:
            raise ValueError("Mask does not have the same shape as the data")

    def multipleMasks(self):
        """Return the current mode of multiple masks support.

        See :meth:`setMultipleMasks`
        """
        return self._multipleMasks

    def setMultipleMasks(self, mode):
        """Set the mode of multiple masks support.

        Available modes:

        - 'single': Edit a single level of mask
        - 'exclusive': Supports to 256 levels of non overlapping masks

        :param str mode: The mode to use
        """
        assert mode in ('exclusive', 'single')
        if mode != self._multipleMasks:
            self._multipleMasks = mode
            self.levelWidget.setVisible(self._multipleMasks != 'single')
            self.clearAllBtn.setVisible(self._multipleMasks != 'single')

    @property
    def maskFileDir(self):
        """The directory from which to load/save mask from/to files."""
        if not os.path.isdir(self._maskFileDir):
            self._maskFileDir = qt.QDir.home().absolutePath()
        return self._maskFileDir

    @maskFileDir.setter
    def maskFileDir(self, maskFileDir):
        self._maskFileDir = str(maskFileDir)

    @property
    def plot(self):
        """The :class:`.PlotWindow` this widget is attached to."""
        return self._plot

    def setDirection(self, direction=qt.QBoxLayout.LeftToRight):
        """Set the direction of the layout of the widget

        :param direction: QBoxLayout direction
        """
        self.layout().setDirection(direction)

    def _initWidgets(self):
        """Create widgets"""
        layout = qt.QBoxLayout(qt.QBoxLayout.LeftToRight)
        layout.addWidget(self._initMaskGroupBox())
        layout.addWidget(self._initDrawGroupBox())
        layout.addWidget(self._initThresholdGroupBox())
        layout.addStretch(1)
        self.setLayout(layout)

    @staticmethod
    def _hboxWidget(*widgets, **kwargs):
        """Place widgets in widget with horizontal layout

        :param widgets: Widgets to position horizontally
        :param bool stretch: True for trailing stretch (default),
                             False for no trailing stretch
        :return: A QWidget with a QHBoxLayout
        """
        stretch = kwargs.get('stretch', True)

        layout = qt.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        if stretch:
            layout.addStretch(1)
        widget = qt.QWidget()
        widget.setLayout(layout)
        return widget

    def _initTransparencyWidget(self):
        """ Init the mask transparency widget """
        transparencyWidget = qt.QWidget(self)
        grid = qt.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.transparencySlider = qt.QSlider(qt.Qt.Horizontal, parent=transparencyWidget)
        self.transparencySlider.setRange(3, 10)
        self.transparencySlider.setValue(8)
        self.transparencySlider.setToolTip(
            'Set the transparency of the mask display')
        self.transparencySlider.valueChanged.connect(self._updateColors)
        grid.addWidget(qt.QLabel('Display:', parent=transparencyWidget), 0, 0)
        grid.addWidget(self.transparencySlider, 0, 1, 1, 3)
        grid.addWidget(qt.QLabel('<small><b>Transparent</b></small>', parent=transparencyWidget), 1, 1)
        grid.addWidget(qt.QLabel('<small><b>Opaque</b></small>', parent=transparencyWidget), 1, 3)
        transparencyWidget.setLayout(grid)
        return transparencyWidget

    def _initMaskGroupBox(self):
        """Init general mask operation widgets"""

        # Mask level
        self.levelSpinBox = qt.QSpinBox()
        self.levelSpinBox.setRange(1, self._maxLevelNumber)
        self.levelSpinBox.setToolTip(
            'Choose which mask level is edited.\n'
            'A mask can have up to 255 non-overlapping levels.')
        self.levelSpinBox.valueChanged[int].connect(self._updateColors)
        self.levelWidget = self._hboxWidget(qt.QLabel('Mask level:'),
                                            self.levelSpinBox)
        # Transparency
        self.transparencyWidget = self._initTransparencyWidget()

        # Buttons group
        invertBtn = qt.QPushButton('Invert')
        invertBtn.setShortcut(qt.Qt.CTRL + qt.Qt.Key_I)
        invertBtn.setToolTip('Invert current mask <b>%s</b>' %
                             invertBtn.shortcut().toString())
        invertBtn.clicked.connect(self._handleInvertMask)

        clearBtn = qt.QPushButton('Clear')
        clearBtn.setShortcut(qt.QKeySequence.Delete)
        clearBtn.setToolTip('Clear current mask level <b>%s</b>' %
                            clearBtn.shortcut().toString())
        clearBtn.clicked.connect(self._handleClearMask)

        invertClearWidget = self._hboxWidget(
            invertBtn, clearBtn, stretch=False)

        undoBtn = qt.QPushButton('Undo')
        undoBtn.setShortcut(qt.QKeySequence.Undo)
        undoBtn.setToolTip('Undo last mask change <b>%s</b>' %
                           undoBtn.shortcut().toString())
        self._mask.sigUndoable.connect(undoBtn.setEnabled)
        undoBtn.clicked.connect(self._mask.undo)

        redoBtn = qt.QPushButton('Redo')
        redoBtn.setShortcut(qt.QKeySequence.Redo)
        redoBtn.setToolTip('Redo last undone mask change <b>%s</b>' %
                           redoBtn.shortcut().toString())
        self._mask.sigRedoable.connect(redoBtn.setEnabled)
        redoBtn.clicked.connect(self._mask.redo)

        undoRedoWidget = self._hboxWidget(undoBtn, redoBtn, stretch=False)

        self.clearAllBtn = qt.QPushButton('Clear all')
        self.clearAllBtn.setToolTip('Clear all mask levels')
        self.clearAllBtn.clicked.connect(self.resetSelectionMask)

        loadBtn = qt.QPushButton('Load...')
        loadBtn.clicked.connect(self._loadMask)

        saveBtn = qt.QPushButton('Save...')
        saveBtn.clicked.connect(self._saveMask)

        self.loadSaveWidget = self._hboxWidget(loadBtn, saveBtn, stretch=False)

        layout = qt.QVBoxLayout()
        layout.addWidget(self.levelWidget)
        layout.addWidget(self.transparencyWidget)
        layout.addWidget(invertClearWidget)
        layout.addWidget(undoRedoWidget)
        layout.addWidget(self.clearAllBtn)
        layout.addWidget(self.loadSaveWidget)
        layout.addStretch(1)

        maskGroup = qt.QGroupBox('Mask')
        maskGroup.setLayout(layout)
        return maskGroup

    def _initDrawGroupBox(self):
        """Init drawing tools widgets"""
        layout = qt.QVBoxLayout()

        # Draw tools
        self.browseAction = qt.QAction(
            icons.getQIcon('normal'), 'Browse', None)
        self.browseAction.setShortcut(qt.QKeySequence(qt.Qt.Key_B))
        self.browseAction.setToolTip(
            'Disables drawing tools, enables zooming interaction mode'
            ' <b>B</b>')
        self.browseAction.setCheckable(True)
        self.browseAction.triggered.connect(self._activeBrowseMode)
        self.addAction(self.browseAction)

        self.rectAction = qt.QAction(
            icons.getQIcon('shape-rectangle'), 'Rectangle selection', None)
        self.rectAction.setToolTip(
            'Rectangle selection tool: (Un)Mask a rectangular region <b>R</b>')
        self.rectAction.setShortcut(qt.QKeySequence(qt.Qt.Key_R))
        self.rectAction.setCheckable(True)
        self.rectAction.triggered.connect(self._activeRectMode)
        self.addAction(self.rectAction)

        self.polygonAction = qt.QAction(
            icons.getQIcon('shape-polygon'), 'Polygon selection', None)
        self.polygonAction.setShortcut(qt.QKeySequence(qt.Qt.Key_S))
        self.polygonAction.setToolTip(
            'Polygon selection tool: (Un)Mask a polygonal region <b>S</b><br>'
            'Left-click to place polygon corners<br>'
            'Right-click to place the last corner')
        self.polygonAction.setCheckable(True)
        self.polygonAction.triggered.connect(self._activePolygonMode)
        self.addAction(self.polygonAction)

        self.pencilAction = qt.QAction(
            icons.getQIcon('draw-pencil'), 'Pencil tool', None)
        self.pencilAction.setShortcut(qt.QKeySequence(qt.Qt.Key_P))
        self.pencilAction.setToolTip(
            'Pencil tool: (Un)Mask using a pencil <b>P</b>')
        self.pencilAction.setCheckable(True)
        self.pencilAction.triggered.connect(self._activePencilMode)
        self.addAction(self.polygonAction)

        self.drawActionGroup = qt.QActionGroup(self)
        self.drawActionGroup.setExclusive(True)
        self.drawActionGroup.addAction(self.browseAction)
        self.drawActionGroup.addAction(self.rectAction)
        self.drawActionGroup.addAction(self.polygonAction)
        self.drawActionGroup.addAction(self.pencilAction)

        self.browseAction.setChecked(True)

        self.drawButtons = {}
        for action in self.drawActionGroup.actions():
            btn = qt.QToolButton()
            btn.setDefaultAction(action)
            self.drawButtons[action.text()] = btn
        container = self._hboxWidget(*self.drawButtons.values())
        layout.addWidget(container)

        # Mask/Unmask radio buttons
        maskRadioBtn = qt.QRadioButton('Mask')
        maskRadioBtn.setToolTip(
            'Drawing masks with current level. Press <b>Ctrl</b> to unmask')
        maskRadioBtn.setChecked(True)

        unmaskRadioBtn = qt.QRadioButton('Unmask')
        unmaskRadioBtn.setToolTip(
            'Drawing unmasks with current level. Press <b>Ctrl</b> to mask')

        self.maskStateGroup = qt.QButtonGroup()
        self.maskStateGroup.addButton(maskRadioBtn, 1)
        self.maskStateGroup.addButton(unmaskRadioBtn, 0)

        self.maskStateWidget = self._hboxWidget(maskRadioBtn, unmaskRadioBtn)
        layout.addWidget(self.maskStateWidget)

        # Connect mask state widget visibility with browse action
        self.maskStateWidget.setHidden(self.browseAction.isChecked())
        self.browseAction.toggled[bool].connect(
            self.maskStateWidget.setHidden)

        # Pencil settings
        self.pencilSetting = self._createPencilSettings(None)
        self.pencilSetting.setVisible(False)
        layout.addWidget(self.pencilSetting)

        layout.addStretch(1)

        drawGroup = qt.QGroupBox('Draw tools')
        drawGroup.setLayout(layout)
        return drawGroup

    def _createPencilSettings(self, parent=None):
        pencilSetting = qt.QWidget(parent)

        self.pencilSpinBox = qt.QSpinBox(parent=pencilSetting)
        self.pencilSpinBox.setRange(1, 1024)
        pencilToolTip = """Set pencil drawing tool size in pixels of the image
            on which to make the mask."""
        self.pencilSpinBox.setToolTip(pencilToolTip)

        self.pencilSlider = qt.QSlider(qt.Qt.Horizontal, parent=pencilSetting)
        self.pencilSlider.setRange(1, 50)
        self.pencilSlider.setToolTip(pencilToolTip)

        pencilLabel = qt.QLabel('Pencil size:', parent=pencilSetting)

        layout = qt.QGridLayout()
        layout.addWidget(pencilLabel, 0, 0)
        layout.addWidget(self.pencilSpinBox, 0, 1)
        layout.addWidget(self.pencilSlider, 1, 1)
        pencilSetting.setLayout(layout)

        self.pencilSpinBox.valueChanged.connect(self._pencilWidthChanged)
        self.pencilSlider.valueChanged.connect(self._pencilWidthChanged)

        return pencilSetting

    def _initThresholdGroupBox(self):
        """Init thresholding widgets"""
        layout = qt.QVBoxLayout()

        # Thresholing

        self.belowThresholdAction = qt.QAction(
            icons.getQIcon('plot-roi-below'), 'Mask below threshold', None)
        self.belowThresholdAction.setToolTip(
            'Mask image where values are below given threshold')
        self.belowThresholdAction.setCheckable(True)
        self.belowThresholdAction.triggered[bool].connect(
            self._belowThresholdActionTriggered)

        self.betweenThresholdAction = qt.QAction(
            icons.getQIcon('plot-roi-between'), 'Mask within range', None)
        self.betweenThresholdAction.setToolTip(
            'Mask image where values are within given range')
        self.betweenThresholdAction.setCheckable(True)
        self.betweenThresholdAction.triggered[bool].connect(
            self._betweenThresholdActionTriggered)

        self.aboveThresholdAction = qt.QAction(
            icons.getQIcon('plot-roi-above'), 'Mask above threshold', None)
        self.aboveThresholdAction.setToolTip(
            'Mask image where values are above given threshold')
        self.aboveThresholdAction.setCheckable(True)
        self.aboveThresholdAction.triggered[bool].connect(
            self._aboveThresholdActionTriggered)

        self.thresholdActionGroup = qt.QActionGroup(self)
        self.thresholdActionGroup.setExclusive(False)
        self.thresholdActionGroup.addAction(self.belowThresholdAction)
        self.thresholdActionGroup.addAction(self.betweenThresholdAction)
        self.thresholdActionGroup.addAction(self.aboveThresholdAction)
        self.thresholdActionGroup.triggered.connect(
            self._thresholdActionGroupTriggered)

        self.loadColormapRangeAction = qt.QAction(
            icons.getQIcon('view-refresh'), 'Set min-max from colormap', None)
        self.loadColormapRangeAction.setToolTip(
            'Set min and max values from current colormap range')
        self.loadColormapRangeAction.setCheckable(False)
        self.loadColormapRangeAction.triggered.connect(
            self._loadRangeFromColormapTriggered)

        widgets = []
        for action in self.thresholdActionGroup.actions():
            btn = qt.QToolButton()
            btn.setDefaultAction(action)
            widgets.append(btn)

        spacer = qt.QWidget()
        spacer.setSizePolicy(qt.QSizePolicy.Expanding,
                             qt.QSizePolicy.Preferred)
        widgets.append(spacer)

        loadColormapRangeBtn = qt.QToolButton()
        loadColormapRangeBtn.setDefaultAction(self.loadColormapRangeAction)
        widgets.append(loadColormapRangeBtn)

        container = self._hboxWidget(*widgets, stretch=False)
        layout.addWidget(container)

        form = qt.QFormLayout()

        self.minLineEdit = qt.QLineEdit()
        self.minLineEdit.setText('0')
        self.minLineEdit.setValidator(qt.QDoubleValidator())
        self.minLineEdit.setEnabled(False)
        form.addRow('Min:', self.minLineEdit)

        self.maxLineEdit = qt.QLineEdit()
        self.maxLineEdit.setText('0')
        self.maxLineEdit.setValidator(qt.QDoubleValidator())
        self.maxLineEdit.setEnabled(False)
        form.addRow('Max:', self.maxLineEdit)

        self.applyMaskBtn = qt.QPushButton('Apply mask')
        self.applyMaskBtn.clicked.connect(self._maskBtnClicked)
        self.applyMaskBtn.setEnabled(False)
        form.addRow(self.applyMaskBtn)

        self.maskNanBtn = qt.QPushButton('Mask not finite values')
        self.maskNanBtn.setToolTip('Mask Not a Number and infinite values')
        self.maskNanBtn.clicked.connect(self._maskNotFiniteBtnClicked)
        form.addRow(self.maskNanBtn)

        thresholdWidget = qt.QWidget()
        thresholdWidget.setLayout(form)
        layout.addWidget(thresholdWidget)

        layout.addStretch(1)

        self.thresholdGroup = qt.QGroupBox('Threshold')
        self.thresholdGroup.setLayout(layout)
        return self.thresholdGroup

    # Handle mask refresh on the plot

    def _updatePlotMask(self):
        """Update mask image in plot"""
        mask = self.getSelectionMask(copy=False)
        if len(mask):
            self.plot.addScatter(self._data_scatter.getXData(),
                                 self._data_scatter.getYData(),
                                 mask,  # Fixme: colormap
                                 legend=self._maskName,
                                 colormap=self._colormap,
                                 z=self._z)
            self._mask_scatter = self.plot._getItem(kind="scatter",
                                                    legend=self._maskName)
        elif self.plot._getItem(kind="scatter",
                                legend=self._maskName) is not None:
            self.plot.remove(self._maskName, kind='scatter')

    # track widget visibility and plot active image changes

    def changeEvent(self, event):
        """Reset drawing action when disabling widget"""
        if (event.type() == qt.QEvent.EnabledChange and
                not self.isEnabled() and
                not self.browseAction.isChecked()):
            self.browseAction.trigger()  # Disable drawing tool

    def showEvent(self, event):
        try:
            self.plot.sigActiveScatterChanged.disconnect(
                self._activeScatterChangedAfterCare)
        except (RuntimeError, TypeError):
            pass
        self._activeScatterChanged()   # Init mask + enable/disable widget
        self.plot.sigActiveScatterChanged.connect(self._activeScatterChanged)

    def hideEvent(self, event):
        self.plot.sigActiveScatterChanged.disconnect(self._activeScatterChanged)
        if not self.browseAction.isChecked():
            self.browseAction.trigger()  # Disable drawing tool

        if len(self.getSelectionMask(copy=False)):
            self.plot.sigActiveScatterChanged.connect(
                self._activeScatterChangedAfterCare)

    def _activeScatterChangedAfterCare(self):
        """Check synchro of active scatter and mask when mask widget is hidden.

        If active image has no more the same size as the mask, the mask is
        removed, otherwise it is adjusted to z.
        """
        # check that content changed was the active scatter
        activeScatter = self.plot.getScatter()

        if activeScatter is None or activeScatter.getLegend() == self._maskName:
            # No active scatter or active scatter is the mask...
            self.plot.sigActiveScatterChanged.disconnect(
                self._activeScatterChangedAfterCare)
        else:
            colormap = activeScatter.getColormap()
            self._defaultOverlayColor = rgba(cursorColorForColormap(colormap['name']))
            self._setMaskColors(self.levelSpinBox.value(),
                                self.transparencySlider.value() /
                                self.transparencySlider.maximum())

            self._z = activeScatter.getZValue() + 1
            self._data_scatter = activeScatter
            if self._data_scatter.getXData(copy=False).shape != self.getSelectionMask(copy=False).shape:
                # scatter has not the same size, remove mask and stop listening
                if self.plot._getItem(kind="scatter", legend=self._maskName):
                    self.plot.remove(self._maskName, kind='scatter')

                self.plot.sigActiveScatterChanged.disconnect(
                    self._activeScatterChangedAfterCare)
            else:
                # Refresh in case z changed
                self._mask.setScatter(self._data_scatter)
                self._updatePlotMask()

    def _activeScatterChanged(self):
        """Update widget and mask according to active scatter changes"""
        activeScatter = self.plot.getScatter()
        if activeScatter is None or activeScatter.getLegend() == self._maskName:
            # No active image or active image is the mask...
            self.setEnabled(False)

            self._data_scatter = None
            self._mask.reset()
            self._mask.commit()

        else:  # There is an active scatter
            self.setEnabled(True)

            colormap = activeScatter.getColormap()
            self._defaultOverlayColor = rgba(cursorColorForColormap(colormap['name']))
            self._setMaskColors(self.levelSpinBox.value(),
                                self.transparencySlider.value() /
                                self.transparencySlider.maximum())

            self._z = activeScatter.getZValue() + 1
            self._data_scatter = activeScatter
            if self._data_scatter.getXData(copy=False).shape != self.getSelectionMask(copy=False).shape:
                self._mask.reset(self._data_scatter.getXData(copy=False).shape)  # cp
                self._mask.setScatter(self._data_scatter)
                self._mask.commit()
            else:
                # Refresh in case z changed
                self._updatePlotMask()

        self._updateInteractiveMode()

    # Handle whole mask operations    # PK cp (checkpoint)

    def load(self, filename):
        """Load a mask from an image file.

        :param str filename: File name from which to load the mask
        :raise Exception: An exception in case of failure
        :raise RuntimeWarning: In case the mask was applied but with some
            import changes to notice
        """
        # TODO: 1D masks
        pass
        # _, extension = os.path.splitext(filename)
        # extension = extension.lower()[1:]
        #
        # if extension == "npy":
        #     try:
        #         mask = numpy.load(filename)
        #     except IOError:
        #         _logger.error("Can't load filename '%s'", filename)
        #         _logger.debug("Backtrace", exc_info=True)
        #         raise RuntimeError('File "%s" is not a numpy file.', filename)
        # elif extension == "edf":
        #     try:
        #         mask = EdfFile(filename, access='r').GetData(0)
        #     except Exception as e:
        #         _logger.error("Can't load filename %s", filename)
        #         _logger.debug("Backtrace", exc_info=True)
        #         raise e
        # elif extension == "msk":
        #     if fabio is None:
        #         raise ImportError("Fit2d mask files can't be read: Fabio module is not available")
        #     try:
        #         mask = fabio.open(filename).data
        #     except Exception as e:
        #         _logger.error("Can't load fit2d mask file")
        #         _logger.debug("Backtrace", exc_info=True)
        #         raise e
        # else:
        #     msg = "Extension '%s' is not supported."
        #     raise RuntimeError(msg % extension)
        #
        # effectiveMaskShape = self.setSelectionMask(mask, copy=False)
        # if effectiveMaskShape is None:
        #     return
        # if mask.shape != effectiveMaskShape:
        #     msg = 'Mask was resized from %s to %s'
        #     msg = msg % (str(mask.shape), str(effectiveMaskShape))
        #     raise RuntimeWarning(msg)

    def _loadMask(self):
        """Open load mask dialog"""
        pass   # todo

        # dialog = qt.QFileDialog(self)
        # dialog.setWindowTitle("Load Mask")
        # dialog.setModal(1)
        # filters = [
        #     'EDF (*.edf)',
        #     'TIFF (*.tif)',
        #     'NumPy binary file (*.npy)',
        #     # Fit2D mask is displayed anyway fabio is here or not
        #     # to show to the user that the option exists
        #     'Fit2D mask (*.msk)',
        # ]
        # dialog.setNameFilters(filters)
        # dialog.setFileMode(qt.QFileDialog.ExistingFile)
        # dialog.setDirectory(self.maskFileDir)
        # if not dialog.exec_():
        #     dialog.close()
        #     return
        #
        # filename = dialog.selectedFiles()[0]
        # dialog.close()
        #
        # self.maskFileDir = os.path.dirname(filename)
        # try:
        #     self.load(filename)
        # except RuntimeWarning as e:
        #     message = e.args[0]
        #     msg = qt.QMessageBox(self)
        #     msg.setIcon(qt.QMessageBox.Warning)
        #     msg.setText("Mask loaded but an operation was applied.\n" + message)
        #     msg.exec_()
        # except Exception as e:
        #     message = e.args[0]
        #     msg = qt.QMessageBox(self)
        #     msg.setIcon(qt.QMessageBox.Critical)
        #     msg.setText("Cannot load mask from file. " + message)
        #     msg.exec_()

    def save(self, filename, kind):
        """Save current mask in a file

        :param str filename: The file where to save to mask
        :param str kind: The kind of file to save in 'edf', 'tif', 'npy'
        :raise Exception: Raised if the process fails
        """
        self._mask.save(filename, kind)

    def _saveMask(self):
        """Open Save mask dialog"""
        pass
        # dialog = qt.QFileDialog(self)
        # dialog.setWindowTitle("Save Mask")
        # dialog.setModal(1)
        # filters = [
        #     'EDF (*.edf)',
        #     'TIFF (*.tif)',
        #     'NumPy binary file (*.npy)',
        #     # Fit2D mask is displayed anyway fabio is here or not
        #     # to show to the user that the option exists
        #     'Fit2D mask (*.msk)',
        # ]
        # dialog.setNameFilters(filters)
        # dialog.setFileMode(qt.QFileDialog.AnyFile)
        # dialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        # dialog.setDirectory(self.maskFileDir)
        # if not dialog.exec_():
        #     dialog.close()
        #     return
        #
        # # convert filter name to extension name with the .
        # extension = dialog.selectedNameFilter().split()[-1][2:-1]
        # filename = dialog.selectedFiles()[0]
        # dialog.close()
        #
        # if not filename.lower().endswith(extension):
        #     filename += extension
        #
        # if os.path.exists(filename):
        #     try:
        #         os.remove(filename)
        #     except IOError:
        #         msg = qt.QMessageBox(self)
        #         msg.setIcon(qt.QMessageBox.Critical)
        #         msg.setText("Cannot save.\n"
        #                     "Input Output Error: %s" % (sys.exc_info()[1]))
        #         msg.exec_()
        #         return
        #
        # self.maskFileDir = os.path.dirname(filename)
        # try:
        #     self.save(filename, extension[1:])
        # except Exception as e:
        #     msg = qt.QMessageBox(self)
        #     msg.setIcon(qt.QMessageBox.Critical)
        #     msg.setText("Cannot save file %s\n%s" % (filename, e.args[0]))
        #     msg.exec_()

    def getCurrentMaskColor(self):
        """Returns the color of the current selected level.

        :rtype: A tuple or a python array
        """
        currentLevel = self.levelSpinBox.value()
        if self._defaultColors[currentLevel]:
            return self._defaultOverlayColor
        else:
            return self._overlayColors[currentLevel].tolist()

    def _setMaskColors(self, level, alpha):
        """Set-up the mask colormap to highlight current mask level.

        :param int level: The mask level to highlight
        :param float alpha: Alpha level of mask in [0., 1.]
        """
        assert 0 < level <= self._maxLevelNumber

        colors = numpy.empty((self._maxLevelNumber + 1, 4), dtype=numpy.float32)

        # Set color
        colors[:, :3] = self._defaultOverlayColor[:3]

        # check if some colors has been directly set by the user
        mask = numpy.equal(self._defaultColors, False)
        colors[mask, :3] = self._overlayColors[mask, :3]

        # Set alpha
        colors[:, -1] = alpha / 2.

        # Set highlighted level color
        colors[level, 3] = alpha

        # Set no mask level
        colors[0] = (0., 0., 0., 0.)

        self._colormap['colors'] = colors

    def resetMaskColors(self, level=None):
        """Reset the mask color at the given level to be defaultColors

        :param level:
            The index of the mask for which we want to reset the color.
            If none we will reset color for all masks.
        """
        if level is None:
            self._defaultColors[level] = True
        else:
            self._defaultColors[:] = True

        self._updateColors()

    def setMaskColors(self, rgb, level=None):
        """Set the masks color

        :param rgb: The rgb color
        :param level:
            The index of the mask for which we want to change the color.
            If none set this color for all the masks
        """
        if level is None:
            self._overlayColors[:] = rgb
            self._defaultColors[:] = False
        else:
            self._overlayColors[level] = rgb
            self._defaultColors[level] = False

        self._updateColors()

    def getMaskColors(self):
        """masks colors getter"""
        return self._overlayColors

    def _updateColors(self, *args):
        """Rebuild mask colormap when selected level or transparency change"""
        self._setMaskColors(self.levelSpinBox.value(),
                            self.transparencySlider.value() /
                            self.transparencySlider.maximum())
        self._updatePlotMask()
        self._updateInteractiveMode()

    def _pencilWidthChanged(self, width):

        old = self.pencilSpinBox.blockSignals(True)
        try:
            self.pencilSpinBox.setValue(width)
        finally:
            self.pencilSpinBox.blockSignals(old)

        old = self.pencilSlider.blockSignals(True)
        try:
            self.pencilSlider.setValue(width)
        finally:
            self.pencilSlider.blockSignals(old)
        self._updateInteractiveMode()

    def _updateInteractiveMode(self):
        """Update the current mode to the same if some cached data have to be
        updated. It is the case for the color for example.
        """
        if self._drawingMode == 'rectangle':
            self._activeRectMode()
        elif self._drawingMode == 'polygon':
            self._activePolygonMode()
        elif self._drawingMode == 'pencil':
            self._activePencilMode()

    def _handleClearMask(self):
        """Handle clear button clicked: reset current level mask"""
        self._mask.clear(self.levelSpinBox.value())
        self._mask.commit()

    def resetSelectionMask(self):
        """Reset the mask"""
        self._mask.reset(
                shape=self._data_scatter.getXData(copy=False).shape)
        self._mask.commit()

    def _handleInvertMask(self):
        """Invert the current mask level selection."""
        self._mask.invert(self.levelSpinBox.value())
        self._mask.commit()

    # Handle drawing tools UI events

    def _interactiveModeChanged(self, source):
        """Handle plot interactive mode changed:

        If changed from elsewhere, disable drawing tool
        """
        if source is not self:
            # Do not trigger browseAction to avoid to call
            # self.plot.setInteractiveMode
            self.browseAction.setChecked(True)
            self._releaseDrawingMode()

    def _releaseDrawingMode(self):
        """Release the drawing mode if is was used"""
        if self._drawingMode is None:
            return
        self.plot.sigPlotSignal.disconnect(self._plotDrawEvent)
        self._drawingMode = None

    def _activeBrowseMode(self):
        """Handle browse action mode triggered by user.

        Set plot interactive mode only when
        the user is triggering the browse action.
        """
        self._releaseDrawingMode()
        self.plot.setInteractiveMode('zoom', source=self)
        self._updateDrawingModeWidgets()

    def _activeRectMode(self):
        """Handle rect action mode triggering"""
        self._releaseDrawingMode()
        self._drawingMode = 'rectangle'
        self.plot.sigPlotSignal.connect(self._plotDrawEvent)
        color = self.getCurrentMaskColor()
        self.plot.setInteractiveMode(
            'draw', shape='rectangle', source=self, color=color)
        self._updateDrawingModeWidgets()

    def _activePolygonMode(self):
        """Handle polygon action mode triggering"""
        self._releaseDrawingMode()
        self._drawingMode = 'polygon'
        self.plot.sigPlotSignal.connect(self._plotDrawEvent)
        color = self.getCurrentMaskColor()
        self.plot.setInteractiveMode('draw', shape='polygon', source=self, color=color)
        self._updateDrawingModeWidgets()

    def _activePencilMode(self):
        """Handle pencil action mode triggering"""
        self._releaseDrawingMode()
        self._drawingMode = 'pencil'
        self.plot.sigPlotSignal.connect(self._plotDrawEvent)
        color = self.getCurrentMaskColor()
        width = self.pencilSpinBox.value()
        self.plot.setInteractiveMode(
            'draw', shape='pencil', source=self, color=color, width=width)
        self._updateDrawingModeWidgets()

    def _updateDrawingModeWidgets(self):
        self.pencilSetting.setVisible(self._drawingMode == 'pencil')

    # Handle plot drawing events

    def _isMasking(self):
        """Returns true if the tool is used for masking, else it is used for
        unmasking.

        :rtype: bool"""
        # First draw event, use current modifiers for all draw sequence
        doMask = (self.maskStateGroup.checkedId() == 1)
        if qt.QApplication.keyboardModifiers() & qt.Qt.ControlModifier:
            doMask = not doMask
        return doMask

    def _plotDrawEvent(self, event):
        """Handle draw events from the plot"""
        if (self._drawingMode is None or
                event['event'] not in ('drawingProgress', 'drawingFinished')):
            return

        if not len(self._data_scatter.getXData(copy=False)):
            return

        level = self.levelSpinBox.value()

        if (self._drawingMode == 'rectangle' and
                event['event'] == 'drawingFinished'):
            doMask = self._isMasking()

            self._mask.updateRectangle(
                level,
                y=event['y'],
                x=event['x'],
                height=abs(event['height']),
                width=abs(event['width']),
                mask=doMask)
            self._mask.commit()

        elif (self._drawingMode == 'polygon' and
                event['event'] == 'drawingFinished'):
            doMask = self._isMasking()
            vertices = event['points']
            vertices = vertices.astype(numpy.int)[:, (1, 0)]  # (y, x)
            self._mask.updatePolygon(level, vertices, doMask)
            self._mask.commit()

        elif self._drawingMode == 'pencil':
            doMask = self._isMasking()
            # convert from plot to array coords
            x, y = event['points'][-1]
            brushSize = self.pencilSpinBox.value()

            if self._lastPencilPos != (y, x):
                if self._lastPencilPos is not None:
                    # Draw the line
                    self._mask.updateLine(
                        level,
                        self._lastPencilPos[0], self._lastPencilPos[1],
                        y, x,
                        brushSize,
                        doMask)

                # Draw the very first, or last point
                self._mask.updateDisk(level, y, x, brushSize / 2., doMask)

            if event['event'] == 'drawingFinished':
                self._mask.commit()
                self._lastPencilPos = None
            else:
                self._lastPencilPos = y, x

    # Handle threshold UI events

    def _belowThresholdActionTriggered(self, triggered):
        if triggered:
            self.minLineEdit.setEnabled(True)
            self.maxLineEdit.setEnabled(False)
            self.applyMaskBtn.setEnabled(True)

    def _betweenThresholdActionTriggered(self, triggered):
        if triggered:
            self.minLineEdit.setEnabled(True)
            self.maxLineEdit.setEnabled(True)
            self.applyMaskBtn.setEnabled(True)

    def _aboveThresholdActionTriggered(self, triggered):
        if triggered:
            self.minLineEdit.setEnabled(False)
            self.maxLineEdit.setEnabled(True)
            self.applyMaskBtn.setEnabled(True)

    def _thresholdActionGroupTriggered(self, triggeredAction):
        """Threshold action group listener."""
        if triggeredAction.isChecked():
            # Uncheck other actions
            for action in self.thresholdActionGroup.actions():
                if action is not triggeredAction and action.isChecked():
                    action.setChecked(False)
        else:
            # Disable min/max edit
            self.minLineEdit.setEnabled(False)
            self.maxLineEdit.setEnabled(False)
            self.applyMaskBtn.setEnabled(False)

    def _maskBtnClicked(self):
        data_values = self._data_scatter.getValueData(copy=False)

        if self.belowThresholdAction.isChecked():
            if len(data_values) and self.minLineEdit.text():
                self._mask.updateBelowThreshold(self.levelSpinBox.value(),
                                                float(self.minLineEdit.text()))
                self._mask.commit()

        elif self.betweenThresholdAction.isChecked():
            if (len(data_values) and
                    self.minLineEdit.text() and self.maxLineEdit.text()):
                min_ = float(self.minLineEdit.text())
                max_ = float(self.maxLineEdit.text())
                self._mask.updateBetweenThresholds(self.levelSpinBox.value(),
                                                   min_, max_)
                self._mask.commit()

        elif self.aboveThresholdAction.isChecked():
            if len(data_values) and self.maxLineEdit.text():
                max_ = float(self.maxLineEdit.text())
                self._mask.updateAboveThreshold(self.levelSpinBox.value(),
                                                max_)
                self._mask.commit()

    def _maskNotFiniteBtnClicked(self):
        """Handle not finite mask button clicked: mask NaNs and inf"""
        self._mask.updateStencil(
            self.levelSpinBox.value(),
            numpy.logical_not(
                    numpy.isfinite(self._data_scatter.getValueData(copy=False))))
        self._mask.commit()

    def _loadRangeFromColormapTriggered(self):
        """Set range from active scatter colormap range"""
        activeScatter = self.plot._getActiveItem(kind="scatter")
        if (activeScatter is not None and
                activeScatter.getLegend() != self._maskName):
            # Update thresholds according to colormap
            colormap = activeScatter.getColormap()
            if colormap['autoscale']:
                min_ = numpy.nanmin(activeScatter.getValueData(copy=False))
                max_ = numpy.nanmax(activeScatter.getValueData(copy=False))
            else:
                min_, max_ = colormap['vmin'], colormap['vmax']
            self.minLineEdit.setText(str(min_))
            self.maxLineEdit.setText(str(max_))


class ScatterMaskToolsDockWidget(qt.QDockWidget):
    """:class:`MaskToolsWidget` embedded in a QDockWidget.

    For integration in a :class:`PlotWindow`.

    :param parent: See :class:`QDockWidget`
    :param plot: The PlotWidget this widget is operating on
    :paran str name: The title of this widget
    """

    def __init__(self, parent=None, plot=None, name='Mask'):
        super(ScatterMaskToolsDockWidget, self).__init__(parent)
        self.setWindowTitle(name)

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setWidget(ScatterMaskToolsWidget(plot=plot))
        self.dockLocationChanged.connect(self._dockLocationChanged)
        self.topLevelChanged.connect(self._topLevelChanged)

    def getSelectionMask(self, copy=True):
        """Get the current mask as a 2D array.

        :param bool copy: True (default) to get a copy of the mask.
                          If False, the returned array MUST not be modified.
        :return: The array of the mask with dimension of the 'active' image.
                 If there is no active image, an empty array is returned.
        :rtype: 2D numpy.ndarray of uint8
        """
        return self.widget().getSelectionMask(copy=copy)

    def setSelectionMask(self, mask, copy=True):
        """Set the mask to a new array.

        :param numpy.ndarray mask: The array to use for the mask.
        :type mask: numpy.ndarray of uint8 of dimension 2, C-contiguous.
                    Array of other types are converted.
        :param bool copy: True (the default) to copy the array,
                          False to use it as is if possible.
        :return: None if failed, shape of mask as 2-tuple if successful.
                 The mask can be cropped or padded to fit active image,
                 the returned shape is that of the active image.
        """
        return self.widget().setSelectionMask(mask, copy=copy)

    def toggleViewAction(self):
        """Returns a checkable action that shows or closes this widget.

        See :class:`QMainWindow`.
        """
        action = super(ScatterMaskToolsDockWidget, self).toggleViewAction()
        action.setIcon(icons.getQIcon('image-mask'))
        action.setToolTip("Display/hide mask tools")
        return action

    def _dockLocationChanged(self, area):
        if area in (qt.Qt.LeftDockWidgetArea, qt.Qt.RightDockWidgetArea):
            direction = qt.QBoxLayout.TopToBottom
        else:
            direction = qt.QBoxLayout.LeftToRight
        self.widget().setDirection(direction)

    def _topLevelChanged(self, topLevel):
        if topLevel:
            self.widget().setDirection(qt.QBoxLayout.LeftToRight)
            self.resize(self.widget().minimumSize())
            self.adjustSize()

    def showEvent(self, event):
        """Make sure this widget is raised when it is shown
        (when it is first created as a tab in PlotWindow or when it is shown
        again after hiding).
        """
        self.raise_()
