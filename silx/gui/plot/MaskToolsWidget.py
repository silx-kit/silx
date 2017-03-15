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
"""Widget providing a set of tools to draw masks on a PlotWidget.

This widget is meant to work with :class:`silx.gui.plot.PlotWidget`.

- :class:`Mask`: Handle mask bitmap update and history
- :class:`MaskToolsWidget`: GUI for :class:`Mask`
- :class:`MaskToolsDockWidget`: DockWidget to integrate in :class:`PlotWindow`
"""

from __future__ import division

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__data__ = "08/06/2016"


import os
import sys
import numpy
import logging

from silx.image import shapes
from .Colors import cursorColorForColormap, rgba
from .. import icons, qt

from silx.third_party.EdfFile import EdfFile
from silx.third_party.TiffIO import TiffIO

try:
    import fabio
except ImportError:
    fabio = None


_logger = logging.getLogger(__name__)


class Mask(qt.QObject):
    """A mask field with update operations.

    Coords follows (row, column) convention and are in mask array coords.

    This is meant for internal use by :class:`MaskToolsWidget`.
    """

    sigChanged = qt.Signal()
    """Signal emitted when the mask has changed"""

    sigUndoable = qt.Signal(bool)
    """Signal emitted when undo becomes possible/impossible"""

    sigRedoable = qt.Signal(bool)
    """Signal emitted when redo becomes possible/impossible"""

    def __init__(self):
        self.historyDepth = 10
        """Maximum number of operation stored in history list for undo"""

        self._mask = numpy.array((), dtype=numpy.uint8)  # Store the mask

        # Init lists for undo/redo
        self._history = []
        self._redo = []

        super(Mask, self).__init__()

    def _notify(self):
        """Notify of mask change."""
        self.sigChanged.emit()

    def getMask(self, copy=True):
        """Get the current mask as a 2D array.

        :param bool copy: True (default) to get a copy of the mask.
                          If False, the returned array MUST not be modified.
        :return: The array of the mask with dimension of the 'active' image.
                 If there is no active image, an empty array is returned.
        :rtype: 2D numpy.ndarray of uint8
        """
        return numpy.array(self._mask, copy=copy)

    def setMask(self, mask, copy=True):
        """Set the mask to a new array.

        :param numpy.ndarray mask: The array to use for the mask.
        :type mask: numpy.ndarray of uint8 of dimension 2, C-contiguous.
                    Array of other types are converted.
        :param bool copy: True (the default) to copy the array,
                          False to use it as is if possible.
        """
        assert len(mask.shape) == 2
        self._mask = numpy.array(mask, copy=copy, order='C', dtype=numpy.uint8)
        self._notify()

    def save(self, filename, kind):
        """Save current mask in a file

        :param str filename: The file where to save to mask
        :param str kind: The kind of file to save in 'edf', 'tif', 'npy',
            or 'msk' (if FabIO is installed)
        :raise Exception: Raised if the file writing fail
        """
        if kind == 'edf':
            edfFile = EdfFile(filename, access="w+")
            edfFile.WriteImage({}, self.getMask(copy=False), Append=0)

        elif kind == 'tif':
            tiffFile = TiffIO(filename, mode='w')
            tiffFile.writeImage(self.getMask(copy=False), software='silx')

        elif kind == 'npy':
            try:
                numpy.save(filename, self.getMask(copy=False))
            except IOError:
                raise RuntimeError("Mask file can't be written")

        elif kind == 'msk':
            if fabio is None:
                raise ImportError("Fit2d mask files can't be written: Fabio module is not available")
            try:
                data = self.getMask(copy=False)
                image = fabio.fabioimage.FabioImage(data=data)
                image = image.convert(fabio.fit2dmaskimage.Fit2dMaskImage)
                image.save(filename)
            except Exception:
                _logger.debug("Backtrace", exc_info=True)
                raise RuntimeError("Mask file can't be written")

        else:
            raise ValueError("Format '%s' is not supported" % kind)

    # History control

    def resetHistory(self):
        """Reset history"""
        self._history = [numpy.array(self._mask, copy=True)]
        self._redo = []
        self.sigUndoable.emit(False)
        self.sigRedoable.emit(False)

    def commit(self):
        """Append the current mask to history if changed"""
        if (not self._history or self._redo or
                not numpy.all(numpy.equal(self._mask, self._history[-1]))):
            if self._redo:
                self._redo = []  # Reset redo as a new action as been performed
                self.sigRedoable[bool].emit(False)

            while len(self._history) >= self.historyDepth:
                self._history.pop(0)
            self._history.append(numpy.array(self._mask, copy=True))

            if len(self._history) == 2:
                self.sigUndoable.emit(True)

    def undo(self):
        """Restore previous mask if any"""
        if len(self._history) > 1:
            self._redo.append(self._history.pop())
            self._mask = numpy.array(self._history[-1], copy=True)
            self._notify()  # Do not store this change in history

            if len(self._redo) == 1:  # First redo
                self.sigRedoable.emit(True)
            if len(self._history) == 1:  # Last value in history
                self.sigUndoable.emit(False)

    def redo(self):
        """Restore previously undone modification if any"""
        if self._redo:
            self._mask = self._redo.pop()
            self._history.append(numpy.array(self._mask, copy=True))
            self._notify()

            if not self._redo:  # No more redo
                self.sigRedoable.emit(False)
            if len(self._history) == 2:  # Something to undo
                self.sigUndoable.emit(True)

    # Whole mask operations

    def clear(self, level):
        """Set all values of the given mask level to 0.

        :param int level: Value of the mask to set to 0.
        """
        assert 0 < level < 256
        self._mask[self._mask == level] = 0
        self._notify()

    def reset(self, shape=None):
        """Reset the mask to zero and change its shape

        :param shape: Shape of the new mask or None to have an empty mask
        :type shape: 2-tuple of int
        """
        if shape is None:
            shape = 0, 0  # Empty 2D array
        assert len(shape) == 2

        shapeChanged = (shape != self._mask.shape)
        self._mask = numpy.zeros(shape, dtype=numpy.uint8)
        if shapeChanged:
            self.resetHistory()

        self._notify()

    def invert(self, level):
        """Invert mask of the given mask level.

        0 values become level and level values become 0.

        :param int level: The level to invert.
        """
        assert 0 < level < 256
        masked = self._mask == level
        self._mask[self._mask == 0] = level
        self._mask[masked] = 0
        self._notify()

    # Drawing operations

    def updateRectangle(self, level, row, col, height, width, mask=True):
        """Mask/Unmask a rectangle of the given mask level.

        :param int level: Mask level to update.
        :param int row: Starting row of the rectangle
        :param int col: Starting column of the rectangle
        :param int height:
        :param int width:
        :param bool mask: True to mask (default), False to unmask.
        """
        assert 0 < level < 256
        selection = self._mask[max(0, row):row + height + 1,
                               max(0, col):col + width + 1]
        if mask:
            selection[:, :] = level
        else:
            selection[selection == level] = 0
        self._notify()

    def updatePolygon(self, level, vertices, mask=True):
        """Mask/Unmask a polygon of the given mask level.

        :param int level: Mask level to update.
        :param vertices: Nx2 array of polygon corners as (row, col)
        :param bool mask: True to mask (default), False to unmask.
        """
        fill = shapes.polygon_fill_mask(vertices, self._mask.shape)
        if mask:
            self._mask[fill != 0] = level
        else:
            self._mask[numpy.logical_and(fill != 0,
                                         self._mask == level)] = 0
        self._notify()

    def updatePoints(self, level, rows, cols, mask=True):
        """Mask/Unmask points with given coordinates.

        :param int level: Mask level to update.
        :param rows: Rows of selected points
        :type rows: 1D numpy.ndarray
        :param cols: Columns of selected points
        :type cols: 1D numpy.ndarray
        :param bool mask: True to mask (default), False to unmask.
        """
        valid = numpy.logical_and(
            numpy.logical_and(rows >= 0, cols >= 0),
            numpy.logical_and(rows < self._mask.shape[0],
                              cols < self._mask.shape[1]))
        rows, cols = rows[valid], cols[valid]

        if mask:
            self._mask[rows, cols] = level
        else:
            inMask = self._mask[rows, cols] == level
            self._mask[rows[inMask], cols[inMask]] = 0
        self._notify()

    def updateStencil(self, level, stencil, mask=True):
        """Mask/Unmask area from boolean mask.

        :param int level: Mask level to update.
        :param stencil: Boolean mask of mask values to update
        :type stencil: numpy.array of same dimension as the mask
        :param bool mask: True to mask (default), False to unmask.
        """
        rows, cols = numpy.nonzero(stencil)
        self.updatePoints(level, rows, cols, mask)

    def updateDisk(self, level, crow, ccol, radius, mask=True):
        """Mask/Unmask a disk of the given mask level.

        :param int level: Mask level to update.
        :param int crow: Disk center row.
        :param int ccol: Disk center column.
        :param float radius: Radius of the disk in mask array unit
        :param bool mask: True to mask (default), False to unmask.
        """
        rows, cols = shapes.circle_fill(crow, ccol, radius)
        self.updatePoints(level, rows, cols, mask)

    def updateLine(self, level, row0, col0, row1, col1, width, mask=True):
        """Mask/Unmask a line of the given mask level.

        :param int level: Mask level to update.
        :param int row0: Row of the starting point.
        :param int col0: Column of the starting point.
        :param int row1: Row of the end point.
        :param int col1: Column of the end point.
        :param int width: Width of the line in mask array unit.
        :param bool mask: True to mask (default), False to unmask.
        """
        rows, cols = shapes.draw_line(row0, col0, row1, col1, width)
        self.updatePoints(level, rows, cols, mask)


class MaskToolsWidget(qt.QWidget):
    """Widget with tools for drawing mask on an image in a PlotWidget."""

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

        self._origin = (0., 0.)  # Mask origin in plot
        self._scale = (1., 1.)  # Mask scale in plot
        self._z = 1  # Mask layer in plot
        self._data = numpy.zeros((0, 0), dtype=numpy.uint8)  # Store image

        self._mask = Mask()
        self._mask.sigChanged.connect(self._updatePlotMask)

        self._drawingMode = None  # Store current drawing mode
        self._lastPencilPos = None

        self._multipleMasks = 'exclusive'

        super(MaskToolsWidget, self).__init__(parent)
        self._initWidgets()

        self._maskFileDir = qt.QDir.home().absolutePath()

        self.plot.sigInteractiveModeChanged.connect(
            self._interactiveModeChanged)

    def getSelectionMask(self, copy=True):
        """Get the current mask as a 2D array.

        :param bool copy: True (default) to get a copy of the mask.
                          If False, the returned array MUST not be modified.
        :return: The array of the mask with dimension of the 'active' image.
                 If there is no active image, an empty array is returned.
        :rtype: 2D numpy.ndarray of uint8
        """
        return self._mask.getMask(copy=copy)

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
        mask = numpy.array(mask, copy=False, dtype=numpy.uint8)
        if len(mask.shape) != 2:
            _logger.error('Not an image, shape: %d', len(mask.shape))
            return None

        if self._data.shape == (0, 0) or mask.shape == self._data.shape:
            self._mask.setMask(mask, copy=copy)
            self._mask.commit()
            return mask.shape
        else:
            _logger.warning('Mask has not the same size as current image.'
                            ' Mask will be cropped or padded to fit image'
                            ' dimensions. %s != %s',
                            str(mask.shape), str(self._data.shape))
            resizedMask = numpy.zeros(self._data.shape, dtype=numpy.uint8)
            height = min(self._data.shape[0], mask.shape[0])
            width = min(self._data.shape[1], mask.shape[1])
            resizedMask[:height, :width] = mask[:height, :width]
            self._mask.setMask(resizedMask, copy=False)
            self._mask.commit()
            return resizedMask.shape

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
        clearBtn.setToolTip('Clear current mask <b>%s</b>' %
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
            self.plot.addImage(mask, legend=self._maskName,
                               colormap=self._colormap,
                               origin=self._origin,
                               scale=self._scale,
                               z=self._z,
                               replace=False, resetzoom=False)
        elif self.plot.getImage(self._maskName):
            self.plot.remove(self._maskName, kind='image')

    # track widget visibility and plot active image changes

    def changeEvent(self, event):
        """Reset drawing action when disabling widget"""
        if (event.type() == qt.QEvent.EnabledChange and
                not self.isEnabled() and
                not self.browseAction.isChecked()):
            self.browseAction.trigger()  # Disable drawing tool

    def showEvent(self, event):
        try:
            self.plot.sigActiveImageChanged.disconnect(
                self._activeImageChangedAfterCare)
        except (RuntimeError, TypeError):
            pass
        self._activeImageChanged()  # Init mask + enable/disable widget
        self.plot.sigActiveImageChanged.connect(self._activeImageChanged)

    def hideEvent(self, event):
        self.plot.sigActiveImageChanged.disconnect(self._activeImageChanged)
        if not self.browseAction.isChecked():
            self.browseAction.trigger()  # Disable drawing tool

        if len(self.getSelectionMask(copy=False)):
            self.plot.sigActiveImageChanged.connect(
                self._activeImageChangedAfterCare)

    def _activeImageChangedAfterCare(self, *args):
        """Check synchro of active image and mask when mask widget is hidden.

        If active image has no more the same size as the mask, the mask is
        removed, otherwise it is adjusted to origin, scale and z.
        """
        activeImage = self.plot.getActiveImage()
        if activeImage is None or activeImage.getLegend() == self._maskName:
            # No active image or active image is the mask...
            self.plot.sigActiveImageChanged.disconnect(
                self._activeImageChangedAfterCare)
        else:
            colormap = activeImage.getColormap()
            self._defaultOverlayColor = rgba(cursorColorForColormap(colormap['name']))
            self._setMaskColors(self.levelSpinBox.value(),
                                self.transparencySlider.value() /
                                self.transparencySlider.maximum())

            self._origin = activeImage.getOrigin()
            self._scale = activeImage.getScale()
            self._z = activeImage.getZValue() + 1
            self._data = activeImage.getData(copy=False)
            if self._data.shape != self.getSelectionMask(copy=False).shape:
                # Image has not the same size, remove mask and stop listening
                if self.plot.getImage(self._maskName):
                    self.plot.remove(self._maskName, kind='image')

                self.plot.sigActiveImageChanged.disconnect(
                    self._activeImageChangedAfterCare)
            else:
                # Refresh in case origin, scale, z changed
                self._updatePlotMask()

    def _activeImageChanged(self, *args):
        """Update widget and mask according to active image changes"""
        activeImage = self.plot.getActiveImage()
        if activeImage is None or activeImage.getLegend() == self._maskName:
            # No active image or active image is the mask...
            self.setEnabled(False)

            self._data = numpy.zeros((0, 0), dtype=numpy.uint8)
            self._mask.reset()
            self._mask.commit()

        else:  # There is an active image
            self.setEnabled(True)

            colormap = activeImage.getColormap()
            self._defaultOverlayColor = rgba(cursorColorForColormap(colormap['name']))
            self._setMaskColors(self.levelSpinBox.value(),
                                self.transparencySlider.value() /
                                self.transparencySlider.maximum())

            self._origin = activeImage.getOrigin()
            self._scale = activeImage.getScale()
            self._z = activeImage.getZValue() + 1
            self._data = activeImage.getData(copy=False)
            if self._data.shape != self.getSelectionMask(copy=False).shape:
                self._mask.reset(self._data.shape)
                self._mask.commit()
            else:
                # Refresh in case origin, scale, z changed
                self._updatePlotMask()

        self._updateInteractiveMode()

    # Handle whole mask operations

    def load(self, filename):
        """Load a mask from an image file.

        :param str filename: File name from which to load the mask
        :raise Exception: An exception in case of failure
        :raise RuntimeWarning: In case the mask was applied but with some
            import changes to notice
        """
        _, extension = os.path.splitext(filename)
        extension = extension.lower()[1:]

        if extension == "npy":
            try:
                mask = numpy.load(filename)
            except IOError:
                _logger.error("Can't load filename '%s'", filename)
                _logger.debug("Backtrace", exc_info=True)
                raise RuntimeError('File "%s" is not a numpy file.', filename)
        elif extension == "edf":
            try:
                mask = EdfFile(filename, access='r').GetData(0)
            except Exception as e:
                _logger.error("Can't load filename %s", filename)
                _logger.debug("Backtrace", exc_info=True)
                raise e
        elif extension == "msk":
            if fabio is None:
                raise ImportError("Fit2d mask files can't be read: Fabio module is not available")
            try:
                mask = fabio.open(filename).data
            except Exception as e:
                _logger.error("Can't load fit2d mask file")
                _logger.debug("Backtrace", exc_info=True)
                raise e
        else:
            msg = "Extension '%s' is not supported."
            raise RuntimeError(msg % extension)

        effectiveMaskShape = self.setSelectionMask(mask, copy=False)
        if effectiveMaskShape is None:
            return
        if mask.shape != effectiveMaskShape:
            msg = 'Mask was resized from %s to %s'
            msg = msg % (str(mask.shape), str(effectiveMaskShape))
            raise RuntimeWarning(msg)

    def _loadMask(self):
        """Open load mask dialog"""
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle("Load Mask")
        dialog.setModal(1)
        filters = [
            'EDF (*.edf)',
            'TIFF (*.tif)',
            'NumPy binary file (*.npy)',
            # Fit2D mask is displayed anyway fabio is here or not
            # to show to the user that the option exists
            'Fit2D mask (*.msk)',
        ]
        dialog.setNameFilters(filters)
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        dialog.setDirectory(self.maskFileDir)
        if not dialog.exec_():
            dialog.close()
            return

        filename = dialog.selectedFiles()[0]
        dialog.close()

        self.maskFileDir = os.path.dirname(filename)
        try:
            self.load(filename)
        except RuntimeWarning as e:
            message = e.args[0]
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Warning)
            msg.setText("Mask loaded but an operation was applied.\n" + message)
            msg.exec_()
        except Exception as e:
            message = e.args[0]
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Cannot load mask from file. " + message)
            msg.exec_()

    def save(self, filename, kind):
        """Save current mask in a file

        :param str filename: The file where to save to mask
        :param str kind: The kind of file to save in 'edf', 'tif', 'npy'
        :raise Exception: Raised if the process fails
        """
        self._mask.save(filename, kind)

    def _saveMask(self):
        """Open Save mask dialog"""
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle("Save Mask")
        dialog.setModal(1)
        filters = [
            'EDF (*.edf)',
            'TIFF (*.tif)',
            'NumPy binary file (*.npy)',
            # Fit2D mask is displayed anyway fabio is here or not
            # to show to the user that the option exists
            'Fit2D mask (*.msk)',
        ]
        dialog.setNameFilters(filters)
        dialog.setFileMode(qt.QFileDialog.AnyFile)
        dialog.setAcceptMode(qt.QFileDialog.AcceptSave)
        dialog.setDirectory(self.maskFileDir)
        if not dialog.exec_():
            dialog.close()
            return

        # convert filter name to extension name with the .
        extension = dialog.selectedNameFilter().split()[-1][2:-1]
        filename = dialog.selectedFiles()[0]
        dialog.close()

        if not filename.lower().endswith(extension):
            filename += extension

        if os.path.exists(filename):
            try:
                os.remove(filename)
            except IOError:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Cannot save.\n"
                            "Input Output Error: %s" % (sys.exc_info()[1]))
                msg.exec_()
                return

        self.maskFileDir = os.path.dirname(filename)
        try:
            self.save(filename, extension[1:])
        except Exception as e:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Cannot save file %s\n%s" % (filename, e.args[0]))
            msg.exec_()

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
        self._mask.reset(shape=self._data.shape)
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

        if not len(self._data):
            return

        level = self.levelSpinBox.value()

        if (self._drawingMode == 'rectangle' and
                event['event'] == 'drawingFinished'):
            # Convert from plot to array coords
            doMask = self._isMasking()
            ox, oy = self._origin
            sx, sy = self._scale

            height = int(abs(event['height'] / sy))
            width = int(abs(event['width'] / sx))

            row = int((event['y'] - oy) / sy)
            if sy < 0:
                row -= height

            col = int((event['x'] - ox) / sx)
            if sx < 0:
                col -= width

            self._mask.updateRectangle(
                level,
                row=row,
                col=col,
                height=height,
                width=width,
                mask=doMask)
            self._mask.commit()

        elif (self._drawingMode == 'polygon' and
                event['event'] == 'drawingFinished'):
            doMask = self._isMasking()
            # Convert from plot to array coords
            vertices = (event['points'] - self._origin) / self._scale
            vertices = vertices.astype(numpy.int)[:, (1, 0)]  # (row, col)
            self._mask.updatePolygon(level, vertices, doMask)
            self._mask.commit()

        elif self._drawingMode == 'pencil':
            doMask = self._isMasking()
            # convert from plot to array coords
            col, row = (event['points'][-1] - self._origin) / self._scale
            col, row = int(col), int(row)
            brushSize = self.pencilSpinBox.value()

            if self._lastPencilPos != (row, col):
                if self._lastPencilPos is not None:
                    # Draw the line
                    self._mask.updateLine(
                        level,
                        self._lastPencilPos[0], self._lastPencilPos[1],
                        row, col,
                        brushSize,
                        doMask)

                # Draw the very first, or last point
                self._mask.updateDisk(level, row, col, brushSize / 2., doMask)

            if event['event'] == 'drawingFinished':
                self._mask.commit()
                self._lastPencilPos = None
            else:
                self._lastPencilPos = row, col

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
        if self.belowThresholdAction.isChecked():
            if len(self._data) and self.minLineEdit.text():
                min_ = float(self.minLineEdit.text())
                self._mask.updateStencil(self.levelSpinBox.value(),
                                         self._data < min_)
                self._mask.commit()

        elif self.betweenThresholdAction.isChecked():
            if (len(self._data) and
                    self.minLineEdit.text() and self.maxLineEdit.text()):
                min_ = float(self.minLineEdit.text())
                max_ = float(self.maxLineEdit.text())
                self._mask.updateStencil(self.levelSpinBox.value(),
                                         numpy.logical_and(min_ <= self._data,
                                                           self._data <= max_))
                self._mask.commit()

        elif self.aboveThresholdAction.isChecked():
            if len(self._data) and self.maxLineEdit.text():
                max_ = float(self.maxLineEdit.text())
                self._mask.updateStencil(self.levelSpinBox.value(),
                                         self._data > max_)
                self._mask.commit()

    def _maskNotFiniteBtnClicked(self):
        """Handle not finite mask button clicked: mask NaNs and inf"""
        self._mask.updateStencil(
            self.levelSpinBox.value(),
            numpy.logical_not(numpy.isfinite(self._data)))
        self._mask.commit()

    def _loadRangeFromColormapTriggered(self):
        """Set range from active image colormap range"""
        activeImage = self.plot.getActiveImage()
        if (activeImage is not None and
                activeImage.getLegend() != self._maskName):
            # Update thresholds according to colormap
            colormap = activeImage.getColormap()
            if colormap['autoscale']:
                min_ = numpy.nanmin(activeImage.getData(copy=False))
                max_ = numpy.nanmax(activeImage.getData(copy=False))
            else:
                min_, max_ = colormap['vmin'], colormap['vmax']
            self.minLineEdit.setText(str(min_))
            self.maxLineEdit.setText(str(max_))


class MaskToolsDockWidget(qt.QDockWidget):
    """:class:`MaskToolsDockWidget` embedded in a QDockWidget.

    For integration in a :class:`PlotWindow`.

    :param parent: See :class:`QDockWidget`
    :param plot: The PlotWidget this widget is operating on
    :paran str name: The title of this widget
    """

    def __init__(self, parent=None, plot=None, name='Mask'):
        super(MaskToolsDockWidget, self).__init__(parent)
        self.setWindowTitle(name)

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.setWidget(MaskToolsWidget(plot=plot))
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
        action = super(MaskToolsDockWidget, self).toggleViewAction()
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
