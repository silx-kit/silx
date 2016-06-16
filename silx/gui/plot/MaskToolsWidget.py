# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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

This widget is meant to work with :class:`PlotWidget`.
"""

from __future__ import division

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__data__ = "08/06/2016"


import numpy
import logging

from silx.image import shapes
from .. import icons, qt

_logger = logging.getLogger(__name__)


# TODO: choose mask color depending on image colormap
# TODO: change pencil interaction to drag
# TODO use a flow layout? or change orientation depending on dock area
# TODO get browse action from PlotWindow? anyway sync with other tools


class Mask(qt.QObject):
    """A mask field with update operations.

    Coords follows (row, column) convention and are in mask array coords.
    """

    sigChanged = qt.Signal()
    """Signal emitted when the mask has changed"""

    def __init__(self):
        self._mask = numpy.array((), dtype=numpy.uint8)  # Store the mask
        super(Mask, self).__init__()

    def _notify(self):
        """Notify of mask changed."""
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

    def clear(self, level):
        """Set all values of the given mask level to 0.

        :param int level: Value of the mask to set to 0.
        """
        assert level > 0 and level < 256
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
        self._mask = numpy.zeros(shape, dtype=numpy.uint8)
        self._notify()

    def invert(self, level):
        """Invert mask of the given mask level.

        0 values become level and level values become 0.

        :param int level: The level to invert.
        """
        assert level > 0 and level < 256
        masked = self._mask == level
        self._mask[self._mask == 0] = level
        self._mask[masked] = 0
        self._notify()

    def updateRectangle(self, level, row, col, height, width, mask=True):
        """Mask/Unmask a rectangle of the given mask level.

        :param int level: Mask level to update.
        :param int row: Starting row of the rectangle
        :param int col: Starting column of the rectangle
        :param int height:
        :param int width:
        :param bool mask: True to mask (default), False to unmask.
        """
        assert level > 0 and level < 256
        selection = self._mask[max(0, row):row+height+1,
                               max(0, col):col+width+1]
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

    def __init__(self, plot, parent=None):
        self._plot = plot
        self._maskName = '__MASK_TOOLS_%d' % id(self)  # Legend of the mask

        self._colormap = {
            'name': None,
            'normalization': 'linear',
            'autoscale': False,
            'vmin': 0, 'vmax': 255,
            'colors': None}
        self._setMaskColors(1, True)

        self._origin = (0., 0.)  # Mask origin in plot
        self._scale = (1., 1.)  # Mask scale in plot
        self._z = 1  # Mask layer in plot
        self._data = numpy.zeros((0, 0), dtype=numpy.uint8)  # Store image

        self._mask = Mask()
        self._mask.sigChanged.connect(self._updatePlotMask)

        self._drawingMode = None  # Store current drawing mode
        self._lastPencilPos = None

        super(MaskToolsWidget, self).__init__(parent)
        self._initWidgets()

    def getMask(self, copy=True):
        """Get the current mask as a 2D array.

        :param bool copy: True (default) to get a copy of the mask.
                          If False, the returned array MUST not be modified.
        :return: The array of the mask with dimension of the 'active' image.
                 If there is no active image, an empty array is returned.
        :rtype: 2D numpy.ndarray of uint8
        """
        return self._mask.getMask(copy=copy)

    @property
    def plot(self):
        """The :class:`.PlotWindow` this widget is attached to."""
        return self._plot

    def _initWidgets(self):
        """Create widgets"""
        layout = qt.QBoxLayout(qt.QBoxLayout.LeftToRight)
        layout.addWidget(self._initMaskGroupBox())
        layout.addWidget(self._initDrawGroupBox())
        layout.addWidget(self._initThresholdGroupBox())
        layout.addStretch(1)
        self.setLayout(layout)

    def _hboxWidget(self, *widgets):
        """Place widgets in widget with horizontal layout

        :params widgets: Widgets to position horizontally
        :return: A QWidget with a QHBoxLayout
        """
        layout = qt.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        for widget in widgets:
            layout.addWidget(widget)
        layout.addStretch(1)
        widget = qt.QWidget()
        widget.setLayout(layout)
        return widget

    def _initMaskGroupBox(self):
        """Init general mask operation widgets"""
        self.levelSpinBox = qt.QSpinBox()
        self.levelSpinBox.setRange(1, 255)
        self.levelSpinBox.setToolTip(
            'Choose which mask level is edited.\n'
            'A mask can have up to 255 non-overlapping levels.')
        self.levelSpinBox.valueChanged[int].connect(self._updateColors)
        levelWidget = self._hboxWidget(qt.QLabel('Mask level:'),
                                       self.levelSpinBox)

        self.transparencyCheckBox = qt.QCheckBox('Transparent display')
        self.transparencyCheckBox.setToolTip(
            'Toggle between transparent and opaque masks display')
        self.transparencyCheckBox.setChecked(True)
        self.transparencyCheckBox.toggled[bool].connect(self._updateColors)

        invertBtn = qt.QPushButton('Invert Level')
        invertBtn.setShortcut(qt.Qt.CTRL + qt.Qt.Key_I)
        invertBtn.setToolTip('Invert current mask level <b>%s</b>' %
                             invertBtn.shortcut().toString())
        invertBtn.clicked.connect(self._handleInvertMask)

        clearBtn = qt.QPushButton('Clear Level')
        clearBtn.setShortcut(qt.QKeySequence.Delete)
        clearBtn.setToolTip('Clear current mask level <b>%s</b>' %
                            clearBtn.shortcut().toString())
        clearBtn.clicked.connect(self._handleClearMask)

        clearAllBtn = qt.QPushButton('Clear All')
        clearAllBtn.setToolTip('Clear all mask levels')
        clearAllBtn.clicked.connect(self.resetMask)

        layout = qt.QVBoxLayout()
        layout.addWidget(levelWidget)
        layout.addWidget(self.transparencyCheckBox)
        layout.addWidget(invertBtn)
        layout.addWidget(clearBtn)
        layout.addWidget(clearAllBtn)
        layout.addStretch()

        maskGroup = qt.QGroupBox('Mask')
        maskGroup.setLayout(layout)
        return maskGroup

    def _initDrawGroupBox(self):
        """Init drawing tools widgets"""
        layout = qt.QVBoxLayout()

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

        container = self._hboxWidget(maskRadioBtn, unmaskRadioBtn)
        layout.addWidget(container)

        # Draw tools
        self.browseAction = qt.QAction(
            icons.getQIcon('normal'), 'Browse', None)
        self.browseAction.setShortcut(qt.QKeySequence(qt.Qt.Key_B))
        self.browseAction.setToolTip(
            'Disables drawing tools, enables zooming interaction mode'
            ' <b>B</b>')
        self.browseAction.setCheckable(True)
        self.browseAction.toggled[bool].connect(self._browseActionToggled)

        self.rectAction = qt.QAction(
            icons.getQIcon('shape-rectangle'), 'Rectangle selection', None)
        self.rectAction.setToolTip(
            'Rectangle selection tool: (Un)Mask a rectangular region <b>R</b>')
        self.rectAction.setShortcut(qt.QKeySequence(qt.Qt.Key_R))
        self.rectAction.setCheckable(True)
        self.rectAction.toggled[bool].connect(self._rectActionToggled)

        self.polygonAction = qt.QAction(
            icons.getQIcon('shape-polygon'), 'Polygon selection', None)
        self.polygonAction.setShortcut(qt.QKeySequence(qt.Qt.Key_S))
        self.polygonAction.setToolTip(
            'Polygon selection tool: (Un)Mask a polygonal region <b>S</b>')
        self.polygonAction.setCheckable(True)
        self.polygonAction.toggled[bool].connect(self._polygonActionToggled)

        self.pencilAction = qt.QAction(
            icons.getQIcon('draw-pencil'), 'Pencil tool', None)
        self.pencilAction.setShortcut(qt.QKeySequence(qt.Qt.Key_P))
        self.pencilAction.setToolTip(
            'Pencil tool: (Un)Mask using a pencil <b>P</b>')
        self.pencilAction.setCheckable(True)
        self.pencilAction.toggled[bool].connect(self._pencilActionToggled)

        self.drawActionGroup = qt.QActionGroup(self)
        self.drawActionGroup.setExclusive(True)
        self.drawActionGroup.addAction(self.browseAction)
        self.drawActionGroup.addAction(self.rectAction)
        self.drawActionGroup.addAction(self.polygonAction)
        self.drawActionGroup.addAction(self.pencilAction)

        self.browseAction.setChecked(True)

        drawButtons = []
        for action in self.drawActionGroup.actions():
            btn = qt.QToolButton()
            btn.setDefaultAction(action)
            drawButtons.append(btn)
        container = self._hboxWidget(*drawButtons)
        layout.addWidget(container)

        self.brushSpinBox = qt.QSpinBox()
        self.brushSpinBox.setRange(1, 1024)
        self.brushSpinBox.setToolTip(
            """Set pencil drawing tool size in pixels of the image
            on which to make the mask.""")

        self.pencilSetting = self._hboxWidget(
            qt.QLabel('Pencil Size:'), self.brushSpinBox)
        self.pencilSetting.setVisible(False)
        layout.addWidget(self.pencilSetting)

        layout.addStretch()

        drawGroup = qt.QGroupBox('Draw tools')
        drawGroup.setLayout(layout)
        return drawGroup

    def _initThresholdGroupBox(self):
        """Init thresholding widgets"""
        layout = qt.QFormLayout()

        self.minLineEdit = qt.QLineEdit()
        self.minLineEdit.setText('0')
        self.minLineEdit.setValidator(qt.QDoubleValidator())
        layout.addRow('Min:', self.minLineEdit)

        self.maxLineEdit = qt.QLineEdit()
        self.maxLineEdit.setText('0')
        self.maxLineEdit.setValidator(qt.QDoubleValidator())
        layout.addRow('Max:', self.maxLineEdit)

        aboveBtn = qt.QPushButton('Mask values > Max')
        aboveBtn.clicked.connect(self._aboveBtnClicked)
        layout.addRow(aboveBtn)

        betweenBtn = qt.QPushButton('Mask values in [Min, Max]')
        betweenBtn.clicked.connect(self._betweenBtnClicked)
        layout.addRow(betweenBtn)

        belowBtn = qt.QPushButton('Mask values < Min')
        belowBtn.clicked.connect(self._belowBtnClicked)
        layout.addRow(belowBtn)

        thresholdGroup = qt.QGroupBox('Threshold')
        thresholdGroup.setLayout(layout)
        return thresholdGroup

    # Handle mask refresh on the plot

    def _updatePlotMask(self):
        """Update mask image in plot"""
        mask = self.getMask(copy=False)
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

    def showEvent(self, event):
        self._activeImageChanged()  # Init mask + enable/disable widget
        self.plot.sigActiveImageChanged.connect(self._activeImageChanged)

    def hideEvent(self, event):
        self.plot.sigActiveImageChanged.disconnect(self._activeImageChanged)
        if self.plot.getImage(self._maskName):
            self.plot.remove(self._maskName, kind='image')

    def _activeImageChanged(self, *args):
        """Update widget and mask according to active image changes"""
        activeImage = self.plot.getActiveImage()
        if activeImage is None or activeImage[1] == self._maskName:
            # No active image or active image is the mask...
            self.setEnabled(False)

            self._data = numpy.zeros((0, 0), dtype=numpy.uint8)
            self._mask.reset()

        else:  # There is an active image
            self.setEnabled(True)

            # Update thresholds according to colormap
            colormap = activeImage[4]['colormap']
            if colormap['autoscale']:
                min_ = numpy.nanmin(activeImage[0])
                max_ = numpy.nanmax(activeImage[0])
            else:
                min_, max_ = colormap['vmin'], colormap['vmax']
            self.minLineEdit.setText(str(min_))
            self.maxLineEdit.setText(str(max_))

            self._origin = activeImage[4]['origin']
            self._scale = activeImage[4]['scale']
            self._z = activeImage[4]['z'] + 1
            self._data = activeImage[0]
            if self._data.shape != self.getMask(copy=False).shape:
                self._mask.reset(self._data.shape)
            else:
                # Refresh in case origin, scale, z changed
                self._updatePlotMask()

    # Handle whole mask operations

    def _setMaskColors(self, level, transparent):
        """Set-up the mask colormap to highlight current mask level.

        :param int level: The mask level to highlight
        :param bool transparent: True to make highlighted color transparent,
                                 False for opaque
        """
        assert level > 0 and level < 256
        colors = numpy.ones((256, 4), dtype=numpy.float32)

        # Set alpha
        colors[:, -1] = 0.5

        # Set highlighted level color
        colors[level] = (0., 0., 0., 0.5 if transparent else 1.)

        # Set no mask level
        colors[0] = (0., 0., 0., 0.)

        self._colormap['colors'] = colors

    def _updateColors(self, *args):
        """Rebuild mask colormap when selected level or transparency change"""
        self._setMaskColors(self.levelSpinBox.value(),
                            self.transparencyCheckBox.isChecked())
        self._updatePlotMask()

    def _handleClearMask(self):
        """Handle clear button clicked: reset current level mask"""
        self._mask.clear(self.levelSpinBox.value())

    def resetMask(self):
        """Reset the mask"""
        self._mask.reset(shape=self._data.shape)

    def _handleInvertMask(self):
        """Invert the current mask level selection."""
        self._mask.invert(self.levelSpinBox.value())

    # Handle drawing tools UI events

    def _browseActionToggled(self, checked):
        """Handle browse action mode triggering"""
        if checked:
            if self._drawingMode:
                self.plot.sigPlotSignal.disconnect(self._plotDrawEvent)
                self._drawingMode = None
            self.plot.setInteractiveMode('zoom')

    def _rectActionToggled(self, checked):
        """Handle rect action mode triggering"""
        if checked:
            self._drawingMode = 'rectangle'
            self.plot.sigPlotSignal.connect(self._plotDrawEvent)
            self.plot.setInteractiveMode('draw', shape='rectangle')

    def _polygonActionToggled(self, checked):
        """Handle polygon action mode triggering"""
        if checked:
            self._drawingMode = 'polygon'
            self.plot.sigPlotSignal.connect(self._plotDrawEvent)
            self.plot.setInteractiveMode('draw', shape='polygon')

    def _pencilActionToggled(self, checked):
        """Handle pencil action mode triggering"""
        if checked:
            self._drawingMode = 'pencil'
            self.plot.sigPlotSignal.connect(self._plotDrawEvent)
            self.plot.setInteractiveMode('draw', shape='line')
        self.pencilSetting.setVisible(checked)

    # Handle plot drawing events

    def _plotDrawEvent(self, event):
        """Handle draw events from the plot"""
        if (self._drawingMode is None or
                event['event'] not in ('drawingProgress', 'drawingFinished')):
            return

        if not len(self._data):
            return

        level = self.levelSpinBox.value()
        doMask = (self.maskStateGroup.checkedId() == 1)
        if qt.QApplication.keyboardModifiers() == qt.Qt.ControlModifier:
            # Invert masking
            doMask = not doMask


        if (self._drawingMode == 'rectangle' and
                event['event'] == 'drawingFinished'):
            # Convert from plot to array coords
            ox, oy = self._origin
            sx, sy = self._scale
            self._mask.updateRectangle(
                level,
                row=int((event['y'] - oy) / sy),
                col=int((event['x'] - ox) / sx),
                height=int(event['height'] / sy),
                width=int(event['width'] / sx),
                mask=doMask)

        elif (self._drawingMode == 'polygon' and
                event['event'] == 'drawingFinished'):
            # Convert from plot to array coords
            vertices = event['points'] / self._scale - self._origin
            vertices = vertices.astype(numpy.int)[:, (1, 0)]  # (row, col)
            self._mask.updatePolygon(level, vertices, doMask)

        elif self._drawingMode == 'pencil':
            # convert from plot to array coords
            col, row = event['points'][-1] / self._scale - self._origin
            row, col = int(row), int(col)
            brushSize = self.brushSpinBox.value()

            # Draw point
            self._mask.updateDisk(level, row, col, brushSize / 2., doMask)

            if self._lastPencilPos and self._lastPencilPos != (row, col):
                # Draw the line
                self._mask.updateLine(
                    level,
                    self._lastPencilPos[0], self._lastPencilPos[1],
                    row, col,
                    brushSize,
                    doMask)

            if event['event'] == 'drawingFinished':
                self._lastPencilPos = None
            else:
                self._lastPencilPos = row, col

    # Handle threshold UI events

    def _aboveBtnClicked(self):
        """Handle select above button"""
        if len(self._data) and self.maxLineEdit.text():
            max_ = float(self.maxLineEdit.text())
            self._mask.updateStencil(self.levelSpinBox.value(),
                                     self._data > max_)

    def _betweenBtnClicked(self):
        """Handle select between button"""
        if (len(self._data) and
                self.minLineEdit.text() and self.maxLineEdit.text()):
            min_ = float(self.minLineEdit.text())
            max_ = float(self.maxLineEdit.text())
            self._mask.updateStencil(self.levelSpinBox.value(),
                                     numpy.logical_and(min_ <= self._data,
                                                       self._data <= max_))

    def _belowBtnClicked(self):
        """Handle select below button"""
        if len(self._data) and self.minLineEdit.text():
            min_ = float(self.minLineEdit.text())
            self._mask.updateStencil(self.levelSpinBox.value(),
                                     self._data < min_)


class MaskToolsDockWidget(qt.QDockWidget):
    """:class:`MaskToolsDockWidget` embedded in a QDockWidget.

    For integration in a :class:`PlotWindow`.

    :param PlotWidget plot: The PlotWidget this widget is operating on
    :paran str name: The title of this widget
    :param parent: See :class:`QDockWidget`
    """

    def __init__(self, plot, name='Mask', parent=None):
        super(MaskToolsDockWidget, self).__init__(parent)
        self.setWindowTitle(name)

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.maskToolsWidget = MaskToolsWidget(plot)
        self.setWidget(self.maskToolsWidget)

    def getMask(self, copy=False):
        """Get the current mask as a 2D array.

        :param bool copy: True (default) to get a copy of the mask.
                          If False, the returned array MUST not be modified.
        :return: The array of the mask with dimension of the 'active' image.
                 If there is no active image, an empty array is returned.
        :rtype: 2D numpy.ndarray of uint8
        """
        return self.maskToolsWidget.getMask(copy=copy)

    def toggleViewAction(self):
        """Returns a checkable action that shows or closes this widget.

        See :class:`QMainWindow`.
        """
        action = super(MaskToolsDockWidget, self).toggleViewAction()
        action.setIcon(icons.getQIcon('image-select-brush'))
        return action
