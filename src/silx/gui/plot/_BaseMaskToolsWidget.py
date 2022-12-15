# /*##########################################################################
#
# Copyright (c) 2017-2022 European Synchrotron Radiation Facility
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
"""This module is a collection of base classes used in modules
:mod:`.MaskToolsWidget` (images) and :mod:`.ScatterMaskToolsWidget`
"""

__authors__ = ["T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "08/12/2020"

import os
import weakref

import numpy

from silx.gui import qt, icons
from silx.gui.widgets.FloatEdit import FloatEdit
from silx.gui.colors import Colormap
from silx.gui.colors import rgba
from .actions.mode import PanModeAction


class BaseMask(qt.QObject):
    """Base class for :class:`ImageMask` and :class:`ScatterMask`

    A mask field with update operations.

    A mask is an array of the same shape as some underlying data. The mask
    array stores integer values in the range 0-255, to allow for 254 levels
    of mask (value 0 is reserved for unmasked data).

    The mask is updated using spatial selection methods: data located inside
    a selected area is masked with a specified mask level.

    """

    sigChanged = qt.Signal()
    """Signal emitted when the mask has changed"""

    sigStateChanged = qt.Signal()
    """Signal emitted for each mask commit/undo/redo operation"""

    sigUndoable = qt.Signal(bool)
    """Signal emitted when undo becomes possible/impossible"""

    sigRedoable = qt.Signal(bool)
    """Signal emitted when redo becomes possible/impossible"""

    def __init__(self, dataItem=None):
        self.historyDepth = 10
        """Maximum number of operation stored in history list for undo"""
        # Init lists for undo/redo
        self._history = []
        self._redo = []

        # Store the mask
        self._mask = numpy.array((), dtype=numpy.uint8)

        # Store the plot item to be masked
        self._dataItem = None
        if dataItem is not None:
            self.setDataItem(dataItem)
            self.reset(self.getDataValues().shape)
        super(BaseMask, self).__init__()

    def setDataItem(self, item):
        """Set a data item

        :param item: A plot item, subclass of :class:`silx.gui.plot.items.Item`
        :return:
        """
        self._dataItem = item

    def getDataItem(self):
        """Returns current plot item the mask is on.

        :rtype: Union[~silx.gui.plot.items.Item,None]
        """
        return self._dataItem

    def getDataValues(self):
        """Return data values, as a numpy array with the same shape
        as the mask.

        This method must be implemented in a subclass, as the way of
        accessing data depends on the data item passed to :meth:`setDataItem`

        :return: Data values associated with the data item.
        :rtype: numpy.ndarray
        """
        raise NotImplementedError("To be implemented in subclass")

    def _notify(self):
        """Notify of mask change."""
        self.sigChanged.emit()

    def getMask(self, copy=True):
        """Get the current mask as a numpy array.

        :param bool copy: True (default) to get a copy of the mask.
                          If False, the returned array MUST not be modified.
        :return: The array of the mask with dimension of the data to be masked.
        :rtype: numpy.ndarray of uint8
        """
        return numpy.array(self._mask, copy=copy)

    def setMask(self, mask, copy=True):
        """Set the mask to a new array.

        :param numpy.ndarray mask: The array to use for the mask.
        :type mask: numpy.ndarray of uint8, C-contiguous.
                    Array of other types are converted.
        :param bool copy: True (the default) to copy the array,
                          False to use it as is if possible.
        """
        self._mask = numpy.array(mask, copy=copy, order='C', dtype=numpy.uint8)
        self._notify()

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
                not numpy.array_equal(self._mask, self._history[-1])):
            if self._redo:
                self._redo = []  # Reset redo as a new action as been performed
                self.sigRedoable[bool].emit(False)

            while len(self._history) >= self.historyDepth:
                self._history.pop(0)
            self._history.append(numpy.array(self._mask, copy=True))

            if len(self._history) == 2:
                self.sigUndoable.emit(True)
        self.sigStateChanged.emit()

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
            self.sigStateChanged.emit()

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
            self.sigStateChanged.emit()

    # Whole mask operations

    def clear(self, level):
        """Set all values of the given mask level to 0.

        :param int level: Value of the mask to set to 0.
        """
        assert 0 < level < 256
        self._mask[self._mask == level] = 0
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

    def reset(self, shape=None):
        """Reset the mask to zero and change its shape.

        :param shape: Shape of the new mask with the correct dimensionality
            with regards to the data dimensionality,
            or None to have an empty mask
        :type shape: tuple of int
        """
        if shape is None:
            # assume dimensionality never changes
            shape = (0,) * len(self._mask.shape)  # empty array
        shapeChanged = (shape != self._mask.shape)
        self._mask = numpy.zeros(shape, dtype=numpy.uint8)
        if shapeChanged:
            self.resetHistory()

        self._notify()

    # To be implemented
    def save(self, filename, kind):
        """Save current mask in a file

        :param str filename: The file where to save to mask
        :param str kind: The kind of file to save (e.g 'npy')
        :raise Exception: Raised if the file writing fail
        """
        raise NotImplementedError("To be implemented in subclass")

    # update thresholds
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
            self._mask[numpy.logical_and(self._mask == level, stencil)] = 0
        self._notify()

    def updateBelowThreshold(self, level, threshold, mask=True):
        """Mask/unmask all points whose values are below a threshold.

        :param int level:
        :param float threshold: Threshold
        :param bool mask: True to mask (default), False to unmask.
        """
        self.updateStencil(level,
                           self.getDataValues() < threshold,
                           mask)

    def updateBetweenThresholds(self, level, min_, max_, mask=True):
        """Mask/unmask all points whose values are in a range.

        :param int level:
        :param float min_: Lower threshold
        :param float max_: Upper threshold
        :param bool mask: True to mask (default), False to unmask.
        """
        stencil = numpy.logical_and(min_ <= self.getDataValues(),
                                    self.getDataValues() <= max_)
        self.updateStencil(level, stencil, mask)

    def updateAboveThreshold(self, level, threshold, mask=True):
        """Mask/unmask all points whose values are above a threshold.

        :param int level: Mask level to update.
        :param float threshold: Threshold.
        :param bool mask: True to mask (default), False to unmask.
        """
        self.updateStencil(level,
                           self.getDataValues() > threshold,
                           mask)

    def updateNotFinite(self, level, mask=True):
        """Mask/unmask all points whose values are not finite.

        :param int level: Mask level to update.
        :param bool mask: True to mask (default), False to unmask.
        """
        self.updateStencil(level,
                           numpy.logical_not(numpy.isfinite(self.getDataValues())),
                           mask)

    # Drawing operations:
    def updateRectangle(self, level, row, col, height, width, mask=True):
        """Mask/Unmask data inside a rectangle, with the given mask level.

        :param int level: Mask level to update, in range 1-255.
        :param row: Starting row/y of the rectangle
        :param col: Starting column/x of the rectangle
        :param height:
        :param width:
        :param bool mask: True to mask (default), False to unmask.
        """
        raise NotImplementedError("To be implemented in subclass")

    def updatePolygon(self, level, vertices, mask=True):
        """Mask/Unmask data inside a polygon, with the given mask level.

        :param int level: Mask level to update.
        :param vertices: Nx2 array of polygon corners as (row, col) / (y, x)
        :param bool mask: True to mask (default), False to unmask.
        """
        raise NotImplementedError("To be implemented in subclass")

    def updatePoints(self, level, rows, cols, mask=True):
        """Mask/Unmask points with given coordinates.

        :param int level: Mask level to update.
        :param rows: Rows/ordinates (y) of selected points
        :type rows: 1D numpy.ndarray
        :param cols: Columns/abscissa (x) of selected points
        :type cols: 1D numpy.ndarray
        :param bool mask: True to mask (default), False to unmask.
        """
        raise NotImplementedError("To be implemented in subclass")

    def updateDisk(self, level, crow, ccol, radius, mask=True):
        """Mask/Unmask data located inside a dick of the given mask level.

        :param int level: Mask level to update.
        :param crow: Disk center row/ordinate (y).
        :param ccol: Disk center column/abscissa.
        :param float radius: Radius of the disk in mask array unit
        :param bool mask: True to mask (default), False to unmask.
        """
        raise NotImplementedError("To be implemented in subclass")

    def updateEllipse(self, level, crow, ccol, radius_r, radius_c, mask=True):
        """Mask/Unmask a disk of the given mask level.

        :param int level: Mask level to update.
        :param int crow: Row of the center of the ellipse
        :param int ccol: Column of the center of the ellipse
        :param float radius_r: Radius of the ellipse in the row
        :param float radius_c: Radius of the ellipse in the column
        :param bool mask: True to mask (default), False to unmask.
        """
        raise NotImplementedError("To be implemented in subclass")

    def updateLine(self, level, row0, col0, row1, col1, width, mask=True):
        """Mask/Unmask a line of the given mask level.

        :param int level: Mask level to update.
        :param row0: Row/y of the starting point.
        :param col0: Column/x of the starting point.
        :param row1: Row/y of the end point.
        :param col1: Column/x of the end point.
        :param width: Width of the line in mask array unit.
        :param bool mask: True to mask (default), False to unmask.
        """
        raise NotImplementedError("To be implemented in subclass")


class BaseMaskToolsWidget(qt.QWidget):
    """Base class for :class:`MaskToolsWidget` (image mask) and
    :class:`scatterMaskToolsWidget`"""

    sigMaskChanged = qt.Signal()
    _maxLevelNumber = 255

    def __init__(self, parent=None, plot=None, mask=None):
        """

        :param parent: Parent QWidget
        :param plot: Plot widget on which to operate
        :param mask: Instance of subclass of :class:`BaseMask`
            (e.g. :class:`ImageMask`)
        """
        super(BaseMaskToolsWidget, self).__init__(parent)
        # register if the user as force a color for the corresponding mask level
        self._defaultColors = numpy.ones((self._maxLevelNumber + 1), dtype=bool)
        # overlays colors set by the user
        self._overlayColors = numpy.zeros((self._maxLevelNumber + 1, 3), dtype=numpy.float32)

        # as parent have to be the first argument of the widget to fit
        # QtDesigner need but here plot can't be None by default.
        assert plot is not None
        self._plotRef = weakref.ref(plot)
        self._maskName = '__MASK_TOOLS_%d' % id(self)  # Legend of the mask

        self._colormap = Colormap(normalization='linear',
                                  vmin=0,
                                  vmax=self._maxLevelNumber)
        self._defaultOverlayColor = rgba('gray')  # Color of the mask
        self._setMaskColors(1, 0.5)  # Set the colormap LUT

        if not isinstance(mask, BaseMask):
            raise TypeError("mask is not an instance of BaseMask")
        self._mask = mask

        self._mask.sigChanged.connect(self._updatePlotMask)
        self._mask.sigChanged.connect(self._emitSigMaskChanged)

        self._drawingMode = None  # Store current drawing mode
        self._lastPencilPos = None
        self._multipleMasks = 'exclusive'

        self._maskFileDir = qt.QDir.current().absolutePath()
        self.plot.sigInteractiveModeChanged.connect(
            self._interactiveModeChanged)

        self._initWidgets()

    def _emitSigMaskChanged(self):
        """Notify mask changes"""
        self.sigMaskChanged.emit()

    def getMaskedItem(self):
        """Returns the item that is currently being masked

        :rtype: Union[~silx.gui.plot.items.Item,None]
        """
        return self._mask.getDataItem()

    def getSelectionMask(self, copy=True):
        """Get the current mask as a numpy array.

        :param bool copy: True (default) to get a copy of the mask.
                          If False, the returned array MUST not be modified.
        :return: The mask (as an array of uint8) with dimension of
                 the 'active' plot item.
                 If there is no active image or scatter, it returns None.
        :rtype: Union[numpy.ndarray,None]
        """
        mask = self._mask.getMask(copy=copy)
        return None if mask.size == 0 else mask

    def setSelectionMask(self, mask):
        """Set the mask: Must be implemented in subclass"""
        raise NotImplementedError()

    def resetSelectionMask(self):
        """Reset the mask: Must be implemented in subclass"""
        raise NotImplementedError()

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
            self._levelWidget.setVisible(self._multipleMasks != 'single')
            self._clearAllBtn.setVisible(self._multipleMasks != 'single')

    def setMaskFileDirectory(self, path):
        """Set the default directory to use by load/save GUI tools

        The directory is also updated by the user, if he change the location
        of the dialog.
        """
        self.maskFileDir = path

    def getMaskFileDirectory(self):
        """Get the default directory used by load/save GUI tools"""
        return self.maskFileDir

    @property
    def maskFileDir(self):
        """The directory from which to load/save mask from/to files."""
        if not os.path.isdir(self._maskFileDir):
            self._maskFileDir = qt.QDir.current().absolutePath()
        return self._maskFileDir

    @maskFileDir.setter
    def maskFileDir(self, maskFileDir):
        self._maskFileDir = str(maskFileDir)

    @property
    def plot(self):
        """The :class:`.PlotWindow` this widget is attached to."""
        plot = self._plotRef()
        if plot is None:
            raise RuntimeError(
                'Mask widget attached to a PlotWidget that no longer exists')
        return plot

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
        layout.addWidget(self._initOtherToolsGroupBox())
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
        transparencyWidget = qt.QWidget(parent=self)
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
        self._levelWidget = self._hboxWidget(qt.QLabel('Mask level:'),
                                            self.levelSpinBox)
        # Transparency
        self._transparencyWidget = self._initTransparencyWidget()

        style = qt.QApplication.style()

        def getIcon(*identifiyers):
            for i in identifiyers:
                if isinstance(i, str):
                    if qt.QIcon.hasThemeIcon(i):
                        return qt.QIcon.fromTheme(i)
                elif isinstance(i, qt.QIcon):
                    return i
                else:
                    return style.standardIcon(i)
            return qt.QIcon()

        undoAction = qt.QAction(self)
        undoAction.setText('Undo')
        icon = getIcon("edit-undo", qt.QStyle.SP_ArrowBack)
        undoAction.setIcon(icon)
        undoAction.setShortcut(qt.QKeySequence.Undo)
        undoAction.setToolTip('Undo last mask change <b>%s</b>' %
                              undoAction.shortcut().toString())
        self._mask.sigUndoable.connect(undoAction.setEnabled)
        undoAction.triggered.connect(self._mask.undo)

        redoAction = qt.QAction(self)
        redoAction.setText('Redo')
        icon = getIcon("edit-redo", qt.QStyle.SP_ArrowForward)
        redoAction.setIcon(icon)
        redoAction.setShortcut(qt.QKeySequence.Redo)
        redoAction.setToolTip('Redo last undone mask change <b>%s</b>' %
                              redoAction.shortcut().toString())
        self._mask.sigRedoable.connect(redoAction.setEnabled)
        redoAction.triggered.connect(self._mask.redo)

        loadAction = qt.QAction(self)
        loadAction.setText('Load...')
        icon = icons.getQIcon("document-open")
        loadAction.setIcon(icon)
        loadAction.setToolTip('Load mask from file')
        loadAction.triggered.connect(self._loadMask)

        saveAction = qt.QAction(self)
        saveAction.setText('Save...')
        icon = icons.getQIcon("document-save")
        saveAction.setIcon(icon)
        saveAction.setToolTip('Save mask to file')
        saveAction.triggered.connect(self._saveMask)

        invertAction = qt.QAction(self)
        invertAction.setText('Invert')
        icon = icons.getQIcon("mask-invert")
        invertAction.setIcon(icon)
        invertAction.setShortcut(qt.QKeySequence(qt.Qt.CTRL | qt.Qt.Key_I))
        invertAction.setToolTip('Invert current mask <b>%s</b>' %
                                invertAction.shortcut().toString())
        invertAction.triggered.connect(self._handleInvertMask)

        clearAction = qt.QAction(self)
        clearAction.setText('Clear')
        icon = icons.getQIcon("mask-clear")
        clearAction.setIcon(icon)
        clearAction.setShortcut(qt.QKeySequence.Delete)
        clearAction.setToolTip('Clear current mask level <b>%s</b>' %
                               clearAction.shortcut().toString())
        clearAction.triggered.connect(self._handleClearMask)

        clearAllAction = qt.QAction(self)
        clearAllAction.setText('Clear all')
        icon = icons.getQIcon("mask-clear-all")
        clearAllAction.setIcon(icon)
        clearAllAction.setToolTip('Clear all mask levels')
        clearAllAction.triggered.connect(self.resetSelectionMask)

        # Buttons group
        margin1 = qt.QWidget(self)
        margin1.setMinimumWidth(6)
        margin2 = qt.QWidget(self)
        margin2.setMinimumWidth(6)

        actions = (loadAction, saveAction, margin1,
                   undoAction, redoAction, margin2,
                   invertAction, clearAction, clearAllAction)
        widgets = []
        for action in actions:
            if isinstance(action, qt.QWidget):
                widgets.append(action)
                continue
            btn = qt.QToolButton()
            btn.setDefaultAction(action)
            widgets.append(btn)
            if action is clearAllAction:
                self._clearAllBtn = btn
        container = self._hboxWidget(*widgets)
        container.layout().setSpacing(1)

        layout = qt.QVBoxLayout()
        layout.addWidget(container)
        layout.addWidget(self._levelWidget)
        layout.addWidget(self._transparencyWidget)
        layout.addStretch(1)

        maskGroup = qt.QGroupBox('Mask')
        maskGroup.setLayout(layout)
        return maskGroup

    def isMaskInteractionActivated(self):
        """Returns true if any mask interaction is activated"""
        return self.drawActionGroup.checkedAction() is not None

    def _initDrawGroupBox(self):
        """Init drawing tools widgets"""
        layout = qt.QVBoxLayout()

        self.browseAction = PanModeAction(self.plot, self.plot)
        self.addAction(self.browseAction)

        # Draw tools
        self.rectAction = qt.QAction(icons.getQIcon('shape-rectangle'),
                                     'Rectangle selection',
                                     self)
        self.rectAction.setToolTip(
                'Rectangle selection tool: (Un)Mask a rectangular region <b>R</b>')
        self.rectAction.setShortcut(qt.QKeySequence(qt.Qt.Key_R))
        self.rectAction.setCheckable(True)
        self.rectAction.triggered.connect(self._activeRectMode)
        self.addAction(self.rectAction)

        self.ellipseAction = qt.QAction(icons.getQIcon('shape-ellipse'),
                                        'Circle selection',
                                        self)
        self.ellipseAction.setToolTip(
                'Rectangle selection tool: (Un)Mask a circle region <b>R</b>')
        self.ellipseAction.setShortcut(qt.QKeySequence(qt.Qt.Key_R))
        self.ellipseAction.setCheckable(True)
        self.ellipseAction.triggered.connect(self._activeEllipseMode)
        self.addAction(self.ellipseAction)

        self.polygonAction = qt.QAction(icons.getQIcon('shape-polygon'),
                                        'Polygon selection',
                                        self)
        self.polygonAction.setShortcut(qt.QKeySequence(qt.Qt.Key_S))
        self.polygonAction.setToolTip(
                'Polygon selection tool: (Un)Mask a polygonal region <b>S</b><br>'
                'Left-click to place new polygon corners<br>'
                'Left-click on first corner to close the polygon')
        self.polygonAction.setCheckable(True)
        self.polygonAction.triggered.connect(self._activePolygonMode)
        self.addAction(self.polygonAction)

        self.pencilAction = qt.QAction(icons.getQIcon('draw-pencil'),
                                       'Pencil tool',
                                       self)
        self.pencilAction.setShortcut(qt.QKeySequence(qt.Qt.Key_P))
        self.pencilAction.setToolTip(
                'Pencil tool: (Un)Mask using a pencil <b>P</b>')
        self.pencilAction.setCheckable(True)
        self.pencilAction.triggered.connect(self._activePencilMode)
        self.addAction(self.pencilAction)

        self.drawActionGroup = qt.QActionGroup(self)
        self.drawActionGroup.setExclusive(True)
        self.drawActionGroup.addAction(self.rectAction)
        self.drawActionGroup.addAction(self.ellipseAction)
        self.drawActionGroup.addAction(self.polygonAction)
        self.drawActionGroup.addAction(self.pencilAction)

        actions = (self.browseAction, self.rectAction, self.ellipseAction,
                   self.polygonAction, self.pencilAction)
        drawButtons = []
        for action in actions:
            btn = qt.QToolButton()
            btn.setDefaultAction(action)
            drawButtons.append(btn)
        container = self._hboxWidget(*drawButtons)
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

        self.maskStateWidget.setHidden(True)

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

        self.belowThresholdAction = qt.QAction(icons.getQIcon('plot-roi-below'),
                                               'Mask below threshold',
                                               self)
        self.belowThresholdAction.setToolTip(
                'Mask image where values are below given threshold')
        self.belowThresholdAction.setCheckable(True)
        self.belowThresholdAction.setChecked(True)

        self.betweenThresholdAction = qt.QAction(icons.getQIcon('plot-roi-between'),
                                                 'Mask within range',
                                                 self)
        self.betweenThresholdAction.setToolTip(
                'Mask image where values are within given range')
        self.betweenThresholdAction.setCheckable(True)

        self.aboveThresholdAction = qt.QAction(icons.getQIcon('plot-roi-above'),
                                               'Mask above threshold',
                                               self)
        self.aboveThresholdAction.setToolTip(
                'Mask image where values are above given threshold')
        self.aboveThresholdAction.setCheckable(True)

        self.thresholdActionGroup = qt.QActionGroup(self)
        self.thresholdActionGroup.setExclusive(True)
        self.thresholdActionGroup.addAction(self.belowThresholdAction)
        self.thresholdActionGroup.addAction(self.betweenThresholdAction)
        self.thresholdActionGroup.addAction(self.aboveThresholdAction)
        self.thresholdActionGroup.triggered.connect(
                self._thresholdActionGroupTriggered)

        self.loadColormapRangeAction = qt.QAction(icons.getQIcon('view-refresh'),
                                                  'Set min-max from colormap',
                                                  self)
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

        spacer = qt.QWidget(parent=self)
        spacer.setSizePolicy(qt.QSizePolicy.Expanding,
                             qt.QSizePolicy.Preferred)
        widgets.append(spacer)

        loadColormapRangeBtn = qt.QToolButton()
        loadColormapRangeBtn.setDefaultAction(self.loadColormapRangeAction)
        widgets.append(loadColormapRangeBtn)

        toolBar = self._hboxWidget(*widgets, stretch=False)

        config = qt.QGridLayout()
        config.setContentsMargins(0, 0, 0, 0)

        self.minLineLabel = qt.QLabel("Min:", self)
        self.minLineEdit = FloatEdit(self, value=0)
        config.addWidget(self.minLineLabel, 0, 0)
        config.addWidget(self.minLineEdit, 0, 1)

        self.maxLineLabel = qt.QLabel("Max:", self)
        self.maxLineEdit = FloatEdit(self, value=0)
        config.addWidget(self.maxLineLabel, 1, 0)
        config.addWidget(self.maxLineEdit, 1, 1)

        self.applyMaskBtn = qt.QPushButton('Apply mask')
        self.applyMaskBtn.clicked.connect(self._maskBtnClicked)

        layout = qt.QVBoxLayout()
        layout.addWidget(toolBar)
        layout.addLayout(config)
        layout.addWidget(self.applyMaskBtn)
        layout.addStretch(1)

        self.thresholdGroup = qt.QGroupBox('Threshold')
        self.thresholdGroup.setLayout(layout)

        # Init widget state
        self._thresholdActionGroupTriggered(self.belowThresholdAction)
        return self.thresholdGroup

        # track widget visibility and plot active image changes

    def _initOtherToolsGroupBox(self):
        layout = qt.QVBoxLayout()

        self.maskNanBtn = qt.QPushButton('Mask not finite values')
        self.maskNanBtn.setToolTip('Mask Not a Number and infinite values')
        self.maskNanBtn.clicked.connect(self._maskNotFiniteBtnClicked)
        layout.addWidget(self.maskNanBtn)
        layout.addStretch(1)

        self.otherToolGroup = qt.QGroupBox('Other tools')
        self.otherToolGroup.setLayout(layout)
        return self.otherToolGroup

    def changeEvent(self, event):
        """Reset drawing action when disabling widget"""
        if (event.type() == qt.QEvent.EnabledChange and
                not self.isEnabled() and
                self.drawActionGroup.checkedAction()):
            # Disable drawing tool by reseting interaction to pan or zoom
            self.plot.resetInteractiveMode()

    def save(self, filename, kind):
        """Save current mask in a file

        :param str filename: The file where to save to mask
        :param str kind: The kind of file to save in 'edf', 'tif', 'npy'
        :raise Exception: Raised if the process fails
        """
        self._mask.save(filename, kind)

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
        colors[:,:3] = self._defaultOverlayColor[:3]

        # check if some colors has been directly set by the user
        mask = numpy.equal(self._defaultColors, False)
        colors[mask,:3] = self._overlayColors[mask,:3]

        # Set alpha
        colors[:, -1] = alpha / 2.

        # Set highlighted level color
        colors[level, 3] = alpha

        # Set no mask level
        colors[0] = (0., 0., 0., 0.)

        self._colormap.setColormapLUT(colors)

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
        rgb = rgba(rgb)[0:3]
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
        elif self._drawingMode == 'ellipse':
            self._activeEllipseMode()
        elif self._drawingMode == 'polygon':
            self._activePolygonMode()
        elif self._drawingMode == 'pencil':
            self._activePencilMode()

    def _handleClearMask(self):
        """Handle clear button clicked: reset current level mask"""
        self._mask.clear(self.levelSpinBox.value())
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
            self.pencilAction.setChecked(False)
            self.rectAction.setChecked(False)
            self.polygonAction.setChecked(False)
            self._releaseDrawingMode()
            self._updateDrawingModeWidgets()

    def _releaseDrawingMode(self):
        """Release the drawing mode if is was used"""
        if self._drawingMode is None:
            return
        self.plot.sigPlotSignal.disconnect(self._plotDrawEvent)
        self._drawingMode = None

    def _activeRectMode(self):
        """Handle rect action mode triggering"""
        self._releaseDrawingMode()
        self._drawingMode = 'rectangle'
        self.plot.sigPlotSignal.connect(self._plotDrawEvent)
        color = self.getCurrentMaskColor()
        self.plot.setInteractiveMode(
            'draw', shape='rectangle', source=self, color=color)
        self._updateDrawingModeWidgets()

    def _activeEllipseMode(self):
        """Handle circle action mode triggering"""
        self._releaseDrawingMode()
        self._drawingMode = 'ellipse'
        self.plot.sigPlotSignal.connect(self._plotDrawEvent)
        color = self.getCurrentMaskColor()
        self.plot.setInteractiveMode(
            'draw', shape='ellipse', source=self, color=color)
        self._updateDrawingModeWidgets()

    def _activePolygonMode(self):
        """Handle polygon action mode triggering"""
        self._releaseDrawingMode()
        self._drawingMode = 'polygon'
        self.plot.sigPlotSignal.connect(self._plotDrawEvent)
        color = self.getCurrentMaskColor()
        self.plot.setInteractiveMode('draw', shape='polygon', source=self, color=color)
        self._updateDrawingModeWidgets()

    def _getPencilWidth(self):
        """Returns the width of the pencil to use in data coordinates`

        :rtype: float
        """
        return self.pencilSpinBox.value()

    def _activePencilMode(self):
        """Handle pencil action mode triggering"""
        self._releaseDrawingMode()
        self._drawingMode = 'pencil'
        self.plot.sigPlotSignal.connect(self._plotDrawEvent)
        color = self.getCurrentMaskColor()
        width = self._getPencilWidth()
        self.plot.setInteractiveMode(
            'draw', shape='pencil', source=self, color=color, width=width)
        self._updateDrawingModeWidgets()

    def _updateDrawingModeWidgets(self):
        self.maskStateWidget.setVisible(self._drawingMode is not None)
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

    # Handle threshold UI events

    def _thresholdActionGroupTriggered(self, triggeredAction):
        """Threshold action group listener."""
        if triggeredAction is self.belowThresholdAction:
            self.minLineLabel.setVisible(True)
            self.maxLineLabel.setVisible(False)
            self.minLineEdit.setVisible(True)
            self.maxLineEdit.setVisible(False)
            self.applyMaskBtn.setText("Mask below")
        elif triggeredAction is self.betweenThresholdAction:
            self.minLineLabel.setVisible(True)
            self.maxLineLabel.setVisible(True)
            self.minLineEdit.setVisible(True)
            self.maxLineEdit.setVisible(True)
            self.applyMaskBtn.setText("Mask between")
        elif triggeredAction is self.aboveThresholdAction:
            self.minLineLabel.setVisible(False)
            self.maxLineLabel.setVisible(True)
            self.minLineEdit.setVisible(False)
            self.maxLineEdit.setVisible(True)
            self.applyMaskBtn.setText("Mask above")
        self.applyMaskBtn.setToolTip(triggeredAction.toolTip())

    def _maskBtnClicked(self):
        if self.belowThresholdAction.isChecked():
            if self.minLineEdit.text():
                self._mask.updateBelowThreshold(self.levelSpinBox.value(),
                                                self.minLineEdit.value())
                self._mask.commit()

        elif self.betweenThresholdAction.isChecked():
            if self.minLineEdit.text() and self.maxLineEdit.text():
                min_ = self.minLineEdit.value()
                max_ = self.maxLineEdit.value()
                self._mask.updateBetweenThresholds(self.levelSpinBox.value(),
                                                   min_, max_)
                self._mask.commit()

        elif self.aboveThresholdAction.isChecked():
            if self.maxLineEdit.text():
                max_ = float(self.maxLineEdit.value())
                self._mask.updateAboveThreshold(self.levelSpinBox.value(),
                                                max_)
                self._mask.commit()

    def _maskNotFiniteBtnClicked(self):
        """Handle not finite mask button clicked: mask NaNs and inf"""
        self._mask.updateNotFinite(
            self.levelSpinBox.value())
        self._mask.commit()


class BaseMaskToolsDockWidget(qt.QDockWidget):
    """Base class for :class:`MaskToolsWidget` and
    :class:`ScatterMaskToolsWidget`.

    For integration in a :class:`PlotWindow`.

    :param parent: See :class:`QDockWidget`
    :paran str name: The title of this widget
    """

    sigMaskChanged = qt.Signal()

    def __init__(self, parent=None, name='Mask', widget=None):
        super(BaseMaskToolsDockWidget, self).__init__(parent)
        self.setWindowTitle(name)

        if not isinstance(widget, BaseMaskToolsWidget):
            raise TypeError("BaseMaskToolsDockWidget requires a MaskToolsWidget")
        self.setWidget(widget)
        self.widget().sigMaskChanged.connect(self._emitSigMaskChanged)

        self.layout().setContentsMargins(0, 0, 0, 0)
        self.dockLocationChanged.connect(self._dockLocationChanged)
        self.topLevelChanged.connect(self._topLevelChanged)

    def _emitSigMaskChanged(self):
        """Notify mask changes"""
        # must be connected to self.widget().sigMaskChanged in child class
        self.sigMaskChanged.emit()

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

    def resetSelectionMask(self):
        """Reset the mask to an array of zeros with the shape of the
        current data."""
        self.widget().resetSelectionMask()

    def toggleViewAction(self):
        """Returns a checkable action that shows or closes this widget.

        See :class:`QMainWindow`.
        """
        action = super(BaseMaskToolsDockWidget, self).toggleViewAction()
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
