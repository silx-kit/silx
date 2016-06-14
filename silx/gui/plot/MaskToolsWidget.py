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


# TODO: split mask management from widget, or overkill?
# TODO: find a cleaner way for handling active image to avoid test it each time
# TODO: sync threshold range with colormap
# TODO: use a colormap with more than 3 colors + change color of current level
# TODO: choose mask color depending on image colormap
# TODO: change pencil interaction to drag


class MaskToolsWidget(qt.QWidget):
    """Widget with tools for drawing mask on an image in a PlotWidget."""

    def __init__(self, plot, parent=None):
        self._plot = plot
        self._maskName = '__MASK_TOOLS_%d' % id(self)  # Legend of the mask
        self._colormap = {
            'name': None,
            'normalization': 'linear',
            'autoscale': False,
            'vmin': 0., 'vmax': 2.,
            'colors': numpy.array(
                ((0, 0, 0, 0), (0.5, 0.5, 0.5, 0.5), (1., 1., 1., 0.5)),
                dtype=numpy.float32)}
        self._mask = numpy.array((), dtype=numpy.uint8)  # Store the mask
        self._drawingMode = None  # Store current drawing mode
        self._lastPencilPos = None

        super(MaskToolsWidget, self).__init__(parent)
        self._initWidgets()

    def showEvent(self, event):
        self._activeImageChanged()  # Init mask + enable/disable widget
        self.plot.sigActiveImageChanged.connect(
            self._activeImageChanged)

    def hideEvent(self, event):
        self.plot.sigActiveImageChanged.disconnect(
            self._activeImageChanged)
        if len(self._mask):
            self.plot.remove(self._maskName, kind='image')

    @property
    def plot(self):
        """The :class:`.PlotWindow` this widget is attached to."""
        return self._plot

    def _refreshPlot(self):
        """Update mask image in plot"""
        if len(self._mask):
            activeImage = self.plot.getActiveImage()
            assert activeImage is not None
            params = activeImage[4]
            self.plot.addImage(self._mask, legend=self._maskName,
                               colormap=self._colormap,
                               origin=params['origin'],
                               scale=params['scale'],
                               z=params['z'] + 1,  # Ensure overlay
                               replace=False, resetzoom=False)
        else:
            self.plot.remove(self._maskName, kind='image')

    def getMask(self, copy=True):
        """Get the current mask as a 2D array.

        :param bool copy: True (default) to get a copy of the mask.
                          If False, the returned array MUST not be modified.
        :return: The array of the mask with dimension of the 'active' image.
                 If there is no active image, an empty array is returned.
        :rtype: 2D numpy.ndarray of uint8
        """
        return numpy.array(self._mask, copy=copy)

    def _initWidgets(self):
        """Create widgets"""
        # TODO use a flow layout? or change orientation depending on dock area
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
            """Choose which mask level is edited.
            A mask can have up to 255 non-overlapping levels.""")
        levelWidget = self._hboxWidget(qt.QLabel('Level:'), self.levelSpinBox)

        invertBtn = qt.QPushButton('Invert')
        invertBtn.setToolTip('Invert current mask level')
        invertBtn.clicked.connect(self._handleInvertMask)

        clearBtn = qt.QPushButton('Clear')
        clearBtn.setToolTip('Clear current mask level')
        clearBtn.clicked.connect(self._handleClearMask)

        clearAllBtn = qt.QPushButton('Clear All')
        clearAllBtn.setToolTip('Clear all mask levels')
        clearAllBtn.clicked.connect(self.resetMask)

        layout = qt.QVBoxLayout()
        layout.addWidget(levelWidget)
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
            'Make drawing tools extend current mask level area')
        maskRadioBtn.setChecked(True)

        unmaskRadioBtn = qt.QRadioButton('Unmask')
        unmaskRadioBtn.setToolTip(
            'Make drawing tools unmask current mask level area')

        self.maskStateGroup = qt.QButtonGroup()
        self.maskStateGroup.addButton(maskRadioBtn, 1)
        self.maskStateGroup.addButton(unmaskRadioBtn, 0)

        container = self._hboxWidget(maskRadioBtn, unmaskRadioBtn)
        layout.addWidget(container)

        # Draw tools
        # TODO get browse action from PlotWindow? anyway sync with other tools
        self.browseAction = qt.QAction(
            icons.getQIcon('normal'), 'Browse', None)
        self.browseAction.setToolTip(
            'Disables drawing tools, enables zooming interaction mode')
        self.browseAction.setCheckable(True)
        self.browseAction.toggled[bool].connect(self._browseActionToggled)

        self.rectAction = qt.QAction(
            icons.getQIcon('shape-rectangle'), 'Rectangle selection', None)
        self.rectAction.setToolTip(
            'Rectangle selection tool: Mask/Unmask a rectangular region')
        self.rectAction.setCheckable(True)
        self.rectAction.toggled[bool].connect(self._rectActionToggled)

        self.polygonAction = qt.QAction(
            icons.getQIcon('shape-polygon'), 'Polygon selection', None)
        self.polygonAction.setToolTip(
            'Polygon selection tool: Mask/Unmask a polygonal region')
        self.polygonAction.setCheckable(True)
        self.polygonAction.toggled[bool].connect(self._polygonActionToggled)

        self.pencilAction = qt.QAction(
            icons.getQIcon('draw-pencil'), 'Pencil tool', None)
        self.pencilAction.setToolTip(
            'Pencil tool: Mask/Unmask using a pencil')
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

        self._minLineEdit = qt.QLineEdit()
        self._minLineEdit.setText('0')
        self._minLineEdit.setValidator(qt.QDoubleValidator())
        layout.addRow('Min:', self._minLineEdit)

        self._maxLineEdit = qt.QLineEdit()
        self._maxLineEdit.setText('0')
        self._maxLineEdit.setValidator(qt.QDoubleValidator())
        layout.addRow('Max:', self._maxLineEdit)

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

    def _activeImageChanged(self, *args):
        """Rest mask if the size of the active image change"""
        activeImage = self.plot.getActiveImage()
        if activeImage is None or activeImage[1] == self._maskName:
            self.setEnabled(False)
            self.resetMask()
        else:
            self.setEnabled(True)
            if activeImage[0].shape != self._mask.shape:
                self.resetMask()
            else:
                self._refreshPlot()  # Refresh in case origin or scale changed

    def _handleClearMask(self):
        """Handle clear button clicked: reset current level mask"""
        maskValue = self.levelSpinBox.value()
        self._mask[self._mask == maskValue] = 0
        self._refreshPlot()

    def resetMask(self):
        """Reset the mask"""
        activeImage = self.plot.getActiveImage()
        if activeImage is None:
            self._mask = numpy.array((), dtype=numpy.uint8)
        else:
            self._mask = numpy.zeros(activeImage[0].shape, dtype=numpy.uint8)
        self._refreshPlot()

    def _handleInvertMask(self):
        """Invert the current mask level selection.

        What was 0 becomes current mask, what was current mask becomes 0.
        """
        maskValue = self.levelSpinBox.value()
        masked = self._mask == maskValue
        self._mask[self._mask == 0] = maskValue
        self._mask[masked] = 0
        self._refreshPlot()

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

    def _plotDrawEvent(self, event):
        """Handle draw events from the plot"""
        if (self._drawingMode is None or
                event['event'] not in ('drawingProgress', 'drawingFinished')):
            return

        activeImage = self.plot.getActiveImage()
        if activeImage is None:
            return

        origin = activeImage[4]['origin']
        scale = activeImage[4]['scale']

        maskValue = self.levelSpinBox.value()

        if (self._drawingMode == 'rectangle' and
                event['event'] == 'drawingFinished'):
            # Convert to array coords
            x = int(event['x'] / scale[0] - origin[0])
            y = int(event['y'] / scale[1] - origin[1])
            width = int(event['width'] / scale[0])
            height = int(event['height'] / scale[1])
            selection = self._mask[max(0, y):y+height+1, max(0, x):x+width+1]
            if self.maskStateGroup.checkedId() == 1:  # mask
                selection[:, :] = maskValue
            else:  # unmask
                selection[selection == maskValue] = 0
            self._refreshPlot()

        elif (self._drawingMode == 'polygon' and
                event['event'] == 'drawingFinished'):

            # Convert points to array coordinates
            points = (event['points'] / scale - origin).astype(numpy.int)
            points = points[:, (1, 0)]  # Switch to y, x convention
            fill = shapes.polygon_fill(points, activeImage[0].shape)
            if self.maskStateGroup.checkedId() == 1:  # mask
                self._mask[fill != 0] = maskValue
            else:  # unmask
                self._mask[numpy.logical_and(fill != 0,
                                             self._mask == maskValue)] = 0
            self._refreshPlot()

        elif self._drawingMode == 'pencil':
            # convert to array coords
            x, y = event['points'][-1] / scale - origin
            x, y = int(x), int(y)
            brushSize = self.brushSpinBox.value()

            # Draw point
            rows, cols = shapes.circle_fill(y, x, brushSize / 2.)
            valid = numpy.logical_and(
                numpy.logical_and(rows >= 0, cols >= 0),
                numpy.logical_and(rows < self._mask.shape[0],
                                  cols < self._mask.shape[1]))
            rows, cols = rows[valid], cols[valid]
            if self.maskStateGroup.checkedId() == 1:  # mask
                self._mask[rows, cols] = maskValue
            else:  # unmask
                inMask = self._mask[rows, cols] == maskValue
                self._mask[rows[inMask], cols[inMask]] = 0

            # Draw line
            if (self._lastPencilPos is not None and
                    self._lastPencilPos != (y, x)):
                linePoints = shapes.draw_line(self._lastPencilPos[0],
                                              self._lastPencilPos[1],
                                              y, x, width=brushSize)
                # Remove points < 0
                linePoints = linePoints[numpy.logical_and(
                    numpy.logical_and(linePoints[:, 0] >= 0,
                                      linePoints[:, 1] >= 0),
                    numpy.logical_and(linePoints[:, 0] < self._mask.shape[0],
                                      linePoints[:, 1] < self._mask.shape[1]))]

                if self.maskStateGroup.checkedId() == 1:  # mask
                    self._mask[linePoints[:, 0], linePoints[:, 1]] = maskValue
                else:  # unmask
                    values = self._mask[linePoints[:, 0], linePoints[:, 1]]
                    linePoints = linePoints[values == maskValue]
                    self._mask[linePoints[:, 0], linePoints[:, 1]] = 0

            if event['event'] == 'drawingFinished':
                self._lastPencilPos = None
            else:
                self._lastPencilPos = y, x

            self._refreshPlot()

    # Handle threshold UI events

    def _aboveBtnClicked(self):
        """Handle select above button"""
        activeImage = self.plot.getActiveImage()
        if (activeImage is not None and
                self._mask.shape == activeImage[0].shape and
                self._maxLineEdit.text()):
            threshold = float(self._maxLineEdit.text())
            self._mask[activeImage[0] > threshold] = 1
            self._refreshPlot()

    def _betweenBtnClicked(self):
        """Handle select between button"""
        activeImage = self.plot.getActiveImage()
        if (activeImage is not None and
                self._mask.shape == activeImage[0].shape and
                self._minLineEdit.text() and self._maxLineEdit.text()):
            min_ = float(self._minLineEdit.text())
            max_ = float(self._maxLineEdit.text())
            self._mask[numpy.logical_and(min_ <= activeImage[0],
                                         activeImage[0] <= max_)] = 1
            self._refreshPlot()

    def _belowBtnClicked(self):
        """Handle select below button"""
        activeImage = self.plot.getActiveImage()
        if (activeImage is not None and
                self._mask.shape == activeImage[0].shape and
                self._minLineEdit.text()):
            threshold = float(self._minLineEdit.text())
            self._mask[activeImage[0] < threshold] = 1
            self._refreshPlot()


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
