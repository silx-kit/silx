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
""":class:`GroupPropertiesWidget` allows to reset properties in a GroupItem."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "24/04/2018"

from ....gui import qt
from ....gui.colors import Colormap
from ....gui.dialog.ColormapDialog import ColormapDialog

from ..items import SymbolMixIn, ColormapMixIn


class GroupPropertiesWidget(qt.QWidget):
    """Set properties of all items in a :class:`GroupItem`

    :param QWidget parent:
    """

    MAX_MARKER_SIZE = 20
    """Maximum value for marker size"""

    MAX_LINE_WIDTH = 10
    """Maximum value for line width"""

    def __init__(self, parent=None):
        super(GroupPropertiesWidget, self).__init__(parent)
        self._group = None
        self.setEnabled(False)

        # Set widgets
        layout = qt.QFormLayout(self)
        self.setLayout(layout)

        # Colormap
        colormapButton = qt.QPushButton('Set...')
        colormapButton.setToolTip("Set colormap for all items")
        colormapButton.clicked.connect(self._colormapButtonClicked)
        layout.addRow('Colormap', colormapButton)

        self._markerComboBox = qt.QComboBox(self)
        self._markerComboBox.addItems(SymbolMixIn.getSupportedSymbolNames())

        # Marker
        markerButton = qt.QPushButton('Set')
        markerButton.setToolTip("Set marker for all items")
        markerButton.clicked.connect(self._markerButtonClicked)

        markerLayout = qt.QHBoxLayout()
        markerLayout.setContentsMargins(0, 0, 0, 0)
        markerLayout.addWidget(self._markerComboBox, 1)
        markerLayout.addWidget(markerButton, 0)

        layout.addRow('Marker', markerLayout)

        # Marker size
        self._markerSizeSlider = qt.QSlider()
        self._markerSizeSlider.setOrientation(qt.Qt.Horizontal)
        self._markerSizeSlider.setSingleStep(1)
        self._markerSizeSlider.setRange(1, self.MAX_MARKER_SIZE)
        self._markerSizeSlider.setValue(1)

        markerSizeButton = qt.QPushButton('Set')
        markerSizeButton.setToolTip("Set marker size for all items")
        markerSizeButton.clicked.connect(self._markerSizeButtonClicked)

        markerSizeLayout = qt.QHBoxLayout()
        markerSizeLayout.setContentsMargins(0, 0, 0, 0)
        markerSizeLayout.addWidget(qt.QLabel('1'))
        markerSizeLayout.addWidget(self._markerSizeSlider, 1)
        markerSizeLayout.addWidget(qt.QLabel(str(self.MAX_MARKER_SIZE)))
        markerSizeLayout.addWidget(markerSizeButton, 0)

        layout.addRow('Marker Size', markerSizeLayout)

        # Line width
        self._lineWidthSlider = qt.QSlider()
        self._lineWidthSlider.setOrientation(qt.Qt.Horizontal)
        self._lineWidthSlider.setSingleStep(1)
        self._lineWidthSlider.setRange(1, self.MAX_LINE_WIDTH)
        self._lineWidthSlider.setValue(1)

        lineWidthButton = qt.QPushButton('Set')
        lineWidthButton.setToolTip("Set line width for all items")
        lineWidthButton.clicked.connect(self._lineWidthButtonClicked)

        lineWidthLayout = qt.QHBoxLayout()
        lineWidthLayout.setContentsMargins(0, 0, 0, 0)
        lineWidthLayout.addWidget(qt.QLabel('1'))
        lineWidthLayout.addWidget(self._lineWidthSlider, 1)
        lineWidthLayout.addWidget(qt.QLabel(str(self.MAX_LINE_WIDTH)))
        lineWidthLayout.addWidget(lineWidthButton, 0)

        layout.addRow('Line Width', lineWidthLayout)

        self._colormapDialog = None  # To store dialog
        self._colormap = Colormap()

    def getGroup(self):
        """Returns the :class:`GroupItem` this widget is attached to.

        :rtype: Union[GroupItem, None]
        """
        return self._group

    def setGroup(self, group):
        """Set the :class:`GroupItem` this widget is attached to.

        :param GroupItem group: GroupItem to control (or None)
        """
        self._group = group
        if group is not None:
            self.setEnabled(True)

    def _colormapButtonClicked(self, checked=False):
        """Handle colormap button clicked"""
        group = self.getGroup()
        if group is None:
            return

        if self._colormapDialog is None:
            self._colormapDialog = ColormapDialog(self)
            self._colormapDialog.setColormap(self._colormap)

        previousColormap = self._colormapDialog.getColormap()
        if self._colormapDialog.exec():
            colormap = self._colormapDialog.getColormap()

            for item in group.visit():
                if isinstance(item, ColormapMixIn):
                    itemCmap = item.getColormap()
                    cmapName = colormap.getName()
                    if cmapName is not None:
                        itemCmap.setName(colormap.getName())
                    else:
                        itemCmap.setColormapLUT(colormap.getColormapLUT())
                    itemCmap.setNormalization(colormap.getNormalization())
                    itemCmap.setGammaNormalizationParameter(
                        colormap.getGammaNormalizationParameter())
                    itemCmap.setVRange(colormap.getVMin(), colormap.getVMax())
        else:
            # Reset colormap
            self._colormapDialog.setColormap(previousColormap)

    def _markerButtonClicked(self, checked=False):
        """Handle marker set button clicked"""
        group = self.getGroup()
        if group is None:
            return

        marker = self._markerComboBox.currentText()
        for item in group.visit():
            if isinstance(item, SymbolMixIn):
                item.setSymbol(marker)

    def _markerSizeButtonClicked(self, checked=False):
        """Handle marker size set button clicked"""
        group = self.getGroup()
        if group is None:
            return

        markerSize = self._markerSizeSlider.value()
        for item in group.visit():
            if isinstance(item, SymbolMixIn):
                item.setSymbolSize(markerSize)

    def _lineWidthButtonClicked(self, checked=False):
        """Handle line width set button clicked"""
        group = self.getGroup()
        if group is None:
            return

        lineWidth = self._lineWidthSlider.value()
        for item in group.visit():
            if hasattr(item, 'setLineWidth'):
                item.setLineWidth(lineWidth)
