#!/usr/bin/env python
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
"""
This script illustrates image ROI selection in a :class:`~silx.gui.plot.PlotWidget`

It uses :class:`~silx.gui.plot.tools.roi.RegionOfInterestManager` and
:class:`~silx.gui.plot.tools.roi.RegionOfInterestTableWidget` to handle the
interactive selection and to display the list of selected ROIs.
"""

import numpy

from silx.gui import qt
from silx.gui.plot import Plot2D
from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot.tools.roi import RegionOfInterestTableWidget


def dummy_image():
    """Create a dummy image"""
    x = numpy.linspace(-1.5, 1.5, 1024)
    xv, yv = numpy.meshgrid(x, x)
    signal = numpy.exp(- (xv ** 2 / 0.15 ** 2
                          + yv ** 2 / 0.25 ** 2))
    # add noise
    signal += 0.3 * numpy.random.random(size=signal.shape)
    return signal


app = qt.QApplication([])  # Start QApplication

# Create the plot widget and add an image
plot = Plot2D()
plot.getDefaultColormap().setName('viridis')
plot.addImage(dummy_image())

# Create the object controlling the ROI selection and set it up
roiManager = RegionOfInterestManager(plot)
roiManager.setColor('pink')  # Set the color of ROI


# Set the name of each created selection
def updateAddedSelection(selection):
    """Called for each added selection: set the name"""
    if selection.getLabel() == '':
        selection.setLabel('ROI %d' % len(roiManager.getSelections()))


roiManager.sigSelectionAdded.connect(updateAddedSelection)

# Add a rectangle selection
roiManager.createSelection('rectangle',
                           points=((50, 50), (200, 200)),
                           label='Initial ROI')


# Create the table widget displaying
roiTable = RegionOfInterestTableWidget()
roiTable.setRegionOfInterestManager(roiManager)

# Create buttons for all ROI 'drawing' modes
btnLayout = qt.QHBoxLayout()  # The layout to store the buttons
btnLayout.addStretch(1)
btnLayout.addWidget(qt.QLabel('Add ROIs:'))

for kind in roiManager.getSupportedSelectionKinds():
    # Create a tool button and associate it with the QAction of each mode
    toolButton = qt.QToolButton()
    toolButton.setDefaultAction(roiManager.getDrawSelectionModeAction(kind))
    btnLayout.addWidget(toolButton, 0, qt.Qt.AlignCenter)

btnLayout.addStretch(1)


# Add the selection table and the buttons to a dock widget
widget = qt.QWidget()
layout = qt.QVBoxLayout()
widget.setLayout(layout)
layout.addWidget(roiTable)
layout.addLayout(btnLayout)

dock = qt.QDockWidget('Image ROI')
dock.setWidget(widget)
plot.addTabbedDockWidget(dock)

# Show the widget and start the application
plot.show()
app.exec_()
