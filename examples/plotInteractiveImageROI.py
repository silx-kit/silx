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
interactive ROI selection and to display the list of ROIs.
"""

import sys
import numpy

from silx.gui import qt
from silx.gui.plot import Plot2D
from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot.tools.roi import RegionOfInterestTableWidget
from silx.gui.plot.items.roi import RectangleROI


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

# Create the object controlling the ROIs and set it up
roiManager = RegionOfInterestManager(plot)
roiManager.setColor('pink')  # Set the color of ROI


# Set the name of each created region of interest
def updateAddedRegionOfInterest(roi):
    """Called for each added region of interest: set the name"""
    if roi.getLabel() == '':
        roi.setLabel('ROI %d' % len(roiManager.getRois()))


roiManager.sigRoiAdded.connect(updateAddedRegionOfInterest)

# Add a rectangular region of interest
roi = RectangleROI()
roi.setGeometry(origin=(50, 50), size=(200, 200))
roi.setLabel('Initial ROI')
roiManager.addRoi(roi)

# Create the table widget displaying
roiTable = RegionOfInterestTableWidget()
roiTable.setRegionOfInterestManager(roiManager)

# Create a toolbar containing buttons for all ROI 'drawing' modes
roiToolbar = qt.QToolBar()  # The layout to store the buttons
roiToolbar.setIconSize(qt.QSize(16, 16))

for roiClass in roiManager.getSupportedRoiClasses():
    # Create a tool button and associate it with the QAction of each mode
    action = roiManager.getInteractionModeAction(roiClass)
    roiToolbar.addAction(action)

# Add the region of interest table and the buttons to a dock widget
widget = qt.QWidget()
layout = qt.QVBoxLayout()
widget.setLayout(layout)
layout.addWidget(roiToolbar)
layout.addWidget(roiTable)

def roiDockVisibilityChanged(visible):
    """Handle change of visibility of the roi dock widget

    If dock becomes hidden, ROI interaction is stopped.
    """
    if not visible:
        roiManager.stop()

dock = qt.QDockWidget('Image ROI')
dock.setWidget(widget)
dock.visibilityChanged.connect(roiDockVisibilityChanged)
plot.addTabbedDockWidget(dock)

# Show the widget and start the application
plot.show()
result = app.exec_()
app.deleteLater()
sys.exit(result)
