#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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
"""This script is a simple example of how to display statistics on a specific
region of interest.

An example on how to define your own statistic is given in the 'plotStats.py'
script.
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "23/07/2019"

from silx.gui import qt
from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot.tools.roi import RegionOfInterestTableWidget
from silx.gui.plot.items.roi import RectangleROI
from silx.gui.plot import Plot2D, Plot1D
from silx.gui.plot.CurvesROIWidget import ROI
from silx.gui.plot.tools.ROIStatsWidget import RoiStatsWidget
from silx.gui.plot.StatsWidget import UpdateModeWidget, UpdateMode
from collections import OrderedDict
import numpy


class _RoiStatsWidget(qt.QMainWindow):
    def __init__(self, parent=None, plot=None):
        assert plot is not None
        qt.QMainWindow.__init__(self, parent)
        self._roiStatsWindow = RoiStatsWidget(plot=plot)
        self.setCentralWidget(self._roiStatsWindow)

        # update mode docker
        self._updateModeControl = UpdateModeWidget(parent=self)
        self._docker = qt.QDockWidget(parent=self)
        self._docker.setWidget(self._updateModeControl)
        self.addDockWidget(qt.Qt.TopDockWidgetArea,
                           self._docker)
        self.setWindowFlags(qt.Qt.Widget)

        # connect signal / slot
        self._updateModeControl.sigUpdateModeChanged.connect(
            self._roiStatsWindow._setUpdateMode)
        self._updateModeControl.sigUpdateRequested.connect(
            self._roiStatsWindow._updateAllStats)

        # expose API
        self.registerROI = self._roiStatsWindow.registerROI
        self.setStats = self._roiStatsWindow.setStats
        self._addRoiStatsItem = self._roiStatsWindow._addRoiStatsItem

        # setup
        self._updateModeControl.setUpdateMode('auto')


class _RoiStatsDisplayExWindow(qt.QMainWindow):
    """
    Simple window to group the different stats actors
    """
    def __init__(self, parent=None):
        qt.QMainWindow.__init__(self, parent)
        self.plot = Plot2D()
        self.setCentralWidget(self.plot)

        # 1D roi management
        self._curveRoiWidget = self.plot.getCurvesRoiDockWidget().widget()
        # 2D - 3D roi manager
        self._regionManager = RegionOfInterestManager(parent=self.plot)

        self._curveRoiWidget = self.plot.getCurvesRoiDockWidget().widget()
        # roi display widget
        self._roiStatsWindow = RoiStatsWidget(plot=self.plot)

        # Create the table widget displaying
        self._2DRoiWidget = RegionOfInterestTableWidget()
        self._2DRoiWidget.setRegionOfInterestManager(self._regionManager)
        self._2DRoiWidget.show()

        # tabWidget for displaying the rois
        self._roisTabWidget = qt.QTabWidget(parent=self)
        self._roisTabWidget.addTab(self._curveRoiWidget, '1D roi(s)')
        self._roisTabWidget.addTab(self._2DRoiWidget, '2D roi(s)')

        # widget for displaying stats results and update mode
        self._statsWidget = _RoiStatsWidget(parent=self, plot=self.plot)

        # create Dock widgets
        self._roisTabWidgetDockWidget = qt.QDockWidget(parent=self)
        self._roisTabWidgetDockWidget.setWidget(self._roisTabWidget)
        self.addDockWidget(qt.Qt.RightDockWidgetArea,
                           self._roisTabWidgetDockWidget)

        # create Dock widgets
        self._roiStatsWindowDockWidget = qt.QDockWidget(parent=self)
        self._roiStatsWindowDockWidget.setWidget(self._statsWidget)
        self.addDockWidget(qt.Qt.RightDockWidgetArea,
                           self._roiStatsWindowDockWidget)

    def setRois(self, rois1D, rois2D):
        self._curveRoiWidget.setRois(rois1D)
        for roi1D in rois1D:
            self._statsWidget.registerROI(roi1D)

        for roi2D in rois2D:
            self._regionManager.addRoi(roi2D)
            self._statsWidget.registerROI(roi2D)

    def setStats(self, stats):
        self._statsWidget.setStats(stats=stats)

    def addItem(self, item, roi):
        self._statsWidget._addRoiStatsItem(roi=roi, item=item)


def main():
    app = qt.QApplication([])

    window = _RoiStatsDisplayExWindow()

    # define some image and curve
    window.plot.addImage(numpy.arange(10000).reshape(100, 100), legend='img1')
    window.plot.addImage(numpy.random.random(10000).reshape(100, 100), legend='img2',
                         origin=(0, 100))
    window.plot.addCurve(x=numpy.linspace(0, 10, 56), y=numpy.arange(56),
                         legend='curve1')

    # define rois
    rectangle_roi = RectangleROI()
    rectangle_roi.setGeometry(origin=(0, 0), size=(20, 20))
    rectangle_roi.setLabel('Initial ROI')
    rectangle_roi2 = RectangleROI()
    rectangle_roi2.setGeometry(origin=(0, 100), size=(50, 50))
    rectangle_roi2.setLabel('ROI second')
    roi1D = ROI(name='range1', fromdata=0, todata=4, type_='energy')
    window.setRois(rois1D=(roi1D,), rois2D=(rectangle_roi, rectangle_roi2))

    # define stats to display
    stats = [
        ('sum', numpy.sum),
        ('mean', numpy.mean),
    ]
    window.setStats(stats)

    # add some couple (plotItem, roi) to be displayed by default
    img_item = window.plot.getImage('img1')
    window.addItem(roi=rectangle_roi, item=img_item)
    curve_item = window.plot.getCurve('curve1')
    window.addItem(item=curve_item, roi=roi1D)

    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
