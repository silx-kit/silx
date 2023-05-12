#!/usr/bin/env python
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
from silx.gui.plot.items.roi import RectangleROI, PolygonROI, ArcROI
from silx.gui.plot import Plot2D
from silx.gui.plot.CurvesROIWidget import ROI
from silx.gui.plot.ROIStatsWidget import ROIStatsWidget
from silx.gui.plot.StatsWidget import UpdateModeWidget
import sys
import argparse
import functools
import numpy
import threading
from silx.gui.utils import concurrent
import random
import time


class UpdateThread(threading.Thread):
    """Thread updating the image of a :class:`~silx.gui.plot.Plot2D`

    :param plot2d: The Plot2D to update."""

    def __init__(self, plot2d):
        self.plot2d = plot2d
        self.running = False
        super(UpdateThread, self).__init__()

    def start(self):
        """Start the update thread"""
        self.running = True
        super(UpdateThread, self).start()

    def run(self):
        """Method implementing thread loop that updates the plot"""
        while self.running:
            time.sleep(1)
            # Run plot update asynchronously
            concurrent.submitToQtMainThread(
                self.plot2d.addImage,
                numpy.random.random(10000).reshape(100, 100),
                resetzoom=False,
                legend=random.choice(('img1', 'img2'))
            )

    def stop(self):
        """Stop the update thread"""
        self.running = False
        self.join(2)


class _RoiStatsWidget(qt.QMainWindow):
    """
    Window used to associate ROIStatsWidget and UpdateModeWidget
    """
    def __init__(self, parent=None, plot=None, mode=None):
        assert plot is not None
        qt.QMainWindow.__init__(self, parent)
        self._roiStatsWindow = ROIStatsWidget(plot=plot)
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
        callback = functools.partial(self._roiStatsWindow._updateAllStats,
                                     is_request=True)
        self._updateModeControl.sigUpdateRequested.connect(callback)

        # expose API
        self.registerROI = self._roiStatsWindow.registerROI
        self.setStats = self._roiStatsWindow.setStats
        self.addItem = self._roiStatsWindow.addItem
        self.removeItem = self._roiStatsWindow.removeItem
        self.setUpdateMode = self._updateModeControl.setUpdateMode

        # setup
        self._updateModeControl.setUpdateMode('auto')


class _RoiStatsDisplayExWindow(qt.QMainWindow):
    """
    Simple window to group the different statistics actors
    """
    def __init__(self, parent=None, mode=None):
        qt.QMainWindow.__init__(self, parent)
        self.plot = Plot2D()
        self.setCentralWidget(self.plot)

        # 1D roi management
        self._curveRoiWidget = self.plot.getCurvesRoiDockWidget().widget()
        # hide last columns which are of no use now
        for index in (5, 6, 7, 8):
            self._curveRoiWidget.roiTable.setColumnHidden(index, True)

        # 2D - 3D roi manager
        self._regionManager = RegionOfInterestManager(parent=self.plot)

        # Create the table widget displaying
        self._2DRoiWidget = RegionOfInterestTableWidget()
        self._2DRoiWidget.setRegionOfInterestManager(self._regionManager)

        # tabWidget for displaying the rois
        self._roisTabWidget = qt.QTabWidget(parent=self)
        if hasattr(self._roisTabWidget, 'setTabBarAutoHide'):
            self._roisTabWidget.setTabBarAutoHide(True)

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
        # move the docker contain in the parent widget
        self.addDockWidget(qt.Qt.RightDockWidgetArea,
                           self._statsWidget._docker)
        self.addDockWidget(qt.Qt.RightDockWidgetArea,
                           self._roiStatsWindowDockWidget)

        # expose API
        self.setUpdateMode = self._statsWidget.setUpdateMode

    def setRois(self, rois1D=None, rois2D=None):
        rois1D = rois1D or ()
        rois2D = rois2D or ()
        self._curveRoiWidget.setRois(rois1D)
        for roi1D in rois1D:
            self._statsWidget.registerROI(roi1D)

        for roi2D in rois2D:
            self._regionManager.addRoi(roi2D)
            self._statsWidget.registerROI(roi2D)

        # update manage tab visibility
        if len(rois2D) > 0:
            self._roisTabWidget.addTab(self._2DRoiWidget, '2D roi(s)')
        if len(rois1D) > 0:
            self._roisTabWidget.addTab(self._curveRoiWidget, '1D roi(s)')

    def setStats(self, stats):
        self._statsWidget.setStats(stats=stats)

    def addItem(self, item, roi):
        self._statsWidget.addItem(roi=roi, plotItem=item)


# define stats to display
STATS = [
    ('sum', numpy.sum),
    ('mean', numpy.mean),
]


def get_1D_rois():
    """return some ROI instance"""
    roi1D = ROI(name='range1', fromdata=0, todata=4, type_='energy')
    roi2D = ROI(name='range2', fromdata=-2, todata=6, type_='energy')
    return roi1D, roi2D


def get_2D_rois():
    """return some RectangleROI instance"""
    rectangle_roi = RectangleROI()
    rectangle_roi.setGeometry(origin=(0, 100), size=(20, 20))
    rectangle_roi.setName('Initial ROI')
    polygon_roi = PolygonROI()
    polygon_points = numpy.array([(0, 10), (10, 20), (45, 30), (35, 0)])
    polygon_roi.setPoints(polygon_points)
    polygon_roi.setName('polygon ROI')
    arc_roi = ArcROI()
    arc_roi.setName('arc ROI')
    arc_roi.setFirstShapePoints(numpy.array([[50, 10], [80, 120]]))
    arc_roi.setGeometry(*arc_roi.getGeometry())
    return rectangle_roi, polygon_roi, arc_roi


def example_curve(mode):
    """set up the roi stats example for curves"""
    app = qt.QApplication([])
    roi_1, roi_2 = get_1D_rois()
    window = _RoiStatsDisplayExWindow()
    window.setRois(rois1D=(roi_2, roi_1))

    # define some image and curve
    window.plot.addCurve(x=numpy.linspace(0, 10, 56), y=numpy.arange(56),
                         legend='curve1', color='blue')
    window.plot.addCurve(x=numpy.linspace(0, 10, 56), y=numpy.random.random_sample(size=56),
                         legend='curve2', color='red')

    window.setStats(STATS)

    # add some couple (plotItem, roi) to be displayed by default
    curve1_item = window.plot.getCurve('curve1')
    window.addItem(item=curve1_item, roi=roi_1)
    window.addItem(item=curve1_item, roi=roi_2)
    curve2_item = window.plot.getCurve('curve2')
    window.addItem(item=curve2_item, roi=roi_2)

    window.setUpdateMode(mode)

    window.show()
    app.exec()


def example_image(mode):
    """set up the roi stats example for images"""
    app = qt.QApplication([])
    rectangle_roi, polygon_roi, arc_roi = get_2D_rois()

    window = _RoiStatsDisplayExWindow()
    window.setRois(rois2D=(rectangle_roi, polygon_roi, arc_roi))
    # Create the thread that calls submitToQtMainThread
    updateThread = UpdateThread(window.plot)
    updateThread.start()  # Start updating the plot

    # define some image and curve
    window.plot.addImage(numpy.arange(10000).reshape(100, 100), legend='img1')
    window.plot.addImage(numpy.random.random(10000).reshape(100, 100), legend='img2',
                         origin=(0, 100))
    window.setStats(STATS)

    # add some couple (plotItem, roi) to be displayed by default
    img1_item = window.plot.getImage('img1')
    img2_item = window.plot.getImage('img2')
    window.addItem(item=img2_item, roi=rectangle_roi)
    window.addItem(item=img1_item, roi=polygon_roi)
    window.addItem(item=img1_item, roi=arc_roi)

    window.setUpdateMode(mode)

    window.show()
    app.exec()
    updateThread.stop()  # Stop updating the plot


def example_curve_image(mode):
    """set up the roi stats example for curves and images"""
    app = qt.QApplication([])

    roi1D_1, roi1D_2 = get_1D_rois()
    rectangle_roi, polygon_roi, arc_roi = get_2D_rois()

    window = _RoiStatsDisplayExWindow()
    window.setRois(rois1D=(roi1D_1, roi1D_2,),
                   rois2D=(rectangle_roi, polygon_roi, arc_roi))

    # define some image and curve
    window.plot.addImage(numpy.arange(10000).reshape(100, 100), legend='img1')
    window.plot.addImage(numpy.random.random(10000).reshape(100, 100),
                         legend='img2', origin=(0, 100))
    window.plot.addCurve(x=numpy.linspace(0, 10, 56), y=numpy.arange(56),
                         legend='curve1')
    window.setStats(STATS)

    # add some couple (plotItem, roi) to be displayed by default
    img_item = window.plot.getImage('img2')
    window.addItem(item=img_item, roi=rectangle_roi)
    curve_item = window.plot.getCurve('curve1')
    window.addItem(item=curve_item, roi=roi1D_1)

    window.setUpdateMode(mode)

    # Create the thread that calls submitToQtMainThread
    updateThread = UpdateThread(window.plot)
    updateThread.start()  # Start updating the plot

    window.show()
    app.exec()
    updateThread.stop()  # Stop updating the plot


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", dest="items", default='curves+images',
                        help="items type(s), can be curve, image, curves+images")
    parser.add_argument('--mode', dest='mode', default='auto',
                        help='valid modes are `auto` or `manual`')
    options = parser.parse_args(argv[1:])

    items = options.items.lower()
    if items == 'curves':
        example_curve(mode=options.mode)
    elif items == 'images':
        example_image(mode=options.mode)
    elif items == 'curves+images':
        example_curve_image(mode=options.mode)
    else:
        raise ValueError('invalid entry for item type')


if __name__ == '__main__':
    main(sys.argv)
