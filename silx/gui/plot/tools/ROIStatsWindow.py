# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
"""This module provides widget for displaying statistics relative to a
Region of interest and an item
"""


__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "22/07/2019"


from silx.gui import qt
from silx.gui import icons
from silx.gui.plot.StatsWidget import StatsTable
from silx.gui.plot.items.roi import RegionOfInterest
from silx.gui.plot.CurvesROIWidget import ROI
import logging

_logger = logging.getLogger(__name__)


class _GetRoiItemCoupleDialog(qt.QDialog):
    """
    Dialog used to know which plot item and which roi he wants
    """
    _COMPATIBLE_KINDS = ('curve', 'image', 'scatter', 'histogram')

    def __init__(self, parent=None, plot=None, rois=None):
        qt.QDialog.__init__(self, parent=parent)
        assert plot is not None
        assert rois is not None
        self._plot = plot
        self._rois = rois

        self.setLayout(qt.QVBoxLayout())

        # define the selection widget
        self._selection_widget = qt.QWidget()
        self._selection_widget.setLayout(qt.QHBoxLayout())
        self._kindCB = qt.QComboBox(parent=self)
        self._selection_widget.layout().addWidget(self._kindCB)
        self._itemCB = qt.QComboBox(parent=self)
        self._selection_widget.layout().addWidget(self._itemCB)
        self._roiCB = qt.QComboBox(parent=self)
        self._selection_widget.layout().addWidget(self._roiCB)
        self.layout().addWidget(self._selection_widget)

        # define modal buttons
        types = qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel
        self._buttonsModal = qt.QDialogButtonBox(parent=self)
        self._buttonsModal.setStandardButtons(types)
        self.layout().addWidget(self._buttonsModal)
        self._buttonsModal.accepted.connect(self.accept)
        self._buttonsModal.rejected.connect(self.reject)

    def _getCompatibleRois(self, kind):
        """Return compatible rois for the given item kind"""
        def is_compatible(roi, kind):
            if isinstance(roi, RegionOfInterest):
                return kind in ('image', 'scatter')
            elif isinstance(roi, ROi):
                return kind in ('curve', 'histogram')
            else:
                raise ValueError('kind not managed')
        return list(filter(lambda x: is_compatible(x, kind), self._rois))

    def exec_(self):
        self._kindCB.clear()
        self._itemCB.clear()
        # filter kind without any items
        self.valid_kinds = {}
        self.valid_rois = {}
        for kind in _GetRoiItemCoupleDialog._COMPATIBLE_KINDS:
            items = self._plot._getItems(kind=kind)
            rois = self._getCompatibleRois(kind=kind)
            if len(items) > 0 and len(rois) > 0:
                self.valid_kinds[kind] = items
                self.valid_rois[kind] = rois

        # filter roi according to kinds
        if len(self.valid_kinds) == 0:
            _logger.warning('no couple item/roi detected for displaying stats')
            return self.reject()

        for kind in self.valid_kinds:
            self._kindCB.addItem(kind)
        self._updateValidItemAndRoi()

        return qt.QDialog.__init__(self)

    def _updateValidItemAndRoi(self):
        self._itemCB.clear()
        self._roiCB.clear()
        kind = self._kindCB.currentText()
        for roi in self.valid_rois[kind]:
            self._roiCB.addItem(roi.name())
        for item in self.valid_kinds[kind]:
            self._kindCB.addItem(item)


class _StatsROITable(StatsTable):
    def __init__(self, parent, plot):
        StatsTable.__init__(self, parent=parent, plot=plot)

    def add(self, item):
        pass

    def remove(self, item):
        pass


class RoiStatsWindow(qt.QMainWindow):
    """
    Main widget for displaying stats item for (roi, plotItem) couple.
    Also provide interface for adding and removing items.
    
    :param Union[qt.QWidget, None] parent: parent qWidget
    :param PlotWindow plot: plot widget containing the items
    :param stats: stats to display
    :param tuple rois: tuple of rois to manage
    """
    def __init__(self, parent=None, plot=None, stats=None, rois=None):
        assert rois is not None
        qt.QMainWindow.__init__(self, parent)

        toolbar = qt.QToolBar(self)
        icon = icons.getQIcon('add')
        self._rois = rois
        self._addAction = qt.QAction(icon, 'add item/roi', toolbar)
        self._addAction.triggered.connect(self.addRoiStatsItem)

        toolbar.addAction(self._addAction)
        self.addToolBar(toolbar)

        self._plot = plot
        self._stats = stats
        self._statsROITable = _StatsROITable(parent=self, plot=self._plot)
        self.setCentralWidget(self._statsROITable)
        self.setWindowFlags(qt.Qt.Widget)

    def setPlot(self):
        self._plot = plot

    def getPlot(self):
        return self._plot

    def setStats(self, stats):
        self._stats = stats

    def getStats(self):
        return self._stats

    def addRoiStatsItem(self):
        """Ask the user what couple ROI / item he want to display"""
        dialog = _GetRoiItemCoupleDialog(parent=self, plot=self._plot,
                                         rois=self._rois)
        if dialog.exec_():
            self._addRoiStatsItem(roi=dialog.getRoi(), item=dialog.getItem())

    def _addRoiStatsItem(self, roi, item):
        statsItem = RoiStatsItemWidget(parent=self, roi=roi, stats=self._stats,
                                       item=item)
        self._addStatsItem(statsItem=statsItem)

    def _addStatsItem(self, statsItem):
        pass

    def showItemKindColumn(self):
        pass


class RoiStatsItemWidget(qt.QWidget):
    """
    Item to display stats regarding the couple (roi, plotItem)
    
    :param Union[qt.QWidget, None] parent: parent qWidget
    :param roi: region of interest to use for statistic calculation
    :type: Union[ROI, RegionOfInterest]
    :param stats: stats to display
    :param item: item on which we want to compute statistics
    """
    def __init__(self, parent=None, roi=None, stats=None, item=None):
        qt.QWidget.__init__(self, parent)
        pass


if __name__ == '__main__':
    from silx.gui.plot.tools.roi import RegionOfInterestManager
    from silx.gui.plot.tools.roi import RegionOfInterestTableWidget
    from silx.gui.plot.items.roi import RectangleROI
    from silx.gui.plot import Plot2D, Plot1D
    from silx.gui.plot.CurvesROIWidget import ROI
    from collections import OrderedDict
    import numpy
    app = qt.QApplication([])

    plot = Plot2D()

    plot.addImage(numpy.arange(10000).reshape(100, 100), legend='img1')
    img_item = plot.getCurve('img1')

    region_manager = RegionOfInterestManager(parent=plot)
    rectangle_roi = RectangleROI()
    rectangle_roi.setGeometry(origin=(0, 0), size=(2, 2))
    rectangle_roi.setLabel('Initial ROI')
    region_manager.addRoi(rectangle_roi)

    # roi = ROI(name='range1', fromdata=0, todata=4, type_='energy')
    # plot.getCurvesRoiDockWidget().setRois((roi,))
    # plot.getCurvesRoiDockWidget().setVisible(True)

    stats = [
        ('sum', numpy.sum),
        ('mean', numpy.mean),
    ]
    roiStatsWindow = RoiStatsWindow(plot=plot, rois=(rectangle_roi, ))
    # roiItem = RoiStatsItemWidget(parent=None, roi=roi, item=curve_item,
    #                                stats=stats)
    # plotRoiStats.addRoiStatsItem(roiItem)
    plot.show()
    roiStatsWindow.show()
    app.exec_()
