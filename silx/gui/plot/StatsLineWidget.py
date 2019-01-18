# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2019 European Synchrotron Radiation Facility
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
Module containing widgets displaying stats from items of a plot.
"""

from __future__ import division

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "18/01/2019"


import numpy
import weakref
import functools
from silx.gui import qt
from silx.gui.plot.stats.stats import StatBase
from silx.gui.plot.items.curve import Curve as CurveItem
from silx.gui.plot.items.image import ImageBase as ImageItem
from silx.gui.plot.items.scatter import Scatter as ScatterItem
from silx.gui.plot.items.histogram import Histogram as HistogramItem
from silx.gui.plot.stats.statshandler import StatsHandler, StatFormatter
from silx.gui.plot import stats as statsmdl
from silx.gui.widgets.FlowLayout import FlowLayout
import logging

logger = logging.getLogger(__name__)


class LineStatsWidget(qt.QWidget):
    """
    Widget made to display stats into a QLayout with for all stat a couple
     (QLabel, QLineEdit) created.
     The the layout can be defined prior of adding any statistic.
    """

    COMPATIBLE_KINDS = {
        'curve': CurveItem,
        'image': ImageItem,
        'scatter': ScatterItem,
        'histogram': HistogramItem
    }

    COMPATIBLE_ITEMS = tuple(COMPATIBLE_KINDS.values())

    def __init__(self, parent=None, plot=None, kind='curve',
                 statsOnVisibleData=False):
        qt.QWidget.__init__(self, parent)
        self._plotRef = None
        self._statsOnVisibleData = statsOnVisibleData
        self._item_kind = kind
        self.callbackImage = None
        self.callbackScatter = None
        self.callbackCurve = None
        """Associate the curve legend to his first item"""
        self._statsHandler = None
        self._statQlineEdit = {}
        """list of legends actually displayed"""
        self._n_statistics_per_row = 4
        """number of statistics displayed per line in the grid layout"""
        self.setPlot(plot)

    def _addItemForStatistic(self, statistic):
        assert isinstance(statistic, StatBase)
        assert statistic.name in self._statsHandler.stats

        if self.layout() is None:
            self.setLayout(FlowLayout())

        assert isinstance(self.layout(), (FlowLayout, qt.QGridLayout, qt.QBoxLayout))

        self.layout().setSpacing(2)
        self.layout().setContentsMargins(2, 2, 2, 2)

        qLabel = qt.QLabel(statistic.name + ':', parent=self)
        qLineEdit = qt.QLineEdit('', parent=self)
        qLineEdit.setReadOnly(True)

        if isinstance(self.layout(), qt.QGridLayout):
            column = len(self._statQlineEdit) % self._n_statistics_per_row
            row = len(self._statQlineEdit) // self._n_statistics_per_row
            self.layout().addWidget(qLabel, row, column*2)
            self.layout().addWidget(qLineEdit, row, column*2+1)
        else:
            # create a mother widget to make sure both will always be displayed
            # side by side
            widget = qt.QWidget(parent=self)
            widget.setLayout(qt.QHBoxLayout())
            widget.layout().setSpacing(0)
            widget.layout().setContentsMargins(0, 0, 0, 0)

            widget.layout().addWidget(qLabel)
            widget.layout().addWidget(qLineEdit)

            self.layout().addWidget(widget)

        self._statQlineEdit[statistic.name] = qLineEdit

    def setStats(self, statsHandler):
        """Set which stats to display and the associated formatting.

        :param StatsHandler statsHandler:
            Set the statistics to be displayed and how to format them using
        """
        _statsHandler = statsHandler
        if statsHandler is None:
            _statsHandler = StatsHandler(statFormatters=())
        if isinstance(_statsHandler, (list, tuple)):
            _statsHandler = StatsHandler(_statsHandler)
        assert isinstance(_statsHandler, StatsHandler)
        self._statsHandler = _statsHandler
        for statName, stat in list(_statsHandler.stats.items()):
            self._addItemForStatistic(stat)
        self._updateStats()

    def getStatsHandler(self):
        """Returns the :class:`StatsHandler` in use.

        :rtype: StatsHandler
        """
        return self._statsHandler

    def setPlot(self, plot):
        """Define the plot to interact with

        :param Union[PlotWidget,None] plot:
            The plot containing the items on which statistics are applied
        """
        self._dealWithPlotConnection(create=False)
        self._plotRef = None if plot is None else weakref.ref(plot)
        self._dealWithPlotConnection(create=True)
        self._updateStats()

    def getPlot(self):
        """Returns the plot attached to this widget

        :rtype: Union[PlotWidget,None]
        """
        return None if self._plotRef is None else self._plotRef()

    def _dealWithPlotConnection(self, create=True):
        """Manage connection to plot signals

        Note: connection on Item are managed by the _removeItem function
        """
        plot = self.getPlot()
        if plot is None:
            return

        if create is True:
            if self.callbackImage is None:
                self.callbackImage = functools.partial(self._activeItemChanged, 'image')
                self.callbackScatter = functools.partial(self._activeItemChanged, 'scatter')
                self.callbackCurve = functools.partial(self._activeItemChanged, 'curve')
            plot.sigActiveImageChanged.connect(self.callbackImage)
            plot.sigActiveScatterChanged.connect(self.callbackScatter)
            plot.sigActiveCurveChanged.connect(self.callbackCurve)
        else:
            if self.callbackImage is not None:
                plot.sigActiveImageChanged.disconnect(self.callbackImage)
                plot.sigActiveScatterChanged.disconnect(self.callbackScatter)
                plot.sigActiveCurveChanged.disconnect(self.callbackCurve)
            self.callbackImage = None
            self.callbackScatter = None
            self.callbackCurve = None

        if create is True:
            plot.sigPlotSignal.connect(self._zoomPlotChanged)
        else:
            plot.sigPlotSignal.disconnect(self._zoomPlotChanged)

    def _activeItemChanged(self, kind, previous, current):
        if kind == self._item_kind:
            self._updateStats()

    def _updateStats(self):
        plot = self.getPlot()
        if plot is not None:
            item = plot._getActiveItem(kind=self._item_kind)
            if(item is not None and self._statsHandler is not None and
               len(self._statsHandler) > 0):
                statsValDict = self._statsHandler.calculate(item,
                                                            plot,
                                                            self._statsOnVisibleData)
                for statName, statVal in list(statsValDict.items()):
                    self._statQlineEdit[statName].setText(statVal)

    def _zoomPlotChanged(self, event):
        if self._statsOnVisibleData is True:
            if 'event' in event and event['event'] == 'limitsChanged':
                self._updateStats()

    def setStatsOnVisibleData(self, b):
        """Toggle computation of statistics on whole data or only visible ones.

        .. warning:: When visible data is activated we will process to a simple
                     filtering of visible data by the user. The filtering is a
                     simple data sub-sampling. No interpolation is made to fit
                     data to boundaries.

        :param bool b: True if we want to apply statistics only on visible data
        """
        if self._statsOnVisibleData != b:
            self._statsOnVisibleData = b
            self._updateStats()


class BasicLineStatsWidget(LineStatsWidget):
    """
    Widget defining a simple set of :class:`Stat` to be displayed on a
    :class:`LineStatsWidget`.
    """
    STATS = StatsHandler((
        (statsmdl.StatMin(), StatFormatter()),
        statsmdl.StatCoordMin(),
        (statsmdl.StatMax(), StatFormatter()),
        statsmdl.StatCoordMax(),
        (('std', numpy.std), StatFormatter()),
        (('mean', numpy.mean), StatFormatter()),
        statsmdl.StatCOM()
    ))

    def __init__(self, parent=None, plot=None, stats=STATS, kind='curve',
                 statsOnVisibleData=False):
        LineStatsWidget.__init__(self, parent=parent, plot=plot, kind=kind,
                                 statsOnVisibleData=statsOnVisibleData)
        if stats is not None:
            self.setStats(stats)


class BasicGridStatsWidget(LineStatsWidget):
    """
    pymca like widget
    """
    STATS = StatsHandler((
        (statsmdl.StatMin(), StatFormatter()),
        statsmdl.StatCoordMin(),
        (statsmdl.StatMax(), StatFormatter()),
        statsmdl.StatCoordMax(),
        (('std', numpy.std), StatFormatter()),
        (('mean', numpy.mean), StatFormatter()),
        statsmdl.StatCOM()
    ))

    def __init__(self, parent=None, plot=None, stats=STATS, kind='curve',
                 statsOnVisibleData=False, width=4):
        LineStatsWidget.__init__(self, parent=parent, plot=plot, kind=kind,
                                 statsOnVisibleData=statsOnVisibleData)
        self._n_statistics_per_line = width
        self.setLayout(qt.QGridLayout())
        if stats is not None:
            self.setStats(stats)
