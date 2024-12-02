# /*##########################################################################
#
# Copyright (c) 2024 European Synchrotron Radiation Facility
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
:mod:`silx.gui.plot.actions.image` provides a set of QAction relative to data processing
and outputs for a :class:`.PlotWidget`.

The following QAction are available:

- :class:`AggregationModeAction`
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "19/09/2024"

import logging
from silx.gui import qt
from silx.gui import icons
from ..items.image_aggregated import ImageDataAggregated

_logger = logging.getLogger(__name__)

class AggregationModeAction(qt.QWidgetAction):
    """Action providing filters for an aggregated image"""

    sigAggregationModeChanged = qt.Signal()
    """Signal emitted when the aggregation mode has changed"""  

    def __init__(self, parent):
        qt.QWidgetAction.__init__(self, parent)

        toolButton = qt.QToolButton(parent)

        filterAction = qt.QAction(self)
        filterAction.setText("No filter")
        filterAction.setCheckable(True)
        filterAction.setChecked(True)
        filterAction.setProperty(
            "aggregation", ImageDataAggregated.Aggregation.NONE
        )
        densityNoFilterAction = filterAction

        filterAction = qt.QAction(self)
        filterAction.setText("Max filter")
        filterAction.setCheckable(True)
        filterAction.setProperty(
            "aggregation", ImageDataAggregated.Aggregation.MAX
        )
        densityMaxFilterAction = filterAction

        filterAction = qt.QAction(self)
        filterAction.setText("Mean filter")
        filterAction.setCheckable(True)
        filterAction.setProperty(
            "aggregation", ImageDataAggregated.Aggregation.MEAN
        )
        densityMeanFilterAction = filterAction

        filterAction = qt.QAction(self)
        filterAction.setText("Min filter")
        filterAction.setCheckable(True)
        filterAction.setProperty(
            "aggregation", ImageDataAggregated.Aggregation.MIN
        )
        densityMinFilterAction = filterAction

        densityGroup = qt.QActionGroup(self)
        densityGroup.setExclusive(True)
        densityGroup.addAction(densityNoFilterAction)
        densityGroup.addAction(densityMaxFilterAction)
        densityGroup.addAction(densityMeanFilterAction)
        densityGroup.addAction(densityMinFilterAction)
        densityGroup.triggered.connect(self._aggregationModeChanged)
        self.__densityGroup = densityGroup

        filterMenu = qt.QMenu(toolButton)
        filterMenu.addAction(densityNoFilterAction)
        filterMenu.addAction(densityMaxFilterAction)
        filterMenu.addAction(densityMeanFilterAction)
        filterMenu.addAction(densityMinFilterAction)

        toolButton.setPopupMode(qt.QToolButton.InstantPopup)
        toolButton.setMenu(filterMenu)
        toolButton.setText("Data filters")
        toolButton.setToolTip("Enable/disable filter on the image")
        icon = icons.getQIcon("aggregation-mode")
        toolButton.setIcon(icon)
        toolButton.setText("Pixel aggregation filter")

        self.setDefaultWidget(toolButton)

    def _aggregationModeChanged(self):
        self.sigAggregationModeChanged.emit()

    def setAggregationMode(self, mode):
        """Set an Aggregated enum from ImageDataAggregated"""
        for a in self.__densityGroup.actions():
            if a.property("aggregation") is mode:
                a.setChecked(True)

    def getAggregationMode(self):
        """Returns an Aggregated enum from ImageDataAggregated"""
        densityAction = self.__densityGroup.checkedAction()
        if densityAction is None:
            return ImageDataAggregated.Aggregation.NONE
        return densityAction.property("aggregation")
