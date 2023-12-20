# /*##########################################################################
#
# Copyright (c) 20023 European Synchrotron Radiation Facility
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
PlotToolButton to measure a distance in a plot
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "30/10/2023"


import logging
import numpy

from silx.gui import icons

from .PlotToolButton import PlotToolButton

from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot.items.roi import LineROI


_logger = logging.getLogger(__name__)


class RulerToolButton(PlotToolButton):
    """
    Button to active measurement between two point of the plot

    An instance of `RulerToolButton` can be added to a plot toolbar like:
    .. code-block:: python

        plot = Plot2D()

        rulerButton = RulerToolButton(parent=plot, plot=plot)
        plot.toolBar().addWidget(rulerButton)
    """

    class RulerROI(LineROI):
        def __init__(self, parent=None):
            super().__init__(parent)
            self._formatFunction = None

        def registerFormatFunction(self, fct):
            """fct is expected to be a function taking (startPoint, endPoint) as parameter"""
            self._formatFunction = fct

        def setEndPoints(self, startPoint, endPoint):
            super().setEndPoints(startPoint=startPoint, endPoint=endPoint)
            if self._formatFunction is not None:
                ruler_text = self._formatFunction(
                    startPoint=startPoint, endPoint=endPoint
                )
                self._updateText(ruler_text)

    def __init__(
        self,
        parent=None,
        plot=None,
        color: str = "yellow",
    ):
        self.__color = color
        super().__init__(parent=parent, plot=plot)
        self.setCheckable(True)
        self._roiManager = None
        self._lastRoiCreated = None
        self.setIcon(icons.getQIcon("ruler"))
        self.toggled.connect(self._callback)
        self._connectPlot(plot)

    def setPlot(self, plot):
        return super().setPlot(plot)

    def _callback(self, *args, **kwargs):
        if not self._roiManager:
            return
        if self._lastRoiCreated is not None:
            self._lastRoiCreated.setVisible(self.isChecked())
        if self.isChecked():
            self._roiManager.start(
                self.RulerROI,
                self,
            )
            self.__interactiveModeStarted(self._roiManager)
        else:
            source = self._roiManager.getInteractionSource()
            if source is self:
                self._roiManager.stop()

    def __interactiveModeStarted(self, roiManager):
        roiManager.sigInteractiveModeFinished.connect(self.__interactiveModeFinished)

    def __interactiveModeFinished(self):
        roiManager = self._roiManager
        if roiManager is not None:
            roiManager.sigInteractiveModeFinished.disconnect(
                self.__interactiveModeFinished
            )
        self.setChecked(False)

    def _connectPlot(self, plot):
        """
        Called when the plot is connected to the widget

        :param plot: :class:`.PlotWidget` instance
        """
        if plot is None:
            return
        self._roiManager = RegionOfInterestManager(plot)
        self._roiManager.setColor(self.__color)  # Set the color of ROI
        self._roiManager.sigRoiAdded.connect(self._registerCurrentROI)

    def _disconnectPlot(self, plot):
        if plot and self._lastRoiCreated is not None:
            self._roiManager.removeRoi(self._lastRoiCreated)
            self._lastRoiCreated = None
        return super()._disconnectPlot(plot)

    def _registerCurrentROI(self, currentRoi):
        if self._lastRoiCreated is None:
            self._lastRoiCreated = currentRoi
            self._lastRoiCreated.registerFormatFunction(self.buildDistanceText)
        elif currentRoi != self._lastRoiCreated and self._roiManager is not None:
            self._roiManager.removeRoi(self._lastRoiCreated)
            self._lastRoiCreated = currentRoi
            self._lastRoiCreated.registerFormatFunction(self.buildDistanceText)

    def buildDistanceText(self, startPoint, endPoint):
        """
        define the text to be displayed by the ruler.
        It can be redefine to modify precision or handle other parameters
        (handling pixel size to display metric distance, display distance on each distance - for non-square pixels...)
        """
        distance = numpy.linalg.norm(endPoint - startPoint)
        return f"{distance: .1f}px"
