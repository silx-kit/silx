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
"""This module provides the base class for profile toolbars."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "07/06/2018"


import logging
import weakref

import numpy

from silx.utils.weakref import WeakMethodProxy
from silx.gui import qt, icons, colors
from silx.gui.plot import PlotWidget, items
from silx.gui.plot.ProfileMainWindow import ProfileMainWindow
from silx.gui.plot.tools.roi import RegionOfInterestManager


_logger = logging.getLogger(__name__)


class _BaseProfileToolBar(qt.QToolBar):
    """Base class for QToolBar plot profiling tools

    :param parent: See :class:`QToolBar`.
    :param plot: :class:`PlotWindow` instance on which to operate.
    :param str title: See :class:`QToolBar`.
    """

    sigProfileChanged = qt.Signal()
    """Signal emitted when the profile has changed"""

    def __init__(self, parent=None, plot=None, title=''):
        super(_BaseProfileToolBar, self).__init__(title, parent)

        self.__profile = None
        self.__profileTitle = ''

        assert isinstance(plot, PlotWidget)
        self._plotRef = weakref.ref(
            plot, WeakMethodProxy(self.__plotDestroyed))

        self._profileWindow = None

        # Set-up interaction manager
        roiManager = RegionOfInterestManager(plot)
        self._roiManagerRef = weakref.ref(roiManager)

        roiManager.sigInteractiveModeFinished.connect(
            self.__interactionFinished)
        roiManager.sigRegionOfInterestChanged.connect(self.updateProfile)
        roiManager.sigRegionOfInterestAdded.connect(self.__roiAdded)

        # Add interactive mode actions
        for kind, icon in (
                ('hline', 'shape-horizontal'),
                ('vline', 'shape-vertical'),
                ('line', 'shape-diagonal')):
            action = roiManager.getInteractionModeAction(kind)
            action.setIcon(icons.getQIcon(icon))
            self.addAction(action)

        # Add clear action
        action = qt.QAction(icons.getQIcon('profile-clear'),
                            'Clear Profile', self)
        action.setToolTip('Clear the profile Region of interest')
        action.setCheckable(False)
        action.triggered.connect(self.clearProfile)
        self.addAction(action)

        # Initialize color
        self._color = None
        self.setColor('red')

        # Listen to plot limits changed
        plot.getXAxis().sigLimitsChanged.connect(self.updateProfile)
        plot.getYAxis().sigLimitsChanged.connect(self.updateProfile)

        # Listen to plot scale
        plot.getXAxis().sigScaleChanged.connect(self.__plotAxisScaleChanged)
        plot.getYAxis().sigScaleChanged.connect(self.__plotAxisScaleChanged)

        self.setDefaultProfileWindowEnabled(True)

    def getProfileData(self, copy=True):
        """Returns the profile data as (x, y) or None

        :param bool copy: True to get a copy,
                          False to get internal arrays (do not modify)
        :rtype: Union[List[numpy.ndarray],None]
        """
        if self.__profile is None:
            return None
        else:
            return (numpy.array(self.__profile[0], copy=copy),
                    numpy.array(self.__profile[1], copy=copy))

    def getProfileTitle(self):
        """Returns the profile title

        :rtype: str
        """
        return self.__profileTitle

    # Handle plot reference

    def __plotDestroyed(self, ref):
        """Handle finalization of PlotWidget

        :param ref: weakref to the plot
        """
        self._plotRef = None
        self.setEnabled(False)  # Profile is pointless
        for action in self.actions():  # TODO useful?
            self.removeAction(action)

    def getPlotWidget(self):
        """The :class:`~silx.gui.plot.PlotWidget` associated to the toolbar.

        :rtype: Union[~silx.gui.plot.PlotWidget,None]
        """
        return None if self._plotRef is None else self._plotRef()

    def _getRoiManager(self):
        """Returns the used ROI manager

        :rtype: RegionOfInterestManager
        """
        return self._roiManagerRef()

    # Profile Plot

    def isDefaultProfileWindowEnabled(self):
        """Returns True if the default floating profile window is used

        :rtype
        """
        return self.getDefaultProfileWindow() is not None

    def setDefaultProfileWindowEnabled(self, enabled):
        """Set whether to use or not the default floating profile window.

        :param bool enabled: True to use, False to disable
        """
        if self.isDefaultProfileWindowEnabled() != enabled:
            if enabled:
                self._profileWindow = ProfileMainWindow(self)
                self._profileWindow.sigClose.connect(self.clearProfile)
                self.sigProfileChanged.connect(self.__updateDefaultProfilePlot)

            else:
                self.sigProfileChanged.disconnect(self.__updateDefaultProfilePlot)
                self._profileWindow.sigClose.disconnect(self.clearProfile)
                self._profileWindow.close()
                self._profileWindow = None

    def getDefaultProfileWindow(self):
        """Returns the default floating profile window if in use else None.

        See :meth:`isDefaultProfileWindowEnabled`

        :rtype: Union[ProfileMainWindow,None]
        """
        return self._profileWindow

    def __updateDefaultProfilePlot(self):
        """Update the plot of the default profile window"""
        profileWindow = self.getDefaultProfileWindow()
        if profileWindow is None:
            return

        profilePlot = profileWindow.getPlot()
        if profilePlot is None:
            return

        profilePlot.clear()
        profilePlot.setGraphTitle(self.getProfileTitle())

        profile = self.getProfileData(copy=False)
        if profile is not None:
            x, y = profile
            profilePlot.addCurve(
                x, y, legend='Profile', color=self._color)

        self._showDefaultProfileWindow()

    def _showDefaultProfileWindow(self):
        """If profile window was created by this toolbar,
        try to avoid overlapping with the toolbar's parent window.
        """
        profileWindow = self.getDefaultProfileWindow()
        roiManager = self._getRoiManager()
        if profileWindow is None or roiManager is None:
            return

        if roiManager.isStarted() and not profileWindow.isVisible():
            profileWindow.show()
            profileWindow.raise_()

            window = self.window()
            winGeom = window.frameGeometry()
            qapp = qt.QApplication.instance()
            desktop = qapp.desktop()
            screenGeom = desktop.availableGeometry(self)
            spaceOnLeftSide = winGeom.left()
            spaceOnRightSide = screenGeom.width() - winGeom.right()

            frameGeometry = profileWindow.frameGeometry()
            profileWindowWidth = frameGeometry.width()
            if profileWindowWidth < spaceOnRightSide:
                # Place profile on the right
                profileWindow.move(winGeom.right(), winGeom.top())
            elif profileWindowWidth < spaceOnLeftSide:
                # Place profile on the left
                profileWindow.move(
                    max(0, winGeom.left() - profileWindowWidth), winGeom.top())

    # Handle plot in log scale

    def __plotAxisScaleChanged(self, scale):
        """Handle change of axis scale in the plot widget"""
        plot = self.getPlotWidget()
        if plot is None:
            return

        xScale = plot.getXAxis().getScale()
        yScale = plot.getYAxis().getScale()

        if xScale == items.Axis.LINEAR and yScale == items.Axis.LINEAR:
            self.setEnabled(True)

        else:
            roiManager = self._getRoiManager()
            if roiManager is not None:
                roiManager.stop()  # Stop interactive mode

            self.clearProfile()
            self.setEnabled(False)

    # Profile color

    def getColor(self):
        """Returns the color used for the profile and ROI

        :rtype: QColor
        """
        return qt.QColor.fromRgbF(*self._color)

    def setColor(self, color):
        """Set the color to use for ROI and profile.

        :param color:
           Either a color name, a QColor, a list of uint8 or float in [0, 1].
        """
        self._color = colors.rgba(color)
        roiManager = self._getRoiManager()
        if roiManager is not None:
            roiManager.setColor(self._color)
            for roi in roiManager.getRegionOfInterests():
                roi.setColor(self._color)
        self.updateProfile()

    # Handle ROI manager

    def __interactionFinished(self, rois):
        """Handle end of interactive mode"""
        self.clearProfile()

        profileWindow = self.getDefaultProfileWindow()
        if profileWindow is not None:
            profileWindow.hide()

    def __roiAdded(self, roi):
        """Handle new ROI"""
        roi.setLabel('Profile')
        roi.setEditable(True)

        # Remove any other ROI
        roiManager = self._getRoiManager()
        if roiManager is not None:
            for regionOfInterest in list(roiManager.getRegionOfInterests()):
                if regionOfInterest is not roi:
                    roiManager.removeRegionOfInterest(regionOfInterest)

    def computeProfile(self, x0, y0, x1, y1):
        """Compute corresponding profile

        Override in subclass to compute profile

        :param float x0: Profile start point X coord
        :param float y0: Profile start point Y coord
        :param float x1: Profile end point X coord
        :param float y1: Profile end point Y coord
        :return: (x, y) profile data or None
        """
        return None

    def computeProfileTitle(self, x0, y0, x1, y1):
        """Compute corresponding plot title

        This can be overridden to change title behavior.

        :param float x0: Profile start point X coord
        :param float y0: Profile start point Y coord
        :param float x1: Profile end point X coord
        :param float y1: Profile end point Y coord
        :return: Title to use
        :rtype: str
        """
        if x0 == x1:
            title = 'X = %g; Y = [%g, %g]' % (x0, y0, y1)
        elif y0 == y1:
            title = 'Y = %g; X = [%g, %g]' % (y0, x0, x1)
        else:
            m = (y1 - y0) / (x1 - x0)
            b = y0 - m * x0
            title = 'Y = %g * X %+g' % (m, b)

        return title

    def updateProfile(self, *args):
        """Update profile according to current ROI"""
        roiManager = self._getRoiManager()
        if roiManager is None:
            roi = None
        else:
            rois = roiManager.getRegionOfInterests()
            roi = None if len(rois) == 0 else rois[0]

        if roi is None:
            self._setProfile(profile=None, title='')
            return

        kind = roi.getKind()

        # Get end points
        if kind == 'line':
            points = roi.getControlPoints()
            x0, y0 = points[0]
            x1, y1 = points[1]

        elif kind in ('hline', 'vline'):
            plot = self.getPlotWidget()
            if plot is None:
                self._setProfile(profile=None, title='')
                return

            if kind == 'hline':
                x0, x1 = plot.getXAxis().getLimits()
                y0 = y1 = roi.getControlPoints()[0, 1]

            elif kind == 'vline':
                x0 = x1 = roi.getControlPoints()[0, 0]
                y0, y1 = plot.getYAxis().getLimits()

        else:
            raise RuntimeError('Unsupported kind: {}'.format(kind))

        if x1 < x0 or (x1 == x0 and y1 < y0):
            # Invert points
            x0, y0, x1, y1 = x1, y1, x0, y0

        profile = self.computeProfile(x0, y0, x1, y1)
        title = self.computeProfileTitle(x0, y0, x1, y1)
        self._setProfile(profile=profile, title=title)

    def _setProfile(self, profile=None, title=''):
        """Set profile data and emit signal.

        :param profile:
        :param str title:
        """
        self.__profile = profile
        self.__profileTitle = title

        self.sigProfileChanged.emit()

    def clearProfile(self):
        """Clear the current line ROI and associated profile"""
        roiManager = self._getRoiManager()
        if roiManager is not None:
            roiManager.clearRegionOfInterests()

        self._setProfile(profile=None, title='')
