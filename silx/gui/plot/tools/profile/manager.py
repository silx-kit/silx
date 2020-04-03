# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2019 European Synchrotron Radiation Facility
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
"""This module prvide a manager to compute profiles.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "28/06/2018"

import logging
import weakref

from silx.gui import qt
from silx.gui import colors
from ....utils.concurrent import submitToQtMainThread

from silx.utils.weakref import WeakMethodProxy
from silx.gui import icons
from silx.gui.plot import PlotWidget
from silx.gui.plot.ProfileMainWindow import ProfileMainWindow as _ProfileMainWindow
from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot import items
from . import rois
from . import core


_logger = logging.getLogger(__name__)


class ProfileMainWindow(_ProfileMainWindow):

    def setRoiProfile(self, roi):
        if roi is None:
            return
        self.__color = colors.rgba(roi.getColor())

    def setProfile(self, profileData):
        if profileData is None:
            plot = self.getPlot()
            plot.clear()
            return

        dataIs3D = False
        if hasattr(profileData, "currentData"):
            # FIXME: currentData is not needed for now, it could be removed
            currentData = profileData.currentData
            dataIs3D = len(currentData.shape) > 2

        profileName = profileData.profileName
        xLabel = profileData.xLabel
        coords = profileData.coords
        profile = profileData.profile

        plot = self.getPlot()

        plot.clear()
        plot.setGraphTitle(profileName)
        plot.getXAxis().setLabel(xLabel)

        if dataIs3D:
            colormap = profileData.colormap
            profileScale = (coords[-1] - coords[0]) / profile.shape[1], 1
            plot.addImage(profile,
                          legend=profileName,
                          colormap=colormap,
                          origin=(coords[0], 0),
                          scale=profileScale)
            plot.getYAxis().setLabel("Frame index (depth)")
        else:
            plot.addCurve(coords,
                                 profile[0],
                                 legend=profileName,
                                 color=self.__color)


class ProfileManager(qt.QObject):
    """Base class for profile management tools

    :param plot: :class:`~silx.gui.plot.PlotWidget` on which to operate.
    :param plot: :class:`~silx.gui.plot.tools.roi.RegionOfInterestManager`
        on which to operate.
    """
    def __init__(self, parent=None, plot=None, roiManager=None):
        super(ProfileManager, self).__init__(parent)

        assert isinstance(plot, PlotWidget)
        self._plotRef = weakref.ref(
            plot, WeakMethodProxy(self.__plotDestroyed))

        # Set-up interaction manager
        if roiManager is None:
            roiManager = RegionOfInterestManager(plot)

        self._roiManagerRef = weakref.ref(roiManager)
        self._rois = []
        self._pendingTasks = []
        """List of ROIs which have to be updated"""

        self._item = None
        """The selected item"""

        self._previousWindowGeometry = []

        # Listen to plot limits changed
        plot.getXAxis().sigLimitsChanged.connect(self.requestUpdateAllProfile)
        plot.getYAxis().sigLimitsChanged.connect(self.requestUpdateAllProfile)

        roiManager.sigInteractiveModeFinished.connect(self.__interactionFinished)
        roiManager.sigRoiAdded.connect(self.__roiAdded)

    def __interactionFinished(self):
        """Handle end of interactive mode"""
        pass

    def __roiAdded(self, roi):
        """Handle new ROI"""
        # Remove any other ROI
        if not isinstance(roi, core.ProfileRoiMixIn):
            return

        # self.clearProfile()

        roi.setName('Profile')
        roi.setEditable(True)
        self.addProfile(roi)

    def createActions(self, parent):
        actions = []

        roiManager = self.getRoiManager()
        action = roiManager.getInteractionModeAction(rois.ProfileImageHorizontalLineROI)
        action.setIcon(icons.getQIcon('shape-horizontal'))
        action.setToolTip('Enables horizontal line profile selection mode')
        actions.append(action)

        roiManager = self.getRoiManager()
        action = roiManager.getInteractionModeAction(rois.ProfileImageVerticalLineROI)
        action.setIcon(icons.getQIcon('shape-vertical'))
        action.setToolTip('Enables vertical line profile selection mode')
        actions.append(action)

        roiManager = self.getRoiManager()
        action = roiManager.getInteractionModeAction(rois.ProfileImageLineROI)
        action.setIcon(icons.getQIcon('shape-diagonal'))
        action.setToolTip('Enables line profile selection mode')
        actions.append(action)

        # Add clear action
        icon = icons.getQIcon('profile-clear')
        action = qt.QAction(icon, 'Clear profile', parent)
        action.setToolTip('Clear the profiles')
        action.setCheckable(False)
        action.triggered.connect(self.clearProfile)
        actions.append(action)

        return actions

    def addProfile(self, profileRoi):
        roiManager = self.getRoiManager()
        if profileRoi not in roiManager.getRois():
            roiManager.addRoi(profileRoi)
        profileRoi._setProfileManager(self)
        self._rois.append(profileRoi)
        self.requestUpdateProfile(profileRoi)

    def _disconnectProfileWindow(self, profileRoi):
        window = profileRoi.getProfileWindow()
        profileRoi.setProfileWindow(None)
        return window

    def removeProfile(self, profileRoi):
        window = self._disconnectProfileWindow(profileRoi)
        if window is not None:
            geometry = window.geometry()
            self._previousWindowGeometry.append(geometry)
            window.deleteLater()
        roiManager = self.getRoiManager()
        roiManager.removeRoi(profileRoi)
        self._rois.remove(profileRoi)

    def clearProfile(self):
        """Clear the associated ROI profile"""
        for roi in list(self._rois):
            self.removeProfile(roi)

    def hasPendingOperations(self):
        return len(self._pendingTasks) > 0

    def requestUpdateAllProfile(self):
        for roi in self._rois:
            self.requestUpdateProfile(roi)

    def requestUpdateProfile(self, profileRoi):
        if profileRoi in self._pendingTasks:
            self._pendingTasks.remove(profileRoi)
        self._pendingTasks.append(profileRoi)
        # FIXME: do it asynchronously
        self.__processTasks()

    def __processTasks(self):
        item = self.getPlotItem()
        if item is None:
            # FIXME: It means the result is None profile window have to be updated
            return
        while len(self._pendingTasks) > 0:
            roi = self._pendingTasks.pop(0)
            profileData = roi.computeProfile(item)
            submitToQtMainThread(self.__profileComputedSafe, roi, profileData)

    def __profileComputedSafe(self, *args, **kwargs):
        try:
            return self.__profileComputed(*args, **kwargs)
        except Exception:
            _logger.error("Backtrace", exc_info=True)

    def __profileComputed(self, roi, profileData):
        window = roi.getProfileWindow()
        if window is None:
            # FIXME: reach geometry from the previous closed window
            window = self.createProfileWindow(roi)
            self.initProfileWindow(window)
            window.show()
            roi.setProfileWindow(window)
        window.setProfile(profileData)

    def __plotDestroyed(self, ref):
        """Handle finalization of PlotWidget

        :param ref: weakref to the plot
        """
        self._plotRef = None
        self._roiManagerRef = None
        self._pendingTasks = []

    def setPlotItem(self, item):
        previous = self.getPlotItem()
        if previous is item:
            return
        if item is None:
            self._item = None
        else:
            item.sigItemChanged.connect(self.__itemChanged)
            self._item = weakref.ref(item)
        self.requestUpdateAllProfile()

    def __itemChanged(self, changeType):
        if changeType in (items.ItemChangedType.DATA,
                          items.ItemChangedType.POSITION,
                          items.ItemChangedType.SCALE):
            self.requestUpdateAllProfile()

    def getPlotItem(self):
        if self._item is None:
            return None
        item = self._item()
        if item is None:
            self._item = None
        return item

    def getPlotWidget(self):
        """The :class:`~silx.gui.plot.PlotWidget` associated to the toolbar.

        :rtype: Union[~silx.gui.plot.PlotWidget,None]
        """
        if self._plotRef is None:
            return None
        plot = self._plotRef()
        if plot is None:
            self._plotRef = None
        return plot

    def getRoiManager(self):
        """Returns the used ROI manager

        :rtype: RegionOfInterestManager
        """
        return self._roiManagerRef()

    def createProfileWindow(self, roi):
        """Create new profile window.
        """
        plot = self.getPlotWidget()
        return ProfileMainWindow(plot)

    def initProfileWindow(self, profileWindow):
        """This function is called just after the profile window creation in
        order to initialize the window location."""
        profileWindow.show()
        profileWindow.raise_()

        if len(self._previousWindowGeometry) > 0:
            geometry = self._previousWindowGeometry.pop()
            profileWindow.setGeometry(geometry)
            return

        window = self.getPlotWidget().window()
        winGeom = window.frameGeometry()
        qapp = qt.QApplication.instance()
        desktop = qapp.desktop()
        screenGeom = desktop.availableGeometry(window)
        spaceOnLeftSide = winGeom.left()
        spaceOnRightSide = screenGeom.width() - winGeom.right()

        frameGeometry = profileWindow.frameGeometry()
        profileWindowWidth = frameGeometry.width()
        if profileWindowWidth < spaceOnRightSide:
            # Place profile on the right
            profileWindow.move(winGeom.right(), winGeom.top())
        elif profileWindowWidth < spaceOnLeftSide:
            # Place profile on the left
            left = max(0, winGeom.left() - profileWindowWidth)
            profileWindow.move(left, winGeom.top())
