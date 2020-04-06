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

from silx.utils.weakref import WeakMethodProxy
from silx.gui import icons
from silx.gui.plot import PlotWidget
from silx.gui.plot.ProfileMainWindow import ProfileMainWindow as _ProfileMainWindow
from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot import items
from silx.gui.qt import silxGlobalThreadPool
from silx.gui.qt import inspect
from . import rois
from . import core
from . import editors


_logger = logging.getLogger(__name__)


class _RunnableComputeProfile(qt.QRunnable):
    """Runner to process profiles"""

    class _Signals(qt.QObject):
        """Signal holder"""
        resultReady = qt.Signal(object, object)
        runnerFinished = qt.Signal(object)

    def autoDelete(self):
        return False

    def __init__(self, threadPool, item, roi):
        """Constructor

        :param LoadingItemWorker worker: Object holding data and signals
        """
        super(_RunnableComputeProfile, self).__init__()
        self._signals = self._Signals()
        self._signals.moveToThread(threadPool.thread())
        self._item = item
        self._roi = roi

    def getRoi(self):
        return self._roi

    @property
    def resultReady(self):
        return self._signals.resultReady

    @property
    def runnerFinished(self):
        return self._signals.runnerFinished

    def run(self):
        """Process the profile computation.
        """
        try:
            profileData = self._roi.computeProfile(self._item)
        except Exception:
            _logger.error("Error while computing profile", exc_info=True)
        else:
            self.resultReady.emit(self._roi, profileData)
        self.runnerFinished.emit(self)


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
        elif hasattr(profileData, "r"):
            plot.addCurve(coords, profile[0], legend=profileName, color="black")
            red = profileData.r
            green = profileData.g
            blue = profileData.b
            plot.addCurve(coords, red[0], legend=profileName+"_r", color="red")
            plot.addCurve(coords, green[0], legend=profileName+"_g", color="green")
            plot.addCurve(coords, blue[0], legend=profileName+"_b", color="blue")
            if hasattr(profileData, "a"):
                alpha = profileData.a
                plot.addCurve(coords, alpha[0], legend=profileName+"_a", color="gray")
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
        self._pendingRunners = []
        """List of ROIs which have to be updated"""

        self._item = None
        """The selected item"""

        self.__singleProfileAtATime = True
        """When it's true, only a single profile is displayed at a time."""

        self._previousWindowGeometry = []

        # Listen to plot limits changed
        plot.getXAxis().sigLimitsChanged.connect(self.requestUpdateAllProfile)
        plot.getYAxis().sigLimitsChanged.connect(self.requestUpdateAllProfile)

        roiManager.sigInteractiveModeFinished.connect(self.__interactionFinished)
        roiManager.sigRoiAdded.connect(self.__roiAdded)
        roiManager.sigRoiAboutToBeRemoved.connect(self.__roiRemoved)

    def setSingleProfile(self, enable):
        self.__singleProfileAtATime = enable

    def isSingleProfile(self):
        return self.__singleProfileAtATime

    def __interactionFinished(self):
        """Handle end of interactive mode"""
        pass

    def __roiAdded(self, roi):
        """Handle new ROI"""
        # Filter out non profile ROIs
        if not isinstance(roi, core.ProfileRoiMixIn):
            return
        self.__addProfile(roi)

    def __roiRemoved(self, roi):
        # Filter out non profile ROIs
        if not isinstance(roi, core.ProfileRoiMixIn):
            return
        self.__removeProfile(roi)

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

    def createEditorAction(self, parent):
        action = editors.ProfileRoiEditAction(parent)
        action.setRoiManager(self.getRoiManager())
        return action

    def __addProfile(self, profileRoi):
        if profileRoi.getFocusProxy() is None:
            if self.__singleProfileAtATime:
                # FIXME: It would be good to reuse the windows to avoid blinking
                self.clearProfile()
            # FIXME: This should be removed
            profileRoi.setName('Profile')
            profileRoi.setEditable(True)

        profileRoi._setProfileManager(self)
        self._rois.append(profileRoi)
        self.requestUpdateProfile(profileRoi)

    def __removeProfile(self, profileRoi):
        window = self._disconnectProfileWindow(profileRoi)
        if window is not None:
            geometry = window.geometry()
            self._previousWindowGeometry.append(geometry)
            window.deleteLater()
        if profileRoi in self._rois:
            self._rois.remove(profileRoi)

    def _disconnectProfileWindow(self, profileRoi):
        window = profileRoi.getProfileWindow()
        profileRoi.setProfileWindow(None)
        return window

    def clearProfile(self):
        """Clear the associated ROI profile"""
        roiManager = self.getRoiManager()
        for roi in list(self._rois):
            if roi.getFocusProxy() is not None:
                # Skip sub ROIs, it will be removed by their parents
                continue
            roiManager.removeRoi(roi)

    def hasPendingOperations(self):
        return len(self._pendingRunners) > 0

    def requestUpdateAllProfile(self):
        for roi in self._rois:
            self.requestUpdateProfile(roi)

    def requestUpdateProfile(self, profileRoi):
        if profileRoi.computeProfile is None:
            return
        threadPool = silxGlobalThreadPool()

        # Clean up deprecated runners
        for runner in list(self._pendingRunners):
            if not inspect.isValid(runner):
                self._pendingRunners.remove(runner)
                continue
            if runner.getRoi() is profileRoi:
                if threadPool.tryTake(runner):
                    self._pendingRunners.remove(runner)

        item = self.getPlotItem()
        if item is None:
            # FIXME: It means the result is None profile window have to be updated
            return

        runner = _RunnableComputeProfile(threadPool, item, profileRoi)
        runner.runnerFinished.connect(self.__cleanUpRunner)
        runner.resultReady.connect(self.__displayResult)
        self._pendingRunners.append(runner)
        threadPool.start(runner)

    def __cleanUpRunner(self, runner):
        if runner in self._pendingRunners:
            self._pendingRunners.remove(runner)

    def __displayResult(self, roi, profileData):
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
        self._pendingRunners = []

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
