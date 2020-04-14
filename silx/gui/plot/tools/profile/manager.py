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
"""This module provides a manager to compute and display profiles.
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
    """Runner to process profiles

    :param qt.QThreadPool threadPool: The thread which will be used to
        execute this runner. It is used to update the used signals
    :param ~silx.gui.plot.items.Item item: Item in which the profile is
        computed
    :param ~silx.gui.plot.tools.profile.core.ProfileRoiMixIn roi: ROI
        defining the profile shape and other characteristics
    """

    class _Signals(qt.QObject):
        """Signal holder"""
        resultReady = qt.Signal(object, object)
        runnerFinished = qt.Signal(object)

    def __init__(self, threadPool, item, roi):
        """Constructor
        """
        super(_RunnableComputeProfile, self).__init__()
        self._signals = self._Signals()
        self._signals.moveToThread(threadPool.thread())
        self._item = item
        self._roi = roi

    def autoDelete(self):
        return False

    def getRoi(self):
        """Returns the ROI in which the runner will compute a profile.

        :rtype: ~silx.gui.plot.tools.profile.core.ProfileRoiMixIn
        """
        return self._roi

    @property
    def resultReady(self):
        """Signal emitted when the result of the computation is available.

        This signal provides 2 values: The ROI, and the computation result.
        """
        return self._signals.resultReady

    @property
    def runnerFinished(self):
        """Signal emitted when runner have finished.

        This signal provides a single value: the runner itself.
        """
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
        """Set the profile ROI which it the source of the following data
        to display.

        :param ProfileRoiMixIn roi: The profile ROI data source
        """
        if roi is None:
            return
        self.__color = colors.rgba(roi.getColor())

    def _setImageProfile(self, data):
        """
        Setup the window to display a new profile data which is represented
        by an image.

        :param core.ImageProfileData data: Computed data profile
        """
        plot = self._getPlot2D()

        plot.clear()
        plot.setGraphTitle(data.title)
        plot.getXAxis().setLabel(data.xLabel)


        coords = data.coords
        colormap = data.colormap
        profileScale = (coords[-1] - coords[0]) / data.profile.shape[1], 1
        plot.addImage(data.profile,
                      legend="profile",
                      colormap=colormap,
                      origin=(coords[0], 0),
                      scale=profileScale)
        plot.getYAxis().setLabel("Frame index (depth)")

        self._showPlot2D()

    def _setCurveProfile(self, data):
        """
        Setup the window to display a new profile data which is represented
        by a curve.

        :param core.CurveProfileData data: Computed data profile
        """
        plot = self._getPlot1D()

        plot.clear()
        plot.setGraphTitle(data.title)
        plot.getXAxis().setLabel(data.xLabel)

        plot.addCurve(data.coords,
                      data.profile,
                      legend="level",
                      color=self.__color)

        self._showPlot1D()

    def _setRgbaProfile(self, data):
        """
        Setup the window to display a new profile data which is represented
        by a curve.

        :param core.RgbaProfileData data: Computed data profile
        """
        plot = self._getPlot1D()

        plot.clear()
        plot.setGraphTitle(data.title)
        plot.getXAxis().setLabel(data.xLabel)

        self._showPlot1D()

        plot.addCurve(data.coords, data.profile,
                      legend="level", color="black")
        plot.addCurve(data.coords, data.profile_r,
                      legend="red", color="red")
        plot.addCurve(data.coords, data.profile_g,
                      legend="green", color="green")
        plot.addCurve(data.coords, data.profile_b,
                      legend="blue", color="blue")
        if data.profile_a is not None:
            plot.addCurve(data.coords, data.profile_a, legend="alpha", color="gray")

    def clear(self):
        """Clear the window profile"""
        plot = self._getPlot1D(init=False)
        if plot is not None:
            plot.clear()
        plot = self._getPlot2D(init=False)
        if plot is not None:
            plot.clear()

    def setProfile(self, data):
        """
        Setup the window to display a new profile data.

        This method dispatch the result to a specific method according to the
        data type.

        :param data: Computed data profile
        """
        if data is None:
            self.clear()
        elif isinstance(data, core.ImageProfileData):
            self._setImageProfile(data)
        elif isinstance(data, core.RgbaProfileData):
            self._setRgbaProfile(data)
        elif isinstance(data, core.CurveProfileData):
            self._setCurveProfile(data)
        else:
            raise TypeError("Unsupported type %s" % type(data))


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

        self._computedProfiles = 0
        """Statistics for tests"""

        self.__itemTypes = []
        """Kind of items to use"""

        self.__tracking = False
        """Is the plot active items are tracked"""

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
        """
        Enable or disable the single profile mode.

        In single mode, the manager enforce a single ROI at the same
        time. A new one will remove the previous one.

        If this mode is not enabled, many ROIs can be created, and many
        profile windows will be displayed.
        """
        self.__singleProfileAtATime = enable

    def isSingleProfile(self):
        """
        Returns true if the manager is in a single profile mode.

        :rtype: bool
        """
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
        """Handle removed ROI"""
        # Filter out non profile ROIs
        if not isinstance(roi, core.ProfileRoiMixIn):
            return
        self.__removeProfile(roi)

    def createProfileAction(self, profileRoiClass, parent=None):
        """Create an action from a class of ProfileRoi

        :param core.ProfileRoiMixIn profileRoiClass: A class of a profile ROI
        :param qt.QObject parent: The parent of the created action.
        :rtype: qt.QAction
        """
        if not issubclass(profileRoiClass, core.ProfileRoiMixIn):
            raise TypeError("Type %s not expected" % type(profileRoiClass))
        roiManager = self.getRoiManager()
        action = roiManager.getInteractionModeAction(profileRoiClass,
                                                     parent=parent)
        if hasattr(profileRoiClass, "ICON"):
            action.setIcon(icons.getQIcon(profileRoiClass.ICON))
        if hasattr(profileRoiClass, "NAME"):
            action.setToolTip('Enables %s selection mode' % profileRoiClass.NAME)
        return action

    def createClearAction(self, parent):
        """Create an action to clean up the plot from the profile ROIs.

        :param qt.QObject parent: The parent of the created action.
        :rtype: qt.QAction
        """
        # Add clear action
        icon = icons.getQIcon('profile-clear')
        action = qt.QAction(icon, 'Clear profile', parent)
        action.setToolTip('Clear the profiles')
        action.setCheckable(False)
        action.triggered.connect(self.clearProfile)
        return action

    def createImageActions(self, parent):
        """Create actions designed for image items. This actions created
        new ROIs.

        :param qt.QObject parent: The parent of the created action.
        :rtype: List[qt.QAction]
        """
        profileClasses = [
            rois.ProfileImageHorizontalLineROI,
            rois.ProfileImageVerticalLineROI,
            rois.ProfileImageLineROI,
            rois.ProfileImageCrossROI,
            ]
        return [self.createProfileAction(pc, parent=parent) for pc in profileClasses]

    def createScatterActions(self, parent):
        """Create actions designed for scatter items. This actions created
        new ROIs.

        :param qt.QObject parent: The parent of the created action.
        :rtype: List[qt.QAction]
        """
        profileClasses = [
            rois.ProfileScatterHorizontalLineROI,
            rois.ProfileScatterVerticalLineROI,
            rois.ProfileScatterLineROI,
            rois.ProfileScatterCrossROI,
            ]
        return [self.createProfileAction(pc, parent=parent) for pc in profileClasses]

    def createScatterSliceActions(self, parent):
        """Create actions designed for regular scatter items. This actions
        created new ROIs.

        This ROIs was designed to use the input data without interpolation,
        like you could do with an image.

        :param qt.QObject parent: The parent of the created action.
        :rtype: List[qt.QAction]
        """
        profileClasses = [
            rois.ProfileScatterHorizontalSliceROI,
            rois.ProfileScatterVerticalSliceROI,
            rois.ProfileScatterCrossSliceROI,
            ]
        return [self.createProfileAction(pc, parent=parent) for pc in profileClasses]

    def createImageStackActions(self, parent):
        """Create actions designed for stack image items. This actions
        created new ROIs.

        This ROIs was designed to create both profile on the displayed image
        and profile on the full stack (2D result).

        :param qt.QObject parent: The parent of the created action.
        :rtype: List[qt.QAction]
        """
        profileClasses = [
            rois.ProfileImageStackHorizontalLineROI,
            rois.ProfileImageStackVerticalLineROI,
            rois.ProfileImageStackLineROI,
            rois.ProfileImageStackCrossROI,
            ]
        return [self.createProfileAction(pc, parent=parent) for pc in profileClasses]

    def createEditorAction(self, parent):
        """Create an action containing GUI to edit the selected profile ROI.

        :param qt.QObject parent: The parent of the created action.
        :rtype: qt.QAction
        """
        action = editors.ProfileRoiEditorAction(parent)
        action.setRoiManager(self.getRoiManager())
        return action

    def setItemType(self, image=False, scatter=False):
        """Set the item type to use and select the active one.

        :param bool image: Image item are allowed
        :param bool scatter: Scatter item are allowed
        """
        self.__itemTypes = []
        plot = self.getPlotWidget()
        item = None
        if image:
            self.__itemTypes.append("image")
            item = plot.getActiveImage()
        if scatter:
            self.__itemTypes.append("scatter")
            if item is None:
                item = plot.getActiveScatter()
        self.setPlotItem(item)

    def setActiveItemTracking(self, tracking):
        """Enable/disable the tracking of the active item of the plot.

        :param bool tracking: Tracking mode
        """
        if self.__tracking == tracking:
            return
        plot = self.getPlotWidget()
        if self.__tracking:
            plot.sigActiveImageChanged.disconnect(self._activeImageChanged)
            plot.sigActiveScatterChanged.disconnect(self._activeScatterChanged)
        self.__tracking = tracking
        if self.__tracking:
            plot.sigActiveImageChanged.connect(self.__activeImageChanged)
            plot.sigActiveScatterChanged.connect(self.__activeScatterChanged)

    def __activeImageChanged(self, previous, legend):
        """Handle plot item selection"""
        if "image" in self.__itemTypes:
            plot = self.getPlotWidget()
            item = plot.getImage(legend)
            self.setPlotItem(item)

    def __activeScatterChanged(self, previous, legend):
        """Handle plot item selection"""
        if "scatter" in self.__itemTypes:
            plot = self.getPlotWidget()
            item = plot.getScatter(legend)
            self.setPlotItem(item)

    def __addProfile(self, profileRoi):
        """Add a new ROI to the manager."""
        if profileRoi.getParentRoi() is None:
            if self.__singleProfileAtATime:
                # FIXME: It would be good to reuse the windows to avoid blinking
                self.clearProfile()

        profileRoi._setProfileManager(self)
        self._rois.append(profileRoi)
        self.requestUpdateProfile(profileRoi)

    def __removeProfile(self, profileRoi):
        """Remove a ROI from the manager."""
        window = self._disconnectProfileWindow(profileRoi)
        if window is not None:
            geometry = window.geometry()
            self._previousWindowGeometry.append(geometry)
            window.deleteLater()
        if profileRoi in self._rois:
            self._rois.remove(profileRoi)

    def _disconnectProfileWindow(self, profileRoi):
        """Handle profile window close."""
        window = profileRoi.getProfileWindow()
        profileRoi.setProfileWindow(None)
        return window

    def clearProfile(self):
        """Clear the associated ROI profile"""
        roiManager = self.getRoiManager()
        for roi in list(self._rois):
            if roi.getParentRoi() is not None:
                # Skip sub ROIs, it will be removed by their parents
                continue
            roiManager.removeRoi(roi)

    def hasPendingOperations(self):
        """Returns true if a thread is still computing a profile.

        :rtype: bool
        """
        return len(self._pendingRunners) > 0

    def requestUpdateAllProfile(self):
        """Request to update the profile of all the managed ROIs.
        """
        for roi in self._rois:
            self.requestUpdateProfile(roi)

    def requestUpdateProfile(self, profileRoi):
        """Request to update a specific profile ROI.

        :param ~core.ProfileRoiMixIn profileRoi:
        """
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
        """Remove a thread pool runner from the list of hold tasks.

        Called at the termination of the runner.
        """
        if runner in self._pendingRunners:
            self._pendingRunners.remove(runner)

    def __displayResult(self, roi, profileData):
        """Display the result of a ROI.

        :param ~core.ProfileRoiMixIn profileRoi: A managed ROI
        :param ~core.CurveProfileData profileData: Computed data profile
        """
        self._computedProfiles = self._computedProfiles + 1
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
        """Set the plot item focused by the profile manager.

        :param ~silx.gui.plot.items.Item item: A plot item
        """
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
        """Handle item changes.
        """
        if changeType in (items.ItemChangedType.DATA,
                          items.ItemChangedType.POSITION,
                          items.ItemChangedType.SCALE):
            self.requestUpdateAllProfile()

    def getPlotItem(self):
        """Returns the item focused by the profile manager.

        :rtype: ~silx.gui.plot.items.Item
        """
        if self._item is None:
            return None
        item = self._item()
        if item is None:
            self._item = None
        return item

    def getPlotWidget(self):
        """The plot associated to the profile manager.

        :rtype: ~silx.gui.plot.PlotWidget
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
        """Create a new profile window.

        :param ~core.ProfileRoiMixIn roi: A managed ROI
        :rtype: ~ProfileMainWindow
        """
        plot = self.getPlotWidget()
        return ProfileMainWindow(plot)

    def initProfileWindow(self, profileWindow):
        """This function is called just after the profile window creation in
        order to initialize the window location.

        :param ~ProfileMainWindow profileWindow:
            The profile window to initialize.
        """
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
