# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018-2020 European Synchrotron Radiation Facility
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
from silx.gui import utils

from silx.utils.weakref import WeakMethodProxy
from silx.gui import icons
from silx.gui.plot import PlotWidget
from silx.gui.plot.tools.roi import RegionOfInterestManager
from silx.gui.plot.tools.roi import CreateRoiModeAction
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


class ProfileWindow(qt.QMainWindow):
    """
    Display a computed profile.

    The content can be described using :meth:`setRoiProfile` if the source of
    the profile is a profile ROI, and :meth:`setProfile` for the data content.
    """

    sigClose = qt.Signal()
    """Emitted by :meth:`closeEvent` (e.g. when the window is closed
    through the window manager's close icon)."""

    def __init__(self, parent=None, backend=None):
        qt.QMainWindow.__init__(self, parent=parent, flags=qt.Qt.Dialog)

        self.setWindowTitle('Profile window')
        self._plot1D = None
        self._plot2D = None
        self._backend = backend
        self._data = None

        widget = qt.QWidget()
        self._layout = qt.QStackedLayout(widget)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(widget)

    def prepareWidget(self, roi):
        """Called before the show to prepare the window to use with
        a specific ROI."""
        if isinstance(roi, rois._DefaultImageStackProfileRoiMixIn):
            profileType = roi.getProfileType()
        else:
            profileType = "1D"
        if profileType == "1D":
            self.getPlot1D()
        elif profileType == "2D":
            self.getPlot2D()

    def createPlot1D(self, parent, backend):
        """Inherit this function to create your own plot to render 1D
        profiles. The default value is a `Plot1D`.

        :param parent: The parent of this widget or None.
        :param backend: The backend to use for the plot.
                        See :class:`PlotWidget` for the list of supported backend.
        :rtype: PlotWidget
        """
        # import here to avoid circular import
        from ...PlotWindow import Plot1D
        plot = Plot1D(parent=parent, backend=backend)
        plot.setDataMargins(yMinMargin=0.1, yMaxMargin=0.1)
        plot.setGraphYLabel('Profile')
        plot.setGraphXLabel('')
        return plot

    def createPlot2D(self, parent, backend):
        """Inherit this function to create your own plot to render 2D
        profiles. The default value is a `Plot2D`.

        :param parent: The parent of this widget or None.
        :param backend: The backend to use for the plot.
                        See :class:`PlotWidget` for the list of supported backend.
        :rtype: PlotWidget
        """
        # import here to avoid circular import
        from ...PlotWindow import Plot2D
        return Plot2D(parent=parent, backend=backend)

    def getPlot1D(self, init=True):
        """Return the current plot used to display curves and create it if it
        does not yet exists and `init` is True. Else returns None."""
        if not init:
            return self._plot1D
        if self._plot1D is None:
            self._plot1D = self.createPlot1D(self, self._backend)
            self._layout.addWidget(self._plot1D)
        return self._plot1D

    def _showPlot1D(self):
        plot = self.getPlot1D()
        self._layout.setCurrentWidget(plot)

    def getPlot2D(self, init=True):
        """Return the current plot used to display image and create it if it
        does not yet exists and `init` is True. Else returns None."""
        if not init:
            return self._plot2D
        if self._plot2D is None:
            self._plot2D = self.createPlot2D(parent=self, backend=self._backend)
            self._layout.addWidget(self._plot2D)
        return self._plot2D

    def _showPlot2D(self):
        plot = self.getPlot2D()
        self._layout.setCurrentWidget(plot)

    def getCurrentPlotWidget(self):
        return self._layout.currentWidget()

    def closeEvent(self, qCloseEvent):
        self.sigClose.emit()
        qCloseEvent.accept()

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
        plot = self.getPlot2D()

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
        plot = self.getPlot1D()

        plot.clear()
        plot.setGraphTitle(data.title)
        plot.getXAxis().setLabel(data.xLabel)
        plot.getYAxis().setLabel(data.yLabel)

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
        plot = self.getPlot1D()

        plot.clear()
        plot.setGraphTitle(data.title)
        plot.getXAxis().setLabel(data.xLabel)
        plot.getYAxis().setLabel(data.yLabel)

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
        plot = self.getPlot1D(init=False)
        if plot is not None:
            plot.clear()
        plot = self.getPlot2D(init=False)
        if plot is not None:
            plot.clear()

    def getProfile(self):
        """Returns the profile data which is displayed"""
        return self.__data

    def setProfile(self, data):
        """
        Setup the window to display a new profile data.

        This method dispatch the result to a specific method according to the
        data type.

        :param data: Computed data profile
        """
        self.__data = data
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


class _ClearAction(qt.QAction):
    """Action to clear the profile manager

    The action is only enabled if something can be cleaned up.
    """

    def __init__(self, parent, profileManager):
        super(_ClearAction, self).__init__(parent)
        self.__profileManager = weakref.ref(profileManager)
        icon = icons.getQIcon('profile-clear')
        self.setIcon(icon)
        self.setText('Clear profile')
        self.setToolTip('Clear the profiles')
        self.setCheckable(False)
        self.setEnabled(False)
        self.triggered.connect(profileManager.clearProfile)
        plot = profileManager.getPlotWidget()
        roiManager = profileManager.getRoiManager()
        plot.sigInteractiveModeChanged.connect(self.__modeUpdated)
        roiManager.sigRoiChanged.connect(self.__roiListUpdated)

    def getProfileManager(self):
        return self.__profileManager()

    def __roiListUpdated(self):
        self.__update()

    def __modeUpdated(self, source):
        self.__update()

    def __update(self):
        profileManager = self.getProfileManager()
        if profileManager is None:
            return
        roiManager = profileManager.getRoiManager()
        if roiManager is None:
            return
        enabled = roiManager.isStarted() or len(roiManager.getRois()) > 0
        self.setEnabled(enabled)


class _StoreLastParamBehavior(qt.QObject):
    """This object allow to store and restore the properties of the ROI
    profiles"""

    def __init__(self, parent):
        assert isinstance(parent, ProfileManager)
        super(_StoreLastParamBehavior, self).__init__(parent=parent)
        self.__properties = {}
        self.__profileRoi = None
        self.__filter = utils.LockReentrant()

    def _roi(self):
        """Return the spied ROI"""
        if self.__profileRoi is None:
            return None
        roi = self.__profileRoi()
        if roi is None:
            self.__profileRoi = None
        return roi

    def setProfileRoi(self, roi):
        """Set a profile ROI to spy.

        :param ProfileRoiMixIn roi: A profile ROI
        """
        previousRoi = self._roi()
        if previousRoi is roi:
            return
        if previousRoi is not None:
            previousRoi.sigProfilePropertyChanged.disconnect(self._profilePropertyChanged)
        self.__profileRoi = None if roi is None else weakref.ref(roi)
        if roi is not None:
            roi.sigProfilePropertyChanged.connect(self._profilePropertyChanged)

    def _profilePropertyChanged(self):
        """Handle changes on the properties defining the profile ROI.
        """
        if self.__filter.locked():
            return
        roi = self.sender()
        self.storeProperties(roi)

    def storeProperties(self, roi):
        if isinstance(roi, (rois._DefaultImageStackProfileRoiMixIn,
                              rois.ProfileImageStackCrossROI)):
            self.__properties["method"] = roi.getProfileMethod()
            self.__properties["line-width"] = roi.getProfileLineWidth()
            self.__properties["type"] = roi.getProfileType()
        elif isinstance(roi, (rois._DefaultImageProfileRoiMixIn,
                            rois.ProfileImageCrossROI)):
            self.__properties["method"] = roi.getProfileMethod()
            self.__properties["line-width"] = roi.getProfileLineWidth()
        elif isinstance(roi, (rois._DefaultScatterProfileRoiMixIn,
                              rois.ProfileScatterCrossROI)):
            self.__properties["npoints"] = roi.getNPoints()

    def restoreProperties(self, roi):
        with self.__filter:
            if isinstance(roi, (rois._DefaultImageStackProfileRoiMixIn,
                                  rois.ProfileImageStackCrossROI)):
                value = self.__properties.get("method", None)
                if value is not None:
                    roi.setProfileMethod(value)
                value = self.__properties.get("line-width", None)
                if value is not None:
                    roi.setProfileLineWidth(value)
                value = self.__properties.get("type", None)
                if value is not None:
                    roi.setProfileType(value)
            elif isinstance(roi, (rois._DefaultImageProfileRoiMixIn,
                                rois.ProfileImageCrossROI)):
                value = self.__properties.get("method", None)
                if value is not None:
                    roi.setProfileMethod(value)
                value = self.__properties.get("line-width", None)
                if value is not None:
                    roi.setProfileLineWidth(value)
            elif isinstance(roi, (rois._DefaultScatterProfileRoiMixIn,
                                  rois.ProfileScatterCrossROI)):
                value = self.__properties.get("npoints", None)
                if value is not None:
                    roi.setNPoints(value)


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

        self.__reentrantResults = {}
        """Store reentrant result to avoid to skip some of them
        cause the implementation uses a QEventLoop."""

        self._profileWindowClass = ProfileWindow
        """Class used to display the profile results"""

        self._computedProfiles = 0
        """Statistics for tests"""

        self.__itemTypes = []
        """Kind of items to use"""

        self.__tracking = False
        """Is the plot active items are tracked"""

        self.__useColorFromCursor = True
        """If true, force the ROI color with the colormap marker color"""

        self._item = None
        """The selected item"""

        self.__singleProfileAtATime = True
        """When it's true, only a single profile is displayed at a time."""

        self._previousWindowGeometry = []

        self._storeProperties = _StoreLastParamBehavior(self)
        """If defined the profile properties of the last ROI are reused to the
        new created ones"""

        # Listen to plot limits changed
        plot.getXAxis().sigLimitsChanged.connect(self.requestUpdateAllProfile)
        plot.getYAxis().sigLimitsChanged.connect(self.requestUpdateAllProfile)

        roiManager.sigInteractiveModeFinished.connect(self.__interactionFinished)
        roiManager.sigInteractiveRoiCreated.connect(self.__roiCreated)
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
        action = CreateRoiModeAction(parent, roiManager, profileRoiClass)
        if hasattr(profileRoiClass, "ICON"):
            action.setIcon(icons.getQIcon(profileRoiClass.ICON))
        if hasattr(profileRoiClass, "NAME"):
            def articulify(word):
                """Add an an/a article in the front of the word"""
                first = word[1] if word[0] == 'h' else word[0]
                if first in "aeiou":
                    return "an " + word
                return "a " + word
            action.setText('Define %s' % articulify(profileRoiClass.NAME))
            action.setToolTip('Enables %s selection mode' % profileRoiClass.NAME)
        action.setSingleShot(True)
        return action

    def createClearAction(self, parent):
        """Create an action to clean up the plot from the profile ROIs.

        :param qt.QObject parent: The parent of the created action.
        :rtype: qt.QAction
        """
        action = _ClearAction(parent, self)
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
            rois.ProfileImageDirectedLineROI,
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

    def setProfileWindowClass(self, profileWindowClass):
        """Set the class which will be instantiated to display profile result.
        """
        self._profileWindowClass = profileWindowClass

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

    def setDefaultColorFromCursorColor(self, enabled):
        """Enabled/disable the use of the colormap cursor color to display the
        ROIs.

        If set, the manager will update the color of the profile ROIs using the
        current colormap cursor color from the selected item.
        """
        self.__useColorFromCursor = enabled

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

    def __roiCreated(self, roi):
        """Handle ROI creation"""
        # Filter out non profile ROIs
        if isinstance(roi, core.ProfileRoiMixIn):
            if self._storeProperties is not None:
                # Initialize the properties with the previous ones
                self._storeProperties.restoreProperties(roi)

    def __addProfile(self, profileRoi):
        """Add a new ROI to the manager."""
        if profileRoi.getFocusProxy() is None:
            if self._storeProperties is not None:
                # Follow changes on properties
                self._storeProperties.setProfileRoi(profileRoi)
            if self.__singleProfileAtATime:
                # FIXME: It would be good to reuse the windows to avoid blinking
                self.clearProfile()

        profileRoi._setProfileManager(self)
        self._updateRoiColor(profileRoi)
        self._rois.append(profileRoi)
        self.requestUpdateProfile(profileRoi)

    def __removeProfile(self, profileRoi):
        """Remove a ROI from the manager."""
        window = self._disconnectProfileWindow(profileRoi)
        if window is not None:
            geometry = window.geometry()
            self._previousWindowGeometry.append(geometry)
            self.clearProfileWindow(window)
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
            if roi.getFocusProxy() is not None:
                # Skip sub ROIs, it will be removed by their parents
                continue
            roiManager.removeRoi(roi)

        if not roiManager.isDrawing():
            # Clean the selected mode
            roiManager.stop()

    def hasPendingOperations(self):
        """Returns true if a thread is still computing or displaying a profile.

        :rtype: bool
        """
        return len(self.__reentrantResults) > 0 or len(self._pendingRunners) > 0

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
        if item is None or not isinstance(item, profileRoi.ITEM_KIND):
            # This item is not compatible with this profile
            profileRoi._setPlotItem(None)
            profileWindow = profileRoi.getProfileWindow()
            if profileWindow is not None:
                profileWindow.setProfile(None)
            return

        profileRoi._setPlotItem(item)
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
        if roi in self.__reentrantResults:
            # Store the data to process it in the main loop
            # And not a sub loop created by initProfileWindow
            # This also remove the duplicated requested
            self.__reentrantResults[roi] = profileData
            return

        self.__reentrantResults[roi] = profileData
        self._computedProfiles = self._computedProfiles + 1
        window = roi.getProfileWindow()
        if window is None:
            plot = self.getPlotWidget()
            window = self.createProfileWindow(plot, roi)
            # roi.profileWindow have to be set before initializing the window
            # Cause the initialization is using QEventLoop
            roi.setProfileWindow(window)
            self.initProfileWindow(window, roi)
            window.show()

        lastData = self.__reentrantResults.pop(roi)
        window.setProfile(lastData)

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
        self._updateRoiColors()
        self.requestUpdateAllProfile()

    def getDefaultColor(self, item):
        """Returns the default ROI color to use according to the given item.

        :param ~silx.gui.plot.items.item.Item item: AN item
        :rtype: qt.QColor
        """
        color = 'pink'
        if isinstance(item, items.ColormapMixIn):
            colormap = item.getColormap()
            name = colormap.getName()
            if name is not None:
                color = colors.cursorColorForColormap(name)
        color = colors.asQColor(color)
        return color

    def _updateRoiColors(self):
        """Update ROI color according to the item selection"""
        if not self.__useColorFromCursor:
            return
        item = self.getPlotItem()
        color = self.getDefaultColor(item)
        for roi in self._rois:
            roi.setColor(color)

    def _updateRoiColor(self, roi):
        """Update a specific ROI according to the current selected item.

        :param RegionOfInterest roi: The ROI to update
        """
        if not self.__useColorFromCursor:
            return
        item = self.getPlotItem()
        color = self.getDefaultColor(item)
        roi.setColor(color)

    def __itemChanged(self, changeType):
        """Handle item changes.
        """
        if changeType in (items.ItemChangedType.DATA,
                          items.ItemChangedType.POSITION,
                          items.ItemChangedType.SCALE):
            self.requestUpdateAllProfile()
        elif changeType == (items.ItemChangedType.COLORMAP):
            self._updateRoiColors()

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

    def getCurrentRoi(self):
        """Returns the currently selected ROI, else None.

        :rtype: core.ProfileRoiMixIn
        """
        roiManager = self.getRoiManager()
        if roiManager is None:
            return None
        roi = roiManager.getCurrentRoi()
        if not isinstance(roi, core.ProfileRoiMixIn):
            return None
        return roi

    def getRoiManager(self):
        """Returns the used ROI manager

        :rtype: RegionOfInterestManager
        """
        return self._roiManagerRef()

    def createProfileWindow(self, plot, roi):
        """Create a new profile window.

        :param ~core.ProfileRoiMixIn roi: The plot containing the raw data
        :param ~core.ProfileRoiMixIn roi: A managed ROI
        :rtype: ~ProfileWindow
        """
        return self._profileWindowClass(plot)

    def initProfileWindow(self, profileWindow, roi):
        """This function is called just after the profile window creation in
        order to initialize the window location.

        :param ~ProfileWindow profileWindow:
            The profile window to initialize.
        """
        # Enforce the use of one of the widgets
        # To have the correct window size
        profileWindow.prepareWidget(roi)
        profileWindow.adjustSize()

        # Trick to avoid blinking while retrieving the right window size
        # Display the window, hide it and wait for some event loops
        profileWindow.show()
        profileWindow.hide()
        eventLoop = qt.QEventLoop(self)
        for _ in range(10):
            if not eventLoop.processEvents():
                break

        profileWindow.show()
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

        profileGeom = profileWindow.frameGeometry()
        profileWidth = profileGeom.width()

        # Align vertically to the center of the window
        top = winGeom.top() + (winGeom.height() - profileGeom.height()) // 2

        margin = 5
        if profileWidth < spaceOnRightSide:
            # Place profile on the right
            left = winGeom.right() + margin
        elif profileWidth < spaceOnLeftSide:
            # Place profile on the left
            left = max(0, winGeom.left() - profileWidth - margin)
        else:
            # Move it as much as possible where there is more space
            if spaceOnLeftSide > spaceOnRightSide:
                left = 0
            else:
                left = screenGeom.width() - profileGeom.width()
        profileWindow.move(left, top)


    def clearProfileWindow(self, profileWindow):
        """Called when a profile window is not anymore needed.

        By default the window will be closed. But it can be
        inherited to change this behavior.
        """
        profileWindow.deleteLater()
