# /*##########################################################################
#
# Copyright (c) 2004-2023 European Synchrotron Radiation Facility
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
"""Utility functions, toolbars and actions  to create profile on images
and stacks of images"""


__authors__ = ["V.A. Sole", "T. Vincent", "P. Knobel", "H. Payno"]
__license__ = "MIT"
__date__ = "12/04/2019"


import weakref

from .. import qt
from . import actions
from .tools.profile import manager
from .tools.profile import rois
from silx.gui.widgets.MultiModeAction import MultiModeAction

from .tools import roi as roi_mdl
from silx.gui.plot import items


class _CustomProfileManager(manager.ProfileManager):
    """This custom profile manager uses a single predefined profile window
    if it is specified. Else the behavior is the same as the default
    ProfileManager"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__profileWindow = None
        self.__specializedProfileWindows = {}

    def setSpecializedProfileWindow(self, roiClass, profileWindow):
        """Set a profile window for a given class or ROI.

        Setting profileWindow to None removes the roiClass from the list.

        :param roiClass:
        :param profileWindow:
        """
        if profileWindow is None:
            self.__specializedProfileWindows.pop(roiClass, None)
        else:
            self.__specializedProfileWindows[roiClass] = profileWindow

    def setProfileWindow(self, profileWindow):
        self.__profileWindow = profileWindow

    def createProfileWindow(self, plot, roi):
        for (
            roiClass,
            specializedProfileWindow,
        ) in self.__specializedProfileWindows.items():
            if isinstance(roi, roiClass):
                return specializedProfileWindow

        if self.__profileWindow is not None:
            return self.__profileWindow
        else:
            return super(_CustomProfileManager, self).createProfileWindow(plot, roi)

    def clearProfileWindow(self, profileWindow):
        for specializedProfileWindow in self.__specializedProfileWindows.values():
            if profileWindow is specializedProfileWindow:
                profileWindow.setProfile(None)
                return

        if self.__profileWindow is not None:
            self.__profileWindow.setProfile(None)
        else:
            return super(_CustomProfileManager, self).clearProfileWindow(profileWindow)


class ProfileToolBar(qt.QToolBar):
    """QToolBar providing profile tools operating on a :class:`PlotWindow`.

    Attributes:

    - plot: Associated :class:`PlotWindow` on which the profile line is drawn.
    - actionGroup: :class:`QActionGroup` of available actions.

    To run the following sample code, a QApplication must be initialized.
    First, create a PlotWindow and add a :class:`ProfileToolBar`.

    >>> from silx.gui.plot import PlotWindow
    >>> from silx.gui.plot.Profile import ProfileToolBar

    >>> plot = PlotWindow()  # Create a PlotWindow
    >>> toolBar = ProfileToolBar(plot=plot)  # Create a profile toolbar
    >>> plot.addToolBar(toolBar)  # Add it to plot
    >>> plot.show()  # To display the PlotWindow with the profile toolbar

    :param plot: :class:`PlotWindow` instance on which to operate.
    :param profileWindow: Plot widget instance where to
                          display the profile curve or None to create one.
    :param parent: See :class:`QToolBar`.
    """

    def __init__(self, parent=None, plot=None, profileWindow=None):
        super(ProfileToolBar, self).__init__(parent)
        assert plot is not None

        self._plotRef = weakref.ref(plot)

        # If a profileWindow is defined,
        # It will be used to display all the profiles
        self._manager = self.createProfileManager(self, plot)
        self._manager.setProfileWindow(profileWindow)
        self._manager.setDefaultColorFromCursorColor(True)
        self._manager.setItemType(image=True)
        self._manager.setActiveItemTracking(True)

        # Actions
        self._browseAction = actions.mode.ZoomModeAction(plot, parent=self)
        self._browseAction.setVisible(False)
        self.freeLineAction = None
        self._createProfileActions()
        self._editor = self._manager.createEditorAction(self)

        # ActionGroup
        self.actionGroup = qt.QActionGroup(self)
        self.actionGroup.addAction(self._browseAction)
        self.actionGroup.addAction(self.hLineAction)
        self.actionGroup.addAction(self.vLineAction)
        self.actionGroup.addAction(self.lineAction)
        self.actionGroup.addAction(self._editor)

        modes = MultiModeAction(self)
        modes.addAction(self.hLineAction)
        modes.addAction(self.vLineAction)
        modes.addAction(self.lineAction)
        if self.freeLineAction is not None:
            modes.addAction(self.freeLineAction)
        modes.addAction(self.crossAction)
        self.__multiAction = modes

        # Add actions to ToolBar
        self.addAction(self._browseAction)
        self.addAction(modes)
        self.addAction(self._editor)
        self.addAction(self.clearAction)

        plot.sigActiveImageChanged.connect(self._activeImageChanged)
        self._activeImageChanged()

    def createProfileManager(self, parent, plot):
        return _CustomProfileManager(parent, plot)

    def _createProfileActions(self):
        self.hLineAction = self._manager.createProfileAction(
            rois.ProfileImageHorizontalLineROI, self
        )
        self.vLineAction = self._manager.createProfileAction(
            rois.ProfileImageVerticalLineROI, self
        )
        self.lineAction = self._manager.createProfileAction(
            rois.ProfileImageLineROI, self
        )
        self.freeLineAction = self._manager.createProfileAction(
            rois.ProfileImageDirectedLineROI, self
        )
        self.crossAction = self._manager.createProfileAction(
            rois.ProfileImageCrossROI, self
        )
        self.clearAction = self._manager.createClearAction(self)

    def getPlotWidget(self):
        """The :class:`.PlotWidget` associated to the toolbar."""
        return self._plotRef()

    def _setRoiActionEnabled(self, itemKind, enabled):
        for action in self.__multiAction.getMenu().actions():
            if not isinstance(action, roi_mdl.CreateRoiModeAction):
                continue
            roiClass = action.getRoiClass()
            if issubclass(itemKind, roiClass.ITEM_KIND):
                action.setEnabled(enabled)

    def _activeImageChanged(self, previous=None, legend=None):
        """Handle active image change to toggle actions"""
        if legend is None:
            self._setRoiActionEnabled(items.ImageStack, False)
            self._setRoiActionEnabled(items.ImageBase, False)
        else:
            plot = self.getPlotWidget()
            image = plot.getActiveImage()
            # Disable for empty image
            enabled = image.getData(copy=False).size > 0
            self._setRoiActionEnabled(type(image), enabled)

    def getProfileManager(self):
        """Return the manager of the profiles.

        :rtype: ProfileManager
        """
        return self._manager

    def clearProfile(self):
        """Remove profile curve and profile area."""
        self._manager.clearProfile()


class Profile3DToolBar(ProfileToolBar):
    def __init__(self, parent=None, stackview=None):
        """QToolBar providing profile tools for an image or a stack of images.

        :param parent: the parent QWidget
        :param stackview: :class:`StackView` instance on which to operate.
        :param parent: See :class:`QToolBar`.
        """
        # TODO: add param profileWindow (specify the plot used for profiles)
        super(Profile3DToolBar, self).__init__(
            parent=parent, plot=stackview.getPlotWidget()
        )

        self.stackView = stackview
        """:class:`StackView` instance"""

    def _createProfileActions(self):
        self.hLineAction = self._manager.createProfileAction(
            rois.ProfileImageStackHorizontalLineROI, self
        )
        self.vLineAction = self._manager.createProfileAction(
            rois.ProfileImageStackVerticalLineROI, self
        )
        self.lineAction = self._manager.createProfileAction(
            rois.ProfileImageStackLineROI, self
        )
        self.crossAction = self._manager.createProfileAction(
            rois.ProfileImageStackCrossROI, self
        )
        self.clearAction = self._manager.createClearAction(self)
