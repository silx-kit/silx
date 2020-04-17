# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2020 European Synchrotron Radiation Facility
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

from .. import icons
from .. import qt
from . import items
from ..colors import cursorColorForColormap
from . import actions
from .PlotToolButtons import ProfileToolButton, ProfileOptionToolButton
from .ProfileMainWindow import ProfileMainWindow
from .tools.profile import core

from silx.utils.deprecation import deprecated


@deprecated(replacement="silx.gui.plot.tools.profile.createProfile", since_version="0.13.0")
def createProfile(roiInfo, currentData, origin, scale, lineWidth, method):
    return core.createProfile(roiInfo, currentData, origin,
                              scale, lineWidth, method)


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
    :param str title: See :class:`QToolBar`.
    :param parent: See :class:`QToolBar`.
    """
    # TODO Make it a QActionGroup instead of a QToolBar

    _POLYGON_LEGEND = '__ProfileToolBar_ROI_Polygon'

    DEFAULT_PROF_METHOD = 'mean'

    def __init__(self, parent=None, plot=None, profileWindow=None,
                 title='Profile Selection'):
        super(ProfileToolBar, self).__init__(title, parent)
        assert plot is not None
        self._plotRef = weakref.ref(plot)

        self._overlayColor = None
        self._defaultOverlayColor = 'red'  # update when active image change
        self._method = self.DEFAULT_PROF_METHOD

        self._roiInfo = None  # Store start and end points and type of ROI

        self._profileWindow = profileWindow
        """User provided plot widget in which the profile curve is plotted.
        None if no custom profile plot was provided."""

        self._profileMainWindow = None
        """Main window providing 2 profile plot widgets for 1D or 2D profiles.
        The window provides two public methods
            - :meth:`setProfileDimensions`
            - :meth:`getPlot`: return handle on the actual plot widget
              currently being used
        None if the user specified a custom profile plot window.
        """

        self.__profileMainWindowNeverShown = True

        if self._profileWindow is None:
            backend = type(plot._backend)
            self._profileMainWindow = ProfileMainWindow(self, backend=backend)

        # Actions
        self._browseAction = actions.mode.ZoomModeAction(self.plot, parent=self)
        self._browseAction.setVisible(False)

        self.hLineAction = qt.QAction(icons.getQIcon('shape-horizontal'),
                                      'Horizontal Profile Mode',
                                      self)
        self.hLineAction.setToolTip(
            'Enables horizontal profile selection mode')
        self.hLineAction.setCheckable(True)
        self.hLineAction.toggled[bool].connect(self._hLineActionToggled)

        self.vLineAction = qt.QAction(icons.getQIcon('shape-vertical'),
                                      'Vertical Profile Mode',
                                      self)
        self.vLineAction.setToolTip(
            'Enables vertical profile selection mode')
        self.vLineAction.setCheckable(True)
        self.vLineAction.toggled[bool].connect(self._vLineActionToggled)

        self.lineAction = qt.QAction(icons.getQIcon('shape-diagonal'),
                                     'Free Line Profile Mode',
                                     self)
        self.lineAction.setToolTip(
            'Enables line profile selection mode')
        self.lineAction.setCheckable(True)
        self.lineAction.toggled[bool].connect(self._lineActionToggled)

        self.clearAction = qt.QAction(icons.getQIcon('profile-clear'),
                                      'Clear Profile',
                                      self)
        self.clearAction.setToolTip(
            'Clear the profile Region of interest')
        self.clearAction.setCheckable(False)
        self.clearAction.triggered.connect(self.clearProfile)

        # ActionGroup
        self.actionGroup = qt.QActionGroup(self)
        self.actionGroup.addAction(self._browseAction)
        self.actionGroup.addAction(self.hLineAction)
        self.actionGroup.addAction(self.vLineAction)
        self.actionGroup.addAction(self.lineAction)

        # Add actions to ToolBar
        self.addAction(self._browseAction)
        self.addAction(self.hLineAction)
        self.addAction(self.vLineAction)
        self.addAction(self.lineAction)
        self.addAction(self.clearAction)

        # Add width spin box to toolbar
        self.addWidget(qt.QLabel('W:'))
        self.lineWidthSpinBox = qt.QSpinBox(self)
        self.lineWidthSpinBox.setRange(1, 1000)
        self.lineWidthSpinBox.setValue(1)
        self.lineWidthSpinBox.valueChanged[int].connect(
            self._lineWidthSpinBoxValueChangedSlot)
        self.addWidget(self.lineWidthSpinBox)

        self.methodsButton = ProfileOptionToolButton(parent=self, plot=self)
        self.__profileOptionToolAction = self.addWidget(self.methodsButton)
        # TODO: add connection with the signal
        self.methodsButton.sigMethodChanged.connect(self.setProfileMethod)

        self.plot.sigInteractiveModeChanged.connect(
            self._interactiveModeChanged)

        # Enable toolbar only if there is an active image
        self.setEnabled(self.plot.getActiveImage(just_legend=True) is not None)
        self.plot.sigActiveImageChanged.connect(
            self._activeImageChanged)

        # listen to the profile window signals to clear profile polygon on close
        if self.getProfileMainWindow() is not None:
            self.getProfileMainWindow().sigClose.connect(self.clearProfile)

    @property
    def plot(self):
        """The :class:`.PlotWidget` associated to the toolbar."""
        return self._plotRef()

    @property
    @deprecated(since_version="0.6.0")
    def browseAction(self):
        return self._browseAction

    @property
    @deprecated(replacement="getProfilePlot", since_version="0.5.0")
    def profileWindow(self):
        return self.getProfilePlot()

    def getProfilePlot(self):
        """Return plot widget in which the profile curve or the
        profile image is plotted.
        """
        if self.getProfileMainWindow() is not None:
            return self.getProfileMainWindow().getPlot()

        # in case the user provided a custom plot for profiles
        return self._profileWindow

    def getProfileMainWindow(self):
        """Return window containing the profile curve widget.
        This can return *None* if a custom profile plot window was
        specified in the constructor.
        """
        return self._profileMainWindow

    def _activeImageChanged(self, previous, legend):
        """Handle active image change: toggle enabled toolbar, update curve"""
        if legend is None:
            self.setEnabled(False)
        else:
            activeImage = self.plot.getActiveImage()

            # Disable for empty image
            self.setEnabled(activeImage.getData(copy=False).size > 0)

            # Update default profile color
            if isinstance(activeImage, items.ColormapMixIn):
                self._defaultOverlayColor = cursorColorForColormap(
                    activeImage.getColormap()['name'])
            else:
                self._defaultOverlayColor = 'black'

            self.updateProfile()

    def _lineWidthSpinBoxValueChangedSlot(self, value):
        """Listen to ROI width widget to refresh ROI and profile"""
        self.updateProfile()

    def _interactiveModeChanged(self, source):
        """Handle plot interactive mode changed:

        If changed from elsewhere, disable drawing tool
        """
        if source is not self:
            self.clearProfile()

            # Uncheck all drawing profile modes
            self.hLineAction.setChecked(False)
            self.vLineAction.setChecked(False)
            self.lineAction.setChecked(False)

            if self.getProfileMainWindow() is not None:
                self.getProfileMainWindow().hide()

    def _hLineActionToggled(self, checked):
        """Handle horizontal line profile action toggle"""
        if checked:
            self.plot.setInteractiveMode('draw', shape='hline',
                                         color=None, source=self)
            self.plot.sigPlotSignal.connect(self._plotWindowSlot)
        else:
            self.plot.sigPlotSignal.disconnect(self._plotWindowSlot)

    def _vLineActionToggled(self, checked):
        """Handle vertical line profile action toggle"""
        if checked:
            self.plot.setInteractiveMode('draw', shape='vline',
                                         color=None, source=self)
            self.plot.sigPlotSignal.connect(self._plotWindowSlot)
        else:
            self.plot.sigPlotSignal.disconnect(self._plotWindowSlot)

    def _lineActionToggled(self, checked):
        """Handle line profile action toggle"""
        if checked:
            self.plot.setInteractiveMode('draw', shape='line',
                                         color=None, source=self)
            self.plot.sigPlotSignal.connect(self._plotWindowSlot)
        else:
            self.plot.sigPlotSignal.disconnect(self._plotWindowSlot)

    def _plotWindowSlot(self, event):
        """Listen to Plot to handle drawing events to refresh ROI and profile.
        """
        if event['event'] not in ('drawingProgress', 'drawingFinished'):
            return

        checkedAction = self.actionGroup.checkedAction()
        if checkedAction == self.hLineAction:
            lineProjectionMode = 'X'
        elif checkedAction == self.vLineAction:
            lineProjectionMode = 'Y'
        elif checkedAction == self.lineAction:
            lineProjectionMode = 'D'
        else:
            return

        roiStart, roiEnd = event['points'][0], event['points'][1]

        self._roiInfo = roiStart, roiEnd, lineProjectionMode
        self.updateProfile()

    @property
    def overlayColor(self):
        """The color to use for the ROI.

        If set to None (the default), the overlay color is adapted to the
        active image colormap and changes if the active image colormap changes.
        """
        return self._overlayColor or self._defaultOverlayColor

    @overlayColor.setter
    def overlayColor(self, color):
        self._overlayColor = color
        self.updateProfile()

    def clearProfile(self):
        """Remove profile curve and profile area."""
        self._roiInfo = None
        self.updateProfile()

    def updateProfile(self):
        """Update the displayed profile and profile ROI.

        This uses the current active image of the plot and the current ROI.
        """
        image = self.plot.getActiveImage()
        if image is None:
            return

        # Clean previous profile area, and previous curve
        self.plot.remove(self._POLYGON_LEGEND, kind='item')
        self.getProfilePlot().clear()
        self.getProfilePlot().setGraphTitle('')
        self.getProfilePlot().getXAxis().setLabel('X')
        self.getProfilePlot().getYAxis().setLabel('Y')

        self._createProfile(currentData=image.getData(copy=False),
                            origin=image.getOrigin(),
                            scale=image.getScale(),
                            colormap=None,  # Not used for 2D data
                            z=image.getZValue(),
                            method=self.getProfileMethod())

    def _createProfile(self, currentData, origin, scale, colormap, z, method):
        """Create the profile line for the the given image.

        :param numpy.ndarray currentData: the image or the stack of images
            on which we compute the profile
        :param origin: (ox, oy) the offset from origin
        :type origin: 2-tuple of float
        :param scale: (sx, sy) the scale to use
        :type scale: 2-tuple of float
        :param dict colormap: The colormap to use
        :param int z: The z layer of the image
        """
        if self._roiInfo is None:
            return

        coords, profile, area, profileName, xLabel = createProfile(
            roiInfo=self._roiInfo,
            currentData=currentData,
            origin=origin,
            scale=scale,
            lineWidth=self.lineWidthSpinBox.value(),
            method=method)

        profilePlot = self.getProfilePlot()

        profilePlot.setGraphTitle(profileName)
        profilePlot.getXAxis().setLabel(xLabel)

        dataIs3D = len(currentData.shape) > 2
        if dataIs3D:
            profileScale = (coords[-1] - coords[0]) / profile.shape[1], 1
            profilePlot.addImage(profile,
                                 legend=profileName,
                                 colormap=colormap,
                                 origin=(coords[0], 0),
                                 scale=profileScale)
            profilePlot.getYAxis().setLabel("Frame index (depth)")
        else:
            profilePlot.addCurve(coords,
                                 profile[0],
                                 legend=profileName,
                                 color=self.overlayColor)

        self.plot.addShape(area[0], area[1],
                           legend=self._POLYGON_LEGEND,
                           color=self.overlayColor,
                           shape='polygon', fill=True,
                           replace=False, z=z + 1)

        self._showProfileMainWindow()

    def _showProfileMainWindow(self):
        """If profile window was created by this toolbar,
        try to avoid overlapping with the toolbar's parent window.
        """
        profileMainWindow = self.getProfileMainWindow()
        if profileMainWindow is not None:
            if self.__profileMainWindowNeverShown:
                # Places the profile window in order to avoid overlapping the plot
                self.__profileMainWindowNeverShown = False
                winGeom = self.window().frameGeometry()
                qapp = qt.QApplication.instance()
                screenGeom = qapp.desktop().availableGeometry(self)
                spaceOnLeftSide = winGeom.left()
                spaceOnRightSide = screenGeom.width() - winGeom.right()

                profileWindowWidth = profileMainWindow.frameGeometry().width()
                if (profileWindowWidth < spaceOnRightSide):
                    # Place profile on the right
                    profileMainWindow.move(winGeom.right(), winGeom.top())
                elif(profileWindowWidth < spaceOnLeftSide):
                    # Place profile on the left
                    profileMainWindow.move(
                        max(0, winGeom.left() - profileWindowWidth), winGeom.top())

                profileMainWindow.raise_()

            profileMainWindow.show()
        else:
            self.getProfilePlot().show()
            self.getProfilePlot().raise_()

    def hideProfileWindow(self):
        """Hide profile window.
        """
        # this method is currently only used by StackView when the perspective
        # is changed
        if self.getProfileMainWindow() is not None:
            self.getProfileMainWindow().hide()

    def setProfileMethod(self, method):
        assert method in ('sum', 'mean')
        self._method = method
        self.updateProfile()

    def getProfileMethod(self):
        return self._method

    def getProfileOptionToolAction(self):
        return self.__profileOptionToolAction


class Profile3DToolBar(ProfileToolBar):
    def __init__(self, parent=None, stackview=None,
                 title='Profile Selection'):
        """QToolBar providing profile tools for an image or a stack of images.

        :param parent: the parent QWidget
        :param stackview: :class:`StackView` instance on which to operate.
        :param str title: See :class:`QToolBar`.
        :param parent: See :class:`QToolBar`.
        """
        # TODO: add param profileWindow (specify the plot used for profiles)
        super(Profile3DToolBar, self).__init__(parent=parent,
                                               plot=stackview.getPlotWidget(),
                                               title=title)
        self._method3D = 'sum'
        self._profileType = None

        self.stackView = stackview
        """:class:`StackView` instance"""

        self.profile3dAction = ProfileToolButton(
            parent=self, plot=self.plot)
        self.profile3dAction.computeProfileIn2D()
        self.profile3dAction.setVisible(True)
        self.addWidget(self.profile3dAction)
        self.profile3dAction.sigDimensionChanged.connect(self._setProfileType)

        # create the 3D toolbar
        self._setProfileType(2)

    def _setProfileType(self, dimensions):
        """Set the profile type: "1D" for a curve (profile on a single image)
        or "2D" for an image (profile on a stack of images).

        :param int dimensions: 1 for a "1D" profile or 2 for a "2D" profile
        """
        # fixme this assumes that we created _profileMainWindow
        self._profileType = "1D" if dimensions == 1 else "2D"
        self.getProfileMainWindow().setProfileType(self._profileType)
        self.updateProfile()

    def updateProfile(self):
        """Method overloaded from :class:`ProfileToolBar`,
        to pass the stack of images instead of just the active image.

        In 1D profile mode, use the regular parent method.
        """
        if self._profileType == "1D":
            super(Profile3DToolBar, self).updateProfile()
        elif self._profileType == "2D":
            stackData = self.stackView.getCurrentView(copy=False,
                                                      returnNumpyArray=True)
            if stackData is None:
                return
            self.plot.remove(self._POLYGON_LEGEND, kind='item')
            self.getProfilePlot().clear()
            self.getProfilePlot().setGraphTitle('')
            self.getProfilePlot().getXAxis().setLabel('X')
            self.getProfilePlot().getYAxis().setLabel('Y')
            self._createProfile(currentData=stackData[0],
                                origin=stackData[1]['origin'],
                                scale=stackData[1]['scale'],
                                colormap=stackData[1]['colormap'],
                                z=stackData[1]['z'],
                                method=self.getProfileMethod())
        else:
            raise ValueError(
                    "Profile type must be 1D or 2D, not %s" % self._profileType)

    def setProfileMethod(self, method):
        assert method in ('sum', 'mean')
        self._method3D = method
        self.updateProfile()

    def getProfileMethod(self):
        return self._method3D
