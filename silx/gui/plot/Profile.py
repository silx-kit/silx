# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
__date__ = "24/07/2018"


import weakref

import numpy

from silx.image.bilinear import BilinearImage

from .. import icons
from .. import qt
from . import items
from ..colors import cursorColorForColormap
from . import actions
from .PlotToolButtons import ProfileToolButton, ProfileOptionToolButton
from .ProfileMainWindow import ProfileMainWindow

from silx.utils.deprecation import deprecated


def _alignedFullProfile(data, origin, scale, position, roiWidth, axis, method):
    """Get a profile along one axis on a stack of images

    :param numpy.ndarray data: 3D volume (stack of 2D images)
        The first dimension is the image index.
    :param origin: Origin of image in plot (ox, oy)
    :param scale: Scale of image in plot (sx, sy)
    :param float position: Position of profile line in plot coords
                           on the axis orthogonal to the profile direction.
    :param int roiWidth: Width of the profile in image pixels.
    :param int axis: 0 for horizontal profile, 1 for vertical.
    :param str method: method to compute the profile. Can be 'mean' or 'sum'
    :return: profile image + effective ROI area corners in plot coords
    """
    assert axis in (0, 1)
    assert len(data.shape) == 3
    assert method in ('mean', 'sum')

    # Convert from plot to image coords
    imgPos = int((position - origin[1 - axis]) / scale[1 - axis])

    if axis == 1:  # Vertical profile
        # Transpose image to always do a horizontal profile
        data = numpy.transpose(data, (0, 2, 1))

    nimages, height, width = data.shape

    roiWidth = min(height, roiWidth)  # Clip roi width to image size

    # Get [start, end[ coords of the roi in the data
    start = int(int(imgPos) + 0.5 - roiWidth / 2.)
    start = min(max(0, start), height - roiWidth)
    end = start + roiWidth

    if start < height and end > 0:
        if method == 'mean':
            _fct = numpy.mean
        elif method == 'sum':
            _fct = numpy.sum
        else:
            raise ValueError('method not managed')
        profile = _fct(data[:, max(0, start):min(end, height), :], axis=1).astype(numpy.float32)
    else:
        profile = numpy.zeros((nimages, width), dtype=numpy.float32)

    # Compute effective ROI in plot coords
    profileBounds = numpy.array(
        (0, width, width, 0),
        dtype=numpy.float32) * scale[axis] + origin[axis]
    roiBounds = numpy.array(
        (start, start, end, end),
        dtype=numpy.float32) * scale[1 - axis] + origin[1 - axis]

    if axis == 0:  # Horizontal profile
        area = profileBounds, roiBounds
    else:  # vertical profile
        area = roiBounds, profileBounds

    return profile, area


def _alignedPartialProfile(data, rowRange, colRange, axis, method):
    """Mean of a rectangular region (ROI) of a stack of images
    along a given axis.

    Returned values and all parameters are in image coordinates.

    :param numpy.ndarray data: 3D volume (stack of 2D images)
        The first dimension is the image index.
    :param rowRange: [min, max[ of ROI rows (upper bound excluded).
    :type rowRange: 2-tuple of int (min, max) with min < max
    :param colRange: [min, max[ of ROI columns (upper bound excluded).
    :type colRange: 2-tuple of int (min, max) with min < max
    :param int axis: The axis along which to take the profile of the ROI.
                     0: Sum rows along columns.
                     1: Sum columns along rows.
    :param str method: method to compute the profile. Can be 'mean' or 'sum'
    :return: Profile image along the ROI as the mean of the intersection
             of the ROI and the image.
    """
    assert axis in (0, 1)
    assert len(data.shape) == 3
    assert rowRange[0] < rowRange[1]
    assert colRange[0] < colRange[1]
    assert method in ('mean', 'sum')

    nimages, height, width = data.shape

    # Range aligned with the integration direction
    profileRange = colRange if axis == 0 else rowRange

    profileLength = abs(profileRange[1] - profileRange[0])

    # Subset of the image to use as intersection of ROI and image
    rowStart = min(max(0, rowRange[0]), height)
    rowEnd = min(max(0, rowRange[1]), height)
    colStart = min(max(0, colRange[0]), width)
    colEnd = min(max(0, colRange[1]), width)

    if method == 'mean':
        _fct = numpy.mean
    elif method == 'sum':
        _fct = numpy.sum
    else:
        raise ValueError('method not managed')

    imgProfile = _fct(data[:, rowStart:rowEnd, colStart:colEnd], axis=axis + 1,
                      dtype=numpy.float32)

    # Profile including out of bound area
    profile = numpy.zeros((nimages, profileLength), dtype=numpy.float32)

    # Place imgProfile in full profile
    offset = - min(0, profileRange[0])
    profile[:, offset:offset + imgProfile.shape[1]] = imgProfile

    return profile


def createProfile(roiInfo, currentData, origin, scale, lineWidth, method):
    """Create the profile line for the the given image.

    :param roiInfo: information about the ROI: start point, end point and
        type ("X", "Y", "D")
    :param numpy.ndarray currentData: the 2D image or the 3D stack of images
        on which we compute the profile.
    :param origin: (ox, oy) the offset from origin
    :type origin: 2-tuple of float
    :param scale: (sx, sy) the scale to use
    :type scale: 2-tuple of float
    :param int lineWidth: width of the profile line
    :param str method: method to compute the profile. Can be 'mean' or 'sum'
    :return: `profile, area, profileName, xLabel`, where:
        - profile is a 2D array of the profiles of the stack of images.
          For a single image, the profile is a curve, so this parameter
          has a shape *(1, len(curve))*
        - area is a tuple of two 1D arrays with 4 values each. They represent
          the effective ROI area corners in plot coords.
        - profileName is a string describing the ROI, meant to be used as
          title of the profile plot
        - xLabel is a string describing the meaning of the X axis on the
          profile plot ("rows", "columns", "distance")

    :rtype: tuple(ndarray, (ndarray, ndarray), str, str)
    """
    if currentData is None or roiInfo is None or lineWidth is None:
        raise ValueError("createProfile called with invalide arguments")

    # force 3D data (stack of images)
    if len(currentData.shape) == 2:
        currentData3D = currentData.reshape((1,) + currentData.shape)
    elif len(currentData.shape) == 3:
        currentData3D = currentData

    roiWidth = max(1, lineWidth)
    roiStart, roiEnd, lineProjectionMode = roiInfo

    if lineProjectionMode == 'X':  # Horizontal profile on the whole image
        profile, area = _alignedFullProfile(currentData3D,
                                            origin, scale,
                                            roiStart[1], roiWidth,
                                            axis=0,
                                            method=method)

        yMin, yMax = min(area[1]), max(area[1]) - 1
        if roiWidth <= 1:
            profileName = 'Y = %g' % yMin
        else:
            profileName = 'Y = [%g, %g]' % (yMin, yMax)
        xLabel = 'Columns'

    elif lineProjectionMode == 'Y':  # Vertical profile on the whole image
        profile, area = _alignedFullProfile(currentData3D,
                                            origin, scale,
                                            roiStart[0], roiWidth,
                                            axis=1,
                                            method=method)

        xMin, xMax = min(area[0]), max(area[0]) - 1
        if roiWidth <= 1:
            profileName = 'X = %g' % xMin
        else:
            profileName = 'X = [%g, %g]' % (xMin, xMax)
        xLabel = 'Rows'

    else:  # Free line profile

        # Convert start and end points in image coords as (row, col)
        startPt = ((roiStart[1] - origin[1]) / scale[1],
                   (roiStart[0] - origin[0]) / scale[0])
        endPt = ((roiEnd[1] - origin[1]) / scale[1],
                 (roiEnd[0] - origin[0]) / scale[0])

        if (int(startPt[0]) == int(endPt[0]) or
                int(startPt[1]) == int(endPt[1])):
            # Profile is aligned with one of the axes

            # Convert to int
            startPt = int(startPt[0]), int(startPt[1])
            endPt = int(endPt[0]), int(endPt[1])

            # Ensure startPt <= endPt
            if startPt[0] > endPt[0] or startPt[1] > endPt[1]:
                startPt, endPt = endPt, startPt

            if startPt[0] == endPt[0]:  # Row aligned
                rowRange = (int(startPt[0] + 0.5 - 0.5 * roiWidth),
                            int(startPt[0] + 0.5 + 0.5 * roiWidth))
                colRange = startPt[1], endPt[1] + 1
                profile = _alignedPartialProfile(currentData3D,
                                                 rowRange, colRange,
                                                 axis=0,
                                                 method=method)

            else:  # Column aligned
                rowRange = startPt[0], endPt[0] + 1
                colRange = (int(startPt[1] + 0.5 - 0.5 * roiWidth),
                            int(startPt[1] + 0.5 + 0.5 * roiWidth))
                profile = _alignedPartialProfile(currentData3D,
                                                 rowRange, colRange,
                                                 axis=1,
                                                 method=method)

            # Convert ranges to plot coords to draw ROI area
            area = (
                numpy.array(
                    (colRange[0], colRange[1], colRange[1], colRange[0]),
                    dtype=numpy.float32) * scale[0] + origin[0],
                numpy.array(
                    (rowRange[0], rowRange[0], rowRange[1], rowRange[1]),
                    dtype=numpy.float32) * scale[1] + origin[1])

        else:  # General case: use bilinear interpolation

            # Ensure startPt <= endPt
            if (startPt[1] > endPt[1] or (
                    startPt[1] == endPt[1] and startPt[0] > endPt[0])):
                startPt, endPt = endPt, startPt

            profile = []
            for slice_idx in range(currentData3D.shape[0]):
                bilinear = BilinearImage(currentData3D[slice_idx, :, :])

                profile.append(bilinear.profile_line(
                    (startPt[0] - 0.5, startPt[1] - 0.5),
                    (endPt[0] - 0.5, endPt[1] - 0.5),
                    roiWidth,
                    method=method))
            profile = numpy.array(profile)

            # Extend ROI with half a pixel on each end, and
            # Convert back to plot coords (x, y)
            length = numpy.sqrt((endPt[0] - startPt[0]) ** 2 +
                                (endPt[1] - startPt[1]) ** 2)
            dRow = (endPt[0] - startPt[0]) / length
            dCol = (endPt[1] - startPt[1]) / length

            # Extend ROI with half a pixel on each end
            startPt = startPt[0] - 0.5 * dRow, startPt[1] - 0.5 * dCol
            endPt = endPt[0] + 0.5 * dRow, endPt[1] + 0.5 * dCol

            # Rotate deltas by 90 degrees to apply line width
            dRow, dCol = dCol, -dRow

            area = (
                numpy.array((startPt[1] - 0.5 * roiWidth * dCol,
                             startPt[1] + 0.5 * roiWidth * dCol,
                             endPt[1] + 0.5 * roiWidth * dCol,
                             endPt[1] - 0.5 * roiWidth * dCol),
                            dtype=numpy.float32) * scale[0] + origin[0],
                numpy.array((startPt[0] - 0.5 * roiWidth * dRow,
                             startPt[0] + 0.5 * roiWidth * dRow,
                             endPt[0] + 0.5 * roiWidth * dRow,
                             endPt[0] - 0.5 * roiWidth * dRow),
                            dtype=numpy.float32) * scale[1] + origin[1])

        y0, x0 = startPt
        y1, x1 = endPt
        if x1 == x0 or y1 == y0:
            profileName = 'From (%g, %g) to (%g, %g)' % (x0, y0, x1, y1)
        else:
            m = (y1 - y0) / (x1 - x0)
            b = y0 - m * x0
            profileName = 'y = %g * x %+g ; width=%d' % (m, b, roiWidth)
        xLabel = 'Distance'

    return profile, area, profileName, xLabel


# ProfileToolBar ##############################################################

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

        if self._profileWindow is None:
            self._profileMainWindow = ProfileMainWindow(self)

        # Actions
        self._browseAction = actions.mode.ZoomModeAction(self.plot, parent=self)
        self._browseAction.setVisible(False)

        self.hLineAction = qt.QAction(
            icons.getQIcon('shape-horizontal'),
            'Horizontal Profile Mode', None)
        self.hLineAction.setToolTip(
            'Enables horizontal profile selection mode')
        self.hLineAction.setCheckable(True)
        self.hLineAction.toggled[bool].connect(self._hLineActionToggled)

        self.vLineAction = qt.QAction(
            icons.getQIcon('shape-vertical'),
            'Vertical Profile Mode', None)
        self.vLineAction.setToolTip(
            'Enables vertical profile selection mode')
        self.vLineAction.setCheckable(True)
        self.vLineAction.toggled[bool].connect(self._vLineActionToggled)

        self.lineAction = qt.QAction(
            icons.getQIcon('shape-diagonal'),
            'Free Line Profile Mode', None)
        self.lineAction.setToolTip(
            'Enables line profile selection mode')
        self.lineAction.setCheckable(True)
        self.lineAction.toggled[bool].connect(self._lineActionToggled)

        self.clearAction = qt.QAction(
            icons.getQIcon('profile-clear'),
            'Clear Profile', None)
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
        self.addWidget(self.methodsButton)
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

        profile, area, profileName, xLabel = createProfile(
            roiInfo=self._roiInfo,
            currentData=currentData,
            origin=origin,
            scale=scale,
            lineWidth=self.lineWidthSpinBox.value(),
            method=method)

        self.getProfilePlot().setGraphTitle(profileName)

        dataIs3D = len(currentData.shape) > 2
        if dataIs3D:
            self.getProfilePlot().addImage(profile,
                                           legend=profileName,
                                           xlabel=xLabel,
                                           ylabel="Frame index (depth)",
                                           colormap=colormap)
        else:
            coords = numpy.arange(len(profile[0]), dtype=numpy.float32)
            # Scale horizontal and vertical profile coordinates
            if self._roiInfo[2] == 'X':
                coords = coords * scale[0] + origin[0]
            elif self._roiInfo[2] == 'Y':
                coords = coords * scale[1] + origin[1]

            self.getProfilePlot().addCurve(coords,
                                           profile[0],
                                           legend=profileName,
                                           xlabel=xLabel,
                                           color=self.overlayColor)

        self.plot.addItem(area[0], area[1],
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

            profileMainWindow.show()
            profileMainWindow.raise_()
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
                                               plot=stackview.getPlot(),
                                               title=title)
        self.stackView = stackview
        """:class:`StackView` instance"""

        self.profile3dAction = ProfileToolButton(
            parent=self, plot=self.plot)
        self.profile3dAction.computeProfileIn2D()
        self.profile3dAction.setVisible(True)
        self.addWidget(self.profile3dAction)
        self.profile3dAction.sigDimensionChanged.connect(self._setProfileType)

        # create the 3D toolbar
        self._profileType = None
        self._setProfileType(2)
        self._method3D = 'sum'

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
