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
"""This module define core objects for profile tools.
"""

__authors__ = ["V.A. Sole", "T. Vincent", "P. Knobel", "H. Payno", "V. Valls"]
__license__ = "MIT"
__date__ = "17/04/2020"

import collections
import numpy
import weakref

from silx.image.bilinear import BilinearImage
from silx.gui import qt


CurveProfileData = collections.namedtuple(
    'CurveProfileData', [
        "coords",
        "profile",
        "title",
        "xLabel",
        "yLabel",
    ])

RgbaProfileData = collections.namedtuple(
    'RgbaProfileData', [
        "coords",
        "profile",
        "profile_r",
        "profile_g",
        "profile_b",
        "profile_a",
        "title",
        "xLabel",
        "yLabel",
    ])

ImageProfileData = collections.namedtuple(
    'ImageProfileData', [
        'coords',
        'profile',
        'title',
        'xLabel',
        'yLabel',
        'colormap',
    ])


class ProfileRoiMixIn:
    """Base mix-in for ROI which can be used to select a profile.

    This mix-in have to be applied to a :class:`~silx.gui.plot.items.roi.RegionOfInterest`
    in order to be usable by a :class:`~silx.gui.plot.tools.profile.manager.ProfileManager`.
    """

    ITEM_KIND = None
    """Define the plot item which can be used with this profile ROI"""

    sigProfilePropertyChanged = qt.Signal()
    """Emitted when a property of this profile have changed"""

    sigPlotItemChanged = qt.Signal()
    """Emitted when the plot item linked to this profile have changed"""

    def __init__(self, parent=None):
        self.__profileWindow = None
        self.__profileManager = None
        self.__plotItem = None
        self.setName("Profile")
        self.setEditable(True)
        self.setSelectable(True)

    def invalidateProfile(self):
        """Must be called by the implementation when the profile have to be
        recomputed."""
        profileManager = self.getProfileManager()
        if profileManager is not None:
            profileManager.requestUpdateProfile(self)

    def invalidateProperties(self):
        """Must be called when a property of the profile have changed."""
        self.sigProfilePropertyChanged.emit()

    def _setPlotItem(self, plotItem):
        """Specify the plot item to use with this profile

        :param `~silx.gui.plot.items.item.Item` plotItem: A plot item
        """
        previousPlotItem = self.getPlotItem()
        if previousPlotItem is plotItem:
            return
        self.__plotItem = weakref.ref(plotItem)
        self.sigPlotItemChanged.emit()

    def getPlotItem(self):
        """Returns the plot item used by this profile

        :rtype: `~silx.gui.plot.items.item.Item`
        """
        if self.__plotItem is None:
            return None
        plotItem = self.__plotItem()
        if plotItem is None:
            self.__plotItem = None
        return plotItem

    def _setProfileManager(self, profileManager):
        self.__profileManager = profileManager

    def getProfileManager(self):
        """
        Returns the profile manager connected to this ROI.

        :rtype: ~silx.gui.plot.tools.profile.manager.ProfileManager
        """
        return self.__profileManager

    def getProfileWindow(self):
        """
        Returns the windows associated to this ROI, else None.

        :rtype: ProfileWindow
        """
        return self.__profileWindow

    def setProfileWindow(self, profileWindow):
        """
        Associate a window to this ROI. Can be None.

        :param ProfileWindow profileWindow: A main window
            to display the profile.
        """
        if profileWindow is self.__profileWindow:
            return
        if self.__profileWindow is not None:
            self.__profileWindow.sigClose.disconnect(self.__profileWindowAboutToClose)
            self.__profileWindow.setRoiProfile(None)
        self.__profileWindow = profileWindow
        if self.__profileWindow is not None:
            self.__profileWindow.sigClose.connect(self.__profileWindowAboutToClose)
            self.__profileWindow.setRoiProfile(self)

    def __profileWindowAboutToClose(self):
        profileManager = self.getProfileManager()
        roiManager = profileManager.getRoiManager()
        try:
            roiManager.removeRoi(self)
        except ValueError:
            pass

    def computeProfile(self, item):
        """
        Compute the profile which will be displayed.

        This method is not called from the main Qt thread, but from a thread
        pool.

        :param ~silx.gui.plot.items.Item item: A plot item
        :rtype: Union[CurveProfileData,ImageProfileData]
        """
        raise NotImplementedError()


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
    :param str method: method to compute the profile. Can be 'mean' or 'sum' or
        'none'
    :return: profile image + effective ROI area corners in plot coords
    """
    assert axis in (0, 1)
    assert len(data.shape) == 3
    assert method in ('mean', 'sum', 'none')

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

    if method == 'none':
        profile = None
    else:
        if start < height and end > 0:
            if method == 'mean':
                fct = numpy.mean
            elif method == 'sum':
                fct = numpy.sum
            else:
                raise ValueError('method not managed')
            profile = fct(data[:, max(0, start):min(end, height), :], axis=1).astype(numpy.float32)
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
        or 'none': to compute everything except the profile
    :return: `coords, profile, area, profileName, xLabel`, where:
        - coords is the X coordinate to use to display the profile
        - profile is a 2D array of the profiles of the stack of images.
          For a single image, the profile is a curve, so this parameter
          has a shape *(1, len(curve))*
        - area is a tuple of two 1D arrays with 4 values each. They represent
          the effective ROI area corners in plot coords.
        - profileName is a string describing the ROI, meant to be used as
          title of the profile plot
        - xLabel the label for X in the profile window

    :rtype: tuple(ndarray,ndarray,(ndarray,ndarray),str)
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

        if method == 'none':
            coords = None
        else:
            coords = numpy.arange(len(profile[0]), dtype=numpy.float32)
            coords = coords * scale[0] + origin[0]

        yMin, yMax = min(area[1]), max(area[1]) - 1
        if roiWidth <= 1:
            profileName = '{ylabel} = %g' % yMin
        else:
            profileName = '{ylabel} = [%g, %g]' % (yMin, yMax)
        xLabel = '{xlabel}'

    elif lineProjectionMode == 'Y':  # Vertical profile on the whole image
        profile, area = _alignedFullProfile(currentData3D,
                                            origin, scale,
                                            roiStart[0], roiWidth,
                                            axis=1,
                                            method=method)

        if method == 'none':
            coords = None
        else:
            coords = numpy.arange(len(profile[0]), dtype=numpy.float32)
            coords = coords * scale[1] + origin[1]

        xMin, xMax = min(area[0]), max(area[0]) - 1
        if roiWidth <= 1:
            profileName = '{xlabel} = %g' % xMin
        else:
            profileName = '{xlabel} = [%g, %g]' % (xMin, xMax)
        xLabel = '{ylabel}'

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
                if method == 'none':
                    profile = None
                else:
                    profile = _alignedPartialProfile(currentData3D,
                                                     rowRange, colRange,
                                                     axis=0,
                                                     method=method)

            else:  # Column aligned
                rowRange = startPt[0], endPt[0] + 1
                colRange = (int(startPt[1] + 0.5 - 0.5 * roiWidth),
                            int(startPt[1] + 0.5 + 0.5 * roiWidth))
                if method == 'none':
                    profile = None
                else:
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

            if method == 'none':
                profile = None
            else:
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
            roiStartPt = startPt[0] - 0.5 * dRow, startPt[1] - 0.5 * dCol
            roiEndPt = endPt[0] + 0.5 * dRow, endPt[1] + 0.5 * dCol

            # Rotate deltas by 90 degrees to apply line width
            dRow, dCol = dCol, -dRow

            area = (
                numpy.array((roiStartPt[1] - 0.5 * roiWidth * dCol,
                             roiStartPt[1] + 0.5 * roiWidth * dCol,
                             roiEndPt[1] + 0.5 * roiWidth * dCol,
                             roiEndPt[1] - 0.5 * roiWidth * dCol),
                            dtype=numpy.float32) * scale[0] + origin[0],
                numpy.array((roiStartPt[0] - 0.5 * roiWidth * dRow,
                             roiStartPt[0] + 0.5 * roiWidth * dRow,
                             roiEndPt[0] + 0.5 * roiWidth * dRow,
                             roiEndPt[0] - 0.5 * roiWidth * dRow),
                            dtype=numpy.float32) * scale[1] + origin[1])

        # Convert start and end points back to plot coords
        y0 = startPt[0] * scale[1] + origin[1]
        x0 = startPt[1] * scale[0] + origin[0]
        y1 = endPt[0] * scale[1] + origin[1]
        x1 = endPt[1] * scale[0] + origin[0]

        if startPt[1] == endPt[1]:
            profileName = '{xlabel} = %g; {ylabel} = [%g, %g]' % (x0, y0, y1)
            if method == 'none':
                coords = None
            else:
                coords = numpy.arange(len(profile[0]), dtype=numpy.float32)
                coords = coords * scale[1] + y0
            xLabel = '{ylabel}'

        elif startPt[0] == endPt[0]:
            profileName = '{ylabel} = %g; {xlabel} = [%g, %g]' % (y0, x0, x1)
            if method == 'none':
                coords = None
            else:
                coords = numpy.arange(len(profile[0]), dtype=numpy.float32)
                coords = coords * scale[0] + x0
            xLabel = '{xlabel}'

        else:
            m = (y1 - y0) / (x1 - x0)
            b = y0 - m * x0
            profileName = '{ylabel} = %g * {xlabel} %+g' % (m, b)
            if method == 'none':
                coords = None
            else:
                coords = numpy.linspace(x0, x1, len(profile[0]),
                                        endpoint=True,
                                        dtype=numpy.float32)
            xLabel = '{xlabel}'

    return coords, profile, area, profileName, xLabel
