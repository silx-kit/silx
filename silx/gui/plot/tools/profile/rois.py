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
"""This module define ROIs for profile tools.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "03/04/2020"

from silx.gui.plot import items
from silx.gui.plot.items import roi as roi_items
from silx.gui.plot.Profile import createProfile
from . import core


class _DefaultImageProfileRoiMixIn(core.ProfileRoiMixIn):
    """Provide common behavior for silx default image profile ROI.
    """
    def __init__(self, parent=None):
        core.ProfileRoiMixIn.__init__(self, parent=parent)
        self.__method = "mean"
        self.__width = 1
        self.sigRegionChanged.connect(self.__regionChanged)

    def __regionChanged(self):
        self.invalidateProfile()

    def setProfileMethod(self, method):
        """
        :param str method: method to compute the profile. Can be 'mean' or 'sum'
        """
        if self.__method == method:
            return
        self.__method = method
        self.invalidateProperties()
        self.invalidateProfile()

    def getProfileMethod(self):
        return self.__method

    def setProfileLineWidth(self, width):
        if self.__width == width:
            return
        self.__width = width
        self._updateArea()
        self.invalidateProperties()
        self.invalidateProfile()

    def getProfileLineWidth(self):
        return self.__width

    def _getRoiInfo(self):
        """Wrapper to allow to reuse the previous Profile code.
    
        It would be good to remove it at one point.
        """
        if isinstance(self, roi_items.HorizontalLineROI):
            lineProjectionMode = 'X'
            y = self.getPosition()
            roiStart = (0, y)
            roiEnd = (1, y)
        elif isinstance(self, roi_items.VerticalLineROI):
            lineProjectionMode = 'Y'
            x = self.getPosition()
            roiStart = (x, 0)
            roiEnd = (x, 1)
        elif isinstance(self, roi_items.LineROI):
            lineProjectionMode = 'D'
            roiStart, roiEnd = self.getEndPoints()
        else:
            assert False
    
        return roiStart, roiEnd, lineProjectionMode

    def computeProfile(self, item):
        if not isinstance(item, items.ImageData):
            raise TypeError("Unexpected class %s" % type(item))

        currentData = item.getData(copy=False)
        origin = item.getOrigin()
        scale = item.getScale()
        colormap = None  # Not used for 2D data
        method = self.getProfileMethod()

        coords, profile, _area, profileName, xLabel = createProfile(
            roiInfo=self._getRoiInfo(),
            currentData=currentData,
            origin=origin,
            scale=scale,
            lineWidth=self.getProfileLineWidth(),
            method=method)

        data = core.ProfileData()
        data.coords = coords
        data.profile = profile
        data.profileName = profileName
        data.xLabel = xLabel
        data.colormap = colormap
        data.currentData = currentData

        return data


class ProfileImageHorizontalLineROI(roi_items.HorizontalLineROI,
                                    _DefaultImageProfileRoiMixIn):
    """ROI for an horizontal profile at a location of an image"""

    ICON = 'shape-horizontal'
    NAME = 'horizontal line profile'

    def __init__(self, parent=None):
        roi_items.HorizontalLineROI.__init__(self, parent=parent)
        _DefaultImageProfileRoiMixIn.__init__(self, parent=parent)


class ProfileImageVerticalLineROI(roi_items.VerticalLineROI,
                                  _DefaultImageProfileRoiMixIn):
    """ROI for a vertical profile at a location of an image"""

    ICON = 'shape-vertical'
    NAME = 'vertical line profile'

    def __init__(self, parent=None):
        roi_items.VerticalLineROI.__init__(self, parent=parent)
        _DefaultImageProfileRoiMixIn.__init__(self, parent=parent)


class ProfileImageLineROI(roi_items.LineROI,
                          _DefaultImageProfileRoiMixIn):
    """ROI for an image profile between 2 points"""

    ICON = 'shape-diagonal'
    NAME = 'line profile'

    def __init__(self, parent=None):
        roi_items.LineROI.__init__(self, parent=parent)
        _DefaultImageProfileRoiMixIn.__init__(self, parent=parent)
