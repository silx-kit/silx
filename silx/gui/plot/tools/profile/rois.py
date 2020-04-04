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

import numpy

from silx.gui import colors

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
        self.__area = None
        self.sigRegionChanged.connect(self.__regionChanged)

    def setProfileWindow(self, profileWindow):
        core.ProfileRoiMixIn.setProfileWindow(self, profileWindow)
        self._updateArea()

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

    def _createAreaItem(self):
        area = items.Shape("polygon")
        color = colors.rgba(self.getColor())
        area.setColor(color)
        area.setFill(True)
        area.setPoints([[0, 0]])  # Else it segfault
        area.setVisible(False)
        self.__area = area
        return area

    def _updateArea(self):
        area = self.__area
        if area is None:
            return
        profileManager = self.getProfileManager()
        if profileManager is None:
            area.setVisible(False)
            return
        item = profileManager.getPlotItem()
        if item is None:
            area.setVisible(False)
            return
        polygon = self._computePolygon(item)
        area.setVisible(True)
        polygon = numpy.array(polygon).T
        area.setPoints(polygon, copy=False)

    def _computePolygon(self, item):
        if not isinstance(item, items.ImageBase):
            raise TypeError("Unexpected class %s" % type(item))

        if isinstance(item, items.ImageData):
            currentData = item.getData(copy=False)
        elif isinstance(item, items.ImageRgba):
            rgba = item.getData(copy=False)
            currentData = rgba[..., 0]

        origin = item.getOrigin()
        scale = item.getScale()
        _coords, _profile, area, _profileName, _xLabel = createProfile(
            roiInfo=self._getRoiInfo(),
            currentData=currentData,
            origin=origin,
            scale=scale,
            lineWidth=self.getProfileLineWidth(),
            method="dry-run")
        return area

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
        if not isinstance(item, items.ImageBase):
            raise TypeError("Unexpected class %s" % type(item))

        origin = item.getOrigin()
        scale = item.getScale()
        colormap = None  # Not used for 2D data
        method = self.getProfileMethod()

        def createProfile2(currentData):
            coords, profile, _area, profileName, xLabel = createProfile(
                roiInfo=self._getRoiInfo(),
                currentData=currentData,
                origin=origin,
                scale=scale,
                lineWidth=self.getProfileLineWidth(),
                method=method)
            return coords, profile, profileName, xLabel

        data = core.ProfileData()

        if isinstance(item, items.ImageData):
            currentData = item.getData(copy=False)
        elif isinstance(item, items.ImageRgba):
            rgba = item.getData(copy=False)
            is_uint8 = rgba.dtype.type == numpy.uint8
            # luminosity
            if is_uint8:
                rgba = rgba.astype(numpy.float)
            currentData = 0.21 * rgba[..., 0] + 0.72 * rgba[..., 1] + 0.07 * rgba[..., 2]

        coords, profile, profileName, xLabel = createProfile2(currentData)

        data.coords = coords
        data.profile = profile
        data.profileName = profileName
        data.xLabel = xLabel
        data.colormap = colormap
        data.currentData = currentData

        if isinstance(item, items.ImageRgba):
            rgba = item.getData(copy=False)
            _coords, r, _profileName, _xLabel = createProfile2(rgba[..., 0])
            _coords, g, _profileName, _xLabel = createProfile2(rgba[..., 1])
            _coords, b, _profileName, _xLabel = createProfile2(rgba[..., 2])
            data.r = r
            data.g = g
            data.b = b
            if rgba.shape[-1] == 4:
                _coords, a, _profileName, _xLabel = createProfile(rgba[..., 3])
                data.a = a

        return data


class ProfileImageHorizontalLineROI(roi_items.HorizontalLineROI,
                                    _DefaultImageProfileRoiMixIn):
    """ROI for an horizontal profile at a location of an image"""

    ICON = 'shape-horizontal'
    NAME = 'horizontal line profile'

    def __init__(self, parent=None):
        roi_items.HorizontalLineROI.__init__(self, parent=parent)
        _DefaultImageProfileRoiMixIn.__init__(self, parent=parent)

    def _updateShape(self):
        """Connect ProfileRoi method with ROI methods"""
        super(ProfileImageHorizontalLineROI, self)._updateShape()
        self._updateArea()

    def _createShapeItems(self, points):
        """Connect ProfileRoi method with ROI methods"""
        result = super(ProfileImageHorizontalLineROI, self)._createShapeItems(points)
        area = self._createAreaItem()
        result.append(area)
        return result


class ProfileImageVerticalLineROI(roi_items.VerticalLineROI,
                                  _DefaultImageProfileRoiMixIn):
    """ROI for a vertical profile at a location of an image"""

    ICON = 'shape-vertical'
    NAME = 'vertical line profile'

    def __init__(self, parent=None):
        roi_items.VerticalLineROI.__init__(self, parent=parent)
        _DefaultImageProfileRoiMixIn.__init__(self, parent=parent)

    def _updateShape(self):
        """Connect ProfileRoi method with ROI methods"""
        super(ProfileImageVerticalLineROI, self)._updateShape()
        self._updateArea()

    def _createShapeItems(self, points):
        """Connect ProfileRoi method with ROI methods"""
        result = super(ProfileImageVerticalLineROI, self)._createShapeItems(points)
        area = self._createAreaItem()
        result.append(area)
        return result

class ProfileImageLineROI(roi_items.LineROI,
                          _DefaultImageProfileRoiMixIn):
    """ROI for an image profile between 2 points"""

    ICON = 'shape-diagonal'
    NAME = 'line profile'

    def __init__(self, parent=None):
        roi_items.LineROI.__init__(self, parent=parent)
        _DefaultImageProfileRoiMixIn.__init__(self, parent=parent)

    def _updateShape(self):
        """Connect ProfileRoi method with ROI methods"""
        super(ProfileImageLineROI, self)._updateShape()
        self._updateArea()

    def _createShapeItems(self, points):
        """Connect ProfileRoi method with ROI methods"""
        result = super(ProfileImageLineROI, self)._createShapeItems(points)
        area = self._createAreaItem()
        result.append(area)
        return result
