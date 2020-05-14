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
"""This module define ROIs for profile tools.

.. inheritance-diagram::
   silx.gui.plot.tools.profile.rois
   :top-classes: silx.gui.plot.tools.profile.core.ProfileRoiMixIn, silx.gui.plot.items.roi.RegionOfInterest
   :parts: 1
   :private-bases:
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "03/04/2020"

import numpy

from silx.gui import colors

from silx.gui.plot import items
from silx.gui.plot.items import roi as roi_items
from . import core
from silx.gui import utils



class _DefaultImageProfileRoiMixIn(core.ProfileRoiMixIn):
    """Provide common behavior for silx default image profile ROI.
    """

    ITEM_KIND = items.ImageBase

    def __init__(self, parent=None):
        core.ProfileRoiMixIn.__init__(self, parent=parent)
        self.__method = "mean"
        self.__width = 1

        area = items.Shape("polygon")
        color = colors.rgba(self.getColor())
        area.setColor(color)
        area.setFill(True)
        area.setOverlay(True)
        area.setPoints([[0, 0], [0, 0]])  # Else it segfault
        self.__area = area
        self.addItem(area)

        self.sigRegionChanged.connect(self.__regionChanged)
        self.sigItemChanged.connect(self._updateAreaProperty)

    def _updateAreaProperty(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.COLOR:
            self._updateItemProperty(event, self, self.__area)
        elif event == items.ItemChangedType.VISIBLE:
            self._updateItemProperty(event, self, self.__area)

    def setProfileWindow(self, profileWindow):
        core.ProfileRoiMixIn.setProfileWindow(self, profileWindow)
        self._updateArea()

    def __regionChanged(self):
        self.invalidateProfile()
        self._updateArea()

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

    def _updateArea(self):
        area = self.__area
        if area is None:
            self.setLineStyle("-")
            return
        profileManager = self.getProfileManager()
        if profileManager is None:
            area.setVisible(False)
            self.setLineStyle("-")
            return
        item = profileManager.getPlotItem()
        if item is None:
            area.setVisible(False)
            self.setLineStyle("-")
            return
        polygon = self._computePolygon(item)
        area.setVisible(True)
        polygon = numpy.array(polygon).T
        self.setLineStyle("--")
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
        _coords, _profile, area, _profileName, _xLabel = core.createProfile(
            roiInfo=self._getRoiInfo(),
            currentData=currentData,
            origin=origin,
            scale=scale,
            lineWidth=self.getProfileLineWidth(),
            method="none")
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
        method = self.getProfileMethod()

        def createProfile2(currentData):
            coords, profile, _area, profileName, xLabel = core.createProfile(
                roiInfo=self._getRoiInfo(),
                currentData=currentData,
                origin=origin,
                scale=scale,
                lineWidth=self.getProfileLineWidth(),
                method=method)
            return coords, profile, profileName, xLabel

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

        if isinstance(item, items.ImageRgba):
            rgba = item.getData(copy=False)
            _coords, r, _profileName, _xLabel = createProfile2(rgba[..., 0])
            _coords, g, _profileName, _xLabel = createProfile2(rgba[..., 1])
            _coords, b, _profileName, _xLabel = createProfile2(rgba[..., 2])
            if rgba.shape[-1] == 4:
                _coords, a, _profileName, _xLabel = createProfile2(rgba[..., 3])
            else:
                a = [None]
            data = core.RgbaProfileData(
                coords=coords,
                profile=profile[0],
                profile_r=r[0],
                profile_g=g[0],
                profile_b=b[0],
                profile_a=a[0],
                title=profileName,
                xLabel=xLabel,
            )
        else:
            data = core.CurveProfileData(
                coords=coords,
                profile=profile[0],
                title=profileName,
                xLabel=xLabel,
            )
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
    """ROI for an image profile between 2 points.

    The X profile of this ROI is the projecting into one of the x/y axes,
    using its scale and its orientation.
    """

    ICON = 'shape-diagonal'
    NAME = 'line profile'

    def __init__(self, parent=None):
        roi_items.LineROI.__init__(self, parent=parent)
        _DefaultImageProfileRoiMixIn.__init__(self, parent=parent)


class ProfileImageDirectedLineROI(roi_items.LineROI,
                                  _DefaultImageProfileRoiMixIn):
    """ROI for an image profile between 2 points.

    The X profile of the line is displayed projected into the line itself,
    using its scale and its orientation. It's the distance from the origin.
    """

    ICON = 'shape-diagonal-directed'
    NAME = 'line projection'

    def __init__(self, parent=None):
        roi_items.LineROI.__init__(self, parent=parent)
        _DefaultImageProfileRoiMixIn.__init__(self, parent=parent)
        self._handleStart.setSymbol('o')

    def computeProfile(self, item):
        if not isinstance(item, items.ImageBase):
            raise TypeError("Unexpected class %s" % type(item))

        from silx.image.bilinear import BilinearImage

        origin = item.getOrigin()
        scale = item.getScale()
        method = self.getProfileMethod()
        roiWidth = self.getProfileLineWidth()
        currentData = item.getData(copy=False)

        roiInfo = self._getRoiInfo()
        roiStart, roiEnd, _lineProjectionMode = roiInfo

        startPt = ((roiStart[1] - origin[1]) / scale[1],
                   (roiStart[0] - origin[0]) / scale[0])
        endPt = ((roiEnd[1] - origin[1]) / scale[1],
                 (roiEnd[0] - origin[0]) / scale[0])

        bilinear = BilinearImage(currentData)
        profile = bilinear.profile_line(
            (startPt[0] - 0.5, startPt[1] - 0.5),
            (endPt[0] - 0.5, endPt[1] - 0.5),
            roiWidth,
            method=method)

        # Compute the line size
        lineSize = numpy.sqrt((roiEnd[1] - roiStart[1])**2 +
                              (roiEnd[0] - roiStart[0])**2)
        coords = numpy.linspace(0, lineSize, len(profile),
                                endpoint=True,
                                dtype=numpy.float32)
        data = core.CurveProfileData(
            coords=coords,
            profile=profile,
            title="Line projection",
            xLabel="Distance",
        )
        return data


class _ProfileCrossROI(roi_items.HandleBasedROI, core.ProfileRoiMixIn):

    """ROI to manage a cross of profiles

    It is managed using 2 sub ROIs for vertical and horizontal.
    """

    _kind = "Cross"
    """Label for this kind of ROI"""

    _plotShape = "point"
    """Plot shape which is used for the first interaction"""

    def __init__(self, parent=None):
        roi_items.HandleBasedROI.__init__(self, parent=parent)
        core.ProfileRoiMixIn.__init__(self, parent=parent)
        self.sigRegionChanged.connect(self.__regionChanged)
        self.sigAboutToBeRemoved.connect(self.__aboutToBeRemoved)
        self.__position = 0, 0
        self.__vline = None
        self.__hline = None
        self.__handle = self.addHandle()
        self.__handleLabel = self.addLabelHandle()
        self.__handleLabel.setText(self.getName())
        self.__inhibitReentance = utils.LockReentrant()
        self.computeProfile = None
        self.sigItemChanged.connect(self.__updateLineProperty)

        # Make sure the marker is over the ROIs
        self.__handle.setZValue(1)
        # Create the vline and the hline
        self._createSubRois()

    def setFirstShapePoints(self, points):
        pos = points[0]
        self.setPosition(pos)

    def getPosition(self):
        """Returns the position of this ROI

        :rtype: numpy.ndarray
        """
        return self.__position

    def setPosition(self, pos):
        """Set the position of this ROI

        :param numpy.ndarray pos: 2d-coordinate of this point
        """
        self.__position = pos
        with utils.blockSignals(self.__handle):
            self.__handle.setPosition(*pos)
        with utils.blockSignals(self.__handleLabel):
            self.__handleLabel.setPosition(*pos)
        self.sigRegionChanged.emit()

    def handleDragUpdated(self, handle, origin, previous, current):
        if handle is self.__handle:
            self.setPosition(current)

    def __updateLineProperty(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.NAME:
            self.__handleLabel.setText(self.getName())
        elif event in [items.ItemChangedType.COLOR,
                     items.ItemChangedType.VISIBLE]:
            lines = []
            if self.__vline:
                lines.append(self.__vline)
            if self.__hline:
                lines.append(self.__hline)
            self._updateItemProperty(event, self, lines)

    def _createLines(self, parent):
        """Inherit this function to return 2 ROI objects for respectivly
        the horizontal, and the vertical lines."""
        raise NotImplementedError()

    def _setProfileManager(self, profileManager):
        core.ProfileRoiMixIn._setProfileManager(self, profileManager)
        # Connecting the vline and the hline
        roiManager = profileManager.getRoiManager()
        roiManager.addRoi(self.__vline)
        roiManager.addRoi(self.__hline)

    def _createSubRois(self):
        hline, vline = self._createLines(parent=None)
        for i, line in enumerate([vline, hline]):
            line.setPosition(self.__position[i])
            line.setEditable(True)
            line.setSelectable(True)
            line.setFocusProxy(self)
            line.setName("")
        self.__vline = vline
        self.__hline = hline
        vline.sigAboutToBeRemoved.connect(self.__vlineRemoved)
        vline.sigRegionChanged.connect(self.__vlineRegionChanged)
        hline.sigAboutToBeRemoved.connect(self.__hlineRemoved)
        hline.sigRegionChanged.connect(self.__hlineRegionChanged)

    def _getLines(self):
        return self.__hline, self.__vline

    def __regionChanged(self):
        if self.__inhibitReentance.locked():
            return
        x, y = self.getPosition()
        hline, vline = self._getLines()
        if hline is None:
            return
        with self.__inhibitReentance:
            hline.setPosition(y)
            vline.setPosition(x)

    def __vlineRegionChanged(self):
        if self.__inhibitReentance.locked():
            return
        pos = self.getPosition()
        vline = self.__vline
        pos = vline.getPosition(), pos[1]
        with self.__inhibitReentance:
            self.setPosition(pos)

    def __hlineRegionChanged(self):
        if self.__inhibitReentance.locked():
            return
        pos = self.getPosition()
        hline = self.__hline
        pos = pos[0], hline.getPosition()
        with self.__inhibitReentance:
            self.setPosition(pos)

    def __aboutToBeRemoved(self):
        vline = self.__vline
        hline = self.__hline
        # Avoid side remove signals
        if hline is not None:
            hline.sigAboutToBeRemoved.disconnect(self.__hlineRemoved)
            hline.sigRegionChanged.disconnect(self.__hlineRegionChanged)
        if vline is not None:
            vline.sigAboutToBeRemoved.disconnect(self.__vlineRemoved)
            vline.sigRegionChanged.disconnect(self.__vlineRegionChanged)
        # Clean up the child
        profileManager = self.getProfileManager()
        roiManager = profileManager.getRoiManager()
        if hline is not None:
            roiManager.removeRoi(hline)
            self.__hline = None
        if vline is not None:
            roiManager.removeRoi(vline)
            self.__vline = None

    def __hlineRemoved(self):
        self.__lineRemoved(isHline=True)

    def __vlineRemoved(self):
        self.__lineRemoved(isHline=False)

    def __lineRemoved(self, isHline):
        """If any of the lines is removed: disconnect this objects, and let the
        other one persist"""
        hline, vline = self._getLines()

        hline.sigAboutToBeRemoved.disconnect(self.__hlineRemoved)
        vline.sigAboutToBeRemoved.disconnect(self.__vlineRemoved)
        hline.sigRegionChanged.disconnect(self.__hlineRegionChanged)
        vline.sigRegionChanged.disconnect(self.__vlineRegionChanged)

        self.__hline = None
        self.__vline = None
        profileManager = self.getProfileManager()
        roiManager = profileManager.getRoiManager()
        if isHline:
            self.__releaseLine(vline)
        else:
            self.__releaseLine(hline)
        roiManager.removeRoi(self)

    def __releaseLine(self, line):
        """Release the line in order to make it independent"""
        line.setFocusProxy(None)
        line.setName(self.getName())
        line.setEditable(self.isEditable())
        line.setSelectable(self.isSelectable())


class ProfileImageCrossROI(_ProfileCrossROI):
    """ROI to manage a cross of profiles

    It is managed using 2 sub ROIs for vertical and horizontal.
    """

    ICON = 'shape-cross'
    NAME = 'cross profile'
    ITEM_KIND = items.ImageBase

    def _createLines(self, parent):
        vline = ProfileImageVerticalLineROI(parent=parent)
        hline = ProfileImageHorizontalLineROI(parent=parent)
        return hline, vline

    def setProfileMethod(self, method):
        """
        :param str method: method to compute the profile. Can be 'mean' or 'sum'
        """
        hline, vline = self._getLines()
        hline.setProfileMethod(method)
        vline.setProfileMethod(method)
        self.invalidateProperties()

    def getProfileMethod(self):
        hline, _vline = self._getLines()
        return hline.getProfileMethod()

    def setProfileLineWidth(self, width):
        hline, vline = self._getLines()
        hline.setProfileLineWidth(width)
        vline.setProfileLineWidth(width)
        self.invalidateProperties()

    def getProfileLineWidth(self):
        hline, _vline = self._getLines()
        return hline.getProfileLineWidth()


class _DefaultScatterProfileRoiMixIn(core.ProfileRoiMixIn):
    """Provide common behavior for silx default scatter profile ROI.
    """

    ITEM_KIND = items.Scatter

    def __init__(self, parent=None):
        core.ProfileRoiMixIn.__init__(self, parent=parent)
        self.__nPoints = 1024
        self.sigRegionChanged.connect(self.__regionChanged)

    def __regionChanged(self):
        self.invalidateProfile()

    # Number of points

    def getNPoints(self):
        """Returns the number of points of the profiles

        :rtype: int
        """
        return self.__nPoints

    def setNPoints(self, npoints):
        """Set the number of points of the profiles

        :param int npoints:
        """
        npoints = int(npoints)
        if npoints < 1:
            raise ValueError("Unsupported number of points: %d" % npoints)
        elif npoints != self.__nPoints:
            self.__nPoints = npoints
            self.invalidateProperties()
            self.invalidateProfile()

    def _computeProfileTitle(self, x0, y0, x1, y1):
        """Compute corresponding plot title

        This can be overridden to change title behavior.

        :param float x0: Profile start point X coord
        :param float y0: Profile start point Y coord
        :param float x1: Profile end point X coord
        :param float y1: Profile end point Y coord
        :return: Title to use
        :rtype: str
        """
        if x0 == x1:
            title = 'X = %g; Y = [%g, %g]' % (x0, y0, y1)
        elif y0 == y1:
            title = 'Y = %g; X = [%g, %g]' % (y0, x0, x1)
        else:
            m = (y1 - y0) / (x1 - x0)
            b = y0 - m * x0
            title = 'Y = %g * X %+g' % (m, b)

        return title

    def _computeProfile(self, scatter, x0, y0, x1, y1):
        """Compute corresponding profile

        :param float x0: Profile start point X coord
        :param float y0: Profile start point Y coord
        :param float x1: Profile end point X coord
        :param float y1: Profile end point Y coord
        :return: (points, values) profile data or None
        """
        future = scatter._getInterpolator()
        interpolator = future.result()
        if interpolator is None:
            return None  # Cannot init an interpolator

        nPoints = self.getNPoints()
        points = numpy.transpose((
            numpy.linspace(x0, x1, nPoints, endpoint=True),
            numpy.linspace(y0, y1, nPoints, endpoint=True)))

        values = interpolator(points)

        if not numpy.any(numpy.isfinite(values)):
            return None  # Profile outside convex hull

        return points, values

    def computeProfile(self, item):
        """Update profile according to current ROI"""
        if not isinstance(item, items.Scatter):
            raise TypeError("Unexpected class %s" % type(item))

        # Get end points
        if isinstance(self, roi_items.LineROI):
            points = self.getEndPoints()
            x0, y0 = points[0]
            x1, y1 = points[1]
        elif isinstance(self, (roi_items.VerticalLineROI, roi_items.HorizontalLineROI)):
            profileManager = self.getProfileManager()
            plot = profileManager.getPlotWidget()

            if isinstance(self, roi_items.HorizontalLineROI):
                x0, x1 = plot.getXAxis().getLimits()
                y0 = y1 = self.getPosition()

            elif isinstance(self, roi_items.VerticalLineROI):
                x0 = x1 = self.getPosition()
                y0, y1 = plot.getYAxis().getLimits()
        else:
            raise RuntimeError('Unsupported ROI for profile: {}'.format(self.__class__))

        if x1 < x0 or (x1 == x0 and y1 < y0):
            # Invert points
            x0, y0, x1, y1 = x1, y1, x0, y0

        profile = self._computeProfile(item, x0, y0, x1, y1)
        if profile is None:
            return None
        title = self._computeProfileTitle(x0, y0, x1, y1)

        points = profile[0]
        values = profile[1]

        if (numpy.abs(points[-1, 0] - points[0, 0]) >
                numpy.abs(points[-1, 1] - points[0, 1])):
            xProfile = points[:, 0]
            xLabel = 'X'
        else:
            xProfile = points[:, 1]
            xLabel = 'Y'

        data = core.CurveProfileData(
            coords=xProfile,
            profile=values,
            xLabel=xLabel,
            title=title
        )
        return data


class ProfileScatterHorizontalLineROI(roi_items.HorizontalLineROI,
                                      _DefaultScatterProfileRoiMixIn):
    """ROI for an horizontal profile at a location of a scatter"""

    ICON = 'shape-horizontal'
    NAME = 'horizontal line profile'

    def __init__(self, parent=None):
        roi_items.HorizontalLineROI.__init__(self, parent=parent)
        _DefaultScatterProfileRoiMixIn.__init__(self, parent=parent)


class ProfileScatterVerticalLineROI(roi_items.VerticalLineROI,
                                    _DefaultScatterProfileRoiMixIn):
    """ROI for an horizontal profile at a location of a scatter"""

    ICON = 'shape-vertical'
    NAME = 'vertical line profile'

    def __init__(self, parent=None):
        roi_items.VerticalLineROI.__init__(self, parent=parent)
        _DefaultScatterProfileRoiMixIn.__init__(self, parent=parent)


class ProfileScatterLineROI(roi_items.LineROI,
                            _DefaultScatterProfileRoiMixIn):
    """ROI for an horizontal profile at a location of a scatter"""

    ICON = 'shape-diagonal'
    NAME = 'line profile'

    def __init__(self, parent=None):
        roi_items.LineROI.__init__(self, parent=parent)
        _DefaultScatterProfileRoiMixIn.__init__(self, parent=parent)


class ProfileScatterCrossROI(_ProfileCrossROI):
    """ROI to manage a cross of profiles for scatters.
    """

    ICON = 'shape-cross'
    NAME = 'cross profile'
    ITEM_KIND = items.Scatter

    def _createLines(self, parent):
        vline = ProfileScatterVerticalLineROI(parent=parent)
        hline = ProfileScatterHorizontalLineROI(parent=parent)
        return hline, vline

    def getNPoints(self):
        """Returns the number of points of the profiles

        :rtype: int
        """
        hline, _vline = self._getLines()
        return hline.getNPoints()

    def setNPoints(self, npoints):
        """Set the number of points of the profiles

        :param int npoints:
        """
        hline, vline = self._getLines()
        hline.setNPoints(npoints)
        vline.setNPoints(npoints)
        self.invalidateProperties()


class _DefaulScatterProfileSliceRoiMixIn(core.ProfileRoiMixIn):
    """Default ROI to allow to slice in the scatter data."""

    ITEM_KIND = items.Scatter

    def __init__(self, parent=None):
        core.ProfileRoiMixIn.__init__(self, parent=parent)

        area = items.Shape("polylines")
        color = colors.rgba(self.getColor())
        area.setColor(color)
        area.setFill(True)
        area.setOverlay(True)
        area.setPoints([[0, 0], [0, 0]])  # Else it segfault
        self.__area = area
        self.addItem(area)

        self.sigRegionChanged.connect(self.__regionChanged)
        self.sigItemChanged.connect(self._updateAreaProperty)

    def _updateAreaProperty(self, event=None, checkVisibility=True):
        if event == items.ItemChangedType.COLOR:
            self._updateItemProperty(event, self, self.__area)
        elif event == items.ItemChangedType.VISIBLE:
            self._updateItemProperty(event, self, self.__area)

    def setProfileWindow(self, profileWindow):
        core.ProfileRoiMixIn.setProfileWindow(self, profileWindow)
        self._updateArea()

    def __regionChanged(self):
        self.invalidateProfile()
        self._updateArea()

    def _updateArea(self):
        area = self.__area
        if area is None:
            self.setLineStyle("-")
            return
        profileManager = self.getProfileManager()
        if profileManager is None:
            self.setLineStyle("-")
            area.setVisible(False)
            return
        item = profileManager.getPlotItem()
        if item is None:
            area.setVisible(False)
            self.setLineStyle("-")
            return
        polylines = self._computePolylines(item)
        if polylines is None or len(polylines) == 0:
            area.setVisible(False)
            self.setLineStyle("-")
            return
        area.setVisible(True)
        self.setLineStyle("--")
        area.setPoints(polylines, copy=False)

    def _computePolylines(self, item):
        if not isinstance(item, items.Scatter):
            raise TypeError("Unexpected class %s" % type(item))

        slicing = self.__getSlice(item)
        if slicing is None:
            return None

        xx, yy, _values, _xx_error, _yy_error = item.getData(copy=False)
        xx, yy = xx[slicing], yy[slicing]
        return numpy.array((xx, yy)).T

    def __getSlice(self, item):
        position = self.getPosition()
        bounds = item.getCurrentVisualizationParameter(items.Scatter.VisualizationParameter.GRID_BOUNDS)
        if isinstance(self, roi_items.HorizontalLineROI):
            axis = 1
        elif isinstance(self, roi_items.VerticalLineROI):
            axis = 0
        else:
            assert False
        if position < bounds[0][axis] or position > bounds[1][axis]:
            # ROI outside of the scatter bound
            return None

        major_order = item.getCurrentVisualizationParameter(items.Scatter.VisualizationParameter.GRID_MAJOR_ORDER)
        assert major_order == 'row'
        max_grid_yy, max_grid_xx = item.getCurrentVisualizationParameter(items.Scatter.VisualizationParameter.GRID_SHAPE)

        xx, yy, _values, _xx_error, _yy_error = item.getData(copy=False)
        if isinstance(self, roi_items.HorizontalLineROI):
            axis = yy
            max_grid_first = max_grid_yy
            max_grid_second = max_grid_xx
            major_axis = major_order == 'column'
        elif isinstance(self, roi_items.VerticalLineROI):
            axis = xx
            max_grid_first = max_grid_xx
            max_grid_second = max_grid_yy
            major_axis = major_order == 'row'
        else:
            assert False

        def argnearest(array, value):
            array = numpy.abs(array - value)
            return numpy.argmin(array)

        if major_axis:
            # slice in the middle of the scatter
            start = max_grid_second // 2 * max_grid_first
            vslice = axis[start:start + max_grid_second]
            index = argnearest(vslice, position)
            slicing = slice(index, None, max_grid_first)
        else:
            # slice in the middle of the scatter
            vslice = axis[max_grid_second // 2::max_grid_second]
            index = argnearest(vslice, position)
            start = index * max_grid_second
            slicing = slice(start, start + max_grid_second)

        return slicing

    def computeProfile(self, item):
        if not isinstance(item, items.Scatter):
            raise TypeError("Unsupported %s item" % type(item))

        slicing = self.__getSlice(item)
        if slicing is None:
            # ROI out of bounds
            return None

        _xx, _yy, values, _xx_error, _yy_error = item.getData(copy=False)
        profile = values[slicing]

        if isinstance(self, roi_items.HorizontalLineROI):
            title = "Horizontal slice"
            xLabel = "Column index"
        elif isinstance(self, roi_items.VerticalLineROI):
            title = "Vertical slice"
            xLabel = "Row index"
        else:
            assert False

        data = core.CurveProfileData(
            coords=numpy.arange(len(profile)),
            profile=profile,
            title=title,
            xLabel=xLabel
        )
        return data


class ProfileScatterHorizontalSliceROI(roi_items.HorizontalLineROI,
                                       _DefaulScatterProfileSliceRoiMixIn):
    """ROI for an horizontal profile at a location of a scatter
    using data slicing.
    """

    ICON = 'slice-horizontal'
    NAME = 'horizontal data slice profile'

    def __init__(self, parent=None):
        roi_items.HorizontalLineROI.__init__(self, parent=parent)
        _DefaulScatterProfileSliceRoiMixIn.__init__(self, parent=parent)


class ProfileScatterVerticalSliceROI(roi_items.VerticalLineROI,
                                       _DefaulScatterProfileSliceRoiMixIn):
    """ROI for a vertical profile at a location of a scatter
    using data slicing.
    """

    ICON = 'slice-vertical'
    NAME = 'vertical data slice profile'

    def __init__(self, parent=None):
        roi_items.VerticalLineROI.__init__(self, parent=parent)
        _DefaulScatterProfileSliceRoiMixIn.__init__(self, parent=parent)


class ProfileScatterCrossSliceROI(_ProfileCrossROI):
    """ROI to manage a cross of slicing profiles on scatters.
    """

    ICON = 'slice-cross'
    NAME = 'cross data slice profile'
    ITEM_KIND = items.Scatter

    def _createLines(self, parent):
        vline = ProfileScatterVerticalSliceROI(parent=parent)
        hline = ProfileScatterHorizontalSliceROI(parent=parent)
        return hline, vline


class _DefaultImageStackProfileRoiMixIn(_DefaultImageProfileRoiMixIn):

    ITEM_KIND = items.ImageStack

    def __init__(self, parent=None):
        super(_DefaultImageStackProfileRoiMixIn, self).__init__(parent=parent)
        self.__profileType = "1D"
        """Kind of profile"""

    def getProfileType(self):
        return self.__profileType

    def setProfileType(self, kind):
        assert kind in ["1D", "2D"]
        if self.__profileType == kind:
            return
        self.__profileType = kind
        self.invalidateProperties()
        self.invalidateProfile()

    def computeProfile(self, item):
        if not isinstance(item, items.ImageStack):
            raise TypeError("Unexpected class %s" % type(item))

        kind = self.getProfileType()
        if kind == "1D":
            result = _DefaultImageProfileRoiMixIn.computeProfile(self, item)
            # z = item.getStackPosition()
            return result

        assert kind == "2D"

        def createProfile2(currentData):
            coords, profile, _area, profileName, xLabel = core.createProfile(
                roiInfo=self._getRoiInfo(),
                currentData=currentData,
                origin=origin,
                scale=scale,
                lineWidth=self.getProfileLineWidth(),
                method=method)
            return coords, profile, profileName, xLabel

        currentData = numpy.array(item.getStackData(copy=False))
        origin = item.getOrigin()
        scale = item.getScale()
        colormap = item.getColormap()
        method = self.getProfileMethod()

        coords, profile, profileName, xLabel = createProfile2(currentData)

        data = core.ImageProfileData(
            coords=coords,
            profile=profile,
            title=profileName,
            xLabel=xLabel,
            yLabel="Y",
            colormap=colormap,
        )
        return data


class ProfileImageStackHorizontalLineROI(roi_items.HorizontalLineROI,
                                         _DefaultImageStackProfileRoiMixIn):
    """ROI for an horizontal profile at a location of a stack of images"""

    ICON = 'shape-horizontal'
    NAME = 'horizontal line profile'

    def __init__(self, parent=None):
        roi_items.HorizontalLineROI.__init__(self, parent=parent)
        _DefaultImageStackProfileRoiMixIn.__init__(self, parent=parent)


class ProfileImageStackVerticalLineROI(roi_items.VerticalLineROI,
                                       _DefaultImageStackProfileRoiMixIn):
    """ROI for an vertical profile at a location of a stack of images"""

    ICON = 'shape-vertical'
    NAME = 'vertical line profile'

    def __init__(self, parent=None):
        roi_items.VerticalLineROI.__init__(self, parent=parent)
        _DefaultImageStackProfileRoiMixIn.__init__(self, parent=parent)


class ProfileImageStackLineROI(roi_items.LineROI,
                               _DefaultImageStackProfileRoiMixIn):
    """ROI for an vertical profile at a location of a stack of images"""

    ICON = 'shape-diagonal'
    NAME = 'line profile'

    def __init__(self, parent=None):
        roi_items.LineROI.__init__(self, parent=parent)
        _DefaultImageStackProfileRoiMixIn.__init__(self, parent=parent)


class ProfileImageStackCrossROI(ProfileImageCrossROI):
    """ROI for an vertical profile at a location of a stack of images"""

    ICON = 'shape-cross'
    NAME = 'cross profile'
    ITEM_KIND = items.ImageStack

    def _createLines(self, parent):
        vline = ProfileImageStackVerticalLineROI(parent=parent)
        hline = ProfileImageStackHorizontalLineROI(parent=parent)
        return hline, vline

    def getProfileType(self):
        hline, _vline = self._getLines()
        return hline.getProfileType()

    def setProfileType(self, kind):
        hline, vline = self._getLines()
        hline.setProfileType(kind)
        vline.setProfileType(kind)
        self.invalidateProperties()
