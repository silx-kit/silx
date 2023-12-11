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
"""This provides profile ROIs.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "09/06/2023"


import numpy

from silx.gui.plot.tools.profile import rois
from silx.gui.plot.tools.profile import core
from .core import _CompareImageItem


COLOR_A = "C0"
COLOR_B = "C8"


class ProfileImageLineROI(rois.ProfileImageLineROI):
    """ROI for a compare image profile between 2 points.

    The X profile of this ROI is the projection into one of the x/y axes,
    using its scale and its orientation.
    """

    def computeProfile(self, item):
        if not isinstance(item, _CompareImageItem):
            raise TypeError("Unexpected class %s" % type(item))

        origin = item.getOrigin()
        scale = item.getScale()
        method = self.getProfileMethod()
        lineWidth = self.getProfileLineWidth()
        roiInfo = self._getRoiInfo()

        def createProfile2(currentData):
            coords, profile, _area, profileName, xLabel = core.createProfile(
                roiInfo=roiInfo,
                currentData=currentData,
                origin=origin,
                scale=scale,
                lineWidth=lineWidth,
                method=method,
            )
            return coords, profile, profileName, xLabel

        currentData1 = item.getImageData1()
        currentData2 = item.getImageData2()

        yLabel = "%s" % str(method).capitalize()
        coords, profile1, title, xLabel = createProfile2(currentData1)
        title = title + "; width = %d" % lineWidth
        _coords, profile2, _title, _xLabel = createProfile2(currentData2)

        profile1.shape = -1
        profile2.shape = -1

        title = title.format(xlabel="width", ylabel="height")
        xLabel = xLabel.format(xlabel="width", ylabel="height")
        yLabel = yLabel.format(xlabel="width", ylabel="height")

        data = core.CurvesProfileData(
            coords=coords,
            profiles=[
                core.CurveProfileDesc(profile1, color=COLOR_A, name="profileA"),
                core.CurveProfileDesc(profile2, color=COLOR_B, name="profileB"),
            ],
            title=title,
            xLabel=xLabel,
            yLabel=yLabel,
        )
        return data


class ProfileImageDirectedLineROI(rois.ProfileImageDirectedLineROI):
    """ROI for a compare image profile between 2 points.

    The X profile of the line is displayed projected into the line itself,
    using its scale and its orientation. It's the distance from the origin.
    """

    def computeProfile(self, item):
        if not isinstance(item, _CompareImageItem):
            raise TypeError("Unexpected class %s" % type(item))

        from silx.image.bilinear import BilinearImage

        origin = item.getOrigin()
        scale = item.getScale()
        method = self.getProfileMethod()
        lineWidth = self.getProfileLineWidth()

        roiInfo = self._getRoiInfo()
        roiStart, roiEnd, _lineProjectionMode = roiInfo

        startPt = (
            (roiStart[1] - origin[1]) / scale[1],
            (roiStart[0] - origin[0]) / scale[0],
        )
        endPt = ((roiEnd[1] - origin[1]) / scale[1], (roiEnd[0] - origin[0]) / scale[0])

        if numpy.array_equal(startPt, endPt):
            return None

        def computeProfile(data):
            bilinear = BilinearImage(data)
            profile = bilinear.profile_line(
                (startPt[0] - 0.5, startPt[1] - 0.5),
                (endPt[0] - 0.5, endPt[1] - 0.5),
                lineWidth,
                method=method,
            )
            return profile

        currentData1 = item.getImageData1()
        currentData2 = item.getImageData2()
        profile1 = computeProfile(currentData1)
        profile2 = computeProfile(currentData2)

        # Compute the line size
        lineSize = numpy.sqrt(
            (roiEnd[1] - roiStart[1]) ** 2 + (roiEnd[0] - roiStart[0]) ** 2
        )
        coords = numpy.linspace(
            0, lineSize, len(profile1), endpoint=True, dtype=numpy.float32
        )

        title = rois._lineProfileTitle(*roiStart, *roiEnd)
        title = title + "; width = %d" % lineWidth
        xLabel = "√({xlabel}²+{ylabel}²)"
        yLabel = str(method).capitalize()

        # Use the axis names from the original plot
        profileManager = self.getProfileManager()
        plot = profileManager.getPlotWidget()
        xLabel = rois._relabelAxes(plot, xLabel)
        title = rois._relabelAxes(plot, title)

        data = core.CurvesProfileData(
            coords=coords,
            profiles=[
                core.CurveProfileDesc(profile1, color=COLOR_A, name="profileA"),
                core.CurveProfileDesc(profile2, color=COLOR_B, name="profileB"),
            ],
            title=title,
            xLabel=xLabel,
            yLabel=yLabel,
        )
        return data
