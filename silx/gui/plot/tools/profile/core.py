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
"""This module define core objects for profile tools.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "03/04/2020"

import collections

from silx.gui import qt


CurveProfileData = collections.namedtuple(
    'CurveProfileData', [
        "coords",
        "profile",
        "title",
        "xLabel",
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
    """Base mix-in for ROI which can be used to select a profile."""

    sigPropertyChanged = qt.Signal()

    def __init__(self, parent=None):
        self.__profileWindow = None
        self.__profileManager = None

    def invalidateProfile(self):
        """Must be called by the implementation when the profile have to be
        recomputed."""
        profileManager = self.getProfileManager()
        if profileManager is not None:
            profileManager.requestUpdateProfile(self)

    def invalidateProperties(self):
        """Must be called when a property of the ROI have changed."""
        self.sigPropertyChanged.emit()

    def _setProfileManager(self, profileManager):
        self.__profileManager = profileManager

    def getProfileManager(self):
        return self.__profileManager

    def getProfileWindow(self):
        return self.__profileWindow

    def setProfileWindow(self, profileWindow):
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
        roiManager.removeRoi(self)

    def computeProfile(self, item):
        raise NotImplementedError()
