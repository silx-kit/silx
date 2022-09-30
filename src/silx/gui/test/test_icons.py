# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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
"""Basic test of Qt icons module."""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "06/09/2017"


import unittest
import weakref
import tempfile
import shutil
import os

import silx.resources
from silx.gui import qt
from silx.gui.utils.testutils import TestCaseQt
from silx.gui import icons


class TestIcons(TestCaseQt):
    """Test to check that icons module."""

    @classmethod
    def setUpClass(cls):
        super(TestIcons, cls).setUpClass()

        cls.tmpDirectory = tempfile.mkdtemp(prefix="resource_")
        os.mkdir(os.path.join(cls.tmpDirectory, "gui"))
        destination = os.path.join(cls.tmpDirectory, "gui", "icons")
        os.mkdir(destination)
        shutil.copy(silx.resources.resource_filename("gui/icons/zoom-in.png"), destination)
        shutil.copy(silx.resources.resource_filename("gui/icons/zoom-out.svg"), destination)

    @classmethod
    def tearDownClass(cls):
        super(TestIcons, cls).tearDownClass()
        shutil.rmtree(cls.tmpDirectory)

    def setUp(self):
        # Store the original configuration
        self._oldResources = dict(silx.resources._RESOURCE_DIRECTORIES)
        silx.resources.register_resource_directory("test", "foo.bar", forced_path=self.tmpDirectory)
        unittest.TestCase.setUp(self)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        # Restiture the original configuration
        silx.resources._RESOURCE_DIRECTORIES = self._oldResources

    def testIcon(self):
        icon = icons.getQIcon("silx:gui/icons/zoom-out")
        self.assertIsNotNone(icon)

    def testPrefix(self):
        icon = icons.getQIcon("silx:gui/icons/zoom-out")
        self.assertIsNotNone(icon)

    def testSvgIcon(self):
        if "svg" not in qt.supportedImageFormats():
            self.skipTest("SVG not supported")
        icon = icons.getQIcon("test:gui/icons/zoom-out")
        self.assertIsNotNone(icon)

    def testPngIcon(self):
        icon = icons.getQIcon("test:gui/icons/zoom-in")
        self.assertIsNotNone(icon)

    def testUnexistingIcon(self):
        self.assertRaises(ValueError, icons.getQIcon, "not-exists")

    def testExistingQPixmap(self):
        icon = icons.getQPixmap("crop")
        self.assertIsNotNone(icon)

    def testUnexistingQPixmap(self):
        self.assertRaises(ValueError, icons.getQPixmap, "not-exists")

    def testCache(self):
        icon1 = icons.getQIcon("crop")
        icon2 = icons.getQIcon("crop")
        self.assertIs(icon1, icon2)

    def testCacheReleased(self):
        icon = icons.getQIcon("crop")
        icon_ref = weakref.ref(icon)
        del icon
        self.assertIsNone(icon_ref())


class TestAnimatedIcons(TestCaseQt):
    """Test to check that icons module."""

    def testProcessWorking(self):
        icon = icons.getWaitIcon()
        self.assertIsNotNone(icon)

    def testProcessWorkingCache(self):
        icon1 = icons.getWaitIcon()
        icon2 = icons.getWaitIcon()
        self.assertIs(icon1, icon2)

    def testMovieIconExists(self):
        if "mng" not in qt.supportedImageFormats():
            self.skipTest("MNG not supported")
        icon = icons.MovieAnimatedIcon("process-working")
        self.assertIsNotNone(icon)

    def testMovieIconNotExists(self):
        self.assertRaises(ValueError, icons.MovieAnimatedIcon, "not-exists")

    def testMultiImageIconExists(self):
        icon = icons.MultiImageAnimatedIcon("process-working")
        self.assertIsNotNone(icon)

    def testPrefixedResourceExists(self):
        icon = icons.MultiImageAnimatedIcon("silx:gui/icons/process-working")
        self.assertIsNotNone(icon)

    def testMultiImageIconNotExists(self):
        self.assertRaises(ValueError, icons.MultiImageAnimatedIcon, "not-exists")
