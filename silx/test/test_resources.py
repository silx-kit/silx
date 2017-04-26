# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Test for resource files management."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "20/04/2017"


import os
import unittest

import silx.resources
from .utils import utilstest


class TestResources(unittest.TestCase):
    def test_resource_dir(self):
        """Get a resource directory"""
        icons_dirname = silx.resources.resource_filename('gui/icons/')
        self.assertTrue(os.path.isdir(icons_dirname))

    def test_resource_file(self):
        """Get a resource file name"""
        filename = silx.resources.resource_filename('gui/icons/colormap.png')
        self.assertTrue(os.path.isfile(filename))

    def test_resource_nonexistent(self):
        """Get a non existent resource"""
        filename = silx.resources.resource_filename('non_existent_file.txt')
        self.assertFalse(os.path.exists(filename))


class TestExternalResources(unittest.TestCase):
    "This is a test for the TestResources"
    def test_tempdir(self):
        "test the temporary directory creation"
        myutilstest = silx.resources.ExternalResources("toto", "http://www.silx.org")
        d = myutilstest.tempdir
        self.assertTrue(os.path.isdir(d))
        self.assertEqual(d, myutilstest.tempdir, 'tmpdir is stable')
        myutilstest.clean_up()
        self.assertFalse(os.path.isdir(d))
        e = myutilstest.tempdir
        self.assertTrue(os.path.isdir(e))
        self.assertEqual(e, myutilstest.tempdir, 'tmpdir is stable')
        self.assertNotEqual(d, e, "tempdir changed")
        myutilstest.clean_up()

    def test_download(self):
        "test the download from silx.org"
        f = utilstest.getfile("lena.png")
        self.assertTrue(os.path.exists(f))
        f = utilstest.getdir("source.tar.gz")
        self.assertTrue(os.path.isfile(f))
        self.assertTrue(os.path.isdir(f[:-7]))

    def test_dowload_all(self):
        "test the download of all files from silx.org"
        l = utilstest.download_all()
        self.assertGreater(len(l), 1, "At least 2 items were downloaded")


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestResources))
    test_suite.addTest(TestExternalResources("test_tempdir"))
    test_suite.addTest(TestExternalResources("test_download")) # order matters !
    test_suite.addTest(TestExternalResources("test_dowload_all"))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
