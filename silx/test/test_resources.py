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
__date__ = "24/08/2017"


import os
import unittest

from silx.third_party import six
import silx.resources
import shutil
from .utils import utilstest
import socket


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


def isSilxWebsiteAvailable():
    try:
        six.moves.urllib.request.urlopen('http://www.silx.org', timeout=1)
        return True
    except six.moves.urllib.error.URLError:
        return False
    except socket.timeout:
        # This exception is still received in Python 2.7
        return False


class TestExternalResources(unittest.TestCase):
    """This is a test for the ExternalResources"""

    @classmethod
    def setUpClass(cls):
        if not isSilxWebsiteAvailable():
            raise unittest.SkipTest("Network or silx website not available")

    def setUp(self):
        self.utilstest = silx.resources.ExternalResources("toto", "http://www.silx.org/pub/silx/")

    def tearDown(self):
        if self.utilstest.data_home:
            shutil.rmtree(self.utilstest.data_home)
        self.utilstest = None

    def test_tempdir(self):
        "test the temporary directory creation"
        d = self.utilstest.tempdir
        self.assertTrue(os.path.isdir(d))
        self.assertEqual(d, self.utilstest.tempdir, 'tmpdir is stable')
        self.utilstest.clean_up()
        self.assertFalse(os.path.isdir(d))
        e = self.utilstest.tempdir
        self.assertTrue(os.path.isdir(e))
        self.assertEqual(e, self.utilstest.tempdir, 'tmpdir is stable')
        self.assertNotEqual(d, e, "tempdir changed")
        self.utilstest.clean_up()

    def test_download(self):
        "test the download from silx.org"
        f = self.utilstest.getfile("lena.png")
        self.assertTrue(os.path.exists(f))
        di = utilstest.getdir("source.tar.gz")
        for fi in di:
            self.assertTrue(os.path.exists(fi))

    def test_download_all(self):
        "test the download of all files from silx.org"
        filename = self.utilstest.getfile("lena.png")
        directory = "source.tar.gz"
        _filelist = self.utilstest.getdir(directory)
        # download file and remove it to create a json mapping file
        os.remove(filename)
        directory_path = os.path.join(self.utilstest.data_home, "source")
        shutil.rmtree(directory_path)
        directory_path = os.path.join(self.utilstest.data_home, directory)
        os.remove(directory_path)
        filelist = self.utilstest.download_all()
        self.assertGreater(len(filelist), 1, "At least 2 items were downloaded")


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loadTests(TestResources))
    test_suite.addTest(loadTests(TestExternalResources))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
