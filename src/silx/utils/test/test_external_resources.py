# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
__date__ = "08/03/2019"


import os
import unittest
import shutil
import socket
import six

from silx.utils.ExternalResources import ExternalResources


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
        self.resources = ExternalResources("toto", "http://www.silx.org/pub/silx/")

    def tearDown(self):
        if self.resources.data_home:
            shutil.rmtree(self.resources.data_home)
        self.resources = None

    def test_download(self):
        "test the download from silx.org"
        f = self.resources.getfile("lena.png")
        self.assertTrue(os.path.exists(f))
        di = self.resources.getdir("source.tar.gz")
        for fi in di:
            self.assertTrue(os.path.exists(fi))

    def test_download_all(self):
        "test the download of all files from silx.org"
        filename = self.resources.getfile("lena.png")
        directory = "source.tar.gz"
        filelist = self.resources.getdir(directory)
        # download file and remove it to create a json mapping file
        os.remove(filename)
        directory_path = os.path.commonprefix(filelist)
        # Make sure we will rmtree a dangerous path like "/"
        self.assertIn(self.resources.data_home, directory_path)
        shutil.rmtree(directory_path)
        filelist = self.resources.download_all()
        self.assertGreater(len(filelist), 1, "At least 2 items were downloaded")


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loadTests(TestExternalResources))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
