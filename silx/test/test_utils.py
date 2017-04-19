# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2017 European Synchrotron Radiation Facility
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
"""Basic test of data downloading."""

__authors__ = ["J. Kieffer"]
__license__ = "MIT"
__date__ = "19/04/2017"

import os
import unittest
from .utils import TestResources, utilstest


class TestUtils(unittest.TestCase):

    def test_tempdir(self):
        "test the temporary directory creation"
        utilstest = TestResources()
        d = utilstest.tempdir
        self.assertTrue(os.path.isdir(d))
        self.assertEqual(d, utilstest.tempdir, 'tmpdir is stable')
        utilstest.clean_up()
        self.assertFalse(os.path.isdir(d))
        e = utilstest.tempdir
        self.assertTrue(os.path.isdir(e))
        self.assertEqual(e, utilstest.tempdir, 'tmpdir is stable')
        self.assertNotEqual(d, e, "tempdir changed")
        utilstest.clean_up()

    def test_download(self):
        "test the download from silx.org"
        f = utilstest.getfile("lena.png")
        self.assertTrue(os.path.exists(f))
        f = utilstest.getdir("source.tar.gz")
        self.assertTrue(os.path.isfile(f))
        self.assertTrue(os.path.isdir(f[:-7]))


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestUtils))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
