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
__date__ = "13/05/2016"


import os
import unittest

import silx.resources


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


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestResources))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
