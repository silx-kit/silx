# coding: utf-8
#/*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
#############################################################################*/
"""Tests for utils module"""
import h5py
import os
import re
import tempfile
import unittest

from silx.io.utils import repr_hdf5_tree, savespec

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "21/03/2016"


class TestReprHDF5Tree(unittest.TestCase):
    """Test displaying the following HDF5 file structure:

        +foo
            +bar
                -spam=<HDF5 dataset "spam": shape (2, 2), type "<i8">
                -tmp=<HDF5 dataset "tmp": shape (3,), type "<i8">
            -data=<HDF5 dataset "data": shape (1,), type "<f8">

    """
    def setUp(self):
        fd, self.h5_fname = tempfile.mkstemp(text=False)
        # Close and delete (we just want the name)
        os.close(fd)
        os.unlink(self.h5_fname)
        self.h5f = h5py.File(self.h5_fname, "w")
        self.h5f["/foo/bar/tmp"] = [1, 2, 3]
        self.h5f["/foo/bar/spam"] = [[1, 2], [3, 4]]
        self.h5f["/foo/data"] = [3.14]
        self.h5f.close()

    def tearDown(self):
        os.unlink(self.h5_fname)

    def assertMatchAnyStringInList(self, pattern, list_of_strings):
        for string_ in list_of_strings:
            if re.match(pattern, string_):
                return None
        raise AssertionError("regex pattern %s does not match any" % pattern +
                             " string in list " + str(list_of_strings))

    def test_repr(self):
        rep = repr_hdf5_tree(self.h5_fname)
        lines = rep.split("\n")

        self.assertIn("+foo", lines)
        self.assertIn("\t+bar", lines)

        self.assertMatchAnyStringInList(
                r'\t\t-tmp=<HDF5 dataset "tmp": shape \(3,\), type "<i[48]">',
                lines)
        self.assertMatchAnyStringInList(
                r'\t\t-spam=<HDF5 dataset "spam": shape \(2, 2\), type "<i[48]">',
                lines)
        self.assertMatchAnyStringInList(
                r'\t-data=<HDF5 dataset "data": shape \(1,\), type "<f[48]">',
                lines)


class TestSaveSpec(unittest.TestCase):
    """Test saving curves as SpecFile:
    """
    def setUp(self):
        fd, self.spec_fname = tempfile.mkstemp(text=False)
        # Close and delete (we just want the name)
        os.close(fd)
        os.unlink(self.spec_fname)

        x = [1, 2, 3]
        xlab = "Abscissa"
        y = [[4, 5, 6], [7, 8, 9]]
        ylabs = "Ordinate1  Ordinate2"

        savespec(self.spec_fname, x, y, xlabel=xlab, ylabels=ylabs,
                 datafmt=["%d", "%.2f"])

    def tearDown(self):
        os.unlink(self.spec_fname)

    def test_spec_format(self):
        expected_spec = r"#F .*\n"
        expected_spec += r"#D .*\n\n"

        expected_spec += "#S 1 Ordinate1\n"
        expected_spec += r"#D .*\n"
        expected_spec += r"#N 2\n"
        expected_spec += r"#L Abscissa  Ordinate1\n"
        expected_spec += r"1  4\.00\n"
        expected_spec += r"2  5\.00\n"
        expected_spec += r"3  6\.00\n"
        expected_spec += r"\n"
        expected_spec += r"#S 2 Ordinate2\n"
        expected_spec += r"#D .*\n"
        expected_spec += r"#N 2\n"
        expected_spec += r"#L Abscissa  Ordinate2\n"
        expected_spec += r"1  7\.00\n"
        expected_spec += r"2  8\.00\n"
        expected_spec += r"3  9\.00\n"
        expected_spec += r"\n"

        specf = open(self.spec_fname)
        actual_spec = specf.read()
        specf.close()

        self.assertRegexpMatches(actual_spec, expected_spec)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestReprHDF5Tree))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestSaveSpec))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")