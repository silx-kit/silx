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
import numpy
import os
import re
import shutil
import tempfile
import unittest

from silx.io.utils import repr_hdf5_tree, savespec, save

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "24/03/2016"


class TestReprHDF5Tree(unittest.TestCase):
    """Test displaying the following HDF5 file structure:

        +foo
            +bar
                -spam=<HDF5 dataset "spam": shape (2, 2), type "<i8">
                -tmp=<HDF5 dataset "tmp": shape (3,), type "<i8">
            -data=<HDF5 dataset "data": shape (1,), type "<f8">

    """
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.h5_fname = os.path.join(self.tempdir, "temp.h5")

        self.h5f = h5py.File(self.h5_fname, "w")
        self.h5f["/foo/bar/tmp"] = [1, 2, 3]
        self.h5f["/foo/bar/spam"] = [[1, 2], [3, 4]]
        self.h5f["/foo/data"] = [3.14]
        self.h5f.close()

    def tearDown(self):
        os.unlink(self.h5_fname)
        shutil.rmtree(self.tempdir)

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

expected_spec = r"""#F .*
#D .*

#S 1 Ordinate1
#D .*
#N 2
#L Abscissa  Ordinate1
1  4\.00
2  5\.00
3  6\.00

#S 2 Ordinate2
#D .*
#N 2
#L Abscissa  Ordinate2
1  7\.00
2  8\.00
3  9\.00
"""

expected_csv = r"""Abscissa;Ordinate1;Ordinate2
1;4\.00;7\.00e\+00
2;5\.00;8\.00e\+00
3;6\.00;9\.00e\+00
"""

class TestSave(unittest.TestCase):
    """Test saving curves as SpecFile:
    """
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.spec_fname = os.path.join(self.tempdir, "savespec.dat")
        self.csv_fname = os.path.join(self.tempdir, "savecsv.dat")
        self.npy_fname = os.path.join(self.tempdir, "savenpy.npy")

        self.x = [1, 2, 3]
        self.xlab = "Abscissa"
        self.y = [[4, 5, 6], [7, 8, 9]]
        self.ylabs = ["Ordinate1", "Ordinate2"]

    def tearDown(self):
        if os.path.isfile(self.spec_fname):
            os.unlink(self.spec_fname)
        if os.path.isfile(self.csv_fname):
            os.unlink(self.csv_fname)
        if os.path.isfile(self.npy_fname):
            os.unlink(self.npy_fname)
        shutil.rmtree(self.tempdir)

    def test_save_csv(self):
        save(self.csv_fname, self.x, self.y,
             xlabel=self.xlab, ylabels=self.ylabs,
             filetype="csv", fmt=["%d", "%.2f", "%.2e"],
             csvdelimiter=";")

        csvf = open(self.csv_fname)
        actual_csv = csvf.read()
        csvf.close()

        self.assertRegexpMatches(actual_csv, expected_csv)

    def test_save_npy(self):
        """npy file is saved with numpy.save after building a numpy array
        and converting it to a named record array"""
        npyf = open(self.npy_fname, "wb")
        save(npyf, self.x, self.y,
             xlabel=self.xlab, ylabels=self.ylabs)
        npyf.close()

        npy_recarray = numpy.load(self.npy_fname)

        self.assertEqual(npy_recarray.shape, (3,))
        self.assertTrue(
                numpy.array_equal(
                        npy_recarray['Ordinate1'],
                        numpy.array((4, 5, 6))))

    def test_savespec_filename(self):
        savespec(self.spec_fname, self.x, self.y,
                 xlabel=self.xlab, ylabels=self.ylabs,
                 fmt=["%d", "%.2f"])

        specf = open(self.spec_fname)
        actual_spec = specf.read()
        specf.close()

        self.assertRegexpMatches(actual_spec, expected_spec)

    def test_save_spec_file_handle(self):
        specf = open(self.spec_fname, "w")
        save(specf, self.x, self.y,
             xlabel=self.xlab, ylabels=self.ylabs,
             filetype="spec", fmt=["%d", "%.2f"])
        specf.close()

        specf = open(self.spec_fname)
        actual_spec = specf.read()
        specf.close()

        self.assertRegexpMatches(actual_spec, expected_spec)


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestReprHDF5Tree))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestSave))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")