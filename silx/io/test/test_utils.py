# coding: utf-8
# /*##########################################################################
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
# ############################################################################*/
"""Tests for utils module"""


import numpy
import os
import re
import shutil
import tempfile
import unittest

from .. import utils

try:
    import h5py
except ImportError:
    h5py_missing = True
else:
    h5py_missing = False
    from ..utils import h5ls


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "27/09/2016"


expected_spec1 = r"""#F .*
#D .*

#S 1 Ordinate1
#D .*
#N 2
#L Abscissa  Ordinate1
1  4\.00
2  5\.00
3  6\.00
"""

expected_spec2 = expected_spec1 + """
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

expected_csv2 = r"""x;y0;y1
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
        self.csv_fname = os.path.join(self.tempdir, "savecsv.csv")
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
        utils.save1D(self.csv_fname, self.x, self.y,
               xlabel=self.xlab, ylabels=self.ylabs,
               filetype="csv", fmt=["%d", "%.2f", "%.2e"],
               csvdelim=";", autoheader=True)

        csvf = open(self.csv_fname)
        actual_csv = csvf.read()
        csvf.close()

        self.assertRegexpMatches(actual_csv, expected_csv)

    def test_save_npy(self):
        """npy file is saved with numpy.save after building a numpy array
        and converting it to a named record array"""
        npyf = open(self.npy_fname, "wb")
        utils.save1D(npyf, self.x, self.y,
               xlabel=self.xlab, ylabels=self.ylabs)
        npyf.close()

        npy_recarray = numpy.load(self.npy_fname)

        self.assertEqual(npy_recarray.shape, (3,))
        self.assertTrue(
                numpy.array_equal(
                        npy_recarray['Ordinate1'],
                        numpy.array((4, 5, 6))))

    def test_savespec_filename(self):
        """Save SpecFile using savespec()"""
        utils.savespec(self.spec_fname, self.x, self.y[0], xlabel=self.xlab,
                 ylabel=self.ylabs[0], fmt=["%d", "%.2f"], close_file=True,
                 scan_number=1)

        specf = open(self.spec_fname)
        actual_spec = specf.read()
        specf.close()

        self.assertRegexpMatches(actual_spec, expected_spec1)

    def test_savespec_file_handle(self):
        """Save SpecFile using savespec(), passing a file handle"""
        # first savespec: open, write file header, save y[0] as scan 1,
        #                 return file handle
        specf = utils.savespec(self.spec_fname, self.x, self.y[0], xlabel=self.xlab,
                         ylabel=self.ylabs[0], fmt=["%d", "%.2f"],
                         close_file=False)

        # second savespec: save y[1] as scan 2, close file
        utils.savespec(specf, self.x, self.y[1], xlabel=self.xlab,
                 ylabel=self.ylabs[1], fmt=["%d", "%.2f"],
                 write_file_header=False, close_file=True,
                 scan_number=2)

        specf = open(self.spec_fname)
        actual_spec = specf.read()
        specf.close()

        self.assertRegexpMatches(actual_spec, expected_spec2)

    def test_save_spec(self):
        """Save SpecFile using save()"""
        utils.save1D(self.spec_fname, self.x, self.y, xlabel=self.xlab,
               ylabels=self.ylabs, filetype="spec", fmt=["%d", "%.2f"])

        specf = open(self.spec_fname)
        actual_spec = specf.read()
        specf.close()
        self.assertRegexpMatches(actual_spec, expected_spec2)

    def test_save_csv_no_labels(self):
        """Save csv using save(), with autoheader=True but
        xlabel=None and ylabels=None
        This is a non-regression test for bug #223"""
        utils.save1D(self.csv_fname, self.x, self.y,
               autoheader=True, fmt=["%d", "%.2f", "%.2e"])

        csvf = open(self.csv_fname)
        actual_csv = csvf.read()
        csvf.close()
        self.assertRegexpMatches(actual_csv, expected_csv2)


def assert_match_any_string_in_list(test, pattern, list_of_strings):
    for string_ in list_of_strings:
        if re.match(pattern, string_):
            return True
    return False


@unittest.skipIf(h5py_missing, "Could not import h5py")
class TestH5Ls(unittest.TestCase):
    """Test displaying the following HDF5 file structure:

        +foo
            +bar
                <HDF5 dataset "spam": shape (2, 2), type "<i8">
                <HDF5 dataset "tmp": shape (3,), type "<i8">
            <HDF5 dataset "data": shape (1,), type "<f8">

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

    def testRepr(self):
        rep = h5ls(self.h5_fname)
        lines = rep.split("\n")

        self.assertIn("+foo", lines)
        self.assertIn("\t+bar", lines)

        self.assertMatchAnyStringInList(
                r'\t\t<HDF5 dataset "tmp": shape \(3,\), type "<i[48]">',
                lines)
        self.assertMatchAnyStringInList(
                r'\t\t<HDF5 dataset "spam": shape \(2, 2\), type "<i[48]">',
                lines)
        self.assertMatchAnyStringInList(
                r'\t<HDF5 dataset "data": shape \(1,\), type "<f[48]">',
                lines)


class TestLoad(unittest.TestCase):
    """Test `silx.io.utils.load` function."""

    def testH5(self):
        if h5py_missing:
            self.skipTest("H5py is missing")

        # create a file
        tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=True)
        tmp.file.close()
        h5 = h5py.File(tmp.name, "w")
        g = h5.create_group("arrays")
        g.create_dataset("scalar", data=10)
        h5.close()

        # load it
        f = utils.load(tmp.name)
        self.assertIsNotNone(f)
        self.assertIsInstance(f, h5py.File)

    def testSpec(self):
        # create a file
        tmp = tempfile.NamedTemporaryFile(mode="w+t", suffix=".dat", delete=True)
        tmp.file.close()
        utils.savespec(tmp.name, [1], [1.1], xlabel="x", ylabel="y",
                       fmt=["%d", "%.2f"], close_file=True, scan_number=1)

        # load it
        f = utils.load(tmp.name)
        self.assertIsNotNone(f)
        self.assertEquals(f.h5py_class, h5py.File)

    def testUnsupported(self):
        # create a file
        tmp = tempfile.NamedTemporaryFile(mode="w+t", suffix=".txt", delete=True)
        tmp.write("Kikoo")
        tmp.close()

        # load it
        self.assertRaises(IOError, utils.load, tmp.name)

    def testNotExists(self):
        # load it
        self.assertRaises(IOError, utils.load, "#$.")


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestSave))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestH5Ls))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestLoad))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
