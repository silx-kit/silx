# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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

import io
import numpy
import os
import re
import shutil
import tempfile
import unittest

from .. import utils
import silx.io.url

try:
    import h5py
    from ..utils import h5ls
except ImportError:
    h5py = None

try:
    import fabio
except ImportError:
    fabio = None


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "12/02/2018"


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

expected_spec2 = expected_spec1 + r"""
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
        self.assertTrue(numpy.array_equal(npy_recarray['Ordinate1'],
                                          numpy.array((4, 5, 6))))

    def test_savespec_filename(self):
        """Save SpecFile using savespec()"""
        utils.savespec(self.spec_fname, self.x, self.y[0], xlabel=self.xlab,
                       ylabel=self.ylabs[0], fmt=["%d", "%.2f"],
                       close_file=True, scan_number=1)

        specf = open(self.spec_fname)
        actual_spec = specf.read()
        specf.close()

        self.assertRegexpMatches(actual_spec, expected_spec1)

    def test_savespec_file_handle(self):
        """Save SpecFile using savespec(), passing a file handle"""
        # first savespec: open, write file header, save y[0] as scan 1,
        #                 return file handle
        specf = utils.savespec(self.spec_fname, self.x, self.y[0],
                               xlabel=self.xlab, ylabel=self.ylabs[0],
                               fmt=["%d", "%.2f"], close_file=False)

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
        self.tempdir = tempfile.mkdtemp()
        self.spec_fname = os.path.join(self.tempdir, "savespec.dat")
        self.csv_fname = os.path.join(self.tempdir, "savecsv.csv")
        self.npy_fname = os.path.join(self.tempdir, "savenpy.npy")

        self.x = [1, 2, 3]
        self.xlab = "Abscissa"
        self.y = [[4, 5, 6], [7, 8, 9]]
        self.ylabs = ["Ordinate1", "Ordinate2"]
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


@unittest.skipIf(h5py is None, "Could not import h5py")
class TestH5Ls(unittest.TestCase):
    """Test displaying the following HDF5 file structure:

        +foo
            +bar
                <HDF5 dataset "spam": shape (2, 2), type "<i8">
                <HDF5 dataset "tmp": shape (3,), type "<i8">
            <HDF5 dataset "data": shape (1,), type "<f8">

    """
    def assertMatchAnyStringInList(self, pattern, list_of_strings):
        for string_ in list_of_strings:
            if re.match(pattern, string_):
                return None
        raise AssertionError("regex pattern %s does not match any" % pattern +
                             " string in list " + str(list_of_strings))

    def testHdf5(self):
        fd, self.h5_fname = tempfile.mkstemp(text=False)
        # Close and delete (we just want the name)
        os.close(fd)
        os.unlink(self.h5_fname)
        self.h5f = h5py.File(self.h5_fname, "w")
        self.h5f["/foo/bar/tmp"] = [1, 2, 3]
        self.h5f["/foo/bar/spam"] = [[1, 2], [3, 4]]
        self.h5f["/foo/data"] = [3.14]
        self.h5f.close()

        rep = h5ls(self.h5_fname)
        lines = rep.split("\n")

        self.assertIn("+foo", lines)
        self.assertIn("\t+bar", lines)

        match = r'\t\t<HDF5 dataset "tmp": shape \(3,\), type "<i[48]">'
        self.assertMatchAnyStringInList(match, lines)
        match = r'\t\t<HDF5 dataset "spam": shape \(2, 2\), type "<i[48]">'
        self.assertMatchAnyStringInList(match, lines)
        match = r'\t<HDF5 dataset "data": shape \(1,\), type "<f[48]">'
        self.assertMatchAnyStringInList(match, lines)

        os.unlink(self.h5_fname)

    # Following test case disabled d/t errors on AppVeyor:
    #     os.unlink(spec_fname)
    # PermissionError: [WinError 32] The process cannot access the file because
    # it is being used by another process: 'C:\\...\\savespec.dat'

    # def testSpec(self):
    #     tempdir = tempfile.mkdtemp()
    #     spec_fname = os.path.join(tempdir, "savespec.dat")
    #
    #     x = [1, 2, 3]
    #     xlab = "Abscissa"
    #     y = [[4, 5, 6], [7, 8, 9]]
    #     ylabs = ["Ordinate1", "Ordinate2"]
    #     utils.save1D(spec_fname, x, y, xlabel=xlab,
    #                  ylabels=ylabs, filetype="spec",
    #                  fmt=["%d", "%.2f"])
    #
    #     rep = h5ls(spec_fname)
    #     lines = rep.split("\n")
    #     self.assertIn("+1.1", lines)
    #     self.assertIn("\t+instrument", lines)
    #
    #     self.assertMatchAnyStringInList(
    #             r'\t\t\t<SPEC dataset "file_header": shape \(\), type "|S60">',
    #             lines)
    #     self.assertMatchAnyStringInList(
    #             r'\t\t<SPEC dataset "Ordinate1": shape \(3L?,\), type "<f4">',
    #             lines)
    #
    #     os.unlink(spec_fname)
    #     shutil.rmtree(tempdir)


class TestOpen(unittest.TestCase):
    """Test `silx.io.utils.open` function."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_directory = tempfile.mkdtemp()
        cls.createResources(cls.tmp_directory)

    @classmethod
    def createResources(cls, directory):

        if h5py is not None:
            cls.h5_filename = os.path.join(directory, "test.h5")
            h5 = h5py.File(cls.h5_filename, mode="w")
            h5["group/group/dataset"] = 50
            h5.close()

        cls.spec_filename = os.path.join(directory, "test.dat")
        utils.savespec(cls.spec_filename, [1], [1.1], xlabel="x", ylabel="y",
                       fmt=["%d", "%.2f"], close_file=True, scan_number=1)

        if fabio is not None:
            cls.edf_filename = os.path.join(directory, "test.edf")
            header = fabio.fabioimage.OrderedDict()
            header["integer"] = "10"
            data = numpy.array([[10, 50], [50, 10]])
            fabiofile = fabio.edfimage.EdfImage(data, header)
            fabiofile.write(cls.edf_filename)

        cls.txt_filename = os.path.join(directory, "test.txt")
        f = io.open(cls.txt_filename, "w+t")
        f.write(u"Kikoo")
        f.close()

        cls.missing_filename = os.path.join(directory, "test.missing")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_directory)

    def testH5(self):
        if h5py is None:
            self.skipTest("H5py is missing")

        f = utils.open(self.h5_filename)
        self.assertIsNotNone(f)
        self.assertIsInstance(f, h5py.File)
        f.close()

    def testH5With(self):
        if h5py is None:
            self.skipTest("H5py is missing")

        with utils.open(self.h5_filename) as f:
            self.assertIsNotNone(f)
            self.assertIsInstance(f, h5py.File)

    def testH5_withPath(self):
        if h5py is None:
            self.skipTest("H5py is missing")

        f = utils.open(self.h5_filename + "::/group/group/dataset")
        self.assertIsNotNone(f)
        self.assertEqual(f.h5py_class, h5py.Dataset)
        self.assertEqual(f[()], 50)
        f.close()

    def testH5With_withPath(self):
        if h5py is None:
            self.skipTest("H5py is missing")

        with utils.open(self.h5_filename + "::/group/group") as f:
            self.assertIsNotNone(f)
            self.assertEqual(f.h5py_class, h5py.Group)
            self.assertIn("dataset", f)

    def testSpec(self):
        f = utils.open(self.spec_filename)
        self.assertIsNotNone(f)
        if h5py is not None:
            self.assertEqual(f.h5py_class, h5py.File)
        f.close()

    def testSpecWith(self):
        with utils.open(self.spec_filename) as f:
            self.assertIsNotNone(f)
            if h5py is not None:
                self.assertEqual(f.h5py_class, h5py.File)

    def testEdf(self):
        if h5py is None:
            self.skipTest("H5py is missing")
        if fabio is None:
            self.skipTest("Fabio is missing")

        f = utils.open(self.edf_filename)
        self.assertIsNotNone(f)
        self.assertEqual(f.h5py_class, h5py.File)
        f.close()

    def testEdfWith(self):
        if h5py is None:
            self.skipTest("H5py is missing")
        if fabio is None:
            self.skipTest("Fabio is missing")

        with utils.open(self.edf_filename) as f:
            self.assertIsNotNone(f)
            self.assertEqual(f.h5py_class, h5py.File)

    def testUnsupported(self):
        self.assertRaises(IOError, utils.open, self.txt_filename)

    def testNotExists(self):
        # load it
        self.assertRaises(IOError, utils.open, self.missing_filename)

    def test_silx_scheme(self):
        if h5py is None:
            self.skipTest("H5py is missing")
        url = silx.io.url.DataUrl(scheme="silx", file_path=self.h5_filename, data_path="/")
        with utils.open(url.path()) as f:
            self.assertIsNotNone(f)
            self.assertTrue(silx.io.utils.is_file(f))

    def test_fabio_scheme(self):
        if h5py is None:
            self.skipTest("H5py is missing")
        if fabio is None:
            self.skipTest("Fabio is missing")
        url = silx.io.url.DataUrl(scheme="fabio", file_path=self.edf_filename)
        self.assertRaises(IOError, utils.open, url.path())

    def test_bad_url(self):
        url = silx.io.url.DataUrl(scheme="sil", file_path=self.h5_filename)
        self.assertRaises(IOError, utils.open, url.path())

    def test_sliced_url(self):
        url = silx.io.url.DataUrl(file_path=self.h5_filename, data_slice=(5,))
        self.assertRaises(IOError, utils.open, url.path())


class TestNodes(unittest.TestCase):
    """Test `silx.io.utils.is_` functions."""
    def test_real_h5py_objects(self):
        if h5py is None:
            self.skipTest("H5py is missing")

        name = tempfile.mktemp(suffix=".h5")
        try:
            with h5py.File(name, "w") as h5file:
                h5group = h5file.create_group("arrays")
                h5dataset = h5group.create_dataset("scalar", data=10)

                self.assertTrue(utils.is_file(h5file))
                self.assertTrue(utils.is_group(h5file))
                self.assertFalse(utils.is_dataset(h5file))

                self.assertFalse(utils.is_file(h5group))
                self.assertTrue(utils.is_group(h5group))
                self.assertFalse(utils.is_dataset(h5group))

                self.assertFalse(utils.is_file(h5dataset))
                self.assertFalse(utils.is_group(h5dataset))
                self.assertTrue(utils.is_dataset(h5dataset))
        finally:
            os.unlink(name)

    def test_h5py_like_file(self):
        if h5py is None:
            self.skipTest("H5py is missing")

        class Foo(object):
            def __init__(self):
                self.h5_class = utils.H5Type.FILE
        obj = Foo()
        self.assertTrue(utils.is_file(obj))
        self.assertTrue(utils.is_group(obj))
        self.assertFalse(utils.is_dataset(obj))

    def test_h5py_like_group(self):
        if h5py is None:
            self.skipTest("H5py is missing")

        class Foo(object):
            def __init__(self):
                self.h5_class = utils.H5Type.GROUP
        obj = Foo()
        self.assertFalse(utils.is_file(obj))
        self.assertTrue(utils.is_group(obj))
        self.assertFalse(utils.is_dataset(obj))

    def test_h5py_like_dataset(self):
        if h5py is None:
            self.skipTest("H5py is missing")

        class Foo(object):
            def __init__(self):
                self.h5_class = utils.H5Type.DATASET
        obj = Foo()
        self.assertFalse(utils.is_file(obj))
        self.assertFalse(utils.is_group(obj))
        self.assertTrue(utils.is_dataset(obj))

    def test_bad(self):
        if h5py is None:
            self.skipTest("H5py is missing")

        class Foo(object):
            def __init__(self):
                pass
        obj = Foo()
        self.assertFalse(utils.is_file(obj))
        self.assertFalse(utils.is_group(obj))
        self.assertFalse(utils.is_dataset(obj))

    def test_bad_api(self):
        if h5py is None:
            self.skipTest("H5py is missing")

        class Foo(object):
            def __init__(self):
                self.h5_class = int
        obj = Foo()
        self.assertFalse(utils.is_file(obj))
        self.assertFalse(utils.is_group(obj))
        self.assertFalse(utils.is_dataset(obj))


class TestGetData(unittest.TestCase):
    """Test `silx.io.utils.get_data` function."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_directory = tempfile.mkdtemp()
        cls.createResources(cls.tmp_directory)

    @classmethod
    def createResources(cls, directory):

        if h5py is not None:
            cls.h5_filename = os.path.join(directory, "test.h5")
            h5 = h5py.File(cls.h5_filename, mode="w")
            h5["group/group/scalar"] = 50
            h5["group/group/array"] = [1, 2, 3, 4, 5]
            h5["group/group/array2d"] = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
            h5.close()

        cls.spec_filename = os.path.join(directory, "test.dat")
        utils.savespec(cls.spec_filename, [1], [1.1], xlabel="x", ylabel="y",
                       fmt=["%d", "%.2f"], close_file=True, scan_number=1)

        if fabio is not None:
            cls.edf_filename = os.path.join(directory, "test.edf")
            cls.edf_multiframe_filename = os.path.join(directory, "test_multi.edf")
            header = fabio.fabioimage.OrderedDict()
            header["integer"] = "10"
            data = numpy.array([[10, 50], [50, 10]])
            fabiofile = fabio.edfimage.EdfImage(data, header)
            fabiofile.write(cls.edf_filename)
            fabiofile.appendFrame(data=data, header=header)
            fabiofile.write(cls.edf_multiframe_filename)

        cls.txt_filename = os.path.join(directory, "test.txt")
        f = io.open(cls.txt_filename, "w+t")
        f.write(u"Kikoo")
        f.close()

        cls.missing_filename = os.path.join(directory, "test.missing")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_directory)

    def test_hdf5_scalar(self):
        if h5py is None:
            self.skipTest("H5py is missing")
        url = "silx:%s?/group/group/scalar" % self.h5_filename
        data = utils.get_data(url=url)
        self.assertEqual(data, 50)

    def test_hdf5_array(self):
        if h5py is None:
            self.skipTest("H5py is missing")
        url = "silx:%s?/group/group/array" % self.h5_filename
        data = utils.get_data(url=url)
        self.assertEqual(data.shape, (5, ))
        self.assertEqual(data[0], 1)

    def test_hdf5_array_slice(self):
        if h5py is None:
            self.skipTest("H5py is missing")
        url = "silx:%s?path=/group/group/array2d&slice=1" % self.h5_filename
        data = utils.get_data(url=url)
        self.assertEqual(data.shape, (5, ))
        self.assertEqual(data[0], 6)

    def test_hdf5_array_slice_out_of_range(self):
        if h5py is None:
            self.skipTest("H5py is missing")
        url = "silx:%s?path=/group/group/array2d&slice=5" % self.h5_filename
        self.assertRaises(ValueError, utils.get_data, url)

    def test_edf_using_silx(self):
        if h5py is None:
            self.skipTest("H5py is missing")
        if fabio is None:
            self.skipTest("fabio is missing")
        url = "silx:%s?/scan_0/instrument/detector_0/data" % self.edf_filename
        data = utils.get_data(url=url)
        self.assertEqual(data.shape, (2, 2))
        self.assertEqual(data[0, 0], 10)

    def test_fabio_frame(self):
        if fabio is None:
            self.skipTest("fabio is missing")
        url = "fabio:%s?slice=1" % self.edf_multiframe_filename
        data = utils.get_data(url=url)
        self.assertEqual(data.shape, (2, 2))
        self.assertEqual(data[0, 0], 10)

    def test_fabio_singleframe(self):
        if fabio is None:
            self.skipTest("fabio is missing")
        url = "fabio:%s?slice=0" % self.edf_filename
        data = utils.get_data(url=url)
        self.assertEqual(data.shape, (2, 2))
        self.assertEqual(data[0, 0], 10)

    def test_fabio_too_much_frames(self):
        if fabio is None:
            self.skipTest("fabio is missing")
        url = "fabio:%s?slice=..." % self.edf_multiframe_filename
        self.assertRaises(ValueError, utils.get_data, url)

    def test_fabio_no_frame(self):
        if fabio is None:
            self.skipTest("fabio is missing")
        url = "fabio:%s" % self.edf_filename
        data = utils.get_data(url=url)
        self.assertEqual(data.shape, (2, 2))
        self.assertEqual(data[0, 0], 10)

    def test_unsupported_scheme(self):
        url = "foo:/foo/bar"
        self.assertRaises(ValueError, utils.get_data, url)

    def test_no_scheme(self):
        if fabio is None:
            self.skipTest("fabio is missing")
        url = "%s?path=/group/group/array2d&slice=5" % self.h5_filename
        self.assertRaises((ValueError, IOError), utils.get_data, url)

    def test_file_not_exists(self):
        url = "silx:/foo/bar"
        self.assertRaises(IOError, utils.get_data, url)


def suite():
    loadTests = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loadTests(TestSave))
    test_suite.addTest(loadTests(TestH5Ls))
    test_suite.addTest(loadTests(TestOpen))
    test_suite.addTest(loadTests(TestNodes))
    test_suite.addTest(loadTests(TestGetData))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
