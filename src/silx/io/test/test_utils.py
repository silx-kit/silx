# /*##########################################################################
# Copyright (C) 2016-2022 European Synchrotron Radiation Facility
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
from ..._version import calc_hexversion
import silx.io.url

import h5py
from ..utils import h5ls
from silx.io import commonh5


import fabio

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "03/12/2020"

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

expected_spec2reg = r"""#F .*
#D .*

#S 1 Ordinate1
#D .*
#N 3
#L Abscissa  Ordinate1  Ordinate2
1  4\.00  7\.00
2  5\.00  8\.00
3  6\.00  9\.00
"""

expected_spec2irr = expected_spec1 + r"""
#S 2 Ordinate2
#D .*
#N 2
#L Abscissa  Ordinate2
1  7\.00
2  8\.00
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
        self.y_irr = [[4, 5, 6], [7, 8]]
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

        self.assertRegex(actual_csv, expected_csv)

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
        self.assertRegex(actual_spec, expected_spec1)

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
        self.assertRegex(actual_spec, expected_spec2)

    def test_save_spec_reg(self):
        """Save SpecFile using save() on a regular pattern"""
        utils.save1D(self.spec_fname, self.x, self.y, xlabel=self.xlab,
                     ylabels=self.ylabs, filetype="spec", fmt=["%d", "%.2f"])

        specf = open(self.spec_fname)
        actual_spec = specf.read()
        specf.close()

        self.assertRegex(actual_spec, expected_spec2reg)

    def test_save_spec_irr(self):
        """Save SpecFile using save() on an irregular pattern"""
        # invalid  test case ?!
        return
        utils.save1D(self.spec_fname, self.x, self.y_irr, xlabel=self.xlab,
                     ylabels=self.ylabs, filetype="spec", fmt=["%d", "%.2f"])

        specf = open(self.spec_fname)
        actual_spec = specf.read()
        specf.close()
        self.assertRegex(actual_spec, expected_spec2irr)

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
        self.assertRegex(actual_csv, expected_csv2)


def assert_match_any_string_in_list(test, pattern, list_of_strings):
    for string_ in list_of_strings:
        if re.match(pattern, string_):
            return True
    return False


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

        cls.h5_filename = os.path.join(directory, "test.h5")
        h5 = h5py.File(cls.h5_filename, mode="w")
        h5["group/group/dataset"] = 50
        h5.close()

        cls.spec_filename = os.path.join(directory, "test.dat")
        utils.savespec(cls.spec_filename, [1], [1.1], xlabel="x", ylabel="y",
                       fmt=["%d", "%.2f"], close_file=True, scan_number=1)

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
        f = utils.open(self.h5_filename)
        self.assertIsNotNone(f)
        self.assertIsInstance(f, h5py.File)
        f.close()

    def testH5With(self):
        with utils.open(self.h5_filename) as f:
            self.assertIsNotNone(f)
            self.assertIsInstance(f, h5py.File)

    def testH5_withPath(self):
        f = utils.open(self.h5_filename + "::/group/group/dataset")
        self.assertIsNotNone(f)
        self.assertEqual(f.h5py_class, h5py.Dataset)
        self.assertEqual(f[()], 50)
        f.close()

    def testH5With_withPath(self):
        with utils.open(self.h5_filename + "::/group/group") as f:
            self.assertIsNotNone(f)
            self.assertEqual(f.h5py_class, h5py.Group)
            self.assertIn("dataset", f)

    def testSpec(self):
        f = utils.open(self.spec_filename)
        self.assertIsNotNone(f)
        self.assertEqual(f.h5py_class, h5py.File)
        f.close()

    def testSpecWith(self):
        with utils.open(self.spec_filename) as f:
            self.assertIsNotNone(f)
            self.assertEqual(f.h5py_class, h5py.File)

    def testEdf(self):
        f = utils.open(self.edf_filename)
        self.assertIsNotNone(f)
        self.assertEqual(f.h5py_class, h5py.File)
        f.close()

    def testEdfWith(self):
        with utils.open(self.edf_filename) as f:
            self.assertIsNotNone(f)
            self.assertEqual(f.h5py_class, h5py.File)

    def testUnsupported(self):
        self.assertRaises(IOError, utils.open, self.txt_filename)

    def testNotExists(self):
        # load it
        self.assertRaises(IOError, utils.open, self.missing_filename)

    def test_silx_scheme(self):
        url = silx.io.url.DataUrl(scheme="silx", file_path=self.h5_filename, data_path="/")
        with utils.open(url.path()) as f:
            self.assertIsNotNone(f)
            self.assertTrue(silx.io.utils.is_file(f))

    def test_fabio_scheme(self):
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

        class Foo(object):

            def __init__(self):
                self.h5_class = utils.H5Type.FILE

        obj = Foo()
        self.assertTrue(utils.is_file(obj))
        self.assertTrue(utils.is_group(obj))
        self.assertFalse(utils.is_dataset(obj))

    def test_h5py_like_group(self):

        class Foo(object):

            def __init__(self):
                self.h5_class = utils.H5Type.GROUP

        obj = Foo()
        self.assertFalse(utils.is_file(obj))
        self.assertTrue(utils.is_group(obj))
        self.assertFalse(utils.is_dataset(obj))

    def test_h5py_like_dataset(self):

        class Foo(object):

            def __init__(self):
                self.h5_class = utils.H5Type.DATASET

        obj = Foo()
        self.assertFalse(utils.is_file(obj))
        self.assertFalse(utils.is_group(obj))
        self.assertTrue(utils.is_dataset(obj))

    def test_bad(self):

        class Foo(object):

            def __init__(self):
                pass

        obj = Foo()
        self.assertFalse(utils.is_file(obj))
        self.assertFalse(utils.is_group(obj))
        self.assertFalse(utils.is_dataset(obj))

    def test_bad_api(self):

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

        cls.h5_filename = os.path.join(directory, "test.h5")
        h5 = h5py.File(cls.h5_filename, mode="w")
        h5["group/group/scalar"] = 50
        h5["group/group/array"] = [1, 2, 3, 4, 5]
        h5["group/group/array2d"] = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        h5.close()

        cls.spec_filename = os.path.join(directory, "test.dat")
        utils.savespec(cls.spec_filename, [1], [1.1], xlabel="x", ylabel="y",
                       fmt=["%d", "%.2f"], close_file=True, scan_number=1)

        cls.edf_filename = os.path.join(directory, "test.edf")
        cls.edf_multiframe_filename = os.path.join(directory, "test_multi.edf")
        header = fabio.fabioimage.OrderedDict()
        header["integer"] = "10"
        data = numpy.array([[10, 50], [50, 10]])
        fabiofile = fabio.edfimage.EdfImage(data, header)
        fabiofile.write(cls.edf_filename)
        fabiofile.append_frame(data=data, header=header)
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
        url = "silx:%s?/group/group/scalar" % self.h5_filename
        data = utils.get_data(url=url)
        self.assertEqual(data, 50)

    def test_hdf5_array(self):
        url = "silx:%s?/group/group/array" % self.h5_filename
        data = utils.get_data(url=url)
        self.assertEqual(data.shape, (5,))
        self.assertEqual(data[0], 1)

    def test_hdf5_array_slice(self):
        url = "silx:%s?path=/group/group/array2d&slice=1" % self.h5_filename
        data = utils.get_data(url=url)
        self.assertEqual(data.shape, (5,))
        self.assertEqual(data[0], 6)

    def test_hdf5_array_slice_out_of_range(self):
        url = "silx:%s?path=/group/group/array2d&slice=5" % self.h5_filename
        # ValueError: h5py 2.x
        # IndexError: h5py 3.x
        self.assertRaises((ValueError, IndexError), utils.get_data, url)

    def test_edf_using_silx(self):
        url = "silx:%s?/scan_0/instrument/detector_0/data" % self.edf_filename
        data = utils.get_data(url=url)
        self.assertEqual(data.shape, (2, 2))
        self.assertEqual(data[0, 0], 10)

    def test_fabio_frame(self):
        url = "fabio:%s?slice=1" % self.edf_multiframe_filename
        data = utils.get_data(url=url)
        self.assertEqual(data.shape, (2, 2))
        self.assertEqual(data[0, 0], 10)

    def test_fabio_singleframe(self):
        url = "fabio:%s?slice=0" % self.edf_filename
        data = utils.get_data(url=url)
        self.assertEqual(data.shape, (2, 2))
        self.assertEqual(data[0, 0], 10)

    def test_fabio_too_much_frames(self):
        url = "fabio:%s?slice=..." % self.edf_multiframe_filename
        self.assertRaises(ValueError, utils.get_data, url)

    def test_fabio_no_frame(self):
        url = "fabio:%s" % self.edf_filename
        data = utils.get_data(url=url)
        self.assertEqual(data.shape, (2, 2))
        self.assertEqual(data[0, 0], 10)

    def test_unsupported_scheme(self):
        url = "foo:/foo/bar"
        self.assertRaises(ValueError, utils.get_data, url)

    def test_no_scheme(self):
        url = "%s?path=/group/group/array2d&slice=5" % self.h5_filename
        self.assertRaises((ValueError, IOError), utils.get_data, url)

    def test_file_not_exists(self):
        url = "silx:/foo/bar"
        self.assertRaises(IOError, utils.get_data, url)


def _h5_py_version_older_than(version):
    v_majeur, v_mineur, v_micro = [int(i) for i in h5py.version.version.split('.')[:3]]
    r_majeur, r_mineur, r_micro = [int(i) for i in version.split('.')]
    return calc_hexversion(v_majeur, v_mineur, v_micro) >= calc_hexversion(r_majeur, r_mineur, r_micro) 


@unittest.skipUnless(_h5_py_version_older_than('2.9.0'), 'h5py version < 2.9.0')
class TestRawFileToH5(unittest.TestCase):
    """Test conversion of .vol file to .h5 external dataset"""

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self._vol_file = os.path.join(self.tempdir, 'test_vol.vol')
        self._file_info = os.path.join(self.tempdir, 'test_vol.info.vol')
        self._dataset_shape = 100, 20, 5
        data = numpy.random.random(self._dataset_shape[0] *
                                   self._dataset_shape[1] *
                                   self._dataset_shape[2]).astype(dtype=numpy.float32).reshape(self._dataset_shape)
        numpy.save(file=self._vol_file, arr=data)
        # those are storing into .noz file
        assert os.path.exists(self._vol_file + '.npy')
        os.rename(self._vol_file + '.npy', self._vol_file)
        self.h5_file = os.path.join(self.tempdir, 'test_h5.h5')
        self.external_dataset_path = '/root/my_external_dataset'
        self._data_url = silx.io.url.DataUrl(file_path=self.h5_file,
                                             data_path=self.external_dataset_path)
        with open(self._file_info, 'w') as _fi:
            _fi.write('NUM_X = %s\n' % self._dataset_shape[2])
            _fi.write('NUM_Y = %s\n' % self._dataset_shape[1])
            _fi.write('NUM_Z = %s\n' % self._dataset_shape[0])

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def check_dataset(self, h5_file, data_path, shape):
        """Make sure the external dataset is valid"""
        with h5py.File(h5_file, 'r') as _file:
            return data_path in _file and _file[data_path].shape == shape

    def test_h5_file_not_existing(self):
        """Test that can create a file with external dataset from scratch"""
        utils.rawfile_to_h5_external_dataset(bin_file=self._vol_file,
                                             output_url=self._data_url,
                                             shape=(100, 20, 5),
                                             dtype=numpy.float32)
        self.assertTrue(self.check_dataset(h5_file=self.h5_file,
                                           data_path=self.external_dataset_path,
                                           shape=self._dataset_shape))
        os.remove(self.h5_file)
        utils.vol_to_h5_external_dataset(vol_file=self._vol_file,
                                         output_url=self._data_url,
                                         info_file=self._file_info)
        self.assertTrue(self.check_dataset(h5_file=self.h5_file,
                                           data_path=self.external_dataset_path,
                                           shape=self._dataset_shape))

    def test_h5_file_existing(self):
        """Test that can add the external dataset from an existing file"""
        with h5py.File(self.h5_file, 'w') as _file:
            _file['/root/dataset1'] = numpy.zeros((100, 100))
            _file['/root/group/dataset2'] = numpy.ones((100, 100))
        utils.rawfile_to_h5_external_dataset(bin_file=self._vol_file,
                                             output_url=self._data_url,
                                             shape=(100, 20, 5),
                                             dtype=numpy.float32)
        self.assertTrue(self.check_dataset(h5_file=self.h5_file,
                                           data_path=self.external_dataset_path,
                                           shape=self._dataset_shape))

    def test_vol_file_not_existing(self):
        """Make sure error is raised if .vol file does not exists"""
        os.remove(self._vol_file)
        utils.rawfile_to_h5_external_dataset(bin_file=self._vol_file,
                                             output_url=self._data_url,
                                             shape=(100, 20, 5),
                                             dtype=numpy.float32)

        self.assertTrue(self.check_dataset(h5_file=self.h5_file,
                                           data_path=self.external_dataset_path,
                                           shape=self._dataset_shape))

    def test_conflicts(self):
        """Test several conflict cases"""
        # test if path already exists
        utils.rawfile_to_h5_external_dataset(bin_file=self._vol_file,
                                             output_url=self._data_url,
                                             shape=(100, 20, 5),
                                             dtype=numpy.float32)
        with self.assertRaises(ValueError):
            utils.rawfile_to_h5_external_dataset(bin_file=self._vol_file,
                                                 output_url=self._data_url,
                                                 shape=(100, 20, 5),
                                                 overwrite=False,
                                                 dtype=numpy.float32)

        utils.rawfile_to_h5_external_dataset(bin_file=self._vol_file,
                                             output_url=self._data_url,
                                             shape=(100, 20, 5),
                                             overwrite=True,
                                             dtype=numpy.float32)

        self.assertTrue(self.check_dataset(h5_file=self.h5_file,
                                           data_path=self.external_dataset_path,
                                           shape=self._dataset_shape))


class TestH5Strings(unittest.TestCase):
    """Test HDF5 str and bytes writing and reading"""

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.mkdtemp()
        cls.vlenstr = h5py.special_dtype(vlen=str)
        cls.vlenbytes = h5py.special_dtype(vlen=bytes)
        try:
            cls.unicode = unicode
        except NameError:
            cls.unicode = str

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir)

    def setUp(self):
        self.file = h5py.File(os.path.join(self.tempdir, 'file.h5'), mode="w")

    def tearDown(self):
        self.file.close()

    @classmethod
    def _make_array(cls, value, n):
        if isinstance(value, bytes):
            dtype = cls.vlenbytes
        elif isinstance(value, cls.unicode):
            dtype = cls.vlenstr
        else:
            return numpy.array([value] * n)
        return numpy.array([value] * n, dtype=dtype)

    @classmethod
    def _get_charset(cls, value):
        if isinstance(value, bytes):
            return h5py.h5t.CSET_ASCII
        elif isinstance(value, cls.unicode):
            return h5py.h5t.CSET_UTF8
        else:
            return None

    def _check_dataset(self, value, result=None):
        # Write+read scalar
        if result:
            decode_ascii = True
        else:
            decode_ascii = False
            result = value
        charset = self._get_charset(value)
        self.file["data"] = value
        data = utils.h5py_read_dataset(self.file["data"], decode_ascii=decode_ascii)
        assert type(data) == type(result), data
        assert data == result, data
        if charset:
            assert self.file["data"].id.get_type().get_cset() == charset

        # Write+read variable length
        self.file["vlen_data"] = self._make_array(value, 2)
        data = utils.h5py_read_dataset(self.file["vlen_data"], decode_ascii=decode_ascii, index=0)
        assert type(data) == type(result), data
        assert data == result, data
        data = utils.h5py_read_dataset(self.file["vlen_data"], decode_ascii=decode_ascii)
        numpy.testing.assert_array_equal(data, [result] * 2)
        if charset:
            assert self.file["vlen_data"].id.get_type().get_cset() == charset

    def _check_attribute(self, value, result=None):
        if result:
            decode_ascii = True
        else:
            decode_ascii = False
            result = value
        self.file.attrs["data"] = value
        data = utils.h5py_read_attribute(self.file.attrs, "data", decode_ascii=decode_ascii)
        assert type(data) == type(result), data
        assert data == result, data

        self.file.attrs["vlen_data"] = self._make_array(value, 2)
        data = utils.h5py_read_attribute(self.file.attrs, "vlen_data", decode_ascii=decode_ascii)
        assert type(data[0]) == type(result), data[0]
        assert data[0] == result, data[0]
        numpy.testing.assert_array_equal(data, [result] * 2)

        data = utils.h5py_read_attributes(self.file.attrs, decode_ascii=decode_ascii)["vlen_data"]
        assert type(data[0]) == type(result), data[0]
        assert data[0] == result, data[0]
        numpy.testing.assert_array_equal(data, [result] * 2)

    def test_dataset_ascii_bytes(self):
        self._check_dataset(b"abc")

    def test_attribute_ascii_bytes(self):
        self._check_attribute(b"abc")

    def test_dataset_ascii_bytes_decode(self):
        self._check_dataset(b"abc", result="abc")

    def test_attribute_ascii_bytes_decode(self):
        self._check_attribute(b"abc", result="abc")

    def test_dataset_ascii_str(self):
        self._check_dataset("abc")

    def test_attribute_ascii_str(self):
        self._check_attribute("abc")

    def test_dataset_utf8_str(self):
        self._check_dataset("\u0101bc")

    def test_attribute_utf8_str(self):
        self._check_attribute("\u0101bc")

    def test_dataset_utf8_bytes(self):
        # 0xC481 is the byte representation of U+0101
        self._check_dataset(b"\xc4\x81bc")

    def test_attribute_utf8_bytes(self):
        # 0xC481 is the byte representation of U+0101
        self._check_attribute(b"\xc4\x81bc")

    def test_dataset_utf8_bytes_decode(self):
        # 0xC481 is the byte representation of U+0101
        self._check_dataset(b"\xc4\x81bc", result="\u0101bc")

    def test_attribute_utf8_bytes_decode(self):
        # 0xC481 is the byte representation of U+0101
        self._check_attribute(b"\xc4\x81bc", result="\u0101bc")

    def test_dataset_latin1_bytes(self):
        # extended ascii character 0xE4
        self._check_dataset(b"\xe423")

    def test_attribute_latin1_bytes(self):
        # extended ascii character 0xE4
        self._check_attribute(b"\xe423")

    def test_dataset_latin1_bytes_decode(self):
        # U+DCE4: surrogate for extended ascii character 0xE4
        self._check_dataset(b"\xe423", result="\udce423")

    def test_attribute_latin1_bytes_decode(self):
        # U+DCE4: surrogate for extended ascii character 0xE4
        self._check_attribute(b"\xe423", result="\udce423")

    def test_dataset_no_string(self):
        self._check_dataset(numpy.int64(10))

    def test_attribute_no_string(self):
        self._check_attribute(numpy.int64(10))


def test_visitall_hdf5(tmp_path):
    """visit HDF5 file content not following links"""
    external_filepath = tmp_path / "external.h5"
    with h5py.File(external_filepath, mode="w") as h5file:
        h5file["target/dataset"] = 50

    filepath = tmp_path / "base.h5"
    with h5py.File(filepath, mode="w") as h5file:
        h5file["group/dataset"] = 50
        h5file["link/soft_link"] = h5py.SoftLink("/group/dataset")
        h5file["link/external_link"] = h5py.ExternalLink("external.h5", "/target/dataset")

    with h5py.File(filepath, mode="r") as h5file:
        visited_items = {}
        for path, item in utils.visitall(h5file):
            if isinstance(item, h5py.Dataset):
                content = item[()]
            elif isinstance(item, h5py.Group):
                content = None
            elif isinstance(item, h5py.SoftLink):
                content = item.path
            elif isinstance(item, h5py.ExternalLink):
                content = item.filename, item.path
            else:
                raise AssertionError("Item should not be present: %s" % path)
            visited_items[path] = (item.__class__, content)

    assert visited_items == {
        "/group": (h5py.Group, None),
        "/group/dataset": (h5py.Dataset, 50),
        "/link": (h5py.Group, None),
        "/link/soft_link": (h5py.SoftLink, "/group/dataset"),
        "/link/external_link": (h5py.ExternalLink, ("external.h5", "/target/dataset")),
    }

def test_visitall_commonh5():
    """Visit commonh5 File object"""
    fobj = commonh5.File("filename.file", mode="w")
    group = fobj.create_group("group")
    dataset = group.create_dataset("dataset", data=numpy.array(50))
    group["soft_link"] = dataset # Create softlink

    visited_items = dict(utils.visitall(fobj))
    assert len(visited_items) == 3
    assert visited_items["/group"] is group
    assert visited_items["/group/dataset"] is dataset
    soft_link = visited_items["/group/soft_link"]
    assert isinstance(soft_link, commonh5.SoftLink)
    assert soft_link.path == "/group/dataset"


def test_match_hdf5(tmp_path):
    """Test match function with HDF5 file"""
    with h5py.File(tmp_path / "test_match.h5", "w") as h5f:
        h5f.create_group("entry_0000/group")
        h5f["entry_0000/data"] = 0
        h5f.create_group("entry_0001/group")
        h5f["entry_0001/data"] = 1
        h5f.create_group("entry_0002")
        h5f["entry_0003"] = 3

        result = list(utils.match(h5f, "/entry_*/*"))

        assert sorted(result) == ['entry_0000/data', 'entry_0000/group', 'entry_0001/data', 'entry_0001/group']


def test_match_commonh5():
    """Test match function with commonh5 objects"""
    with commonh5.File("filename.file", mode="w") as fobj:
        fobj.create_group("entry_0000/group")
        fobj["entry_0000/data"] = 0
        fobj.create_group("entry_0001/group")
        fobj["entry_0001/data"] = 1
        fobj.create_group("entry_0002")
        fobj["entry_0003"] = 3

        result = list(utils.match(fobj, "/entry_*/*"))

        assert sorted(result) == ['entry_0000/data', 'entry_0000/group', 'entry_0001/data', 'entry_0001/group']
