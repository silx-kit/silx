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
"""Tests for fabioh5 wrapper"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "02/07/2018"

import os
import logging
import numpy
import unittest
import tempfile
import shutil

_logger = logging.getLogger(__name__)

import fabio
import h5py

from .. import commonh5
from .. import fabioh5


class TestFabioH5(unittest.TestCase):

    def setUp(self):

        header = {
            "integer": "-100",
            "float": "1.0",
            "string": "hi!",
            "list_integer": "100 50 0",
            "list_float": "1.0 2.0 3.5",
            "string_looks_like_list": "2000 hi!",
        }
        data = numpy.array([[10, 11], [12, 13], [14, 15]], dtype=numpy.int64)
        self.fabio_image = fabio.numpyimage.NumpyImage(data, header)
        self.h5_image = fabioh5.File(fabio_image=self.fabio_image)

    def test_main_groups(self):
        self.assertEqual(self.h5_image.h5py_class, h5py.File)
        self.assertEqual(self.h5_image["/"].h5py_class, h5py.File)
        self.assertEqual(self.h5_image["/scan_0"].h5py_class, h5py.Group)
        self.assertEqual(self.h5_image["/scan_0/instrument"].h5py_class, h5py.Group)
        self.assertEqual(self.h5_image["/scan_0/measurement"].h5py_class, h5py.Group)

    def test_wrong_path_syntax(self):
        # result tested with a default h5py file
        self.assertRaises(ValueError, lambda: self.h5_image[""])

    def test_wrong_root_name(self):
        # result tested with a default h5py file
        self.assertRaises(KeyError, lambda: self.h5_image["/foo"])

    def test_wrong_root_path(self):
        # result tested with a default h5py file
        self.assertRaises(KeyError, lambda: self.h5_image["/foo/foo"])

    def test_wrong_name(self):
        # result tested with a default h5py file
        self.assertRaises(KeyError, lambda: self.h5_image["foo"])

    def test_wrong_path(self):
        # result tested with a default h5py file
        self.assertRaises(KeyError, lambda: self.h5_image["foo/foo"])

    def test_single_frame(self):
        data = numpy.arange(2 * 3)
        data.shape = 2, 3
        fabio_image = fabio.edfimage.edfimage(data=data)
        h5_image = fabioh5.File(fabio_image=fabio_image)

        dataset = h5_image["/scan_0/instrument/detector_0/data"]
        self.assertEqual(dataset.h5py_class, h5py.Dataset)
        self.assertTrue(isinstance(dataset[()], numpy.ndarray))
        self.assertEqual(dataset.dtype.kind, "i")
        self.assertEqual(dataset.shape, (2, 3))
        self.assertEqual(dataset[...][0, 0], 0)
        self.assertEqual(dataset.attrs["interpretation"], "image")

    def test_multi_frames(self):
        data = numpy.arange(2 * 3)
        data.shape = 2, 3
        fabio_image = fabio.edfimage.edfimage(data=data)
        fabio_image.append_frame(data=data)
        h5_image = fabioh5.File(fabio_image=fabio_image)

        dataset = h5_image["/scan_0/instrument/detector_0/data"]
        self.assertEqual(dataset.h5py_class, h5py.Dataset)
        self.assertTrue(isinstance(dataset[()], numpy.ndarray))
        self.assertEqual(dataset.dtype.kind, "i")
        self.assertEqual(dataset.shape, (2, 2, 3))
        self.assertEqual(dataset[...][0, 0, 0], 0)
        self.assertEqual(dataset.attrs["interpretation"], "image")

    def test_heterogeneous_frames(self):
        """Frames containing 2 images with different sizes and a cube"""
        data1 = numpy.arange(2 * 3)
        data1.shape = 2, 3
        data2 = numpy.arange(2 * 5)
        data2.shape = 2, 5
        data3 = numpy.arange(2 * 5 * 1)
        data3.shape = 2, 5, 1
        fabio_image = fabio.edfimage.edfimage(data=data1)
        fabio_image.append_frame(data=data2)
        fabio_image.append_frame(data=data3)
        h5_image = fabioh5.File(fabio_image=fabio_image)

        dataset = h5_image["/scan_0/instrument/detector_0/data"]
        self.assertEqual(dataset.h5py_class, h5py.Dataset)
        self.assertTrue(isinstance(dataset[()], numpy.ndarray))
        self.assertEqual(dataset.dtype.kind, "i")
        self.assertEqual(dataset.shape, (3, 2, 5, 1))
        self.assertEqual(dataset[...][0, 0, 0], 0)
        self.assertEqual(dataset.attrs["interpretation"], "image")

    def test_single_3d_frame(self):
        """Image source contains a cube"""
        data = numpy.arange(2 * 3 * 4)
        data.shape = 2, 3, 4
        # Do not provide the data to the constructor to avoid slicing of the
        # data. In this way the result stay a cube, and not a multi-frame
        fabio_image = fabio.edfimage.edfimage()
        fabio_image.data = data
        h5_image = fabioh5.File(fabio_image=fabio_image)

        dataset = h5_image["/scan_0/instrument/detector_0/data"]
        self.assertEqual(dataset.h5py_class, h5py.Dataset)
        self.assertTrue(isinstance(dataset[()], numpy.ndarray))
        self.assertEqual(dataset.dtype.kind, "i")
        self.assertEqual(dataset.shape, (2, 3, 4))
        self.assertEqual(dataset[...][0, 0, 0], 0)
        self.assertEqual(dataset.attrs["interpretation"], "image")

    def test_metadata_int(self):
        dataset = self.h5_image["/scan_0/instrument/detector_0/others/integer"]
        self.assertEqual(dataset.h5py_class, h5py.Dataset)
        self.assertEqual(dataset[()], -100)
        self.assertEqual(dataset.dtype.kind, "i")
        self.assertEqual(dataset.shape, (1,))

    def test_metadata_float(self):
        dataset = self.h5_image["/scan_0/instrument/detector_0/others/float"]
        self.assertEqual(dataset.h5py_class, h5py.Dataset)
        self.assertEqual(dataset[()], 1.0)
        self.assertEqual(dataset.dtype.kind, "f")
        self.assertEqual(dataset.shape, (1,))

    def test_metadata_string(self):
        dataset = self.h5_image["/scan_0/instrument/detector_0/others/string"]
        self.assertEqual(dataset.h5py_class, h5py.Dataset)
        self.assertEqual(dataset[()], numpy.string_("hi!"))
        self.assertEqual(dataset.dtype.type, numpy.string_)
        self.assertEqual(dataset.shape, (1,))

    def test_metadata_list_integer(self):
        dataset = self.h5_image["/scan_0/instrument/detector_0/others/list_integer"]
        self.assertEqual(dataset.h5py_class, h5py.Dataset)
        self.assertEqual(dataset.dtype.kind, "u")
        self.assertEqual(dataset.shape, (1, 3))
        self.assertEqual(dataset[0, 0], 100)
        self.assertEqual(dataset[0, 1], 50)

    def test_metadata_list_float(self):
        dataset = self.h5_image["/scan_0/instrument/detector_0/others/list_float"]
        self.assertEqual(dataset.h5py_class, h5py.Dataset)
        self.assertEqual(dataset.dtype.kind, "f")
        self.assertEqual(dataset.shape, (1, 3))
        self.assertEqual(dataset[0, 0], 1.0)
        self.assertEqual(dataset[0, 1], 2.0)

    def test_metadata_list_looks_like_list(self):
        dataset = self.h5_image["/scan_0/instrument/detector_0/others/string_looks_like_list"]
        self.assertEqual(dataset.h5py_class, h5py.Dataset)
        self.assertEqual(dataset[()], numpy.string_("2000 hi!"))
        self.assertEqual(dataset.dtype.type, numpy.string_)
        self.assertEqual(dataset.shape, (1,))

    def test_float_32(self):
        float_list = [u'1.2', u'1.3', u'1.4']
        data = numpy.array([[0, 0], [0, 0]], dtype=numpy.int8)
        fabio_image = None
        for float_item in float_list:
            header = {"float_item": float_item}
            if fabio_image is None:
                fabio_image = fabio.edfimage.EdfImage(data=data, header=header)
            else:
                fabio_image.append_frame(data=data, header=header)
        h5_image = fabioh5.File(fabio_image=fabio_image)
        data = h5_image["/scan_0/instrument/detector_0/others/float_item"]
        # There is no equality between items
        self.assertEqual(len(data), len(set(data)))
        # At worst a float32
        self.assertIn(data.dtype.kind, ['d', 'f'])
        self.assertLessEqual(data.dtype.itemsize, 32 / 8)

    def test_float_64(self):
        float_list = [
            u'1469117129.082226',
            u'1469117136.684986', u'1469117144.312749', u'1469117151.892507',
            u'1469117159.474265', u'1469117167.100027', u'1469117174.815799',
            u'1469117182.437561', u'1469117190.094326', u'1469117197.721089']
        data = numpy.array([[0, 0], [0, 0]], dtype=numpy.int8)
        fabio_image = None
        for float_item in float_list:
            header = {"time_of_day": float_item}
            if fabio_image is None:
                fabio_image = fabio.edfimage.EdfImage(data=data, header=header)
            else:
                fabio_image.append_frame(data=data, header=header)
        h5_image = fabioh5.File(fabio_image=fabio_image)
        data = h5_image["/scan_0/instrument/detector_0/others/time_of_day"]
        # There is no equality between items
        self.assertEqual(len(data), len(set(data)))
        # At least a float64
        self.assertIn(data.dtype.kind, ['d', 'f'])
        self.assertGreaterEqual(data.dtype.itemsize, 64 / 8)

    def test_mixed_float_size__scalar(self):
        # We expect to have a precision of 32 bits
        float_list = [u'1.2', u'1.3001']
        expected_float_result = [1.2, 1.3001]
        data = numpy.array([[0, 0], [0, 0]], dtype=numpy.int8)
        fabio_image = None
        for float_item in float_list:
            header = {"float_item": float_item}
            if fabio_image is None:
                fabio_image = fabio.edfimage.EdfImage(data=data, header=header)
            else:
                fabio_image.append_frame(data=data, header=header)
        h5_image = fabioh5.File(fabio_image=fabio_image)
        data = h5_image["/scan_0/instrument/detector_0/others/float_item"]
        # At worst a float32
        self.assertIn(data.dtype.kind, ['d', 'f'])
        self.assertLessEqual(data.dtype.itemsize, 32 / 8)
        for computed, expected in zip(data, expected_float_result):
            numpy.testing.assert_almost_equal(computed, expected, 5)

    def test_mixed_float_size__list(self):
        # We expect to have a precision of 32 bits
        float_list = [u'1.2 1.3001']
        expected_float_result = numpy.array([[1.2, 1.3001]])
        data = numpy.array([[0, 0], [0, 0]], dtype=numpy.int8)
        fabio_image = None
        for float_item in float_list:
            header = {"float_item": float_item}
            if fabio_image is None:
                fabio_image = fabio.edfimage.EdfImage(data=data, header=header)
            else:
                fabio_image.append_frame(data=data, header=header)
        h5_image = fabioh5.File(fabio_image=fabio_image)
        data = h5_image["/scan_0/instrument/detector_0/others/float_item"]
        # At worst a float32
        self.assertIn(data.dtype.kind, ['d', 'f'])
        self.assertLessEqual(data.dtype.itemsize, 32 / 8)
        for computed, expected in zip(data, expected_float_result):
            numpy.testing.assert_almost_equal(computed, expected, 5)

    def test_mixed_float_size__list_of_list(self):
        # We expect to have a precision of 32 bits
        float_list = [u'1.2 1.3001', u'1.3001 1.3001']
        expected_float_result = numpy.array([[1.2, 1.3001], [1.3001, 1.3001]])
        data = numpy.array([[0, 0], [0, 0]], dtype=numpy.int8)
        fabio_image = None
        for float_item in float_list:
            header = {"float_item": float_item}
            if fabio_image is None:
                fabio_image = fabio.edfimage.EdfImage(data=data, header=header)
            else:
                fabio_image.append_frame(data=data, header=header)
        h5_image = fabioh5.File(fabio_image=fabio_image)
        data = h5_image["/scan_0/instrument/detector_0/others/float_item"]
        # At worst a float32
        self.assertIn(data.dtype.kind, ['d', 'f'])
        self.assertLessEqual(data.dtype.itemsize, 32 / 8)
        for computed, expected in zip(data, expected_float_result):
            numpy.testing.assert_almost_equal(computed, expected, 5)

    def test_ub_matrix(self):
        """Data from mediapix.edf"""
        header = {}
        header["UB_mne"] = 'UB0 UB1 UB2 UB3 UB4 UB5 UB6 UB7 UB8'
        header["UB_pos"] = '1.99593e-16 2.73682e-16 -1.54 -1.08894 1.08894 1.6083e-16 1.08894 1.08894 9.28619e-17'
        header["sample_mne"] = 'U0 U1 U2 U3 U4 U5'
        header["sample_pos"] = '4.08 4.08 4.08 90 90 90'
        data = numpy.array([[0, 0], [0, 0]], dtype=numpy.int8)
        fabio_image = fabio.edfimage.EdfImage(data=data, header=header)
        h5_image = fabioh5.File(fabio_image=fabio_image)
        sample = h5_image["/scan_0/sample"]
        self.assertIsNotNone(sample)
        self.assertEqual(sample.attrs["NXclass"], "NXsample")

        d = sample['unit_cell_abc']
        expected = numpy.array([4.08, 4.08, 4.08])
        self.assertIsNotNone(d)
        self.assertEqual(d.shape, (3, ))
        self.assertIn(d.dtype.kind, ['d', 'f'])
        numpy.testing.assert_array_almost_equal(d[...], expected)

        d = sample['unit_cell_alphabetagamma']
        expected = numpy.array([90.0, 90.0, 90.0])
        self.assertIsNotNone(d)
        self.assertEqual(d.shape, (3, ))
        self.assertIn(d.dtype.kind, ['d', 'f'])
        numpy.testing.assert_array_almost_equal(d[...], expected)

        d = sample['ub_matrix']
        expected = numpy.array([[[1.99593e-16, 2.73682e-16, -1.54],
                                 [-1.08894, 1.08894, 1.6083e-16],
                                 [1.08894, 1.08894, 9.28619e-17]]])
        self.assertIsNotNone(d)
        self.assertEqual(d.shape, (1, 3, 3))
        self.assertIn(d.dtype.kind, ['d', 'f'])
        numpy.testing.assert_array_almost_equal(d[...], expected)

    def test_interpretation_mca_edf(self):
        """EDF files with two or more headers starting with "MCA"
        must have @interpretation = "spectrum" an the data."""
        header = {
            "Title": "zapimage  samy -4.975 -5.095 80 500 samz -4.091 -4.171 70 0",
            "MCA a": -23.812,
            "MCA b": 2.7107,
            "MCA c": 8.1164e-06}

        data = numpy.array([[0, 0], [0, 0]], dtype=numpy.int8)
        fabio_image = fabio.edfimage.EdfImage(data=data, header=header)
        h5_image = fabioh5.File(fabio_image=fabio_image)

        data_dataset = h5_image["/scan_0/measurement/image_0/data"]
        self.assertEqual(data_dataset.attrs["interpretation"], "spectrum")

        data_dataset = h5_image["/scan_0/instrument/detector_0/data"]
        self.assertEqual(data_dataset.attrs["interpretation"], "spectrum")

        data_dataset = h5_image["/scan_0/measurement/image_0/info/data"]
        self.assertEqual(data_dataset.attrs["interpretation"], "spectrum")

    def test_get_api(self):
        result = self.h5_image.get("scan_0", getclass=True, getlink=True)
        self.assertIs(result, h5py.HardLink)
        result = self.h5_image.get("scan_0", getclass=False, getlink=True)
        self.assertIsInstance(result, h5py.HardLink)
        result = self.h5_image.get("scan_0", getclass=True, getlink=False)
        self.assertIs(result, h5py.Group)
        result = self.h5_image.get("scan_0", getclass=False, getlink=False)
        self.assertIsInstance(result, commonh5.Group)

    def test_detector_link(self):
        detector1 = self.h5_image["/scan_0/instrument/detector_0"]
        detector2 = self.h5_image["/scan_0/measurement/image_0/info"]
        self.assertIsNot(detector1, detector2)
        self.assertEqual(list(detector1.items()), list(detector2.items()))
        self.assertEqual(self.h5_image.get(detector2.name, getlink=True).path, detector1.name)

    def test_detector_data_link(self):
        data1 = self.h5_image["/scan_0/instrument/detector_0/data"]
        data2 = self.h5_image["/scan_0/measurement/image_0/data"]
        self.assertIsNot(data1, data2)
        self.assertIs(data1._get_data(), data2._get_data())
        self.assertEqual(self.h5_image.get(data2.name, getlink=True).path, data1.name)

    def test_dirty_header(self):
        """Test that it does not fail"""
        try:
            header = {}
            header["foo"] = b'abc'
            data = numpy.array([[0, 0], [0, 0]], dtype=numpy.int8)
            fabio_image = fabio.edfimage.edfimage(data=data, header=header)
            header = {}
            header["foo"] = b'a\x90bc\xFE'
            fabio_image.append_frame(data=data, header=header)
        except Exception as e:
            _logger.error(e.args[0])
            _logger.debug("Backtrace", exc_info=True)
            self.skipTest("fabio do not allow to create the resource")

        h5_image = fabioh5.File(fabio_image=fabio_image)
        scan_header_path = "/scan_0/instrument/file/scan_header"
        self.assertIn(scan_header_path, h5_image)
        data = h5_image[scan_header_path]
        self.assertIsInstance(data[...], numpy.ndarray)

    def test_unicode_header(self):
        """Test that it does not fail"""
        try:
            header = {}
            header["foo"] = b'abc'
            data = numpy.array([[0, 0], [0, 0]], dtype=numpy.int8)
            fabio_image = fabio.edfimage.edfimage(data=data, header=header)
            header = {}
            header["foo"] = u'abc\u2764'
            fabio_image.append_frame(data=data, header=header)
        except Exception as e:
            _logger.error(e.args[0])
            _logger.debug("Backtrace", exc_info=True)
            self.skipTest("fabio do not allow to create the resource")

        h5_image = fabioh5.File(fabio_image=fabio_image)
        scan_header_path = "/scan_0/instrument/file/scan_header"
        self.assertIn(scan_header_path, h5_image)
        data = h5_image[scan_header_path]
        self.assertIsInstance(data[...], numpy.ndarray)


class TestFabioH5MultiFrames(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        names = ["A", "B", "C", "D"]
        values = [["32000", "-10", "5.0", "1"],
                  ["-32000", "-10", "5.0", "1"]]

        fabio_file = None

        for i in range(10):
            header = {
                "image_id": "%d" % i,
                "integer": "-100",
                "float": "1.0",
                "string": "hi!",
                "list_integer": "100 50 0",
                "list_float": "1.0 2.0 3.5",
                "string_looks_like_list": "2000 hi!",
                "motor_mne": " ".join(names),
                "motor_pos": " ".join(values[i % len(values)]),
                "counter_mne": " ".join(names),
                "counter_pos": " ".join(values[i % len(values)])
            }
            for iname, name in enumerate(names):
                header[name] = values[i % len(values)][iname]

            data = numpy.array([[i, 11], [12, 13], [14, 15]], dtype=numpy.int64)
            if fabio_file is None:
                fabio_file = fabio.edfimage.EdfImage(data=data, header=header)
            else:
                fabio_file.append_frame(data=data, header=header)

        cls.fabio_file = fabio_file
        cls.fabioh5 = fabioh5.File(fabio_image=fabio_file)

    def test_others(self):
        others = self.fabioh5["/scan_0/instrument/detector_0/others"]
        dataset = others["A"]
        self.assertGreaterEqual(dataset.dtype.itemsize, 1)
        self.assertEqual(dataset.dtype.kind, "i")
        dataset = others["B"]
        self.assertGreaterEqual(dataset.dtype.itemsize, 1)
        self.assertEqual(dataset.dtype.kind, "i")
        dataset = others["C"]
        self.assertGreaterEqual(dataset.dtype.itemsize, 1)
        self.assertEqual(dataset.dtype.kind, "f")
        dataset = others["D"]
        self.assertGreaterEqual(dataset.dtype.itemsize, 1)
        self.assertEqual(dataset.dtype.kind, "u")

    def test_positioners(self):
        counters = self.fabioh5["/scan_0/instrument/positioners"]
        # At least 32 bits, no unsigned values
        dataset = counters["A"]
        self.assertGreaterEqual(dataset.dtype.itemsize, 4)
        self.assertEqual(dataset.dtype.kind, "i")
        dataset = counters["B"]
        self.assertGreaterEqual(dataset.dtype.itemsize, 4)
        self.assertEqual(dataset.dtype.kind, "i")
        dataset = counters["C"]
        self.assertGreaterEqual(dataset.dtype.itemsize, 4)
        self.assertEqual(dataset.dtype.kind, "f")
        dataset = counters["D"]
        self.assertGreaterEqual(dataset.dtype.itemsize, 4)
        self.assertEqual(dataset.dtype.kind, "i")

    def test_counters(self):
        counters = self.fabioh5["/scan_0/measurement"]
        # At least 32 bits, no unsigned values
        dataset = counters["A"]
        self.assertGreaterEqual(dataset.dtype.itemsize, 4)
        self.assertEqual(dataset.dtype.kind, "i")
        dataset = counters["B"]
        self.assertGreaterEqual(dataset.dtype.itemsize, 4)
        self.assertEqual(dataset.dtype.kind, "i")
        dataset = counters["C"]
        self.assertGreaterEqual(dataset.dtype.itemsize, 4)
        self.assertEqual(dataset.dtype.kind, "f")
        dataset = counters["D"]
        self.assertGreaterEqual(dataset.dtype.itemsize, 4)
        self.assertEqual(dataset.dtype.kind, "i")


class TestFabioH5WithEdf(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.tmp_directory = tempfile.mkdtemp()

        cls.edf_filename = os.path.join(cls.tmp_directory, "test.edf")

        header = {
            "integer": "-100",
            "float": "1.0",
            "string": "hi!",
            "list_integer": "100 50 0",
            "list_float": "1.0 2.0 3.5",
            "string_looks_like_list": "2000 hi!",
        }
        data = numpy.array([[10, 11], [12, 13], [14, 15]], dtype=numpy.int64)
        fabio_image = fabio.edfimage.edfimage(data, header)
        fabio_image.write(cls.edf_filename)

        cls.fabio_image = fabio.open(cls.edf_filename)
        cls.h5_image = fabioh5.File(fabio_image=cls.fabio_image)

    @classmethod
    def tearDownClass(cls):
        cls.fabio_image = None
        cls.h5_image = None
        shutil.rmtree(cls.tmp_directory)

    def test_reserved_format_metadata(self):
        if fabio.hexversion < 327920:  # 0.5.0 final
            self.skipTest("fabio >= 0.5.0 final is needed")

        # The EDF contains reserved keys in the header
        self.assertIn("HeaderID", self.fabio_image.header)
        # We do not expose them in FabioH5
        self.assertNotIn("/scan_0/instrument/detector_0/others/HeaderID", self.h5_image)


class _TestableFrameData(fabioh5.FrameData):
    """Allow to test if the full data is reached."""
    def _create_data(self):
        raise RuntimeError("Not supposed to be called")


class TestFabioH5WithFileSeries(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.tmp_directory = tempfile.mkdtemp()

        cls.edf_filenames = []

        for i in range(10):
            filename = os.path.join(cls.tmp_directory, "test_%04d.edf" % i)
            cls.edf_filenames.append(filename)

            header = {
                "image_id": "%d" % i,
                "integer": "-100",
                "float": "1.0",
                "string": "hi!",
                "list_integer": "100 50 0",
                "list_float": "1.0 2.0 3.5",
                "string_looks_like_list": "2000 hi!",
            }
            data = numpy.array([[i, 11], [12, 13], [14, 15]], dtype=numpy.int64)
            fabio_image = fabio.edfimage.edfimage(data, header)
            fabio_image.write(filename)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_directory)

    def _testH5Image(self, h5_image):
        # test data
        dataset = h5_image["/scan_0/instrument/detector_0/data"]
        self.assertEqual(dataset.h5py_class, h5py.Dataset)
        self.assertTrue(isinstance(dataset[()], numpy.ndarray))
        self.assertEqual(dataset.dtype.kind, "i")
        self.assertEqual(dataset.shape, (10, 3, 2))
        self.assertEqual(list(dataset[:, 0, 0]), list(range(10)))
        self.assertEqual(dataset.attrs["interpretation"], "image")
        # test metatdata
        dataset = h5_image["/scan_0/instrument/detector_0/others/image_id"]
        self.assertEqual(list(dataset[...]), list(range(10)))

    def testFileList(self):
        h5_image = fabioh5.File(file_series=self.edf_filenames)
        self._testH5Image(h5_image)

    def testFileSeries(self):
        file_series = fabioh5._FileSeries(self.edf_filenames)
        h5_image = fabioh5.File(file_series=file_series)
        self._testH5Image(h5_image)

    def testFrameDataCache(self):
        file_series = fabioh5._FileSeries(self.edf_filenames)
        reader = fabioh5.FabioReader(file_series=file_series)
        frameData = _TestableFrameData("foo", reader)
        self.assertEqual(frameData.dtype.kind, "i")
        self.assertEqual(frameData.shape, (10, 3, 2))
