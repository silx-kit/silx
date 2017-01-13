# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2017 European Synchrotron Radiation Facility
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
__date__ = "13/01/2017"

import logging
import numpy
import unittest

_logger = logging.getLogger(__name__)


try:
    import fabio
except ImportError:
    fabio = None

try:
    import h5py
except ImportError:
    h5py = None

if fabio is not None and h5py is not None:
    from .. import fabioh5


class TestFabioH5(unittest.TestCase):

    def setUp(self):
        if fabio is None:
            self.skipTest("fabio is needed")
        if h5py is None:
            self.skipTest("h5py is needed")

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
        self.assertEquals(self.h5_image.h5py_class, h5py.File)
        self.assertEquals(self.h5_image["/"].h5py_class, h5py.File)
        self.assertEquals(self.h5_image["/scan_0"].h5py_class, h5py.Group)
        self.assertEquals(self.h5_image["/scan_0/instrument"].h5py_class, h5py.Group)
        self.assertEquals(self.h5_image["/scan_0/measurement"].h5py_class, h5py.Group)

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

    def test_frames(self):
        dataset = self.h5_image["/scan_0/instrument/detector_0/data"]
        self.assertEquals(dataset.h5py_class, h5py.Dataset)
        self.assertTrue(isinstance(dataset[()], numpy.ndarray))
        self.assertEquals(dataset.dtype.kind, "i")
        self.assertEquals(dataset.shape, (1, 3, 2))
        self.assertEquals(dataset.attrs["interpretation"], "image")

    def test_metadata_int(self):
        dataset = self.h5_image["/scan_0/instrument/detector_0/others/integer"]
        self.assertEquals(dataset.h5py_class, h5py.Dataset)
        self.assertEquals(dataset[()], -100)
        self.assertEquals(dataset.dtype.kind, "i")
        self.assertEquals(dataset.shape, (1,))

    def test_metadata_float(self):
        dataset = self.h5_image["/scan_0/instrument/detector_0/others/float"]
        self.assertEquals(dataset.h5py_class, h5py.Dataset)
        self.assertEquals(dataset[()], 1.0)
        self.assertEquals(dataset.dtype.kind, "f")
        self.assertEquals(dataset.shape, (1,))

    def test_metadata_string(self):
        dataset = self.h5_image["/scan_0/instrument/detector_0/others/string"]
        self.assertEquals(dataset.h5py_class, h5py.Dataset)
        self.assertEquals(dataset[()], "hi!")
        self.assertEquals(dataset.dtype.type, numpy.string_)
        self.assertEquals(dataset.shape, (1,))

    def test_metadata_list_integer(self):
        dataset = self.h5_image["/scan_0/instrument/detector_0/others/list_integer"]
        self.assertEquals(dataset.h5py_class, h5py.Dataset)
        self.assertEquals(dataset.dtype.kind, "u")
        self.assertEquals(dataset.shape, (1, 3))
        self.assertEquals(dataset[0, 0], 100)
        self.assertEquals(dataset[0, 1], 50)

    def test_metadata_list_float(self):
        dataset = self.h5_image["/scan_0/instrument/detector_0/others/list_float"]
        self.assertEquals(dataset.h5py_class, h5py.Dataset)
        self.assertEquals(dataset.dtype.kind, "f")
        self.assertEquals(dataset.shape, (1, 3))
        self.assertEquals(dataset[0, 0], 1.0)
        self.assertEquals(dataset[0, 1], 2.0)

    def test_metadata_list_looks_like_list(self):
        dataset = self.h5_image["/scan_0/instrument/detector_0/others/string_looks_like_list"]
        self.assertEquals(dataset.h5py_class, h5py.Dataset)
        self.assertEquals(dataset[()], "2000 hi!")
        self.assertEquals(dataset.dtype.type, numpy.string_)
        self.assertEquals(dataset.shape, (1,))


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestFabioH5))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
