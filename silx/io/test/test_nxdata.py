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
"""Tests for NXdata parsing"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "27/01/2018"

try:
    import h5py
except ImportError:
    h5py = None
import numpy
import tempfile
import unittest
from .. import nxdata

from silx.third_party import six

text_dtype = h5py.special_dtype(vlen=six.text_type)


@unittest.skipIf(h5py is None, "silx.io.nxdata tests depend on h5py")
class TestNXdata(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.NamedTemporaryFile(prefix="nxdata_examples_", suffix=".h5", delete=True)
        tmp.file.close()
        self.h5fname = tmp.name
        self.h5f = h5py.File(tmp.name, "w")

        # SCALARS
        g0d = self.h5f.create_group("scalars")

        g0d0 = g0d.create_group("0D_scalar")
        g0d0.attrs["NX_class"] = "NXdata"
        g0d0.attrs["signal"] = "scalar"
        g0d0.create_dataset("scalar", data=10)

        g0d1 = g0d.create_group("2D_scalars")
        g0d1.attrs["NX_class"] = "NXdata"
        g0d1.attrs["signal"] = "scalars"
        ds = g0d1.create_dataset("scalars", data=numpy.arange(3 * 10).reshape((3, 10)))
        ds.attrs["interpretation"] = "scalar"

        g0d1 = g0d.create_group("4D_scalars")
        g0d1.attrs["NX_class"] = "NXdata"
        g0d1.attrs["signal"] = "scalars"
        ds = g0d1.create_dataset("scalars", data=numpy.arange(2 * 2 * 3 * 10).reshape((2, 2, 3, 10)))
        ds.attrs["interpretation"] = "scalar"

        # SPECTRA
        g1d = self.h5f.create_group("spectra")

        g1d0 = g1d.create_group("1D_spectrum")
        g1d0.attrs["NX_class"] = "NXdata"
        g1d0.attrs["signal"] = "count"
        g1d0.attrs["auxiliary_signals"] = numpy.array(["count2", "count3"],
                                                      dtype=text_dtype)
        g1d0.attrs["axes"] = "energy_calib"
        g1d0.attrs["uncertainties"] = numpy.array(["energy_errors", ],
                                                  dtype=text_dtype)
        g1d0.create_dataset("count", data=numpy.arange(10))
        g1d0.create_dataset("count2", data=0.5 * numpy.arange(10))
        d = g1d0.create_dataset("count3", data=0.4 * numpy.arange(10))
        d.attrs["long_name"] = "3rd counter"
        g1d0.create_dataset("title", data="Title as dataset (like nexpy)")
        g1d0.create_dataset("energy_calib", data=(10, 5))  # 10 * idx + 5
        g1d0.create_dataset("energy_errors", data=3.14 * numpy.random.rand(10))

        g1d1 = g1d.create_group("2D_spectra")
        g1d1.attrs["NX_class"] = "NXdata"
        g1d1.attrs["signal"] = "counts"
        ds = g1d1.create_dataset("counts", data=numpy.arange(3 * 10).reshape((3, 10)))
        ds.attrs["interpretation"] = "spectrum"

        g1d2 = g1d.create_group("4D_spectra")
        g1d2.attrs["NX_class"] = "NXdata"
        g1d2.attrs["signal"] = "counts"
        g1d2.attrs["axes"] = numpy.array(["energy", ], dtype=text_dtype)
        ds = g1d2.create_dataset("counts", data=numpy.arange(2 * 2 * 3 * 10).reshape((2, 2, 3, 10)))
        ds.attrs["interpretation"] = "spectrum"
        ds = g1d2.create_dataset("errors", data=4.5 * numpy.random.rand(2, 2, 3, 10))
        ds = g1d2.create_dataset("energy", data=5 + 10 * numpy.arange(15),
                                 shuffle=True, compression="gzip")
        ds.attrs["long_name"] = "Calibrated energy"
        ds.attrs["first_good"] = 3
        ds.attrs["last_good"] = 12
        g1d2.create_dataset("energy_errors", data=10 * numpy.random.rand(15))

        # IMAGES
        g2d = self.h5f.create_group("images")

        g2d0 = g2d.create_group("2D_regular_image")
        g2d0.attrs["NX_class"] = "NXdata"
        g2d0.attrs["signal"] = "image"
        g2d0.attrs["auxiliary_signals"] = "image2"
        g2d0.attrs["axes"] = numpy.array(["rows_calib", "columns_coordinates"],
                                         dtype=text_dtype)
        g2d0.create_dataset("image", data=numpy.arange(4 * 6).reshape((4, 6)))
        g2d0.create_dataset("image2", data=numpy.arange(4 * 6).reshape((4, 6)))
        ds = g2d0.create_dataset("rows_calib", data=(10, 5))
        ds.attrs["long_name"] = "Calibrated Y"
        g2d0.create_dataset("columns_coordinates", data=0.5 + 0.02 * numpy.arange(6))

        g2d1 = g2d.create_group("2D_irregular_data")
        g2d1.attrs["NX_class"] = "NXdata"
        g2d1.attrs["signal"] = "data"
        g2d1.attrs["title"] = "Title as group attr"
        g2d1.attrs["axes"] = numpy.array(["rows_coordinates", "columns_coordinates"],
                                         dtype=text_dtype)
        g2d1.create_dataset("data", data=numpy.arange(64 * 128).reshape((64, 128)))
        g2d1.create_dataset("rows_coordinates", data=numpy.arange(64) + numpy.random.rand(64))
        g2d1.create_dataset("columns_coordinates", data=numpy.arange(128) + 2.5 * numpy.random.rand(128))

        g2d2 = g2d.create_group("3D_images")
        g2d2.attrs["NX_class"] = "NXdata"
        g2d2.attrs["signal"] = "images"
        ds = g2d2.create_dataset("images", data=numpy.arange(2 * 4 * 6).reshape((2, 4, 6)))
        ds.attrs["interpretation"] = "image"

        g2d3 = g2d.create_group("5D_images")
        g2d3.attrs["NX_class"] = "NXdata"
        g2d3.attrs["signal"] = "images"
        g2d3.attrs["axes"] = numpy.array(["rows_coordinates", "columns_coordinates"],
                                         dtype=text_dtype)
        ds = g2d3.create_dataset("images", data=numpy.arange(2 * 2 * 2 * 4 * 6).reshape((2, 2, 2, 4, 6)))
        ds.attrs["interpretation"] = "image"
        g2d3.create_dataset("rows_coordinates", data=5 + 10 * numpy.arange(4))
        g2d3.create_dataset("columns_coordinates", data=0.5 + 0.02 * numpy.arange(6))

        g2d4 = g2d.create_group("RGBA_image")
        g2d4.attrs["NX_class"] = "NXdata"
        g2d4.attrs["signal"] = "image"
        g2d4.attrs["axes"] = numpy.array(["rows_calib", "columns_coordinates"],
                                         dtype=text_dtype)
        rgba_image = numpy.linspace(0, 1, num=7*8*3).reshape((7, 8, 3))
        rgba_image[:, :, 1] = 1 - rgba_image[:, :, 1]      # invert G channel to add some color
        ds = g2d4.create_dataset("image", data=rgba_image)
        ds.attrs["interpretation"] = "rgba-image"
        ds = g2d4.create_dataset("rows_calib", data=(10, 5))
        ds.attrs["long_name"] = "Calibrated Y"
        g2d4.create_dataset("columns_coordinates", data=0.5+0.02*numpy.arange(8))

        # SCATTER
        g = self.h5f.create_group("scatters")

        gd0 = g.create_group("x_y_scatter")
        gd0.attrs["NX_class"] = "NXdata"
        gd0.attrs["signal"] = "y"
        gd0.attrs["axes"] = numpy.array(["x", ], dtype=text_dtype)
        gd0.create_dataset("y", data=numpy.random.rand(128) - 0.5)
        gd0.create_dataset("x", data=2 * numpy.random.rand(128))
        gd0.create_dataset("x_errors", data=0.05 * numpy.random.rand(128))
        gd0.create_dataset("errors", data=0.05 * numpy.random.rand(128))

        gd1 = g.create_group("x_y_value_scatter")
        gd1.attrs["NX_class"] = "NXdata"
        gd1.attrs["signal"] = "values"
        gd1.attrs["axes"] = numpy.array(["x", "y"], dtype=text_dtype)
        gd1.create_dataset("values", data=3.14 * numpy.random.rand(128))
        gd1.create_dataset("y", data=numpy.random.rand(128))
        gd1.create_dataset("y_errors", data=0.02 * numpy.random.rand(128))
        gd1.create_dataset("x", data=numpy.random.rand(128))
        gd1.create_dataset("x_errors", data=0.02 * numpy.random.rand(128))

    def tearDown(self):
        self.h5f.close()

    def testValidity(self):
        for group in self.h5f:
            for subgroup in self.h5f[group]:
                self.assertTrue(
                        nxdata.is_valid_nxdata(self.h5f[group][subgroup]),
                        "%s/%s not found to be a valid NXdata group" % (group, subgroup))

    def testScalars(self):
        nxd = nxdata.NXdata(self.h5f["scalars/0D_scalar"])
        self.assertTrue(nxd.signal_is_0d)
        self.assertEqual(nxd.signal[()], 10)
        self.assertEqual(nxd.axes_names, [])
        self.assertEqual(nxd.axes_dataset_names, [])
        self.assertEqual(nxd.axes, [])
        self.assertIsNone(nxd.errors)
        self.assertFalse(nxd.is_scatter or nxd.is_x_y_value_scatter)
        self.assertIsNone(nxd.interpretation)

        nxd = nxdata.NXdata(self.h5f["scalars/2D_scalars"])
        self.assertTrue(nxd.signal_is_2d)
        self.assertEqual(nxd.signal[1, 2], 12)
        self.assertEqual(nxd.axes_names, [None, None])
        self.assertEqual(nxd.axes_dataset_names, [None, None])
        self.assertEqual(nxd.axes, [None, None])
        self.assertIsNone(nxd.errors)
        self.assertFalse(nxd.is_scatter or nxd.is_x_y_value_scatter)
        self.assertEqual(nxd.interpretation, "scalar")

        nxd = nxdata.NXdata(self.h5f["scalars/4D_scalars"])
        self.assertFalse(nxd.signal_is_0d or nxd.signal_is_1d or
                         nxd.signal_is_2d or nxd.signal_is_3d)
        self.assertEqual(nxd.signal[1, 0, 1, 4], 74)
        self.assertEqual(nxd.axes_names, [None, None, None, None])
        self.assertEqual(nxd.axes_dataset_names, [None, None, None, None])
        self.assertEqual(nxd.axes, [None, None, None, None])
        self.assertIsNone(nxd.errors)
        self.assertFalse(nxd.is_scatter or nxd.is_x_y_value_scatter)
        self.assertEqual(nxd.interpretation, "scalar")

    def testSpectra(self):
        nxd = nxdata.NXdata(self.h5f["spectra/1D_spectrum"])
        self.assertTrue(nxd.signal_is_1d)
        self.assertTrue(nxd.is_curve)
        self.assertTrue(numpy.array_equal(numpy.array(nxd.signal),
                                          numpy.arange(10)))
        self.assertEqual(nxd.axes_names, ["energy_calib"])
        self.assertEqual(nxd.axes_dataset_names, ["energy_calib"])
        self.assertEqual(nxd.axes[0][0], 10)
        self.assertEqual(nxd.axes[0][1], 5)
        self.assertIsNone(nxd.errors)
        self.assertFalse(nxd.is_scatter or nxd.is_x_y_value_scatter)
        self.assertIsNone(nxd.interpretation)
        self.assertEqual(nxd.title, "Title as dataset (like nexpy)")

        self.assertEqual(nxd.auxiliary_signals_dataset_names,
                         ["count2", "count3"])
        self.assertEqual(nxd.auxiliary_signals_names,
                         ["count2", "3rd counter"])
        self.assertAlmostEqual(nxd.auxiliary_signals[1][2],
                               0.8)      # numpy.arange(10) * 0.4

        nxd = nxdata.NXdata(self.h5f["spectra/2D_spectra"])
        self.assertTrue(nxd.signal_is_2d)
        self.assertTrue(nxd.is_curve)
        self.assertEqual(nxd.axes_names, [None, None])
        self.assertEqual(nxd.axes_dataset_names, [None, None])
        self.assertEqual(nxd.axes, [None, None])
        self.assertIsNone(nxd.errors)
        self.assertFalse(nxd.is_scatter or nxd.is_x_y_value_scatter)
        self.assertEqual(nxd.interpretation, "spectrum")

        nxd = nxdata.NXdata(self.h5f["spectra/4D_spectra"])
        self.assertFalse(nxd.signal_is_0d or nxd.signal_is_1d or
                         nxd.signal_is_2d or nxd.signal_is_3d)
        self.assertTrue(nxd.is_curve)
        self.assertEqual(nxd.axes_names,
                         [None, None, None, "Calibrated energy"])
        self.assertEqual(nxd.axes_dataset_names,
                         [None, None, None, "energy"])
        self.assertEqual(nxd.axes[:3], [None, None, None])
        self.assertEqual(nxd.axes[3].shape, (10, ))    # dataset shape (15, ) sliced [3:12]
        self.assertIsNotNone(nxd.errors)
        self.assertEqual(nxd.errors.shape, (2, 2, 3, 10))
        self.assertFalse(nxd.is_scatter or nxd.is_x_y_value_scatter)
        self.assertEqual(nxd.interpretation, "spectrum")
        self.assertEqual(nxd.get_axis_errors("energy").shape,
                         (10,))
        # test getting axis errors by long_name
        self.assertTrue(numpy.array_equal(nxd.get_axis_errors("Calibrated energy"),
                                          nxd.get_axis_errors("energy")))
        self.assertTrue(numpy.array_equal(nxd.get_axis_errors(b"Calibrated energy"),
                                          nxd.get_axis_errors("energy")))

    def testImages(self):
        nxd = nxdata.NXdata(self.h5f["images/2D_regular_image"])
        self.assertTrue(nxd.signal_is_2d)
        self.assertTrue(nxd.is_image)
        self.assertEqual(nxd.axes_names, ["Calibrated Y", "columns_coordinates"])
        self.assertEqual(list(nxd.axes_dataset_names),
                         ["rows_calib", "columns_coordinates"])
        self.assertIsNone(nxd.errors)
        self.assertFalse(nxd.is_scatter or nxd.is_x_y_value_scatter)
        self.assertIsNone(nxd.interpretation)
        self.assertEqual(len(nxd.auxiliary_signals), 1)
        self.assertEqual(nxd.auxiliary_signals_names, ["image2"])

        nxd = nxdata.NXdata(self.h5f["images/2D_irregular_data"])
        self.assertTrue(nxd.signal_is_2d)
        self.assertTrue(nxd.is_image)

        self.assertEqual(nxd.axes_dataset_names, nxd.axes_names)
        self.assertEqual(list(nxd.axes_dataset_names),
                         ["rows_coordinates", "columns_coordinates"])
        self.assertEqual(len(nxd.axes), 2)
        self.assertIsNone(nxd.errors)
        self.assertFalse(nxd.is_scatter or nxd.is_x_y_value_scatter)
        self.assertIsNone(nxd.interpretation)
        self.assertEqual(nxd.title, "Title as group attr")

        nxd = nxdata.NXdata(self.h5f["images/5D_images"])
        self.assertTrue(nxd.is_image)
        self.assertFalse(nxd.signal_is_0d or nxd.signal_is_1d or
                         nxd.signal_is_2d or nxd.signal_is_3d)
        self.assertEqual(nxd.axes_names,
                         [None, None, None, 'rows_coordinates', 'columns_coordinates'])
        self.assertEqual(nxd.axes_dataset_names,
                         [None, None, None, 'rows_coordinates', 'columns_coordinates'])
        self.assertIsNone(nxd.errors)
        self.assertFalse(nxd.is_scatter or nxd.is_x_y_value_scatter)
        self.assertEqual(nxd.interpretation, "image")

        nxd = nxdata.NXdata(self.h5f["images/RGBA_image"])
        self.assertTrue(nxd.is_image)
        self.assertEqual(nxd.interpretation, "rgba-image")
        self.assertTrue(nxd.signal_is_3d)
        self.assertEqual(nxd.axes_names, ["Calibrated Y",
                                          "columns_coordinates",
                                          None])
        self.assertEqual(list(nxd.axes_dataset_names),
                         ["rows_calib", "columns_coordinates", None])

    def testScatters(self):
        nxd = nxdata.NXdata(self.h5f["scatters/x_y_scatter"])
        self.assertTrue(nxd.signal_is_1d)
        self.assertEqual(nxd.axes_names, ["x"])
        self.assertEqual(nxd.axes_dataset_names,
                         ["x"])
        self.assertIsNotNone(nxd.errors)
        self.assertEqual(nxd.get_axis_errors("x").shape,
                         (128, ))
        self.assertTrue(nxd.is_scatter)
        self.assertFalse(nxd.is_x_y_value_scatter)
        self.assertIsNone(nxd.interpretation)

        nxd = nxdata.NXdata(self.h5f["scatters/x_y_value_scatter"])
        self.assertFalse(nxd.signal_is_1d)
        self.assertTrue(nxd.axes_dataset_names,
                        nxd.axes_names)
        self.assertEqual(nxd.axes_dataset_names,
                         ["x", "y"])
        self.assertEqual(nxd.get_axis_errors("x").shape,
                         (128, ))
        self.assertEqual(nxd.get_axis_errors("y").shape,
                         (128, ))
        self.assertEqual(len(nxd.axes), 2)
        self.assertIsNone(nxd.errors)
        self.assertTrue(nxd.is_scatter)
        self.assertTrue(nxd.is_x_y_value_scatter)
        self.assertIsNone(nxd.interpretation)


@unittest.skipIf(h5py is None, "silx.io.nxdata tests depend on h5py")
class TestLegacyNXdata(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.NamedTemporaryFile(prefix="nxdata_legacy_examples_",
                                          suffix=".h5", delete=True)
        tmp.file.close()
        self.h5fname = tmp.name
        self.h5f = h5py.File(tmp.name, "w")

    def tearDown(self):
        self.h5f.close()

    def testSignalAttrOnDataset(self):
        g = self.h5f.create_group("2D")
        g.attrs["NX_class"] = "NXdata"

        ds0 = g.create_dataset("image0",
                               data=numpy.arange(4 * 6).reshape((4, 6)))
        ds0.attrs["signal"] = 1
        ds0.attrs["long_name"] = "My first image"

        ds1 = g.create_dataset("image1",
                               data=numpy.arange(4 * 6).reshape((4, 6)))
        ds1.attrs["signal"] = "2"
        ds1.attrs["long_name"] = "My 2nd image"

        ds2 = g.create_dataset("image2",
                               data=numpy.arange(4 * 6).reshape((4, 6)))
        ds2.attrs["signal"] = 3

        nxd = nxdata.NXdata(self.h5f["2D"])

        self.assertEqual(nxd.signal_dataset_name, "image0")
        self.assertEqual(nxd.signal_name, "My first image")
        self.assertEqual(nxd.signal.shape,
                         (4, 6))

        self.assertEqual(len(nxd.auxiliary_signals), 2)
        self.assertEqual(nxd.auxiliary_signals[1].shape,
                         (4, 6))

        self.assertEqual(nxd.auxiliary_signals_dataset_names,
                         ["image1", "image2"])
        self.assertEqual(nxd.auxiliary_signals_names,
                         ["My 2nd image", "image2"])

    def testAxesOnSignalDataset(self):
        g = self.h5f.create_group("2D")
        g.attrs["NX_class"] = "NXdata"

        ds0 = g.create_dataset("image0",
                               data=numpy.arange(4 * 6).reshape((4, 6)))
        ds0.attrs["signal"] = 1
        ds0.attrs["axes"] = "yaxis:xaxis"

        ds1 = g.create_dataset("yaxis",
                               data=numpy.arange(4))
        ds2 = g.create_dataset("xaxis",
                               data=numpy.arange(6))

        nxd = nxdata.NXdata(self.h5f["2D"])

        self.assertEqual(nxd.axes_dataset_names,
                         ["yaxis", "xaxis"])
        self.assertTrue(numpy.array_equal(nxd.axes[0],
                                          numpy.arange(4)))
        self.assertTrue(numpy.array_equal(nxd.axes[1],
                                          numpy.arange(6)))

    def testAxesOnAxesDatasets(self):
        g = self.h5f.create_group("2D")
        g.attrs["NX_class"] = "NXdata"

        ds0 = g.create_dataset("image0",
                               data=numpy.arange(4 * 6).reshape((4, 6)))
        ds0.attrs["signal"] = 1
        ds1 = g.create_dataset("yaxis",
                               data=numpy.arange(4))
        ds1.attrs["axis"] = 0
        ds2 = g.create_dataset("xaxis",
                               data=numpy.arange(6))
        ds2.attrs["axis"] = "1"

        nxd = nxdata.NXdata(self.h5f["2D"])
        self.assertEqual(nxd.axes_dataset_names,
                         ["yaxis", "xaxis"])
        self.assertTrue(numpy.array_equal(nxd.axes[0],
                                          numpy.arange(4)))
        self.assertTrue(numpy.array_equal(nxd.axes[1],
                                          numpy.arange(6)))

    def testAsciiUndefinedAxesAttrs(self):
        """Some files may not be using utf8 for str attrs"""
        g = self.h5f.create_group("bytes_attrs")
        g.attrs["NX_class"] = b"NXdata"
        g.attrs["signal"] = b"image0"
        g.attrs["axes"] = b"yaxis", b"."

        g.create_dataset("image0",
                         data=numpy.arange(4 * 6).reshape((4, 6)))
        g.create_dataset("yaxis",
                         data=numpy.arange(4))

        nxd = nxdata.NXdata(self.h5f["bytes_attrs"])
        self.assertEqual(nxd.axes_dataset_names,
                         ["yaxis", None])


class TestSaveNXdata(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.NamedTemporaryFile(prefix="nxdata",
                                          suffix=".h5", delete=True)
        tmp.file.close()
        self.h5fname = tmp.name

    def testSimpleSave(self):
        sig = numpy.array([0, 1, 2])
        a0 = numpy.array([2, 3, 4])
        a1 = numpy.array([3, 4, 5])
        nxdata.save_NXdata(filename=self.h5fname,
                           signal=sig,
                           axes=[a0, a1],
                           signal_name="sig",
                           axes_names=["a0", "a1"],
                           nxentry_name="a",
                           nxdata_name="mydata")

        h5f = h5py.File(self.h5fname, "r")
        self.assertTrue(nxdata.is_valid_nxdata(h5f["a/mydata"]))

        nxd = nxdata.NXdata(h5f["/a/mydata"])
        self.assertTrue(numpy.array_equal(nxd.signal,
                                          sig))
        self.assertTrue(numpy.array_equal(nxd.axes[0],
                                          a0))

        h5f.close()

    def testSimplestSave(self):
        sig = numpy.array([0, 1, 2])
        nxdata.save_NXdata(filename=self.h5fname,
                           signal=sig)

        h5f = h5py.File(self.h5fname, "r")

        self.assertTrue(nxdata.is_valid_nxdata(h5f["/entry/data0"]))

        nxd = nxdata.NXdata(h5f["/entry/data0"])
        self.assertTrue(numpy.array_equal(nxd.signal,
                                          sig))
        h5f.close()

    def testSaveDefaultAxesNames(self):
        sig = numpy.array([0, 1, 2])
        a0 = numpy.array([2, 3, 4])
        a1 = numpy.array([3, 4, 5])
        nxdata.save_NXdata(filename=self.h5fname,
                           signal=sig,
                           axes=[a0, a1],
                           signal_name="sig",
                           axes_names=None,
                           axes_long_names=["a", "b"],
                           nxentry_name="a",
                           nxdata_name="mydata")

        h5f = h5py.File(self.h5fname, "r")
        self.assertTrue(nxdata.is_valid_nxdata(h5f["a/mydata"]))

        nxd = nxdata.NXdata(h5f["/a/mydata"])
        self.assertTrue(numpy.array_equal(nxd.signal,
                                          sig))
        self.assertTrue(numpy.array_equal(nxd.axes[0],
                                          a0))
        self.assertEqual(nxd.axes_dataset_names,
                         [u"dim0", u"dim1"])
        self.assertEqual(nxd.axes_names,
                         [u"a", u"b"])

        h5f.close()

    def testSaveToExistingEntry(self):
        h5f = h5py.File(self.h5fname, "w")
        g = h5f.create_group("myentry")
        g.attrs["NX_class"] = "NXentry"
        h5f.close()

        sig = numpy.array([0, 1, 2])
        a0 = numpy.array([2, 3, 4])
        a1 = numpy.array([3, 4, 5])
        nxdata.save_NXdata(filename=self.h5fname,
                           signal=sig,
                           axes=[a0, a1],
                           signal_name="sig",
                           axes_names=["a0", "a1"],
                           nxentry_name="myentry",
                           nxdata_name="toto")

        h5f = h5py.File(self.h5fname, "r")
        self.assertTrue(nxdata.is_valid_nxdata(h5f["myentry/toto"]))

        nxd = nxdata.NXdata(h5f["myentry/toto"])
        self.assertTrue(numpy.array_equal(nxd.signal,
                                          sig))
        self.assertTrue(numpy.array_equal(nxd.axes[0],
                                          a0))
        h5f.close()


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestNXdata))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestLegacyNXdata))
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestSaveNXdata))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest="suite")
