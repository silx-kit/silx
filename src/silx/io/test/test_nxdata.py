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
"""Tests for NXdata parsing"""

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "24/03/2020"


import h5py
import numpy
import pytest

from .. import nxdata
from ..dictdump import dicttoh5


text_dtype = h5py.special_dtype(vlen=str)


class TestNXdata:
    def test0DScalar(self, tmp_path):
        with h5py.File(tmp_path / "nxdata_0D_scalar.h5", "w") as h5f:
            group = h5f.create_group("0D_scalar")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "scalar"
            group.create_dataset("scalar", data=10)
            group.create_dataset("scalar_errors", data=0.1)

            assert nxdata.is_valid_nxdata(group)

            nxd = nxdata.NXdata(group)

            assert nxd.signal_is_0d
            assert nxd.signal[()] == 10
            assert nxd.axes_names == []
            assert nxd.axes_dataset_names == []
            assert nxd.axes == []
            assert nxd.errors is not None
            assert not nxd.is_scatter
            assert not nxd.is_x_y_value_scatter
            assert nxd.interpretation is None

    def test2DScalars(self, tmp_path):
        with h5py.File(tmp_path / "nxdata_2D_scalars.h5", "w") as h5f:
            group = h5f.create_group("2D_scalars")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "scalars"
            ds = group.create_dataset(
                "scalars", data=numpy.arange(3 * 10).reshape((3, 10))
            )
            ds.attrs["interpretation"] = "scalar"

            assert nxdata.is_valid_nxdata(group)

            nxd = nxdata.NXdata(group)

            assert nxd.signal_is_2d
            assert nxd.signal[1, 2] == 12
            assert nxd.axes_names == [None, None]
            assert nxd.axes_dataset_names == [None, None]
            assert nxd.axes == [None, None]
            assert nxd.errors is None
            assert not nxd.is_scatter
            assert not nxd.is_x_y_value_scatter
            assert nxd.interpretation == "scalar"

    def test4DScalars(self, tmp_path):
        with h5py.File(tmp_path / "nxdata_1D_spectrum.h5", "w") as h5f:
            group = h5f.create_group("4D_scalars")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "scalars"
            ds = group.create_dataset(
                "scalars", data=numpy.arange(2 * 2 * 3 * 10).reshape((2, 2, 3, 10))
            )
            ds.attrs["interpretation"] = "scalar"

            assert nxdata.is_valid_nxdata(group)

            nxd = nxdata.NXdata(group)

            assert not nxd.signal_is_0d
            assert not nxd.signal_is_1d
            assert not nxd.signal_is_2d
            assert not nxd.signal_is_3d
            assert nxd.signal[1, 0, 1, 4] == 74
            assert nxd.axes_names == [None, None, None, None]
            assert nxd.axes_dataset_names == [None, None, None, None]
            assert nxd.axes == [None, None, None, None]
            assert nxd.errors is None
            assert not nxd.is_scatter
            assert not nxd.is_x_y_value_scatter
            assert nxd.interpretation == "scalar"

    def test1DSpectrum(self, tmp_path):
        with h5py.File(tmp_path / "nxdata_1D_spectrum.h5", "w") as h5f:
            group = h5f.create_group("1D_spectrum")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "count"
            group.attrs["auxiliary_signals"] = numpy.array(
                ["count2", "count3"], dtype=text_dtype
            )
            group.attrs["axes"] = "energy_calib"
            group.attrs["uncertainties"] = numpy.array(
                [
                    "energy_errors",
                ],
                dtype=text_dtype,
            )
            group.create_dataset("count", data=numpy.arange(10))
            group.create_dataset("count2", data=0.5 * numpy.arange(10))
            d = group.create_dataset("count3", data=0.4 * numpy.arange(10))
            d.attrs["long_name"] = "3rd counter"
            group.create_dataset("title", data="Title as dataset (like nexpy)")
            group.create_dataset("energy_calib", data=(10, 5))  # 10 * idx + 5
            group.create_dataset("energy_errors", data=3.14 * numpy.random.rand(10))

            assert nxdata.is_valid_nxdata(group)

            nxd = nxdata.NXdata(group)

            assert nxd.signal_is_1d
            assert nxd.is_curve
            assert numpy.array_equal(numpy.array(nxd.signal), numpy.arange(10))
            assert nxd.axes_names == ["energy_calib"]
            assert nxd.axes_dataset_names == ["energy_calib"]
            assert nxd.axes[0][0] == 10
            assert nxd.axes[0][1] == 5
            assert nxd.errors is None
            assert not nxd.is_scatter
            assert not nxd.is_x_y_value_scatter
            assert nxd.interpretation is None
            assert nxd.title == "Title as dataset (like nexpy)"

            assert nxd.auxiliary_signals_dataset_names == ["count2", "count3"]
            assert nxd.auxiliary_signals_names == ["count2", "3rd counter"]
            assert numpy.isclose(nxd.auxiliary_signals[1][2], 0.8)

    def test2DSpectra(self, tmp_path):
        with h5py.File(tmp_path / "nxdata_2D_spectra.h5", "w") as h5f:
            group = h5f.create_group("2D_spectra")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "counts"
            ds = group.create_dataset(
                "counts", data=numpy.arange(3 * 10).reshape((3, 10))
            )
            ds.attrs["interpretation"] = "spectrum"

            assert nxdata.is_valid_nxdata(group)

            nxd = nxdata.NXdata(group)

            assert nxd.signal_is_2d
            assert nxd.is_curve
            assert nxd.axes_names == [None, None]
            assert nxd.axes_dataset_names == [None, None]
            assert nxd.axes == [None, None]
            assert nxd.errors is None
            assert not nxd.is_scatter
            assert not nxd.is_x_y_value_scatter
            assert nxd.interpretation == "spectrum"

    def test4DSpectra(self, tmp_path):
        with h5py.File(tmp_path / "nxdata_2D_regular_image.h5", "w") as h5f:
            group = h5f.create_group("4D_spectra")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "counts"
            group.attrs["axes"] = numpy.array(
                [
                    "energy",
                ],
                dtype=text_dtype,
            )
            ds = group.create_dataset(
                "counts", data=numpy.arange(2 * 2 * 3 * 10).reshape((2, 2, 3, 10))
            )
            ds.attrs["interpretation"] = "spectrum"
            ds = group.create_dataset(
                "errors", data=4.5 * numpy.random.rand(2, 2, 3, 10)
            )
            ds = group.create_dataset(
                "energy",
                data=5 + 10 * numpy.arange(15),
                shuffle=True,
                compression="gzip",
            )
            ds.attrs["long_name"] = "Calibrated energy"
            ds.attrs["first_good"] = 3
            ds.attrs["last_good"] = 12
            group.create_dataset("energy_errors", data=10 * numpy.random.rand(15))

            assert nxdata.is_valid_nxdata(group)

            nxd = nxdata.NXdata(group)

            assert not nxd.signal_is_0d
            assert not nxd.signal_is_1d
            assert not nxd.signal_is_2d
            assert not nxd.signal_is_3d
            assert nxd.is_curve
            assert nxd.axes_names == [None, None, None, "Calibrated energy"]
            assert nxd.axes_dataset_names == [None, None, None, "energy"]
            assert nxd.axes[:3] == [None, None, None]
            assert nxd.axes[3].shape == (10,)  # dataset shape (15, ) sliced [3:12]
            assert nxd.errors is not None
            assert nxd.errors.shape == (2, 2, 3, 10)
            assert not nxd.is_scatter
            assert not nxd.is_x_y_value_scatter
            assert nxd.interpretation == "spectrum"
            assert nxd.get_axis_errors("energy").shape == (10,)
            # test getting axis errors by long_name
            assert numpy.array_equal(
                nxd.get_axis_errors("Calibrated energy"), nxd.get_axis_errors("energy")
            )
            assert numpy.array_equal(
                nxd.get_axis_errors(b"Calibrated energy"), nxd.get_axis_errors("energy")
            )

    def test2DRegularImage(self, tmp_path):
        with h5py.File(tmp_path / "nxdata_2D_regular_image.h5", "w") as h5f:
            group = h5f.create_group("2D_regular_image")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "image"
            group.attrs["auxiliary_signals"] = "image2"
            group.attrs["axes"] = numpy.array(
                ["rows_calib", "columns_coordinates"], dtype=text_dtype
            )
            group.create_dataset("image", data=numpy.arange(4 * 6).reshape((4, 6)))
            group.create_dataset("image2", data=numpy.arange(4 * 6).reshape((4, 6)))
            ds = group.create_dataset("rows_calib", data=(10, 5))
            ds.attrs["long_name"] = "Calibrated Y"
            group.create_dataset(
                "columns_coordinates", data=0.5 + 0.02 * numpy.arange(6)
            )

            assert nxdata.is_valid_nxdata(group)

            nxd = nxdata.NXdata(group)

            assert nxd.signal_is_2d
            assert nxd.is_image
            assert nxd.axes_names == ["Calibrated Y", "columns_coordinates"]
            assert list(nxd.axes_dataset_names) == ["rows_calib", "columns_coordinates"]

            assert nxd.errors is None
            assert not nxd.is_scatter
            assert not nxd.is_x_y_value_scatter
            assert nxd.interpretation is None
            assert len(nxd.auxiliary_signals) == 1
            assert nxd.auxiliary_signals_names == ["image2"]

    def test2DIrregularImage(self, tmp_path):
        with h5py.File(tmp_path / "nxdata_2D_irregular_data.h5", "w") as h5f:
            group = h5f.create_group("2D_irregular_data")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "data"
            group.attrs["title"] = "Title as group attr"
            group.attrs["axes"] = numpy.array(
                ["rows_coordinates", "columns_coordinates"], dtype=text_dtype
            )
            group.create_dataset("data", data=numpy.arange(64 * 128).reshape((64, 128)))
            group.create_dataset(
                "rows_coordinates", data=numpy.arange(64) + numpy.random.rand(64)
            )
            group.create_dataset(
                "columns_coordinates",
                data=numpy.arange(128) + 2.5 * numpy.random.rand(128),
            )

            assert nxdata.is_valid_nxdata(group)

            nxd = nxdata.NXdata(group)

            assert nxd.signal_is_2d
            assert nxd.is_image
            assert nxd.axes_dataset_names == nxd.axes_names
            assert list(nxd.axes_dataset_names) == [
                "rows_coordinates",
                "columns_coordinates",
            ]
            assert len(nxd.axes) == 2
            assert nxd.errors is None
            assert not nxd.is_scatter
            assert not nxd.is_x_y_value_scatter
            assert nxd.interpretation is None
            assert nxd.title == "Title as group attr"

    def test3DImages(self, tmp_path):
        with h5py.File(tmp_path / "nxdata_3D_images.h5", "w") as h5f:
            group = h5f.create_group("3D_images")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "images"
            ds = group.create_dataset(
                "images", data=numpy.arange(2 * 4 * 6).reshape((2, 4, 6))
            )
            ds.attrs["interpretation"] = "image"

            assert nxdata.is_valid_nxdata(group)

    def test5DImages(self, tmp_path):
        with h5py.File(tmp_path / "nxdata_5D_images.h5", "w") as h5f:
            group = h5f.create_group("5D_images")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "images"
            group.attrs["axes"] = numpy.array(
                ["rows_coordinates", "columns_coordinates"], dtype=text_dtype
            )
            ds = group.create_dataset(
                "images", data=numpy.arange(2 * 2 * 2 * 4 * 6).reshape((2, 2, 2, 4, 6))
            )
            ds.attrs["interpretation"] = "image"
            group.create_dataset("rows_coordinates", data=5 + 10 * numpy.arange(4))
            group.create_dataset(
                "columns_coordinates", data=0.5 + 0.02 * numpy.arange(6)
            )

            assert nxdata.is_valid_nxdata(group)

            nxd = nxdata.NXdata(group)

            assert nxd.is_image
            assert not nxd.signal_is_0d
            assert not nxd.signal_is_1d
            assert not nxd.signal_is_2d
            assert not nxd.signal_is_3d
            assert nxd.axes_names == [
                None,
                None,
                None,
                "rows_coordinates",
                "columns_coordinates",
            ]
            assert nxd.axes_dataset_names == [
                None,
                None,
                None,
                "rows_coordinates",
                "columns_coordinates",
            ]
            assert nxd.errors is None
            assert not nxd.is_scatter
            assert not nxd.is_x_y_value_scatter
            assert nxd.interpretation == "image"

    def testRGBAImage(self, tmp_path):
        with h5py.File(tmp_path / "nxdata_rgba_image.h5", "w") as h5f:
            group = h5f.create_group("rgba_image")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "image"
            group.attrs["axes"] = numpy.array(
                ["rows_calib", "columns_coordinates"], dtype=text_dtype
            )
            rgba_image = numpy.linspace(0, 1, num=7 * 8 * 3).reshape((7, 8, 3))
            rgba_image[:, :, 1] = (
                1 - rgba_image[:, :, 1]
            )  # invert G channel to add some color
            ds = group.create_dataset("image", data=rgba_image)
            ds.attrs["interpretation"] = "rgba-image"
            ds = group.create_dataset("rows_calib", data=(10, 5))
            ds.attrs["long_name"] = "Calibrated Y"
            group.create_dataset(
                "columns_coordinates", data=0.5 + 0.02 * numpy.arange(8)
            )

            assert nxdata.is_valid_nxdata(group)

            nxd = nxdata.NXdata(group)

            assert nxd.is_image
            assert nxd.interpretation == "rgba-image"
            assert nxd.signal_is_3d
            assert nxd.axes_names == ["Calibrated Y", "columns_coordinates", None]
            assert list(nxd.axes_dataset_names) == [
                "rows_calib",
                "columns_coordinates",
                None,
            ]

    def testXYScatter(self, tmp_path):
        with h5py.File(tmp_path / "nxdata_x_y_scatter.h5", "w") as h5f:
            group = h5f.create_group("x_y_scatter")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "y"
            group.attrs["axes"] = numpy.array(["x"], dtype=text_dtype)
            group.create_dataset("y", data=numpy.random.rand(128) - 0.5)
            group.create_dataset("x", data=2 * numpy.random.rand(128))
            group.create_dataset("x_errors", data=0.05 * numpy.random.rand(128))
            group.create_dataset("errors", data=0.05 * numpy.random.rand(128))

            assert nxdata.is_valid_nxdata(group)

            nxd = nxdata.NXdata(group)

            assert nxd.signal_is_1d
            assert nxd.axes_names == ["x"]
            assert nxd.axes_dataset_names == ["x"]
            assert nxd.errors is not None
            assert nxd.get_axis_errors("x").shape == (128,)
            assert nxd.is_scatter
            assert not nxd.is_x_y_value_scatter
            assert nxd.interpretation is None

    def testXYValueScatter(self, tmp_path):
        with h5py.File(tmp_path / "nxdata_x_y_value_scatter.h5", "w") as h5f:
            group = h5f.create_group("x_y_value_scatter")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "values"
            group.attrs["axes"] = numpy.array(["x", "y"], dtype=text_dtype)
            group.create_dataset("values", data=3.14 * numpy.random.rand(128))
            group.create_dataset("y", data=numpy.random.rand(128))
            group.create_dataset("y_errors", data=0.02 * numpy.random.rand(128))
            group.create_dataset("x", data=numpy.random.rand(128))
            group.create_dataset("x_errors", data=0.02 * numpy.random.rand(128))

            assert nxdata.is_valid_nxdata(group)

            nxd = nxdata.NXdata(group)

            assert not nxd.signal_is_1d
            assert nxd.axes_dataset_names
            assert nxd.axes_names
            assert nxd.axes_dataset_names == ["x", "y"]
            assert nxd.get_axis_errors("x").shape == (128,)
            assert nxd.get_axis_errors("y").shape == (128,)
            assert len(nxd.axes) == 2
            assert nxd.errors is None
            assert nxd.is_scatter
            assert nxd.is_x_y_value_scatter
            assert nxd.interpretation is None


class TestLegacyNXdata:
    def testSignalAttrOnDataset(self, tmp_path):
        with h5py.File(tmp_path / "nxdata.h5", "w") as h5f:
            group = h5f.create_group("2D")
            group.attrs["NX_class"] = "NXdata"

            ds0 = group.create_dataset(
                "image0", data=numpy.arange(4 * 6).reshape((4, 6))
            )
            ds0.attrs["signal"] = 1
            ds0.attrs["long_name"] = "My first image"

            ds1 = group.create_dataset(
                "image1", data=numpy.arange(4 * 6).reshape((4, 6))
            )
            ds1.attrs["signal"] = "2"
            ds1.attrs["long_name"] = "My 2nd image"

            ds2 = group.create_dataset(
                "image2", data=numpy.arange(4 * 6).reshape((4, 6))
            )
            ds2.attrs["signal"] = 3

            nxd = nxdata.NXdata(group)

            assert nxd.signal_dataset_name == "image0"
            assert nxd.signal_name == "My first image"
            assert nxd.signal.shape == (4, 6)

            assert len(nxd.auxiliary_signals) == 2
            assert nxd.auxiliary_signals[1].shape == (4, 6)

            assert nxd.auxiliary_signals_dataset_names == ["image1", "image2"]
            assert nxd.auxiliary_signals_names == ["My 2nd image", "image2"]

    def testAxesOnSignalDataset(self, tmp_path):
        with h5py.File(tmp_path / "nxdata.h5", "w") as h5f:
            group = h5f.create_group("2D")
            group.attrs["NX_class"] = "NXdata"

            ds0 = group.create_dataset(
                "image0", data=numpy.arange(4 * 6).reshape((4, 6))
            )
            ds0.attrs["signal"] = 1
            ds0.attrs["axes"] = "yaxis:xaxis"

            group.create_dataset("yaxis", data=numpy.arange(4))
            group.create_dataset("xaxis", data=numpy.arange(6))

            nxd = nxdata.NXdata(group)

            assert nxd.axes_dataset_names == ["yaxis", "xaxis"]
            assert numpy.array_equal(nxd.axes[0], numpy.arange(4))
            assert numpy.array_equal(nxd.axes[1], numpy.arange(6))

    def testAxesOnAxesDatasets(self, tmp_path):
        with h5py.File(tmp_path / "nxdata.h5", "w") as h5f:
            group = h5f.create_group("2D")
            group.attrs["NX_class"] = "NXdata"

            ds0 = group.create_dataset(
                "image0", data=numpy.arange(4 * 6).reshape((4, 6))
            )
            ds0.attrs["signal"] = 1
            ds1 = group.create_dataset("yaxis", data=numpy.arange(4))
            ds1.attrs["axis"] = 0
            ds2 = group.create_dataset("xaxis", data=numpy.arange(6))
            ds2.attrs["axis"] = "1"

            nxd = nxdata.NXdata(group)

            assert nxd.axes_dataset_names == ["yaxis", "xaxis"]
            assert numpy.array_equal(nxd.axes[0], numpy.arange(4))
            assert numpy.array_equal(nxd.axes[1], numpy.arange(6))

    def testAsciiUndefinedAxesAttrs(self, tmp_path):
        """Some files may not be using utf8 for str attrs"""
        with h5py.File(tmp_path / "nxdata.h5", "w") as h5f:
            group = h5f.create_group("bytes_attrs")
            group.attrs["NX_class"] = b"NXdata"
            group.attrs["signal"] = b"image0"
            group.attrs["axes"] = b"yaxis", b"."

            group.create_dataset("image0", data=numpy.arange(4 * 6).reshape((4, 6)))
            group.create_dataset("yaxis", data=numpy.arange(4))

            nxd = nxdata.NXdata(group)

            assert nxd.axes_dataset_names == ["yaxis", None]


class TestSaveNXdata:
    def testSimpleSave(self, tmp_path):
        filename = str(tmp_path / "nxdata_simple.h5")

        sig = numpy.array([0, 1, 2])
        a0 = numpy.array([2, 3, 4])
        a1 = numpy.array([3, 4, 5])
        nxdata.save_NXdata(
            filename=filename,
            signal=sig,
            axes=[a0, a1],
            signal_name="sig",
            axes_names=["a0", "a1"],
            nxentry_name="a",
            nxdata_name="mydata",
        )

        with h5py.File(filename, "r") as h5f:
            assert nxdata.is_valid_nxdata(h5f["a/mydata"])

            nxd = nxdata.NXdata(h5f["/a/mydata"])

            assert numpy.array_equal(nxd.signal, sig)
            assert numpy.array_equal(nxd.axes[0], a0)

    def testSimplestSave(self, tmp_path):
        filename = str(tmp_path / "nxdata_simplest.h5")

        sig = numpy.array([0, 1, 2])
        nxdata.save_NXdata(filename=filename, signal=sig)

        with h5py.File(filename, "r") as h5f:
            assert nxdata.is_valid_nxdata(h5f["/entry/data0"])

            nxd = nxdata.NXdata(h5f["/entry/data0"])

            assert numpy.array_equal(nxd.signal, sig)

    def testSaveDefaultAxesNames(self, tmp_path):
        filename = str(tmp_path / "nxdata_default_axes_names.h5")

        sig = numpy.array([0, 1, 2])
        a0 = numpy.array([2, 3, 4])
        a1 = numpy.array([3, 4, 5])
        nxdata.save_NXdata(
            filename=filename,
            signal=sig,
            axes=[a0, a1],
            signal_name="sig",
            axes_names=None,
            axes_long_names=["a", "b"],
            nxentry_name="a",
            nxdata_name="mydata",
        )

        with h5py.File(filename, "r") as h5f:
            assert nxdata.is_valid_nxdata(h5f["a/mydata"])

            nxd = nxdata.NXdata(h5f["/a/mydata"])

            assert numpy.array_equal(nxd.signal, sig)
            assert numpy.array_equal(nxd.axes[0], a0)
            assert nxd.axes_dataset_names == ["dim0", "dim1"]
            assert nxd.axes_names == ["a", "b"]

    def testSaveToExistingEntry(self, tmp_path):
        filename = str(tmp_path / "nxdata_save_to_existing_path.h5")

        with h5py.File(filename, "w") as h5f:
            group = h5f.create_group("myentry")
            group.attrs["NX_class"] = "NXentry"

        sig = numpy.array([0, 1, 2])
        a0 = numpy.array([2, 3, 4])
        a1 = numpy.array([3, 4, 5])
        nxdata.save_NXdata(
            filename=filename,
            signal=sig,
            axes=[a0, a1],
            signal_name="sig",
            axes_names=["a0", "a1"],
            nxentry_name="myentry",
            nxdata_name="toto",
        )

        with h5py.File(filename, "r") as h5f:
            assert nxdata.is_valid_nxdata(h5f["myentry/toto"])

            nxd = nxdata.NXdata(h5f["myentry/toto"])

            assert numpy.array_equal(nxd.signal, sig)
            assert numpy.array_equal(nxd.axes[0], a0)


class TestGetDefault:
    """Test silx.io.nxdata.get_default function"""

    @pytest.fixture
    def hdf5_file(self, tmp_path):
        with h5py.File(tmp_path / "test_file.h5", "w") as h5f:
            yield h5f

    def testDirectPath(self, hdf5_file):
        dicttoh5(
            {
                ("", "default"): "/nxentry/nxprocess/nxdata",
                "nxentry": {
                    "nxprocess": {
                        "nxdata": {
                            ("", "NX_class"): "NXdata",
                            ("", "signal"): "data",
                            "data": (1, 2, 3),
                        }
                    }
                },
            },
            hdf5_file,
        )
        default = nxdata.get_default(hdf5_file)
        assert isinstance(default, nxdata.NXdata)
        assert default.group.name == "/nxentry/nxprocess/nxdata"

    def testAbsolutePath(self, hdf5_file):
        dicttoh5(
            {
                ("", "default"): "/nxentry",
                "nxentry": {
                    ("", "default"): "/nxentry/nxprocess/nxdata",
                    "nxprocess": {
                        "nxdata": {
                            ("", "NX_class"): "NXdata",
                            ("", "signal"): "data",
                            "data": (1, 2, 3),
                        }
                    },
                },
            },
            hdf5_file,
        )
        default = nxdata.get_default(hdf5_file)
        assert isinstance(default, nxdata.NXdata)
        assert default.group.name == "/nxentry/nxprocess/nxdata"

    def testRelativePath(self, hdf5_file):
        dicttoh5(
            {
                ("", "default"): "nxentry",
                "nxentry": {
                    ("", "default"): "nxdata",
                    "nxdata": {
                        ("", "NX_class"): "NXdata",
                        ("", "signal"): "data",
                        "data": (1, 2, 3),
                    },
                },
            },
            hdf5_file,
        )
        default = nxdata.get_default(hdf5_file)
        assert isinstance(default, nxdata.NXdata)
        assert default.group.name == "/nxentry/nxdata"

    def testRelativePathSubdir(self, hdf5_file):
        dicttoh5(
            {
                ("", "default"): "nxentry",
                "nxentry": {
                    ("", "default"): "nxprocess/nxdata",
                    "nxprocess": {
                        "nxdata": {
                            ("", "NX_class"): "NXdata",
                            ("", "signal"): "data",
                            "data": (1, 2, 3),
                        }
                    },
                },
            },
            hdf5_file,
        )
        default = nxdata.get_default(hdf5_file)
        assert isinstance(default, nxdata.NXdata)
        assert default.group.name == "/nxentry/nxprocess/nxdata"

    def testRecursiveAbsolutePath(self, hdf5_file):
        dicttoh5(
            {
                ("", "default"): "/nxentry",
                "nxentry": {
                    ("", "default"): "/nxentry/nxprocess",
                    "nxprocess": {
                        ("", "default"): "/nxentry/nxprocess/nxdata",
                        "nxdata": {
                            ("", "NX_class"): "NXdata",
                            ("", "signal"): "data",
                            "data": (1, 2, 3),
                        },
                    },
                },
            },
            hdf5_file,
        )
        default = nxdata.get_default(hdf5_file)
        assert isinstance(default, nxdata.NXdata)
        assert default.group.name == "/nxentry/nxprocess/nxdata"

    def testRecursiveRelativePath(self, hdf5_file):
        dicttoh5(
            {
                ("", "default"): "nxentry",
                "nxentry": {
                    ("", "default"): "nxprocess",
                    "nxprocess": {
                        ("", "default"): "nxdata",
                        "nxdata": {
                            ("", "NX_class"): "NXdata",
                            ("", "signal"): "data",
                            "data": (1, 2, 3),
                        },
                    },
                },
            },
            hdf5_file,
        )
        default = nxdata.get_default(hdf5_file)
        assert isinstance(default, nxdata.NXdata)
        assert default.group.name == "/nxentry/nxprocess/nxdata"

    def testLoop(self, hdf5_file):
        """Infinite loop of @default"""
        dicttoh5(
            {
                ("", "default"): "/nxentry",
                "nxentry": {
                    ("", "default"): "/nxentry",
                },
            },
            hdf5_file,
        )
        default = nxdata.get_default(hdf5_file)
        assert default is None


def test_units(tmp_path):
    with h5py.File(tmp_path / "nx.h5", "w") as h5file:
        nxdata_grp = h5file.create_group("NXdata")
        nxdata_grp.attrs["NX_class"] = "NXdata"
        signal = nxdata_grp.create_dataset(
            "Temperature", data=numpy.random.random((10, 20))
        )
        x = nxdata_grp.create_dataset("Latitude", data=numpy.linspace(0, 1, 10))
        y = nxdata_grp.create_dataset("Longitude", data=numpy.linspace(0, 40, 20))
        nxdata_grp.attrs["signal"] = "Temperature"
        nxdata_grp.attrs["axes"] = ["Latitude", "Longitude"]
        signal.attrs["units"] = "K"
        x.attrs["units"] = "deg"
        y.attrs["units"] = "sec"

        nxd = nxdata.NXdata(nxdata_grp)
        assert nxd.signal_name == "Temperature (K)"
        assert len(nxd.axes_names) == 2
        assert nxd.axes_names[0] == "Latitude (deg)"
        assert nxd.axes_names[1] == "Longitude (sec)"
