#!/usr/bin/env python3
# /*##########################################################################
#
# Copyright (c) 2016-2020 European Synchrotron Radiation Facility
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
"""Script generating an HDF5 containing most of special structures supported by
the format.
"""

__license__ = "MIT"
__date__ = "26/03/2019"


import json
import logging
import sys

logging.basicConfig()
logger = logging.getLogger("create_h5file")


try:
    import hdf5plugin
except ImportError:
    logger.error("Backtrace", exc_info=True)

import h5py
import numpy


if sys.version_info.major < 3:
    raise RuntimeError("Python 2 is not supported")


def str_to_utf8(text):
    """Convert str or sequence of str to type compatible with h5py

    :param Union[str,List[str]] text:
    :rtype: numpy.ndarray
    """
    return numpy.array(text, dtype=h5py.special_dtype(vlen=str))


def store_subdimensions(group, data, dtype, prefix=None):
    """Creates datasets in given group with data

    :param h5py.Group group: Group where to add the datasets
    :param numpy.ndarray data: The data to use
    :param Union[numpy.dtype,str] dtype:
    :param Union[str,None] prefix: String to use as datasets name prefix
    """
    try:
        dtype = numpy.dtype(dtype)
    except TypeError as e:
        logger.error("Cannot create datasets for dtype: %s", str(dtype))
        logger.error(e)
        return

    if prefix is None:
        prefix = str(dtype)

    if hasattr(h5py, "Empty"):
        basename = prefix + "_empty"
        try:
            group[basename] = h5py.Empty(dtype=numpy.dtype(dtype))
        except (RuntimeError, ValueError) as e:
            logger.error("Error while creating %s in %s" % (basename, str(group)))
            logger.error(e)
    else:
        logger.warning("h5py.Empty not available")

    data = data.astype(dtype)
    data.shape = -1
    basename = prefix + "_d0"
    try:
        group[basename] = data[0]
    except RuntimeError as e:
        logger.error("Error while creating %s in %s" % (basename, str(group)))
        logger.error(e)

    shapes = [10, 4, 4, 4]
    for i in range(1, 4):
        shape = shapes[:i]
        shape.append(-1)
        reversed(shape)
        shape = tuple(shape)
        data.shape = shape
        basename = prefix + "_d%d" % i
        try:
            group[basename] = data
        except RuntimeError as e:
            logger.error("Error while creating %s in %s" % (basename, str(group)))
            logger.error(e)


def create_hdf5_types(group):
    print("- Creating HDF types...")

    main_group = group.create_group("HDF5")

    # H5T_INTEGER

    int_data = numpy.random.randint(-100, 100, size=10 * 4 * 4 * 4)
    uint_data = numpy.random.randint(0, 100, size=10 * 4 * 4 * 4)

    group = main_group.create_group("integer_little_endian")
    for size in (1, 2, 4, 8):
        store_subdimensions(
            group, int_data, "<i" + str(size), prefix="int" + str(size * 8)
        )
        store_subdimensions(
            group, uint_data, "<u" + str(size), prefix="uint" + str(size * 8)
        )

    group = main_group.create_group("integer_big_endian")
    for size in (1, 2, 4, 8):
        store_subdimensions(
            group, int_data, ">i" + str(size), prefix="int" + str(size * 8)
        )
        store_subdimensions(
            group, uint_data, ">u" + str(size), prefix="uint" + str(size * 8)
        )

    # H5T_FLOAT

    float_data = numpy.random.rand(10 * 4 * 4 * 4)
    group = main_group.create_group("float_little_endian")

    for size in (2, 4, 8):
        store_subdimensions(
            group, float_data, "<f" + str(size), prefix="float" + str(size * 8)
        )

    group = main_group.create_group("float_big_endian")

    for size in (2, 4, 8):
        store_subdimensions(
            group, float_data, ">f" + str(size), prefix="float" + str(size * 8)
        )

    # H5T_TIME

    main_group.create_group("time")

    # H5T_STRING

    group = main_group.create_group("text")
    group.create_dataset("ascii_vlen", data=b"abcd", dtype=h5py.string_dtype("ascii"))
    group.create_dataset(
        "bad_ascii_vlen", data=b"ab\xEFcd\xFF", dtype=h5py.string_dtype("ascii")
    )
    group.create_dataset(
        "ascii_fixed", data=b"Fixed ascii", dtype=h5py.string_dtype("ascii", 20)
    )
    group.create_dataset(
        "array_ascii_vlen",
        data=["a", "bc", "def", "ghij", "klmn"],
        dtype=h5py.string_dtype("ascii"),
    )
    group.create_dataset(
        "array_ascii_fixed",
        data=["abcde", "fghij"],
        dtype=h5py.string_dtype("ascii", 5),
    )

    group.create_dataset(
        "utf8_vlen", data="me \u2661 tu", dtype=h5py.string_dtype("utf-8")
    )
    group.create_dataset(
        "utf8_fixed",
        data="me \u2661 tu".encode("utf-8"),
        dtype=h5py.string_dtype("utf-8", 20),
    )
    group.create_dataset(
        "array_utf8_vlen",
        data=["a", "bc", "def", "ghij", "klmn"],
        dtype=h5py.string_dtype("utf-8"),
    )
    group.create_dataset(
        "array_utf8_fixed", data=["abcde", "fghij"], dtype=h5py.string_dtype("utf-8", 5)
    )

    # H5T_BITFIELD

    main_group.create_group("bitfield")

    # H5T_OPAQUE

    group = main_group.create_group("opaque")

    main_group["opaque/ascii"] = numpy.void(b"abcd")
    main_group["opaque/utf8"] = numpy.void("i \u2661 my mother".encode("utf-8"))
    main_group["opaque/thing"] = numpy.void(b"\x10\x20\x30\x40\xF0")
    main_group["opaque/big_thing"] = numpy.void(b"\x10\x20\x30\x40\xF0" * 100000)

    data = numpy.void(b"\x10\x20\x30\x40\xFF" * 20)
    data = numpy.array([data] * 10 * 4 * 4 * 4, numpy.void)
    store_subdimensions(group, data, "void")

    # H5T_COMPOUND

    a = numpy.array(
        [(1, 2.0, "Hello"), (2, 3.0, "World")],
        dtype=[("foo", "i4"), ("bar", "f4"), ("baz", "S10")],
    )

    b = numpy.zeros(3, dtype="3int8, float32, (2,3)float64")

    c = numpy.zeros(
        3, dtype=("i4", [("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")])
    )

    d = numpy.zeros(
        3, dtype=[("x", "f4"), ("y", numpy.float32), ("value", "f4", (2, 2))]
    )

    e = numpy.zeros(3, dtype={"names": ["col1", "col2"], "formats": ["i4", "f4"]})

    f = numpy.array(
        [(1.5, 2.5, (1.0, 2.0)), (3.0, 4.0, (4.0, 5.0)), (1.0, 3.0, (2.0, 6.0))],
        dtype=[("x", "f4"), ("y", numpy.float32), ("value", "f4", (2, 2))],
    )

    main_group["compound/numpy_example_a"] = a
    main_group["compound/numpy_example_b"] = b
    main_group["compound/numpy_example_c"] = c
    main_group["compound/numpy_example_d"] = d
    main_group["compound/numpy_example_e"] = e
    main_group["compound/numpy_example_f"] = f

    dt = numpy.dtype([("start", numpy.uint32), ("stop", numpy.uint32)])
    vlen_dt = h5py.special_dtype(vlen=dt)
    data = numpy.array([[(1, 2), (2, 3)], [(3, 5), (5, 8), (8, 9)]], vlen_dt)
    dataset = main_group.create_dataset("compound/vlen", data.shape, data.dtype)
    for i, row in enumerate(data):
        dataset[i] = row

    # numpy complex is a H5T_COMPOUND

    real_data = numpy.random.rand(10 * 4 * 4 * 4)
    imaginary_data = numpy.random.rand(10 * 4 * 4 * 4)
    complex_data = real_data + imaginary_data * 1j

    group = main_group.create_group("compound/numpy_complex_little_endian")
    for size in (8, 16, 32):
        store_subdimensions(
            group, complex_data, "<c" + str(size), prefix="complex" + str(size * 8)
        )

    group = main_group.create_group("compound/numpy_complex_big_endian")
    for size in (8, 16, 32):
        store_subdimensions(
            group, complex_data, ">c" + str(size), prefix="complex" + str(size * 8)
        )

    # H5T_REFERENCE

    ref_dt = h5py.special_dtype(ref=h5py.Reference)
    group = main_group.create_group("reference")
    data = group.create_dataset("data", data=numpy.random.rand(10, 10))

    group.create_dataset("ref_0d", data=data.ref, dtype=ref_dt)
    group.create_dataset("ref_1d", data=[data.ref, None], dtype=ref_dt)
    group.create_dataset("regionref_0d", data=data.regionref[0:10, 0:5], dtype=ref_dt)
    group.create_dataset("regionref_1d", data=[data.regionref[0:10, 0:5]], dtype=ref_dt)

    # H5T_ENUM

    enum_dt = h5py.special_dtype(enum=("i", {"RED": 0, "GREEN": 1, "BLUE": 42}))
    group = main_group.create_group("enum")
    uint_data = numpy.random.randint(0, 100, size=10 * 4 * 4 * 4)
    uint_data.shape = 10, 4, 4, 4

    group.create_dataset("color_0d", data=numpy.array(42, dtype=enum_dt))
    group.create_dataset("color_1d", data=numpy.array([0, 1, 100, 42], dtype=enum_dt))
    group.create_dataset("color_4d", data=numpy.array(uint_data, dtype=enum_dt))

    # numpy bool is a H5T_ENUM

    bool_data = uint_data < 50
    bool_group = main_group.create_group("enum/numpy_boolean")
    store_subdimensions(bool_group, bool_data, "bool")

    # H5T_VLEN

    group = main_group.create_group("vlen")
    text = "i \u2661 my dad"

    unicode_vlen_dt = h5py.special_dtype(vlen=str)
    group.create_dataset("unicode", data=numpy.array(text, dtype=unicode_vlen_dt))
    group.create_dataset("unicode_1d", data=numpy.array([text], dtype=unicode_vlen_dt))


def set_silx_style(group, axes_scales=None, signal_scale=None):
    """Set SILX_style attribute

    :param group: NXdata group to set SILX_style for.
    :param axes_scales: Scale to use for the axes.
        Used for plot axis scales.
    :param signal_scale: Scale to use for the signal.
        Used either for plot axis scale or colormap normalization.
    """
    style = {}
    if axes_scales is not None:
        style["axes_scale_types"] = axes_scales
    if signal_scale is not None:
        style["signal_scale_type"] = signal_scale
    if style:
        group.attrs["SILX_style"] = json.dumps(style)


def create_nxdata_group(group):
    print("- Creating NXdata types...")

    main_group = group.create_group("NxData")

    # SCALARS
    g0d = main_group.create_group("scalars")

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
    ds = g0d1.create_dataset(
        "scalars", data=numpy.arange(2 * 2 * 3 * 10).reshape((2, 2, 3, 10))
    )
    ds.attrs["interpretation"] = "scalar"

    # SPECTRA
    g1d = main_group.create_group("spectra")

    g1d0 = g1d.create_group("1D_spectrum")
    g1d0.attrs["NX_class"] = "NXdata"
    g1d0.attrs["signal"] = "count"
    g1d0.attrs["axes"] = "energy_calib"
    g1d0.attrs["uncertainties"] = str_to_utf8(("energy_errors",))
    g1d0.create_dataset("count", data=numpy.arange(10))
    g1d0.create_dataset("energy_calib", data=(10, 5))  # 10 * idx + 5
    g1d0.create_dataset("energy_errors", data=3.14 * numpy.random.rand(10))
    set_silx_style(g1d0, axes_scales=["linear"], signal_scale="log")

    g1d1 = g1d.create_group("2D_spectra")
    g1d1.attrs["NX_class"] = "NXdata"
    g1d1.attrs["signal"] = "counts"
    ds = g1d1.create_dataset("counts", data=numpy.arange(3 * 10).reshape((3, 10)))
    ds.attrs["interpretation"] = "spectrum"
    set_silx_style(g1d1, axes_scales=["linear"], signal_scale="log")

    g1d2 = g1d.create_group("4D_spectra")
    g1d2.attrs["NX_class"] = "NXdata"
    g1d2.attrs["signal"] = "counts"
    g1d2.attrs["axes"] = str_to_utf8(("energy",))
    ds = g1d2.create_dataset(
        "counts", data=numpy.arange(2 * 2 * 3 * 10).reshape((2, 2, 3, 10))
    )
    ds.attrs["interpretation"] = "spectrum"
    ds = g1d2.create_dataset("errors", data=4.5 * numpy.random.rand(2, 2, 3, 10))
    ds = g1d2.create_dataset(
        "energy", data=5 + 10 * numpy.arange(15), shuffle=True, compression="gzip"
    )
    ds.attrs["long_name"] = "Calibrated energy"
    ds.attrs["first_good"] = 3
    ds.attrs["last_good"] = 12
    g1d2.create_dataset("energy_errors", data=10 * numpy.random.rand(15))
    set_silx_style(g1d2, axes_scales=["linear"], signal_scale="log")

    # IMAGES
    g2d = main_group.create_group("images")

    g2d0 = g2d.create_group("2D_regular_image")
    g2d0.attrs["NX_class"] = "NXdata"
    g2d0.attrs["signal"] = "image"
    g2d0.attrs["axes"] = str_to_utf8(("rows_calib", "columns_coordinates"))
    g2d0.create_dataset("image", data=numpy.arange(4 * 6).reshape((4, 6)))
    ds = g2d0.create_dataset("rows_calib", data=(10, 5))
    ds.attrs["long_name"] = "Calibrated Y"
    g2d0.create_dataset("columns_coordinates", data=0.5 + 0.02 * numpy.arange(6))
    set_silx_style(g1d2, axes_scales=["linear"], signal_scale="log")

    g2d1 = g2d.create_group("2D_irregular_data")
    g2d1.attrs["NX_class"] = "NXdata"
    g2d1.attrs["signal"] = "data"
    g2d1.attrs["axes"] = str_to_utf8(("rows_coordinates", "columns_coordinates"))
    g2d1.create_dataset("data", data=numpy.arange(64 * 128).reshape((64, 128)))
    g2d1.create_dataset(
        "rows_coordinates", data=numpy.arange(64) + numpy.random.rand(64)
    )
    g2d1.create_dataset(
        "columns_coordinates", data=numpy.arange(128) + 2.5 * numpy.random.rand(128)
    )
    set_silx_style(g2d1, axes_scales=["linear", "log"], signal_scale="log")

    g2d2 = g2d.create_group("3D_images")
    g2d2.attrs["NX_class"] = "NXdata"
    g2d2.attrs["signal"] = "images"
    ds = g2d2.create_dataset("images", data=numpy.arange(2 * 4 * 6).reshape((2, 4, 6)))
    ds.attrs["interpretation"] = "image"
    set_silx_style(g2d2, signal_scale="log")

    g2d3 = g2d.create_group("5D_images")
    g2d3.attrs["NX_class"] = "NXdata"
    g2d3.attrs["signal"] = "images"
    g2d3.attrs["axes"] = str_to_utf8(("rows_coordinates", "columns_coordinates"))
    ds = g2d3.create_dataset(
        "images", data=numpy.arange(2 * 2 * 2 * 4 * 6).reshape((2, 2, 2, 4, 6))
    )
    ds.attrs["interpretation"] = "image"
    g2d3.create_dataset("rows_coordinates", data=5 + 10 * numpy.arange(4))
    g2d3.create_dataset("columns_coordinates", data=0.5 + 0.02 * numpy.arange(6))
    set_silx_style(g2d3, signal_scale="log")

    y = numpy.arange(-5, 10).reshape(-1, 1)
    x = numpy.arange(-5, 10).reshape(1, -1)
    data = x**2 + y**2
    data = data + (data.max() - data) * 1j

    g2d3 = g2d.create_group("2D_complex_image")
    g2d3.attrs["NX_class"] = "NXdata"
    g2d3.attrs["signal"] = "image"
    g2d3.attrs["axes"] = str_to_utf8(("rows", "columns"))
    g2d3.create_dataset("image", data=data)
    g2d3.create_dataset("rows", data=0.5 + 0.02 * numpy.arange(data.shape[0]))
    g2d3.create_dataset("columns", data=0.5 + 0.02 * numpy.arange(data.shape[1]))
    set_silx_style(g2d3, signal_scale="log")

    # SCATTER
    g = main_group.create_group("scatters")

    gd0 = g.create_group("x_y_scatter")
    gd0.attrs["NX_class"] = "NXdata"
    gd0.attrs["signal"] = "y"
    gd0.attrs["axes"] = str_to_utf8(("x",))
    gd0.create_dataset("y", data=numpy.random.rand(128) - 0.5)
    gd0.create_dataset("x", data=2 * numpy.random.rand(128))
    gd0.create_dataset("x_errors", data=0.05 * numpy.random.rand(128))
    gd0.create_dataset("errors", data=0.05 * numpy.random.rand(128))
    set_silx_style(gd0, axes_scales=["linear"], signal_scale="log")

    gd1 = g.create_group("x_y_value_scatter")
    gd1.attrs["NX_class"] = "NXdata"
    gd1.attrs["signal"] = "values"
    gd1.attrs["axes"] = str_to_utf8(("x", "y"))
    gd1.create_dataset("values", data=3.14 * numpy.random.rand(128))
    gd1.create_dataset("y", data=numpy.random.rand(128))
    gd1.create_dataset("y_errors", data=0.02 * numpy.random.rand(128))
    gd1.create_dataset("x", data=numpy.random.rand(128))
    gd1.create_dataset("x_errors", data=0.02 * numpy.random.rand(128))
    set_silx_style(gd1, axes_scales=["linear", "log"], signal_scale="log")

    # NDIM > 3
    g = main_group.create_group("cubes")

    gd0 = g.create_group("3D_cube")
    gd0.attrs["NX_class"] = "NXdata"
    gd0.attrs["signal"] = "cube"
    gd0.attrs["axes"] = str_to_utf8(("img_idx", "rows_coordinates", "cols_coordinates"))
    gd0.create_dataset("cube", data=numpy.arange(4 * 5 * 6).reshape((4, 5, 6)))
    gd0.create_dataset("img_idx", data=numpy.arange(4))
    gd0.create_dataset("rows_coordinates", data=0.1 * numpy.arange(5))
    gd0.create_dataset("cols_coordinates", data=[0.2, 0.3])  # linear calibration
    set_silx_style(gd0, axes_scales=["log", "linear"], signal_scale="log")

    gd1 = g.create_group("5D")
    gd1.attrs["NX_class"] = "NXdata"
    gd1.attrs["signal"] = "hypercube"
    data = numpy.arange(2 * 3 * 4 * 5 * 6).reshape((2, 3, 4, 5, 6))
    gd1.create_dataset("hypercube", data=data)
    set_silx_style(gd1, axes_scales=["log", "linear"], signal_scale="log")


FILTER_LZF = 32000
FILTER_LZ4 = 32004
FILTER_CBF = 32006
FILTER_BITSHUFFLE = 32008
FILTER_BITSHUFFLE_COMPRESS_LZ4 = 2
FILTER_USER = 32768
"""First id for non-distributed filter"""

encoded_data = [
    (
        "lzf",
        {"compression": FILTER_LZF},
        b"\x01\x00\x00\x80\x00\x00\x01\x80\x06\x01\x00\x02\xa0\x07\x00\x03\xa0\x07"
        b"\x00\x04\xa0\x07\x00\x05\xa0\x07\x00\x06\xa0\x07\x00\x07\xa0\x07\x00\x08"
        b"\xa0\x07\x00\t\xa0\x07\x00\n\xa0\x07\x00\x0b\xa0\x07\x00\x0c\xa0\x07\x00"
        b"\r\xa0\x07\x00\x0e\xa0\x07\x00\x0f`\x07\x01\x00\x00",
    ),
    (
        "bitshuffle+lz4",
        {
            "compression": FILTER_BITSHUFFLE,
            "compression_opts": (0, FILTER_BITSHUFFLE_COMPRESS_LZ4),
        },
        b"\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00 \x00\x00\x00\x00\x13\x9f\xaa\xaa"
        b"\xcc\xcc\xf0\xf0\x00\xff\x00\x01\x00_P\x00\x00\x00\x00\x00",
    ),
    ("cbf", {"compression": FILTER_CBF}, None),
    (
        "lz4",
        {"compression": FILTER_LZ4},
        b"\x00\x00\x00\x00\x00\x00\x00\x80\x00\x00\x00\x80\x00\x00\x00E\x13\x00\x01"
        b"\x00\x13\x01\x08\x00\x13\x02\x08\x00\x13\x03\x08\x00\x13\x04\x08\x00\x13"
        b"\x05\x08\x00\x13\x06\x08\x00\x13\x07\x08\x00\x13\x08\x08\x00\x13\t\x08"
        b"\x00\x13\n\x08\x00\x13\x0b\x08\x00\x13\x0c\x08\x00\x13\r\x08\x00\x13\x0e"
        b"\x08\x00\x80\x0f\x00\x00\x00\x00\x00\x00\x00",
    ),
    (
        "corrupted_datasets/bitshuffle+lz4",
        {
            "compression": FILTER_BITSHUFFLE,
            "compression_opts": (0, FILTER_BITSHUFFLE_COMPRESS_LZ4),
        },
        b"\xFF\x01\x00\x01" * 10,
    ),
    (
        "corrupted_datasets/unavailable_filter",
        {"compression": FILTER_USER + 100},
        b"\xFF\x01\x00\x01" * 10,
    ),
]


def display_encoded_data():
    reference = numpy.arange(16, dtype=int).reshape(4, 4)

    import os
    import tempfile

    tempdir = tempfile.mkdtemp()
    filename = os.path.join(tempdir, "encode.h5")
    for info in encoded_data:
        name, compression_args, stored_data = info
        print("m" * 20)
        print(name)
        try:
            with h5py.File(filename, "a") as h5:
                dataset = h5.create_dataset(
                    name, data=reference, chunks=reference.shape, **compression_args
                )
                offsets = (0,) * len(reference.shape)
                filter_mask = [0xFFFF]
                data = dataset.id.read_direct_chunk(
                    offsets=offsets, filter_mask=filter_mask
                )
            if data != stored_data:
                print("- Data changed")
            print("- Data:")
            print(data)
        except Exception:
            logger.error("Backtrace", exc_info=True)


def create_encoded_data(parent_group):
    print("- Creating encoded data...")

    group = parent_group.create_group("encoded")

    reference = numpy.arange(16).reshape(4, 4)
    group["uncompressed"] = reference

    group.create_dataset("zipped", data=reference, compression="gzip")
    group.create_dataset("zipped_max", data=reference, compression="gzip")

    compressions = ["szip"]
    for compression in compressions:
        try:
            group.create_dataset(compression, data=reference, compression=compression)
        except Exception:
            logger.warning(
                "Filter '%s' is not available. Dataset skipped.", compression
            )

    group.create_dataset("shuffle_filter", data=reference, shuffle=True)
    group.create_dataset("fletcher32_filter", data=reference, fletcher32=True)

    for info in encoded_data:
        name, compression_args, data = info
        if data is None:
            logger.warning(
                "No compressed data for dataset '%s'. Dataset skipped.", name
            )
            continue
        try:
            dataset = group.create_dataset(
                name,
                shape=reference.shape,
                maxshape=reference.shape,
                dtype=reference.dtype,
                chunks=reference.shape,
                **compression_args,
            )
        except Exception:
            logger.debug("Backtrace", exc_info=True)
            logger.error("Error while creating dataset '%s'", name)
            continue
        try:
            offsets = (0,) * len(reference.shape)
            dataset.id.write_direct_chunk(
                offsets=offsets, data=data, filter_mask=0x0000
            )
        except Exception:
            logger.debug("Backtrace", exc_info=True)
            logger.error("Error filling dataset '%s'", name)
            continue


def create_links(h5):
    print("- Creating links...")

    main_group = h5.create_group("links")
    dataset = main_group.create_dataset("dataset", data=numpy.int64(10))
    group = main_group.create_group("group")
    group["something_inside"] = numpy.int64(20)

    main_group["hard_link_to_group"] = main_group["group"]
    main_group["hard_link_to_dataset"] = main_group["dataset"]

    main_group.create_group("hard_recursive_link")
    main_group.create_group("hard_recursive_link2")
    main_group["hard_recursive_link/link"] = main_group["hard_recursive_link2"]
    main_group["hard_recursive_link2/link"] = main_group["hard_recursive_link"]

    main_group["soft_link_to_group"] = h5py.SoftLink(group.name)
    main_group["soft_link_to_dataset"] = h5py.SoftLink(dataset.name)
    main_group["soft_link_to_nothing"] = h5py.SoftLink("/foo/bar/2000")
    main_group["soft_link_to_group_link"] = h5py.SoftLink(
        main_group.name + "/soft_link_to_group"
    )
    main_group["soft_link_to_dataset_link"] = h5py.SoftLink(
        main_group.name + "/soft_link_to_dataset"
    )
    main_group["soft_link_to_itself"] = h5py.SoftLink(
        main_group.name + "/soft_link_to_itself"
    )

    # External links to self file
    main_group["external_link_to_group"] = h5py.ExternalLink(
        h5.file.filename, group.name
    )
    main_group["external_link_to_dataset"] = h5py.ExternalLink(
        h5.file.filename, dataset.name
    )
    main_group["external_link_to_nothing"] = h5py.ExternalLink(
        h5.file.filename, "/foo/bar/2000"
    )
    main_group["external_link_to_missing_file"] = h5py.ExternalLink(
        h5.file.filename + "_unknown", "/"
    )
    main_group["external_link_to_group_link"] = h5py.ExternalLink(
        h5.file.filename, main_group.name + "/soft_link_to_group"
    )
    main_group["external_link_to_dataset_link"] = h5py.ExternalLink(
        h5.file.filename, main_group.name + "/soft_link_to_dataset"
    )
    main_group["external_link_to_itself"] = h5py.ExternalLink(
        h5.file.filename, main_group.name + "/external_link_to_itself"
    )
    main_group["external_link_to_recursive_link2"] = h5py.ExternalLink(
        h5.file.filename, main_group.name + "/external_link_to_recursive_link3"
    )
    main_group["external_link_to_recursive_link3"] = h5py.ExternalLink(
        h5.file.filename, main_group.name + "/external_link_to_recursive_link2"
    )
    main_group["external_link_to_soft_recursive"] = h5py.ExternalLink(
        h5.file.filename, main_group.name + "/soft_link_to_itself"
    )


def create_image_types(h5):
    print("- Creating images...")
    raw = numpy.arange(10 * 10).reshape((10, 10))
    u8 = (raw * 255 / raw.max()).astype(numpy.uint8)
    f32 = (raw / raw.max()).astype(numpy.float32)
    f64 = (raw / raw.max()).astype(numpy.float64)

    u8_t = numpy.swapaxes(u8, 0, 1)
    f32_t = numpy.swapaxes(f32, 0, 1)
    f64_t = numpy.swapaxes(f64, 0, 1)

    main_group = h5.create_group("images")

    main_group["intensity_uint8"] = u8
    main_group["intensity_float32"] = f32
    main_group["intensity_float64"] = f64
    main_group["rgb_uint8"] = numpy.stack((u8, u8_t, numpy.flip(u8, 0))).T
    main_group["rgb_float32"] = numpy.stack((f32, f32_t, numpy.flip(f32, 0))).T
    main_group["rgb_float64"] = numpy.stack((f64, f64_t, numpy.flip(f64, 0))).T
    main_group["rgba_uint8"] = numpy.stack(
        (u8, u8_t, numpy.flip(u8, 0), numpy.flip(u8_t))
    ).T
    main_group["rgba_float32"] = numpy.stack(
        (f32, f32_t, numpy.flip(f32, 0), numpy.flip(f32_t))
    ).T
    main_group["rgba_float64"] = numpy.stack(
        (f64, f64_t, numpy.flip(f64, 0), numpy.flip(f64_t))
    ).T


def create_file():
    filename = "all_types.h5"
    print("Creating file '%s'..." % filename)
    with h5py.File(filename, "w") as h5:
        create_hdf5_types(h5)
        create_nxdata_group(h5)
        create_encoded_data(h5)
        create_links(h5)
        create_image_types(h5)


def main():
    print("Begin")
    create_file()
    # display_encoded_data()
    print("End")


if __name__ == "__main__":
    main()
