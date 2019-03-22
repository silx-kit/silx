#!/usr/bin/python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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
__date__ = "22/03/2019"


import numpy
import h5py
import six
import logging

logging.basicConfig()
logger = logging.getLogger("create_h5file")


def store_subdimentions(group, data, dtype):

    if hasattr(h5py, "Empty"):
        basename = str(dtype) + "_empty"
        try:
            group[basename] = h5py.Empty(dtype=numpy.dtype(dtype))
        except (RuntimeError, ValueError) as e:
            logger.error("Error while creating %s" % basename)
            logger.error(e)
    else:
        logger.warning("h5py.Empty not available")

    data = data.astype(dtype)
    data.shape = -1
    basename = str(dtype) + "_d0"
    try:
        group[basename] = data[0]
    except RuntimeError as e:
        logger.error("Error while creating %s" % basename)
        logger.error(e)

    shapes = [10, 4, 4, 4]
    for i in range(1, 4):
        shape = shapes[:i]
        shape.append(-1)
        reversed(shape)
        shape = tuple(shape)
        data.shape = shape
        basename = str(dtype) + "_d%d" % i
        try:
            group[basename] = data
        except RuntimeError as e:
            logger.error("Error while creating %s" % basename)
            logger.error(e)


def create_hdf5_types(group):
    print("- Creating HDF types...")

    main_group = group.create_group("HDF5")

    # H5T_INTEGER

    int_data = numpy.random.randint(-100, 100, size=10 * 4 * 4 * 4)
    uint_data = numpy.random.randint(0, 100, size=10 * 4 * 4 * 4)
    group = main_group.create_group("integer")

    store_subdimentions(group, int_data, "int8")
    store_subdimentions(group, int_data, "int16")
    store_subdimentions(group, int_data, "int32")
    store_subdimentions(group, int_data, "int64")
    store_subdimentions(group, uint_data, "uint8")
    store_subdimentions(group, uint_data, "uint16")
    store_subdimentions(group, uint_data, "uint32")
    store_subdimentions(group, uint_data, "uint64")
    store_subdimentions(group, int_data, numpy.dtype(">i4"))
    store_subdimentions(group, int_data, numpy.dtype("<i4"))

    # H5T_FLOAT

    float_data = numpy.random.rand(10 * 4 * 4 * 4)
    group = main_group.create_group("float")

    store_subdimentions(group, float_data, "float16")
    store_subdimentions(group, float_data, "float32")
    store_subdimentions(group, float_data, "float64")
    store_subdimentions(group, float_data, "float128")
    store_subdimentions(group, float_data, ">f4")
    store_subdimentions(group, float_data, "<f4")

    # H5T_TIME

    main_group.create_group("time")

    # H5T_STRING

    main_group["text/ascii"] = b"abcd"
    main_group["text/bad_ascii"] = b"ab\xEFcd\xFF"
    main_group["text/utf8"] = u"me \u2661 tu"

    # H5T_BITFIELD

    main_group.create_group("bitfield")

    # H5T_OPAQUE

    group = main_group.create_group("opaque")

    main_group["opaque/ascii"] = numpy.void(b"abcd")
    main_group["opaque/utf8"] = numpy.void(u"i \u2661 my mother".encode("utf-8"))
    main_group["opaque/thing"] = numpy.void(b"\x10\x20\x30\x40\xF0")
    main_group["opaque/big_thing"] = numpy.void(b"\x10\x20\x30\x40\xF0" * 100000)

    data = numpy.void(b"\x10\x20\x30\x40\xFF" * 20)
    data = numpy.array([data] * 10 * 4 * 4 * 4, numpy.void)
    store_subdimentions(group, data, "void")

    # H5T_COMPOUND

    a = numpy.array([(1, 2., 'Hello'), (2, 3., "World")],
                    dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'S10')])

    b = numpy.zeros(3, dtype='3int8, float32, (2,3)float64')

    c = numpy.zeros(3, dtype=('i4', [('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')]))

    d = numpy.zeros(3, dtype=[('x', 'f4'), ('y', numpy.float32), ('value', 'f4', (2, 2))])

    e = numpy.zeros(3, dtype={'names': ['col1', 'col2'], 'formats': ['i4', 'f4']})

    f = numpy.array([(1.5, 2.5, (1.0, 2.0)), (3., 4., (4., 5.)), (1., 3., (2., 6.))],
                    dtype=[('x', 'f4'), ('y', numpy.float32), ('value', 'f4', (2, 2))])

    main_group["compound/numpy_example_a"] = a
    main_group["compound/numpy_example_b"] = b
    main_group["compound/numpy_example_c"] = c
    main_group["compound/numpy_example_d"] = d
    main_group["compound/numpy_example_e"] = e
    main_group["compound/numpy_example_f"] = f

    dt = numpy.dtype([('start', numpy.uint32), ('stop', numpy.uint32)])
    vlen_dt = h5py.special_dtype(vlen=dt)
    data = numpy.array([[(1, 2), (2, 3)], [(3, 5), (5, 8), (8, 9)]], vlen_dt)
    dataset = main_group.create_dataset("compound/vlen", data.shape, data.dtype)
    for i, row in enumerate(data):
        dataset[i] = row

    # numpy complex is a H5T_COMPOUND

    complex_group = main_group.create_group("compound/numpy_complex")

    real_data = numpy.random.rand(10 * 4 * 4 * 4)
    imaginary_data = numpy.random.rand(10 * 4 * 4 * 4)
    complex_data = real_data + imaginary_data * 1j

    store_subdimentions(complex_group, complex_data, "complex64")
    store_subdimentions(complex_group, complex_data, "complex128")
    store_subdimentions(complex_group, complex_data, "complex256")
    store_subdimentions(complex_group, complex_data, ">c8")
    store_subdimentions(complex_group, complex_data, "<c8")

    # H5T_REFERENCE

    ref_dt = h5py.special_dtype(ref=h5py.Reference)
    group = main_group.create_group("reference")
    data = group.create_dataset("data", data=numpy.random.rand(10, 10))

    group.create_dataset("ref_0d", data=data.ref, dtype=ref_dt)
    group.create_dataset("ref_1d", data=[data.ref, None], dtype=ref_dt)
    group.create_dataset("regionref_0d", data=data.regionref[0:10, 0:5], dtype=ref_dt)
    group.create_dataset("regionref_1d", data=[data.regionref[0:10, 0:5]], dtype=ref_dt)

    # H5T_ENUM

    enum_dt = h5py.special_dtype(enum=('i', {"RED": 0, "GREEN": 1, "BLUE": 42}))
    group = main_group.create_group("enum")
    uint_data = numpy.random.randint(0, 100, size=10 * 4 * 4 * 4)
    uint_data.shape = 10, 4, 4, 4

    group.create_dataset("color_0d", data=numpy.array(42, dtype=enum_dt))
    group.create_dataset("color_1d", data=numpy.array([0, 1, 100, 42], dtype=enum_dt))
    group.create_dataset("color_4d", data=numpy.array(uint_data, dtype=enum_dt))

    # numpy bool is a H5T_ENUM

    bool_data = uint_data < 50
    bool_group = main_group.create_group("enum/numpy_boolean")
    store_subdimentions(bool_group, bool_data, "bool")

    # H5T_VLEN

    group = main_group.create_group("vlen")
    text = u"i \u2661 my dad"

    unicode_vlen_dt = h5py.special_dtype(vlen=six.text_type)
    group.create_dataset("unicode", data=numpy.array(text, dtype=unicode_vlen_dt))
    group.create_dataset("unicode_1d", data=numpy.array([text], dtype=unicode_vlen_dt))


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
    ds = g0d1.create_dataset("scalars", data=numpy.arange(2 * 2 * 3 * 10).reshape((2, 2, 3, 10)))
    ds.attrs["interpretation"] = "scalar"

    # SPECTRA
    g1d = main_group.create_group("spectra")

    g1d0 = g1d.create_group("1D_spectrum")
    g1d0.attrs["NX_class"] = "NXdata"
    g1d0.attrs["signal"] = "count"
    g1d0.attrs["axes"] = "energy_calib"
    g1d0.attrs["uncertainties"] = b"energy_errors",
    g1d0.create_dataset("count", data=numpy.arange(10))
    g1d0.create_dataset("energy_calib", data=(10, 5))     # 10 * idx + 5
    g1d0.create_dataset("energy_errors", data=3.14 * numpy.random.rand(10))

    g1d1 = g1d.create_group("2D_spectra")
    g1d1.attrs["NX_class"] = "NXdata"
    g1d1.attrs["signal"] = "counts"
    ds = g1d1.create_dataset("counts", data=numpy.arange(3 * 10).reshape((3, 10)))
    ds.attrs["interpretation"] = "spectrum"

    g1d2 = g1d.create_group("4D_spectra")
    g1d2.attrs["NX_class"] = "NXdata"
    g1d2.attrs["signal"] = "counts"
    g1d2.attrs["axes"] = b"energy",
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
    g2d = main_group.create_group("images")

    g2d0 = g2d.create_group("2D_regular_image")
    g2d0.attrs["NX_class"] = "NXdata"
    g2d0.attrs["signal"] = "image"
    g2d0.attrs["axes"] = b"rows_calib", b"columns_coordinates"
    g2d0.create_dataset("image", data=numpy.arange(4 * 6).reshape((4, 6)))
    ds = g2d0.create_dataset("rows_calib", data=(10, 5))
    ds.attrs["long_name"] = "Calibrated Y"
    g2d0.create_dataset("columns_coordinates", data=0.5 + 0.02 * numpy.arange(6))

    g2d1 = g2d.create_group("2D_irregular_data")
    g2d1.attrs["NX_class"] = "NXdata"
    g2d1.attrs["signal"] = "data"
    g2d1.attrs["axes"] = b"rows_coordinates", b"columns_coordinates"
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
    g2d3.attrs["axes"] = b"rows_coordinates", b"columns_coordinates"
    ds = g2d3.create_dataset("images", data=numpy.arange(2 * 2 * 2 * 4 * 6).reshape((2, 2, 2, 4, 6)))
    ds.attrs["interpretation"] = "image"
    g2d3.create_dataset("rows_coordinates", data=5 + 10 * numpy.arange(4))
    g2d3.create_dataset("columns_coordinates", data=0.5 + 0.02 * numpy.arange(6))

    y = numpy.arange(-5, 10).reshape(-1, 1)
    x = numpy.arange(-5, 10).reshape(1, -1)
    data = (x**2 + y**2)
    data = data + (data.max() - data) * 1j

    g2d3 = g2d.create_group("2D_complex_image")
    g2d3.attrs["NX_class"] = "NXdata"
    g2d3.attrs["signal"] = "image"
    g2d3.attrs["axes"] = b"rows", b"columns"
    g2d3.create_dataset("image", data=data)
    g2d3.create_dataset("rows", data=0.5 + 0.02 * numpy.arange(data.shape[0]))
    g2d3.create_dataset("columns", data=0.5 + 0.02 * numpy.arange(data.shape[1]))

    # SCATTER
    g = main_group.create_group("scatters")

    gd0 = g.create_group("x_y_scatter")
    gd0.attrs["NX_class"] = "NXdata"
    gd0.attrs["signal"] = "y"
    gd0.attrs["axes"] = b"x",
    gd0.create_dataset("y", data=numpy.random.rand(128) - 0.5)
    gd0.create_dataset("x", data=2 * numpy.random.rand(128))
    gd0.create_dataset("x_errors", data=0.05 * numpy.random.rand(128))
    gd0.create_dataset("errors", data=0.05 * numpy.random.rand(128))

    gd1 = g.create_group("x_y_value_scatter")
    gd1.attrs["NX_class"] = "NXdata"
    gd1.attrs["signal"] = "values"
    gd1.attrs["axes"] = b"x", b"y"
    gd1.create_dataset("values", data=3.14 * numpy.random.rand(128))
    gd1.create_dataset("y", data=numpy.random.rand(128))
    gd1.create_dataset("y_errors", data=0.02 * numpy.random.rand(128))
    gd1.create_dataset("x", data=numpy.random.rand(128))
    gd1.create_dataset("x_errors", data=0.02 * numpy.random.rand(128))

    # NDIM > 3
    g = main_group.create_group("cubes")

    gd0 = g.create_group("3D_cube")
    gd0.attrs["NX_class"] = "NXdata"
    gd0.attrs["signal"] = "cube"
    gd0.attrs["axes"] = b"img_idx", b"rows_coordinates", b"cols_coordinates"
    gd0.create_dataset("cube", data=numpy.arange(4 * 5 * 6).reshape((4, 5, 6)))
    gd0.create_dataset("img_idx", data=numpy.arange(4))
    gd0.create_dataset("rows_coordinates", data=0.1 * numpy.arange(5))
    gd0.create_dataset("cols_coordinates", data=[0.2, 0.3])  # linear calibration

    gd1 = g.create_group("5D")
    gd1.attrs["NX_class"] = "NXdata"
    gd1.attrs["signal"] = "hypercube"
    data = numpy.arange(2 * 3 * 4 * 5 * 6).reshape((2, 3, 4, 5, 6))
    gd1.create_dataset("hypercube", data=data)


def create_all_types2():
    filename = "all_types.h5"
    print("Creating file '%s'..." % filename)
    with h5py.File(filename, "w") as h5:
        create_hdf5_types(h5)
        create_nxdata_group(h5)


def create_all_types():
    with h5py.File("../types.h5", "w") as h5:
        g = h5.create_group("arrays")
        g.create_dataset("scalar", data=10)
        g.create_dataset("list", data=[10])
        g.create_dataset("image", data=[[10]])
        g.create_dataset("cube", data=[[[10]]])
        g.create_dataset("hypercube", data=[[[[10]]]])

        g = h5.create_group("dtypes")
        g.create_dataset("int32", data=numpy.int32(10))
        g.create_dataset("int64", data=numpy.int64(10))
        g.create_dataset("float32", data=numpy.float32(10))
        g.create_dataset("float64", data=numpy.float64(10))
        g.create_dataset("string_", data=numpy.string_("Hi!"))
        # g.create_dataset("string0",data=numpy.string0("Hi!\x00"))
        g.create_dataset("bool", data=True)
        g.create_dataset("bool2", data=False)


def create_all_links():
    with h5py.File("../links.h5", "w") as h5:
        g = h5.create_group("group")
        g.create_dataset("dataset", data=numpy.int64(10))
        h5.create_dataset("dataset", data=numpy.int64(10))

        h5["hard_link_to_group"] = h5["/group"]
        h5["hard_link_to_dataset"] = h5["/dataset"]

        h5["soft_link_to_group"] = h5py.SoftLink("/group")
        h5["soft_link_to_dataset"] = h5py.SoftLink("/dataset")
        h5["soft_link_to_nothing"] = h5py.SoftLink("/foo/bar/2000")

        h5["external_link_to_group"] = h5py.ExternalLink("types.h5", "/arrays")
        h5["external_link_to_dataset"] = h5py.ExternalLink("types.h5", "/arrays/cube")
        h5["external_link_to_nothing"] = h5py.ExternalLink("types.h5", "/foo/bar/2000")
        h5["external_link_to_missing_file"] = h5py.ExternalLink("missing_file.h5", "/")


def create_recursive_links():
    with h5py.File("../links_recursive.h5", "w") as h5:
        g = h5.create_group("group")
        g.create_dataset("dataset", data=numpy.int64(10))
        h5.create_dataset("dataset", data=numpy.int64(10))

        h5["hard_recursive_link"] = h5["/group"]
        g["recursive"] = h5["hard_recursive_link"]
        h5["hard_link_to_dataset"] = h5["/dataset"]

        h5["soft_link_to_group"] = h5py.SoftLink("/group")
        h5["soft_link_to_link"] = h5py.SoftLink("/soft_link_to_group")
        h5["soft_link_to_itself"] = h5py.SoftLink("/soft_link_to_itself")


def create_external_recursive_links():

    with h5py.File("../links_external_recursive.h5", "w") as h5:
        g = h5.create_group("group")
        g.create_dataset("dataset", data=numpy.int64(10))
        h5["soft_link_to_group"] = h5py.SoftLink("/group")
        h5["external_link_to_link"] = h5py.ExternalLink("links_external_recursive_2.h5", "/soft_link_to_group")
        h5["external_link_to_recursive_link"] = h5py.ExternalLink("links_external_recursive_2.h5", "/external_link_to_recursive_link")

    with h5py.File("../links_external_recursive_2.h5", "w") as h5:
        g = h5.create_group("group")
        g.create_dataset("dataset", data=numpy.int64(10))
        h5["soft_link_to_group"] = h5py.SoftLink("/group")
        h5["external_link_to_link"] = h5py.ExternalLink("links_external_recursive.h5", "/soft_link_to_group")
        h5["external_link_to_recursive_link"] = h5py.ExternalLink("links_external_recursive.h5", "/external_link_to_recursive_link")


def main():
    print("Begin")
    # create_all_types()
    # create_all_links()
    # create_recursive_links()
    # create_external_recursive_links()
    create_all_types2()
    print("End")


if __name__ == "__main__":
    main()
