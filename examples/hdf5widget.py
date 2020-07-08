#!/usr/bin/env python
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
"""Qt Hdf5 widget examples"""

import logging
import sys
import tempfile

import numpy
import six

logging.basicConfig()
_logger = logging.getLogger("hdf5widget")
"""Module logger"""

try:
    # it should be loaded before h5py
    import hdf5plugin  # noqa
except ImportError:
    message = "Module 'hdf5plugin' is not installed. It supports some hdf5"\
        + " compressions. You can install it using \"pip install hdf5plugin\"."
    _logger.warning(message)
import h5py

import silx.gui.hdf5
import silx.utils.html
from silx.gui import qt
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.widgets.ThreadPoolPushButton import ThreadPoolPushButton

import fabio


_file_cache = {}


def str_attrs(str_list):
    """Return a numpy array of unicode strings"""
    text_dtype = h5py.special_dtype(vlen=six.text_type)
    return numpy.array(str_list, dtype=text_dtype)


def get_hdf5_with_all_types():
    ID = "alltypes"
    if ID in _file_cache:
        return _file_cache[ID].name

    tmp = tempfile.NamedTemporaryFile(prefix=ID + "_", suffix=".h5", delete=True)
    tmp.file.close()
    h5 = h5py.File(tmp.name, "w")

    g = h5.create_group("arrays")
    g.create_dataset("scalar", data=10)
    g.create_dataset("list", data=numpy.arange(10))
    base_image = numpy.arange(10**2).reshape(10, 10)
    images = [ base_image,
               base_image.T,
               base_image.size - 1 - base_image,
               base_image.size - 1 - base_image.T]
    dtype = images[0].dtype
    data = numpy.empty((10 * 10, 10, 10), dtype=dtype)
    for i in range(10 * 10):
        data[i] = images[i % 4]
    data.shape = 10, 10, 10, 10
    g.create_dataset("image", data=data[0, 0])
    g.create_dataset("cube", data=data[0])
    g.create_dataset("hypercube", data=data)
    g = h5.create_group("dtypes")
    g.create_dataset("int32", data=numpy.int32(10))
    g.create_dataset("int64", data=numpy.int64(10))
    g.create_dataset("float32", data=numpy.float32(10))
    g.create_dataset("float64", data=numpy.float64(10))
    g.create_dataset("string_", data=numpy.string_("Hi!"))
    # g.create_dataset("string0",data=numpy.string0("Hi!\x00"))

    g.create_dataset("bool", data=True)
    g.create_dataset("bool2", data=False)
    h5.close()

    _file_cache[ID] = tmp
    return tmp.name


def get_hdf5_with_all_links():
    ID = "alllinks"
    if ID in _file_cache:
        return _file_cache[ID].name

    tmp = tempfile.NamedTemporaryFile(prefix=ID + "_", suffix=".h5", delete=True)
    tmp.file.close()
    h5 = h5py.File(tmp.name, "w")

    g = h5.create_group("group")
    g.create_dataset("dataset", data=numpy.int64(10))
    h5.create_dataset("dataset", data=numpy.int64(10))

    h5["hard_link_to_group"] = h5["/group"]
    h5["hard_link_to_dataset"] = h5["/dataset"]

    h5["soft_link_to_group"] = h5py.SoftLink("/group")
    h5["soft_link_to_dataset"] = h5py.SoftLink("/dataset")
    h5["soft_link_to_nothing"] = h5py.SoftLink("/foo/bar/2000")

    alltypes_filename = get_hdf5_with_all_types()

    h5["external_link_to_group"] = h5py.ExternalLink(alltypes_filename, "/arrays")
    h5["external_link_to_dataset"] = h5py.ExternalLink(alltypes_filename, "/arrays/cube")
    h5["external_link_to_nothing"] = h5py.ExternalLink(alltypes_filename, "/foo/bar/2000")
    h5["external_link_to_missing_file"] = h5py.ExternalLink("missing_file.h5", "/")
    h5.close()

    _file_cache[ID] = tmp
    return tmp.name


def get_hdf5_with_1000_datasets():
    ID = "dataset1000"
    if ID in _file_cache:
        return _file_cache[ID].name

    tmp = tempfile.NamedTemporaryFile(prefix=ID + "_", suffix=".h5", delete=True)
    tmp.file.close()
    h5 = h5py.File(tmp.name, "w")

    g = h5.create_group("group")
    for i in range(1000):
        g.create_dataset("dataset%i" % i, data=numpy.int64(10))
    h5.close()

    _file_cache[ID] = tmp
    return tmp.name


def get_hdf5_with_10000_datasets():
    ID = "dataset10000"
    if ID in _file_cache:
        return _file_cache[ID].name

    tmp = tempfile.NamedTemporaryFile(prefix=ID + "_", suffix=".h5", delete=True)
    tmp.file.close()
    h5 = h5py.File(tmp.name, "w")

    g = h5.create_group("group")
    for i in range(10000):
        g.create_dataset("dataset%i" % i, data=numpy.int64(10))
    h5.close()

    _file_cache[ID] = tmp
    return tmp.name


def get_hdf5_with_100000_datasets():
    ID = "dataset100000"
    if ID in _file_cache:
        return _file_cache[ID].name

    tmp = tempfile.NamedTemporaryFile(prefix=ID + "_", suffix=".h5", delete=True)
    tmp.file.close()
    h5 = h5py.File(tmp.name, "w")

    g = h5.create_group("group")
    for i in range(100000):
        g.create_dataset("dataset%i" % i, data=numpy.int64(10))
    h5.close()

    _file_cache[ID] = tmp
    return tmp.name


def get_hdf5_with_recursive_links():
    ID = "recursive_links"
    if ID in _file_cache:
        return _file_cache[ID].name

    tmp = tempfile.NamedTemporaryFile(prefix=ID + "_", suffix=".h5", delete=True)
    tmp.file.close()
    h5 = h5py.File(tmp.name, "w")

    g = h5.create_group("group")
    g.create_dataset("dataset", data=numpy.int64(10))
    h5.create_dataset("dataset", data=numpy.int64(10))

    h5["hard_recursive_link"] = h5["/group"]
    g["recursive"] = h5["hard_recursive_link"]
    h5["hard_link_to_dataset"] = h5["/dataset"]

    h5["soft_link_to_group"] = h5py.SoftLink("/group")
    h5["soft_link_to_link"] = h5py.SoftLink("/soft_link_to_group")
    h5["soft_link_to_itself"] = h5py.SoftLink("/soft_link_to_itself")
    h5.close()

    _file_cache[ID] = tmp
    return tmp.name


def get_hdf5_with_external_recursive_links():
    ID = "external_recursive_links"
    if ID in _file_cache:
        return _file_cache[ID][0].name

    tmp1 = tempfile.NamedTemporaryFile(prefix=ID + "_", suffix=".h5", delete=True)
    tmp1.file.close()
    h5_1 = h5py.File(tmp1.name, "w")

    tmp2 = tempfile.NamedTemporaryFile(prefix=ID + "_", suffix=".h5", delete=True)
    tmp2.file.close()
    h5_2 = h5py.File(tmp2.name, "w")

    g = h5_1.create_group("group")
    g.create_dataset("dataset", data=numpy.int64(10))
    h5_1["soft_link_to_group"] = h5py.SoftLink("/group")
    h5_1["external_link_to_link"] = h5py.ExternalLink(tmp2.name, "/soft_link_to_group")
    h5_1["external_link_to_recursive_link"] = h5py.ExternalLink(tmp2.name, "/external_link_to_recursive_link")
    h5_1.close()

    g = h5_2.create_group("group")
    g.create_dataset("dataset", data=numpy.int64(10))
    h5_2["soft_link_to_group"] = h5py.SoftLink("/group")
    h5_2["external_link_to_link"] = h5py.ExternalLink(tmp1.name, "/soft_link_to_group")
    h5_2["external_link_to_recursive_link"] = h5py.ExternalLink(tmp1.name, "/external_link_to_recursive_link")
    h5_2.close()

    _file_cache[ID] = (tmp1, tmp2)
    return tmp1.name


def get_hdf5_with_nxdata():
    ID = "nxdata"
    if ID in _file_cache:
        return _file_cache[ID].name

    tmp = tempfile.NamedTemporaryFile(prefix=ID + "_", suffix=".h5", delete=True)
    tmp.file.close()
    h5 = h5py.File(tmp.name, "w")

    # SCALARS
    g0d = h5.create_group("scalars")

    g0d0 = g0d.create_group("0D_scalar")
    g0d0.attrs["NX_class"] = u"NXdata"
    g0d0.attrs["signal"] = u"scalar"
    g0d0.create_dataset("scalar", data=10)

    g0d1 = g0d.create_group("2D_scalars")
    g0d1.attrs["NX_class"] = u"NXdata"
    g0d1.attrs["signal"] = u"scalars"
    ds = g0d1.create_dataset("scalars", data=numpy.arange(3*10).reshape((3, 10)))
    ds.attrs["interpretation"] = u"scalar"

    g0d1 = g0d.create_group("4D_scalars")
    g0d1.attrs["NX_class"] = u"NXdata"
    g0d1.attrs["signal"] = u"scalars"
    ds = g0d1.create_dataset("scalars", data=numpy.arange(2*2*3*10).reshape((2, 2, 3, 10)))
    ds.attrs["interpretation"] = u"scalar"

    # SPECTRA
    g1d = h5.create_group("spectra")

    g1d0 = g1d.create_group("1D_spectrum")
    g1d0.attrs["NX_class"] = u"NXdata"
    g1d0.attrs["signal"] = u"count"
    g1d0.attrs["auxiliary_signals"] = str_attrs(["count.5", "count2"])
    g1d0.attrs["axes"] = u"energy_calib"
    g1d0.attrs["uncertainties"] = str_attrs(["energy_errors"])
    g1d0.create_dataset("count", data=numpy.arange(10))
    g1d0.create_dataset("count.5", data=.5*numpy.arange(10))
    d2 = g1d0.create_dataset("count2", data=2*numpy.arange(10))
    d2.attrs["long_name"] = u"count multiplied by 2"
    g1d0.create_dataset("energy_calib", data=(10, 5))     # 10 * idx + 5
    g1d0.create_dataset("energy_errors", data=3.14*numpy.random.rand(10))
    g1d0.create_dataset("title", data="Title example provided as dataset")

    g1d1 = g1d.create_group("2D_spectra")
    g1d1.attrs["NX_class"] = u"NXdata"
    g1d1.attrs["signal"] = u"counts"
    ds = g1d1.create_dataset("counts", data=numpy.arange(3*10).reshape((3, 10)))
    ds.attrs["interpretation"] = u"spectrum"

    g1d2 = g1d.create_group("4D_spectra")
    g1d2.attrs["NX_class"] = u"NXdata"
    g1d2.attrs["signal"] = u"counts"
    g1d2.attrs["axes"] = str_attrs(["energy"])
    ds = g1d2.create_dataset("counts", data=numpy.arange(2*2*3*10).reshape((2, 2, 3, 10)))
    ds.attrs["interpretation"] = u"spectrum"
    g1d2.create_dataset("errors", data=4.5*numpy.random.rand(2, 2, 3, 10))
    ds = g1d2.create_dataset("energy", data=5+10*numpy.arange(15),
                             shuffle=True, compression="gzip")
    ds.attrs["long_name"] = u"Calibrated energy"
    ds.attrs["first_good"] = 3
    ds.attrs["last_good"] = 12
    g1d2.create_dataset("energy_errors", data=10*numpy.random.rand(15))

    # IMAGES
    g2d = h5.create_group("images")

    g2d0 = g2d.create_group("2D_regular_image")
    g2d0.attrs["NX_class"] = u"NXdata"
    g2d0.attrs["signal"] = u"image"
    g2d0.attrs["auxiliary_signals"] = str_attrs(["image2", "image3"])
    g2d0.attrs["axes"] = str_attrs(["rows_calib", "columns_coordinates"])
    g2d0.attrs["title"] = u"Title example provided as group attr"
    g2d0.create_dataset("image", data=numpy.arange(4*6).reshape((4, 6)))
    g2d0.create_dataset("image2", data=1/(1.+numpy.arange(4*6).reshape((4, 6))))
    ds = g2d0.create_dataset("image3", data=-numpy.arange(4*6).reshape((4, 6)))
    ds.attrs["long_name"] = u"3rd image (2nd auxiliary)"
    ds = g2d0.create_dataset("rows_calib", data=(10, 5))
    ds.attrs["long_name"] = u"Calibrated Y"
    g2d0.create_dataset("columns_coordinates", data=0.5+0.02*numpy.arange(6))

    g2d4 = g2d.create_group("RGBA_image")
    g2d4.attrs["NX_class"] = u"NXdata"
    g2d4.attrs["signal"] = u"image"
    g2d4.attrs["auxiliary_signals"] = u"squared image"
    g2d4.attrs["axes"] = str_attrs(["rows_calib", "columns_coordinates"])
    rgba_image = numpy.linspace(0, 1, num=7*8*3).reshape((7, 8, 3))
    rgba_image[:, :, 1] = 1 - rgba_image[:, :, 1]      # invert G channel to add some color
    ds = g2d4.create_dataset("image", data=rgba_image)
    ds.attrs["interpretation"] = u"rgba-image"
    ds = g2d4.create_dataset("squared image", data=rgba_image**2)
    ds.attrs["interpretation"] = u"rgba-image"
    ds = g2d4.create_dataset("rows_calib", data=(10, 5))
    ds.attrs["long_name"] = u"Calibrated Y"
    g2d4.create_dataset("columns_coordinates", data=0.5+0.02*numpy.arange(8))

    g2d1 = g2d.create_group("2D_irregular_data")
    g2d1.attrs["NX_class"] = u"NXdata"
    g2d1.attrs["signal"] = u"data"
    g2d1.attrs["axes"] = str_attrs(["rows_coordinates", "columns_coordinates"])
    g2d1.create_dataset("data", data=numpy.arange(64*128).reshape((64, 128)))
    g2d1.create_dataset("rows_coordinates", data=numpy.arange(64) + numpy.random.rand(64))
    g2d1.create_dataset("columns_coordinates", data=numpy.arange(128) + 2.5 * numpy.random.rand(128))

    g2d2 = g2d.create_group("3D_images")
    g2d2.attrs["NX_class"] = u"NXdata"
    g2d2.attrs["signal"] = u"images"
    ds = g2d2.create_dataset("images", data=numpy.arange(2*4*6).reshape((2, 4, 6)))
    ds.attrs["interpretation"] = u"image"

    g2d3 = g2d.create_group("5D_images")
    g2d3.attrs["NX_class"] = u"NXdata"
    g2d3.attrs["signal"] = u"images"
    g2d3.attrs["axes"] = str_attrs(["rows_coordinates", "columns_coordinates"])
    ds = g2d3.create_dataset("images", data=numpy.arange(2*2*2*4*6).reshape((2, 2, 2, 4, 6)))
    ds.attrs["interpretation"] = u"image"
    g2d3.create_dataset("rows_coordinates", data=5+10*numpy.arange(4))
    g2d3.create_dataset("columns_coordinates", data=0.5+0.02*numpy.arange(6))

    # SCATTER
    g = h5.create_group("scatters")

    gd0 = g.create_group("x_y_scatter")
    gd0.attrs["NX_class"] = u"NXdata"
    gd0.attrs["signal"] = u"y"
    gd0.attrs["axes"] = str_attrs(["x"])
    gd0.attrs["title"] = u"simple y = f(x) scatters cannot be distinguished from curves"
    gd0.create_dataset("y", data=numpy.random.rand(128) - 0.5)
    gd0.create_dataset("x", data=2*numpy.random.rand(128))
    gd0.create_dataset("x_errors", data=0.05*numpy.random.rand(128))
    gd0.create_dataset("errors", data=0.05*numpy.random.rand(128))

    gd1 = g.create_group("x_y_value_scatter")
    gd1.attrs["NX_class"] = u"NXdata"
    gd1.attrs["signal"] = u"values"
    gd1.attrs["auxiliary_signals"] = str_attrs(["values.5", "values2"])
    gd1.attrs["axes"] = str_attrs(["x", "y"])
    gd1.attrs["title"] = u"x, y, values scatter with asymmetric y_errors"
    gd1.create_dataset("values", data=3.14*numpy.random.rand(128))
    gd1.create_dataset("values.5", data=0.5*3.14*numpy.random.rand(128))
    gd1.create_dataset("values2", data=2.*3.14*numpy.random.rand(128))
    gd1.create_dataset("y", data=numpy.random.rand(128))
    y_errors = [0.03*numpy.random.rand(128), 0.04*numpy.random.rand(128)]
    gd1.create_dataset("y_errors", data=y_errors)
    ds = gd1.create_dataset("x", data=2*numpy.random.rand(128))
    ds.attrs["long_name"] = u"horizontal axis"
    gd1.create_dataset("x_errors", data=0.02*numpy.random.rand(128))

    # NDIM > 3
    g = h5.create_group("cubes")

    gd0 = g.create_group("3D_cube")
    gd0.attrs["NX_class"] = u"NXdata"
    gd0.attrs["signal"] = u"cube"
    gd0.attrs["axes"] = str_attrs(["img_idx", "rows_coordinates", "cols_coordinates"])
    gd0.create_dataset("cube", data=numpy.arange(4*5*6).reshape((4, 5, 6)))
    gd0.create_dataset("img_idx", data=numpy.arange(4))
    gd0.create_dataset("rows_coordinates", data=0.1*numpy.arange(5))
    gd0.create_dataset("cols_coordinates", data=[0.2, 0.3])  # linear calibration

    gd1 = g.create_group("5D")
    gd1.attrs["NX_class"] = u"NXdata"
    gd1.attrs["signal"] = u"hypercube"
    gd1.create_dataset("hypercube",
                       data=numpy.arange(2*3*4*5*6).reshape((2, 3, 4, 5, 6)))

    gd2 = g.create_group("3D_nonlinear_scaling")
    gd2.attrs["NX_class"] = u"NXdata"
    gd2.attrs["signal"] = u"cube"
    gd2.attrs["axes"] = str_attrs(["img_idx", "rows_coordinates", "cols_coordinates"])
    gd2.create_dataset("cube", data=numpy.arange(4*5*6).reshape((4, 5, 6)))
    gd2.create_dataset("img_idx", data=numpy.array([2., -0.1, 8, 3.14]))
    gd2.create_dataset("rows_coordinates", data=0.1*numpy.arange(5))
    gd2.create_dataset("cols_coordinates", data=[0.1, 0.6, 0.7, 8., 8.1, 8.2])


    # invalid NXdata
    g = h5.create_group("invalid")
    g0 = g.create_group("invalid NXdata")
    g0.attrs["NX_class"] = u"NXdata"

    g1 = g.create_group("invalid NXentry")
    g1.attrs["NX_class"] = u"NXentry"
    g1.attrs["default"] = u"missing NXdata group"

    g2 = g.create_group("invalid NXroot")
    g2.attrs["NX_class"] = u"NXroot"
    g2.attrs["default"] = u"invalid NXentry in NXroot"
    g20 = g2.create_group("invalid NXentry in NXroot")
    g20.attrs["NX_class"] = u"NXentry"
    g20.attrs["default"] = u"missing NXdata group"

    h5.close()

    _file_cache[ID] = tmp
    return tmp.name


def get_edf_with_all_types():
    ID = "alltypesedf"
    if ID in _file_cache:
        return _file_cache[ID].name

    tmp = tempfile.NamedTemporaryFile(prefix=ID + "_", suffix=".edf", delete=True)

    header = fabio.fabioimage.OrderedDict()
    header["integer"] = "10"
    header["float"] = "10.5"
    header["string"] = "Hi!"
    header["integer_list"] = "10 20 50"
    header["float_list"] = "1.1 3.14 500.12"
    header["motor_pos"] = "10 2.5 a1"
    header["motor_mne"] = "integer_position float_position named_position"

    data = numpy.array([[10, 50], [50, 10]])
    fabiofile = fabio.edfimage.EdfImage(data, header)
    fabiofile.write(tmp.name)

    _file_cache[ID] = tmp
    return tmp.name


def get_edf_with_100000_frames():
    ID = "frame100000"
    if ID in _file_cache:
        return _file_cache[ID].name

    tmp = tempfile.NamedTemporaryFile(prefix=ID + "_", suffix=".edf", delete=True)

    fabiofile = None
    for framre_id in range(100000):
        data = numpy.array([[framre_id, 50], [50, 10]])
        if fabiofile is None:
            header = fabio.fabioimage.OrderedDict()
            header["nb_frames"] = "100000"
            fabiofile = fabio.edfimage.EdfImage(data, header)
        else:
            header = fabio.fabioimage.OrderedDict()
            header["frame_nb"] = framre_id
            fabiofile.append_frame(fabio.edfimage.Frame(data, header, framre_id))
    fabiofile.write(tmp.name)

    _file_cache[ID] = tmp
    return tmp.name


class Hdf5TreeViewExample(qt.QMainWindow):
    """
    This window show an example of use of a Hdf5TreeView.

    The tree is initialized with a list of filenames. A panel allow to play
    with internal property configuration of the widget, and a text screen
    allow to display events.
    """

    def __init__(self, filenames=None):
        """
        :param files_: List of HDF5 or Spec files (pathes or
            :class:`silx.io.spech5.SpecH5` or :class:`h5py.File`
            instances)
        """
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Silx HDF5 widget example")

        self.__asyncload = False
        self.__treeview = silx.gui.hdf5.Hdf5TreeView(self)
        """Silx HDF5 TreeView"""
        self.__text = qt.QTextEdit(self)
        """Widget displaying information"""

        self.__dataViewer = DataViewerFrame(self)
        vSpliter = qt.QSplitter(qt.Qt.Vertical)
        vSpliter.addWidget(self.__dataViewer)
        vSpliter.addWidget(self.__text)
        vSpliter.setSizes([10, 0])

        spliter = qt.QSplitter(self)
        spliter.addWidget(self.__treeview)
        spliter.addWidget(vSpliter)
        spliter.setStretchFactor(1, 1)

        main_panel = qt.QWidget(self)
        layout = qt.QVBoxLayout()
        layout.addWidget(spliter)
        layout.addWidget(self.createTreeViewConfigurationPanel(self, self.__treeview))
        layout.setStretchFactor(spliter, 1)
        main_panel.setLayout(layout)

        self.setCentralWidget(main_panel)

        # append all files to the tree
        for file_name in filenames:
            self.__treeview.findHdf5TreeModel().appendFile(file_name)

        self.__treeview.activated.connect(self.displayData)
        self.__treeview.activated.connect(lambda index: self.displayEvent("activated", index))
        self.__treeview.clicked.connect(lambda index: self.displayEvent("clicked", index))
        self.__treeview.doubleClicked.connect(lambda index: self.displayEvent("doubleClicked", index))
        self.__treeview.entered.connect(lambda index: self.displayEvent("entered", index))
        self.__treeview.pressed.connect(lambda index: self.displayEvent("pressed", index))

        self.__treeview.addContextMenuCallback(self.customContextMenu)
        # lambda function will never be called cause we store it as weakref
        self.__treeview.addContextMenuCallback(lambda event: None)
        # you have to store it first
        self.__store_lambda = lambda event: self.closeAndSyncCustomContextMenu(event)
        self.__treeview.addContextMenuCallback(self.__store_lambda)

    def displayData(self):
        """Called to update the dataviewer with the selected data.
        """
        selected = list(self.__treeview.selectedH5Nodes())
        if len(selected) == 1:
            # Update the viewer for a single selection
            data = selected[0]
            # data is a hdf5.H5Node object
            # data.h5py_object is a Group/Dataset object (from h5py, spech5, fabioh5)
            # The dataviewer can display both
            self.__dataViewer.setData(data)

    def displayEvent(self, eventName, index):
        """Called to log event in widget
        """
        def formatKey(name, value):
            name, value = silx.utils.html.escape(str(name)), silx.utils.html.escape(str(value))
            return "<li><b>%s</b>: %s</li>" % (name, value)

        text = "<html>"
        text += "<h1>Event</h1>"
        text += "<ul>"
        text += formatKey("name", eventName)
        text += formatKey("index", type(index))
        text += "</ul>"

        text += "<h1>Selected HDF5 objects</h1>"

        for h5_obj in self.__treeview.selectedH5Nodes():
            text += "<h2>HDF5 object</h2>"
            text += "<ul>"
            text += formatKey("local_filename", h5_obj.local_file.filename)
            text += formatKey("local_basename", h5_obj.local_basename)
            text += formatKey("local_name", h5_obj.local_name)
            text += formatKey("real_filename", h5_obj.file.filename)
            text += formatKey("real_basename", h5_obj.basename)
            text += formatKey("real_name", h5_obj.name)

            text += formatKey("obj", h5_obj.ntype)
            text += formatKey("dtype", getattr(h5_obj, "dtype", None))
            text += formatKey("shape", getattr(h5_obj, "shape", None))
            text += formatKey("attrs", getattr(h5_obj, "attrs", None))
            if hasattr(h5_obj, "attrs"):
                text += "<ul>"
                if len(h5_obj.attrs) == 0:
                    text += "<li>empty</li>"
                for key, value in h5_obj.attrs.items():
                    text += formatKey(key, value)
                text += "</ul>"
            text += "</ul>"

        text += "</html>"
        self.__text.setHtml(text)

    def useAsyncLoad(self, useAsync):
        self.__asyncload = useAsync

    def __fileCreated(self, filename):
        if self.__asyncload:
            self.__treeview.findHdf5TreeModel().insertFileAsync(filename)
        else:
            self.__treeview.findHdf5TreeModel().insertFile(filename)

    def customContextMenu(self, event):
        """Called to populate the context menu

        :param silx.gui.hdf5.Hdf5ContextMenuEvent event: Event
            containing expected information to populate the context menu
        """
        selectedObjects = event.source().selectedH5Nodes()
        menu = event.menu()

        hasDataset = False
        for obj in selectedObjects:
            if obj.ntype is h5py.Dataset:
                hasDataset = True
                break

        if not menu.isEmpty():
            menu.addSeparator()

        if hasDataset:
            action = qt.QAction("Do something on the datasets", event.source())
            menu.addAction(action)

    def closeAndSyncCustomContextMenu(self, event):
        """Called to populate the context menu

        :param silx.gui.hdf5.Hdf5ContextMenuEvent event: Event
            containing expected information to populate the context menu
        """
        selectedObjects = event.source().selectedH5Nodes()
        menu = event.menu()

        if not menu.isEmpty():
            menu.addSeparator()

        for obj in selectedObjects:
            if obj.ntype is h5py.File:
                action = qt.QAction("Remove %s" % obj.local_filename, event.source())
                action.triggered.connect(lambda: self.__treeview.findHdf5TreeModel().removeH5pyObject(obj.h5py_object))
                menu.addAction(action)
                action = qt.QAction("Synchronize %s" % obj.local_filename, event.source())
                action.triggered.connect(lambda: self.__treeview.findHdf5TreeModel().synchronizeH5pyObject(obj.h5py_object))
                menu.addAction(action)

    def __hdf5ComboChanged(self, index):
        function = self.__hdf5Combo.itemData(index)
        self.__createHdf5Button.setCallable(function)

    def __edfComboChanged(self, index):
        function = self.__edfCombo.itemData(index)
        self.__createEdfButton.setCallable(function)

    def createTreeViewConfigurationPanel(self, parent, treeview):
        """Create a configuration panel to allow to play with widget states"""
        panel = qt.QWidget(parent)
        panel.setLayout(qt.QHBoxLayout())

        content = qt.QGroupBox("Create HDF5", panel)
        content.setLayout(qt.QVBoxLayout())
        panel.layout().addWidget(content)

        combo = qt.QComboBox()
        combo.addItem("Containing all types", get_hdf5_with_all_types)
        combo.addItem("Containing all links", get_hdf5_with_all_links)
        combo.addItem("Containing 1000 datasets", get_hdf5_with_1000_datasets)
        combo.addItem("Containing 10000 datasets", get_hdf5_with_10000_datasets)
        combo.addItem("Containing 100000 datasets", get_hdf5_with_100000_datasets)
        combo.addItem("Containing recursive links", get_hdf5_with_recursive_links)
        combo.addItem("Containing external recursive links", get_hdf5_with_external_recursive_links)
        combo.addItem("Containing NXdata groups", get_hdf5_with_nxdata)
        combo.activated.connect(self.__hdf5ComboChanged)
        content.layout().addWidget(combo)

        button = ThreadPoolPushButton(content, text="Create")
        button.setCallable(combo.itemData(combo.currentIndex()))
        button.succeeded.connect(self.__fileCreated)
        content.layout().addWidget(button)

        self.__hdf5Combo = combo
        self.__createHdf5Button = button

        asyncload = qt.QCheckBox("Async load", content)
        asyncload.setChecked(self.__asyncload)
        asyncload.toggled.connect(lambda: self.useAsyncLoad(asyncload.isChecked()))
        content.layout().addWidget(asyncload)

        content.layout().addStretch(1)

        content = qt.QGroupBox("Create EDF", panel)
        content.setLayout(qt.QVBoxLayout())
        panel.layout().addWidget(content)

        combo = qt.QComboBox()
        combo.addItem("Containing all types", get_edf_with_all_types)
        combo.addItem("Containing 100000 datasets", get_edf_with_100000_frames)
        combo.activated.connect(self.__edfComboChanged)
        content.layout().addWidget(combo)

        button = ThreadPoolPushButton(content, text="Create")
        button.setCallable(combo.itemData(combo.currentIndex()))
        button.succeeded.connect(self.__fileCreated)
        content.layout().addWidget(button)

        self.__edfCombo = combo
        self.__createEdfButton = button

        content.layout().addStretch(1)

        option = qt.QGroupBox("Tree options", panel)
        option.setLayout(qt.QVBoxLayout())
        panel.layout().addWidget(option)

        sorting = qt.QCheckBox("Enable sorting", option)
        sorting.setChecked(treeview.selectionMode() == qt.QAbstractItemView.MultiSelection)
        sorting.toggled.connect(lambda: treeview.setSortingEnabled(sorting.isChecked()))
        option.layout().addWidget(sorting)

        multiselection = qt.QCheckBox("Multi-selection", option)
        multiselection.setChecked(treeview.selectionMode() == qt.QAbstractItemView.MultiSelection)
        switch_selection = lambda: treeview.setSelectionMode(
            qt.QAbstractItemView.MultiSelection if multiselection.isChecked()
            else qt.QAbstractItemView.SingleSelection)
        multiselection.toggled.connect(switch_selection)
        option.layout().addWidget(multiselection)

        filedrop = qt.QCheckBox("Drop external file", option)
        filedrop.setChecked(treeview.findHdf5TreeModel().isFileDropEnabled())
        filedrop.toggled.connect(lambda: treeview.findHdf5TreeModel().setFileDropEnabled(filedrop.isChecked()))
        option.layout().addWidget(filedrop)

        filemove = qt.QCheckBox("Reorder files", option)
        filemove.setChecked(treeview.findHdf5TreeModel().isFileMoveEnabled())
        filemove.toggled.connect(lambda: treeview.findHdf5TreeModel().setFileMoveEnabled(filedrop.isChecked()))
        option.layout().addWidget(filemove)

        option.layout().addStretch(1)

        option = qt.QGroupBox("Header options", panel)
        option.setLayout(qt.QVBoxLayout())
        panel.layout().addWidget(option)

        autosize = qt.QCheckBox("Auto-size headers", option)
        autosize.setChecked(treeview.header().hasAutoResizeColumns())
        autosize.toggled.connect(lambda: treeview.header().setAutoResizeColumns(autosize.isChecked()))
        option.layout().addWidget(autosize)

        columnpopup = qt.QCheckBox("Popup to hide/show columns", option)
        columnpopup.setChecked(treeview.header().hasHideColumnsPopup())
        columnpopup.toggled.connect(lambda: treeview.header().setEnableHideColumnsPopup(columnpopup.isChecked()))
        option.layout().addWidget(columnpopup)

        define_columns = qt.QComboBox()
        define_columns.addItem("Default columns", treeview.findHdf5TreeModel().COLUMN_IDS)
        define_columns.addItem("Only name and Value", [treeview.findHdf5TreeModel().NAME_COLUMN, treeview.findHdf5TreeModel().VALUE_COLUMN])
        define_columns.activated.connect(lambda index: treeview.header().setSections(define_columns.itemData(index)))
        option.layout().addWidget(define_columns)

        option.layout().addStretch(1)

        panel.layout().addStretch(1)

        return panel


def main(filenames):
    """
    :param filenames: list of file paths
    """
    app = qt.QApplication([])
    sys.excepthook = qt.exceptionHandler
    window = Hdf5TreeViewExample(filenames)
    window.show()
    result = app.exec_()
    # remove ending warnings relative to QTimer
    app.deleteLater()
    sys.exit(result)


if __name__ == "__main__":
    main(sys.argv[1:])
