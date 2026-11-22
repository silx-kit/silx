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
__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "29/01/2018"

import os
import tempfile
from contextlib import contextmanager

import numpy

from silx.gui.data.NumpyAxesSelector import NumpyAxesSelector
from silx.gui.utils.testutils import SignalListener
from silx.gui.utils.testutils import TestCaseQt

import h5py


class TestNumpyAxesSelector(TestCaseQt):
    def test_creation(self):
        data = numpy.arange(3 * 3 * 3)
        data = data.reshape(3, 3, 3)
        widget = NumpyAxesSelector()
        widget.setVisible(True)

    def test_none(self):
        data = numpy.arange(3 * 3 * 3)
        widget = NumpyAxesSelector()
        widget.setData(data)
        widget.setData(None)
        result = widget.selectedData()
        self.assertIsNone(result)

    def test_output_samedim(self):
        data = numpy.arange(3 * 3 * 3)
        data = data.reshape(3, 3, 3)
        expectedResult = data

        widget = NumpyAxesSelector()
        widget.setAxisNames(["x", "y", "z"])
        widget.setData(data)
        result = widget.selectedData()
        self.assertTrue(numpy.array_equal(result, expectedResult))

    def test_output_moredim(self):
        data = numpy.arange(3 * 3 * 3 * 3)
        data = data.reshape(3, 3, 3, 3)
        expectedResult = data

        widget = NumpyAxesSelector()
        widget.setAxisNames(["x", "y", "z", "boum"])
        widget.setData(data[0])
        result = widget.selectedData()
        self.assertIsNone(result)
        widget.setData(data)
        result = widget.selectedData()
        self.assertTrue(numpy.array_equal(result, expectedResult))

    def test_output_lessdim(self):
        data = numpy.arange(3 * 3 * 3)
        data = data.reshape(3, 3, 3)
        expectedResult = data[0]

        widget = NumpyAxesSelector()
        widget.setAxisNames(["y", "x"])
        widget.setData(data)
        result = widget.selectedData()
        self.assertTrue(numpy.array_equal(result, expectedResult))

    def test_output_1dim(self):
        data = numpy.arange(3 * 3 * 3)
        data = data.reshape(3, 3, 3)
        expectedResult = data[0, 0, 0]

        widget = NumpyAxesSelector()
        widget.setData(data)
        result = widget.selectedData()
        self.assertTrue(numpy.array_equal(result, expectedResult))

    @contextmanager
    def h5_temporary_file(self):
        # create tmp file
        fd, tmp_name = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        data = numpy.arange(3 * 3 * 3)
        data = data.reshape(3, 3, 3)
        # create h5 data
        h5file = h5py.File(tmp_name, "w")
        h5file["data"] = data
        yield h5file
        # clean up
        h5file.close()
        os.unlink(tmp_name)

    def test_h5py_dataset(self):
        with self.h5_temporary_file() as h5file:
            dataset = h5file["data"]
            expectedResult = dataset[0]

            widget = NumpyAxesSelector()
            widget.setData(dataset)
            widget.setAxisNames(["y", "x"])
            result = widget.selectedData()
            self.assertTrue(numpy.array_equal(result, expectedResult))

    def test_data_event(self):
        data = numpy.arange(3 * 3 * 3)
        widget = NumpyAxesSelector()
        listener = SignalListener()
        widget.dataChanged.connect(listener)
        widget.setData(data)
        widget.setData(None)
        self.assertEqual(listener.callCount(), 2)

    def test_selected_data_event(self):
        data = numpy.arange(3 * 3 * 3)
        data = data.reshape(3, 3, 3)
        widget = NumpyAxesSelector()
        listener = SignalListener()
        widget.selectionChanged.connect(listener)
        widget.setData(data)
        widget.setAxisNames(["x"])
        widget.setData(None)
        self.assertEqual(listener.callCount(), 3)
        listener.clear()


def test_selection_read_failure_discards_previous_data(tmp_path):
    """A selection whose data cannot be read shows a blank instead of stale data.

    Frame 0 is stored uncompressed and can be read; frame 2 needs the HDF5
    filter 32033 which is not installed, so reading it raises OSError.
    """
    from h5py import h5d, h5p, h5s, h5t, h5z

    filename = str(tmp_path / "missing_filter.h5")
    dcpl = h5p.create(h5p.DATASET_CREATE)
    dcpl.set_chunk((1, 4, 4))
    dcpl.set_filter(32033, h5z.FLAG_OPTIONAL)
    with h5py.File(filename, "w") as h5file:
        h5d.create(
            h5file.id,
            b"data",
            h5t.NATIVE_UINT16,
            h5s.create_simple((3, 4, 4)),
            dcpl,
        )
        raw = numpy.ones((4, 4), numpy.uint16).tobytes()
        dataset = h5file["data"]
        dataset.id.write_direct_chunk((0, 0, 0), raw, filter_mask=1)
        dataset.id.write_direct_chunk((2, 0, 0), b"\0" * 100, filter_mask=0)

    with h5py.File(filename, "r") as h5file:
        widget = NumpyAxesSelector()
        widget.setAxisNames(["x", "y"])
        widget.setData(h5file["data"])

        # Frame 0 can be read.
        widget.setSelection((0, slice(None), slice(None)))
        assert widget.selectedData().sum() == 16

        # Frame 2 needs the missing filter: the read raises. The widget must
        # not keep the previous selection nor let the exception escape.
        widget.setSelection((2, slice(None), slice(None)))
        selected = widget.selectedData()
        assert selected is not None
        assert selected.shape == (4, 4)
        assert selected.sum() == 0
