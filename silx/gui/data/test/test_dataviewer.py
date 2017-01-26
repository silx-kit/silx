# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
__date__ = "26/01/2017"

import os
import tempfile
import unittest
from contextlib import contextmanager

import numpy
from ..DataViewer import DataViewer

from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.test.utils import SignalListener
from silx.gui.test.utils import TestCaseQt

try:
    import h5py
except ImportError:
    h5py = None


class AbstractDataViewerTests(TestCaseQt):

    def create_widget(self):
        raise NotImplementedError()

    @contextmanager
    def h5_temporary_file(self):
        # create tmp file
        fd, tmp_name = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        data = numpy.arange(3 * 3 * 3)
        data.shape = 3, 3, 3
        # create h5 data
        h5file = h5py.File(tmp_name, "w")
        h5file["data"] = data
        yield h5file
        # clean up
        h5file.close()
        os.unlink(tmp_name)

    def test_text_data(self):
        data_list = ["aaa", int, 8, self]
        widget = self.create_widget()
        for data in data_list:
            widget.setData(data)
            self.assertEqual(DataViewer.RAW_MODE, widget.displayMode())

    def test_plot_1d_data(self):
        data = numpy.arange(3 ** 1)
        data.shape = [3] * 1
        widget = self.create_widget()
        widget.setData(data)
        self.assertEqual(DataViewer.PLOT1D_MODE, widget.displayMode())

    def test_plot_2d_data(self):
        data = numpy.arange(3 ** 2)
        data.shape = [3] * 2
        widget = self.create_widget()
        widget.setData(data)
        self.assertEqual(DataViewer.PLOT2D_MODE, widget.displayMode())

    def test_plot_3d_data(self):
        data = numpy.arange(3 ** 3)
        data.shape = [3] * 3
        widget = self.create_widget()
        widget.setData(data)
        availableModes = set([v.modeId() for v in widget.currentAvailableViews()])
        try:
            import silx.gui.plot3d  # noqa
            self.assertIn(DataViewer.PLOT3D_MODE, availableModes)
        except ImportError:
            self.assertIn(DataViewer.STACK_MODE, availableModes)
        self.assertEqual(DataViewer.PLOT2D_MODE, widget.displayMode())

    def test_array_1d_data(self):
        data = numpy.array(["aaa"] * (3 ** 1))
        data.shape = [3] * 1
        widget = self.create_widget()
        widget.setData(data)
        self.assertEqual(DataViewer.RAW_MODE, widget.displayedView().modeId())

    def test_array_2d_data(self):
        data = numpy.array(["aaa"] * (3 ** 2))
        data.shape = [3] * 2
        widget = self.create_widget()
        widget.setData(data)
        self.assertEqual(DataViewer.RAW_MODE, widget.displayedView().modeId())

    def test_array_4d_data(self):
        data = numpy.array(["aaa"] * (3 ** 4))
        data.shape = [3] * 4
        widget = self.create_widget()
        widget.setData(data)
        self.assertEqual(DataViewer.RAW_MODE, widget.displayedView().modeId())

    def test_record_4d_data(self):
        data = numpy.zeros(3 ** 4, dtype='3int8, float32, (2,3)float64')
        data.shape = [3] * 4
        widget = self.create_widget()
        widget.setData(data)
        self.assertEqual(DataViewer.RAW_MODE, widget.displayedView().modeId())

    def test_3d_h5_dataset(self):
        if h5py is None:
            self.skipTest("h5py library is not available")
        with self.h5_temporary_file() as h5file:
            dataset = h5file["data"]
            widget = self.create_widget()
            widget.setData(dataset)

    def test_data_event(self):
        listener = SignalListener()
        widget = self.create_widget()
        widget.dataChanged.connect(listener)
        widget.setData(10)
        widget.setData(None)
        self.assertEquals(listener.callCount(), 2)

    def test_display_mode_event(self):
        listener = SignalListener()
        widget = self.create_widget()
        widget.displayedViewChanged.connect(listener)
        widget.setData(10)
        widget.setData(None)
        modes = [v.modeId() for v in listener.arguments(argumentIndex=0)]
        self.assertEquals(modes, [DataViewer.RAW_MODE, DataViewer.EMPTY_MODE])
        listener.clear()

    def test_change_display_mode(self):
        data = numpy.arange(10 ** 4)
        data.shape = [10] * 4
        widget = self.create_widget()
        widget.setData(data)
        widget.setDisplayMode(DataViewer.PLOT1D_MODE)
        self.assertEquals(widget.displayedView().modeId(), DataViewer.PLOT1D_MODE)
        widget.setDisplayMode(DataViewer.PLOT2D_MODE)
        self.assertEquals(widget.displayedView().modeId(), DataViewer.PLOT2D_MODE)
        widget.setDisplayMode(DataViewer.RAW_MODE)
        self.assertEquals(widget.displayedView().modeId(), DataViewer.RAW_MODE)
        widget.setDisplayMode(DataViewer.EMPTY_MODE)
        self.assertEquals(widget.displayedView().modeId(), DataViewer.EMPTY_MODE)


class TestDataViewer(AbstractDataViewerTests):
    def create_widget(self):
        return DataViewer()


class TestDataViewerFrame(AbstractDataViewerTests):
    def create_widget(self):
        return DataViewerFrame()


def suite():
    test_suite = unittest.TestSuite()
    loadTestsFromTestCase = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite.addTest(loadTestsFromTestCase(TestDataViewer))
    test_suite.addTest(loadTestsFromTestCase(TestDataViewerFrame))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
