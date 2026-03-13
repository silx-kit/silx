# /*##########################################################################
#
# Copyright (c) 2016-2022 European Synchrotron Radiation Facility
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
__date__ = "19/02/2019"

import pytest

import numpy
from ..DataViewer import DataViewer
from ..DataViews import DataView
from .. import DataViews

from silx.gui import qt

from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.utils.testutils import SignalListener
from silx.gui.utils.testutils import TestCaseQt

import h5py


class _DataViewMock(DataView):
    """Dummy view to display nothing"""

    def __init__(self, parent):
        DataView.__init__(self, parent)

    def axesNames(self, data, info):
        return []

    def createWidget(self, parent):
        return qt.QLabel(parent)

    def getDataPriority(self, data, info):
        return 0


# https://docs.pytest.org/en/stable/how-to/fixtures.html#parametrizing-fixtures
@pytest.fixture(params=[DataViewer, DataViewerFrame])
def data_viewer_widget(request):
    widget = request.param(parent=None)
    yield widget


def test_text_data(data_viewer_widget):
    data_list = ["aaa", int, 8, object]
    for data in data_list:
        data_viewer_widget.setData(data)
        assert DataViews.RAW_MODE == data_viewer_widget.displayMode()


def test_plot_1d_data(data_viewer_widget):
    data = numpy.arange(3**1)
    data.shape = [3] * 1
    data_viewer_widget.setData(data)
    availableModes = {v.modeId() for v in data_viewer_widget.currentAvailableViews()}
    assert DataViews.RAW_MODE == data_viewer_widget.displayMode()
    assert DataViews.PLOT1D_MODE in availableModes


def test_image_data(data_viewer_widget):
    data = numpy.arange(3**2)
    data.shape = [3] * 2
    data_viewer_widget.setData(data)
    availableModes = {v.modeId() for v in data_viewer_widget.currentAvailableViews()}
    assert DataViews.RAW_MODE == data_viewer_widget.displayMode()
    assert DataViews.PLOT2D_MODE in availableModes


def test_image_bool(data_viewer_widget):
    data = numpy.zeros((10, 10), dtype=bool)
    data[::2, ::2] = True
    data_viewer_widget.setData(data)
    availableModes = {v.modeId() for v in data_viewer_widget.currentAvailableViews()}
    assert DataViews.RAW_MODE == data_viewer_widget.displayMode()
    assert DataViews.PLOT2D_MODE in availableModes


def test_image_complex_data(data_viewer_widget):
    data = numpy.arange(3**2, dtype=numpy.complex64)
    data.shape = [3] * 2
    data_viewer_widget.setData(data)
    availableModes = {v.modeId() for v in data_viewer_widget.currentAvailableViews()}
    assert DataViews.RAW_MODE == data_viewer_widget.displayMode()
    assert DataViews.COMPLEX_PLOT2D_MODE in availableModes


def test_plot_3d_data(data_viewer_widget):
    data = numpy.arange(3**3)
    data.shape = [3] * 3
    data_viewer_widget.setData(data)
    availableModes = {v.modeId() for v in data_viewer_widget.currentAvailableViews()}
    try:
        import silx.gui.plot3d  # noqa

        assert DataViews.PLOT3D_MODE in availableModes
    except ImportError:
        assert DataViews.PLOT2D_MODE in availableModes
    assert DataViews.RAW_MODE == data_viewer_widget.displayMode()


def test_array_1d_data(data_viewer_widget):
    data = numpy.array(["aaa"] * (3**1))
    data.shape = [3] * 1
    data_viewer_widget.setData(data)
    assert DataViews.RAW_MODE == data_viewer_widget.displayedView().modeId()


def test_array_2d_data(data_viewer_widget):
    data = numpy.array(["aaa"] * (3**2))
    data.shape = [3] * 2
    data_viewer_widget.setData(data)
    assert DataViews.RAW_MODE == data_viewer_widget.displayedView().modeId()


def test_array_4d_data(data_viewer_widget):
    data = numpy.array(["aaa"] * (3**4))
    data.shape = [3] * 4
    data_viewer_widget.setData(data)
    assert DataViews.RAW_MODE == data_viewer_widget.displayedView().modeId()


def test_record_4d_data(data_viewer_widget):
    data = numpy.zeros(3**4, dtype="3int8, float32, (2,3)float64")
    data.shape = [3] * 4
    data_viewer_widget.setData(data)
    assert DataViews.RAW_MODE == data_viewer_widget.displayedView().modeId()


def test_3d_h5_dataset(data_viewer_widget, tmp_path):
    data = numpy.arange(3 * 3 * 3)
    data.shape = 3, 3, 3
    with h5py.File(tmp_path / "tmp.h5", "w") as h5file:
        h5file["data"] = data

    with h5py.File(tmp_path / "tmp.h5", "r") as h5file:
        dataset = h5file["data"]
        data_viewer_widget.setData(dataset)


def test_data_event(data_viewer_widget):
    listener = SignalListener()
    data_viewer_widget.dataChanged.connect(listener)
    data_viewer_widget.setData(10)
    data_viewer_widget.setData(None)
    assert listener.callCount() == 2


def test_display_mode_event(data_viewer_widget):
    listener = SignalListener()
    data_viewer_widget.displayedViewChanged.connect(listener)
    data_viewer_widget.setData(10)
    data_viewer_widget.setData(None)
    modes = [v.modeId() for v in listener.arguments(argumentIndex=0)]
    assert modes == [DataViews.RAW_MODE, DataViews.EMPTY_MODE]
    listener.clear()


def test_create_default_views(data_viewer_widget):
    views = data_viewer_widget.createDefaultViews()
    assert len(views) > 0


def test_add_view(data_viewer_widget):
    view = _DataViewMock(data_viewer_widget)
    data_viewer_widget.addView(view)
    assert view in data_viewer_widget.availableViews()
    assert view in data_viewer_widget.currentAvailableViews()


def test_remove_view(data_viewer_widget):
    data_viewer_widget.setData("foobar")
    view = data_viewer_widget.currentAvailableViews()[0]
    data_viewer_widget.removeView(view)
    assert view not in data_viewer_widget.availableViews()
    assert view not in data_viewer_widget.currentAvailableViews()


def test_replace_view(data_viewer_widget):
    view = _DataViewMock(data_viewer_widget)
    data_viewer_widget.replaceView(DataViews.RAW_MODE, view)
    assert data_viewer_widget.getViewFromModeId(DataViews.RAW_MODE) is None
    assert view in data_viewer_widget.availableViews()
    assert view in data_viewer_widget.currentAvailableViews()


def test_replace_view_in_composite(data_viewer_widget):
    # replace a view that is a child of a composite view
    view = _DataViewMock(data_viewer_widget)
    replaced = data_viewer_widget.replaceView(DataViews.NXDATA_INVALID_MODE, view)
    assert replaced
    nxdata_view = data_viewer_widget.getViewFromModeId(DataViews.NXDATA_MODE)
    assert DataViews.NXDATA_INVALID_MODE not in [
        v.modeId() for v in nxdata_view.getViews()
    ]
    assert view in nxdata_view.getViews()


class _TestAbstractDataViewer(TestCaseQt):
    __test__ = False  # ignore abstract class

    def create_widget(self):
        # Avoid to raise an error when testing the full module
        self.skipTest("Not implemented")

    def test_change_display_mode(self):
        listener = SignalListener()
        data = numpy.arange(10**4)
        data.shape = [10] * 4
        widget = self.create_widget()
        widget.selectionChanged.connect(listener)
        widget.setData(data)

        widget.setDisplayMode(DataViews.PLOT1D_MODE)
        assert widget.displayedView().modeId() == DataViews.PLOT1D_MODE
        self.qWait(200)
        assert listener.arguments() == [((0, 0, 0, slice(None)), None)]
        listener.clear()

        widget.setDisplayMode(DataViews.PLOT2D_MODE)
        self.assertEqual(widget.displayedView().modeId(), DataViews.PLOT2D_MODE)
        self.qWait(200)
        assert listener.arguments() == [((0, 0, slice(None), slice(None)), None)]
        listener.clear()

        widget.setDisplayMode(DataViews.RAW_MODE)
        self.assertEqual(widget.displayedView().modeId(), DataViews.RAW_MODE)
        self.qWait(200)
        # Changing from 2D to 2D view: Selection didn't changed
        assert listener.callCount() == 0

        widget.setDisplayMode(DataViews.EMPTY_MODE)
        self.assertEqual(widget.displayedView().modeId(), DataViews.EMPTY_MODE)
        self.qWait(200)
        assert listener.arguments() == [(None, None)]
        listener.clear()


class TestDataViewer(_TestAbstractDataViewer):
    __test__ = True  # because _TestAbstractDataViewer is ignored

    def create_widget(self):
        return DataViewer()


class TestDataViewerFrame(_TestAbstractDataViewer):
    __test__ = True  # because _TestAbstractDataViewer is ignored

    def create_widget(self):
        return DataViewerFrame()


class TestDataView(TestCaseQt):
    def createComplexData(self):
        line = [1, 2j, 3 + 3j, 4]
        image = [line, line, line, line]
        cube = [image, image, image, image]
        data = numpy.array(cube, dtype=numpy.complex64)
        return data

    def createDataViewWithData(self, dataViewClass, data):
        viewer = dataViewClass(None)
        widget = viewer.getWidget()
        viewer.setData(data)
        return widget

    def testCurveWithComplex(self):
        data = self.createComplexData()
        dataViewClass = DataViews._Plot1dView
        widget = self.createDataViewWithData(dataViewClass, data[0, 0])
        self.qWaitForWindowExposed(widget)

    def testImageWithComplex(self):
        data = self.createComplexData()
        dataViewClass = DataViews._Plot2dView
        widget = self.createDataViewWithData(dataViewClass, data[0])
        self.qWaitForWindowExposed(widget)

    @pytest.mark.usefixtures("use_opengl")
    def testCubeWithComplex(self):
        try:
            import silx.gui.plot3d  # noqa
        except ImportError:
            self.skipTest("OpenGL not available")
        data = self.createComplexData()
        dataViewClass = DataViews._Plot3dView
        widget = self.createDataViewWithData(dataViewClass, data)
        self.qWaitForWindowExposed(widget)
