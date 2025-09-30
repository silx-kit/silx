import h5py
import numpy

from silx.gui.data import DataViews
from silx.gui.data.DataViewer import DataViewer
from silx.gui.plot.items import ImageDataAggregated, ImageRgba


def test_image_no_interpretation(qWidgetFactory, tmp_path):
    widget: DataViewer = qWidgetFactory(DataViewer)
    with h5py.File(tmp_path / "test.h5", "w") as h5file:
        h5file.attrs["NX_class"] = "NXdata"
        h5file.attrs["signal"] = "signal"
        h5file.create_dataset(name="signal", data=numpy.random.random((100, 100)))

        widget.setData(h5file["/"])

        currentCompositeView = widget.currentAvailableViews()[0]
        assert isinstance(currentCompositeView, DataViews._NXdataView)
        currentView = currentCompositeView.getCurrentView()
        assert isinstance(currentView, DataViews._NXdataImageView)
        plot = currentView.getWidget().getPlot()
        isinstance(plot.getImage("signal"), ImageDataAggregated)


def test_rgb_image_with_interpretation(qWidgetFactory, tmp_path):
    widget: DataViewer = qWidgetFactory(DataViewer)
    with h5py.File(tmp_path / "test.h5", "w") as h5file:
        h5file.attrs["NX_class"] = "NXdata"
        h5file.attrs["signal"] = "rgb"
        signal = h5file.create_dataset(
            name="rgb", data=numpy.random.random((100, 100, 3))
        )
        signal.attrs["interpretation"] = "rgb-image"

        widget.setData(h5file["/"])

        currentCompositeView = widget.currentAvailableViews()[0]
        assert isinstance(currentCompositeView, DataViews._NXdataView)
        currentView = currentCompositeView.getCurrentView()
        assert isinstance(currentView, DataViews._NXdataImageView)
        plot = currentView.getWidget().getPlot()
        isinstance(plot.getImage("rgb"), ImageRgba)
