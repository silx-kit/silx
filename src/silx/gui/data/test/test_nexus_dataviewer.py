import h5py
import numpy

from silx.gui.data import DataViews
from silx.gui.data.DataViewer import DataViewer
from silx.gui.plot.items import ImageDataAggregated, ImageRgba
from silx.gui.plot3d.items import Scatter3D


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


def test_rgb_image_with_interpretation(qapp, qWidgetFactory, tmp_path):
    widget: DataViewer = qWidgetFactory(DataViewer)
    with h5py.File(tmp_path / "test.h5", "w") as h5file:
        h5file.attrs["NX_class"] = "NXdata"
        h5file.attrs["signal"] = "rgb"
        signal = h5file.create_dataset(
            name="rgb", data=numpy.random.random((100, 100, 3))
        )
        signal.attrs["interpretation"] = "rgb-image"

        widget.setData(h5file["/"])

        qapp.processEvents()

        currentCompositeView = widget.currentAvailableViews()[0]
        assert isinstance(currentCompositeView, DataViews._NXdataView)
        currentView = currentCompositeView.getCurrentView()
        assert isinstance(currentView, DataViews._NXDataRgbaImageView)
        plot = currentView.getWidget().getPlot()
        assert isinstance(plot.getImage("rgb"), ImageRgba)


def test_3d_scatter(qapp, qWidgetFactory, tmp_path):
    widget: DataViewer = qWidgetFactory(DataViewer)

    x = numpy.arange(500)

    with h5py.File(tmp_path / "scatter.h5", "w") as h5file:
        h5file.attrs["signal"] = "intensity"
        h5file.attrs["auxiliary_signals"] = ["sizes"]
        h5file.attrs["axes"] = ("x", "y", "z")
        h5file.attrs["NX_class"] = "NXdata"
        h5file.create_dataset("intensity", data=numpy.random.random((len(x))))
        h5file.create_dataset("sizes", data=numpy.arange(len(x)))
        h5file.create_dataset("x", data=x)
        h5file.create_dataset("y", data=x)
        h5file.create_dataset("z", data=x)

        widget.setData(h5file["/"])

        qapp.processEvents()

        currentCompositeView = widget.currentAvailableViews()[0]
        assert isinstance(currentCompositeView, DataViews._NXdataView)
        currentView = currentCompositeView.getCurrentView()
        assert isinstance(currentView, DataViews._NxDataScatter3D)

        sceneWindow = currentView.getWidget()
        plotItems = sceneWindow.getSceneWidget().getItems()
        assert len(plotItems) == 1

        plotItem = plotItems[0]
        assert isinstance(plotItem, Scatter3D)
        numpy.testing.assert_equal(plotItem.getXData(), x)
        assert len(plotItem.getSymbolSize()) == len(x)
