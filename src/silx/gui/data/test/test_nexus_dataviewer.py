import h5py
import numpy
import pytest
import sys


from silx.gui.data import DataViews
from silx.gui.data.DataViewer import DataViewer
from silx.gui.plot import Plot2D
from silx.gui.plot.items import ImageDataAggregated, ImageRgba
from silx.gui.plot3d.items import Scatter3D


def test_image_no_interpretation(qapp, qWidgetFactory, tmp_path):
    widget: DataViewer = qWidgetFactory(DataViewer)
    with h5py.File(tmp_path / "test.h5", "w") as h5file:
        h5file.attrs["NX_class"] = "NXdata"
        h5file.attrs["signal"] = "signal"
        h5file.create_dataset(name="signal", data=numpy.random.random((10, 100)))
        h5file.create_dataset(name="y", data=numpy.arange(-5, 5))
        h5file.attrs["axes"] = ["y", "."]

        widget.setData(h5file["/"])

        qapp.processEvents()

        viewClasses = tuple(view.__class__ for view in widget.currentAvailableViews())
        assert viewClasses == (
            DataViews._NXdataImageView,
            DataViews._NXdataCurveView,
            DataViews._Hdf5View,
        )
        imageView = widget.currentAvailableViews()[0]
        plot: Plot2D = imageView.getWidget().getPlot()
        assert isinstance(plot.getImage("signal"), ImageDataAggregated)
        # Disable keep aspect ratio so the limits of the axis match the limits of the image
        plot.setKeepDataAspectRatio(False)
        assert plot.getYAxis().getLimits() == (-5, 5)


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

        viewClasses = tuple(view.__class__ for view in widget.currentAvailableViews())
        assert viewClasses == (
            DataViews._NXDataRgbaImageView,
            DataViews._NXdataImageView,
            DataViews._NXdataCurveView,
            DataViews._NXdataVolumeView,
            DataViews._Hdf5View,
        )
        rgbImageView = widget.currentAvailableViews()[0]
        plot = rgbImageView.getWidget().getPlot()
        assert isinstance(plot.getImage("rgb"), ImageRgba)


def test_image_with_non_affine_axis_becomes_scatter(qapp, qWidgetFactory, tmp_path):
    widget: DataViewer = qWidgetFactory(DataViewer)
    with h5py.File(tmp_path / "test.h5", "w") as h5file:
        h5file.attrs["NX_class"] = "NXdata"
        h5file.attrs["signal"] = "signal"
        h5file.create_dataset(name="signal", data=numpy.random.random((10, 10)))
        h5file.create_dataset(name="non_affine_x", data=numpy.logspace(0, 1, 10))
        h5file.create_dataset(name="y", data=numpy.arange(10))
        h5file.attrs["axes"] = ["y", "non_affine_x"]

        widget.setData(h5file["/"])

        qapp.processEvents()

        imageView = widget.currentAvailableViews()[0]
        plot: Plot2D = imageView.getWidget().getPlot()
        assert plot.getImage() is None
        assert plot.getScatter() is not None


@pytest.mark.skipif(
    sys.version_info.major == 3 and sys.version_info.minor == 14,
    reason="Triggers segfault on Python 3.14. To be fixed",
)
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

        viewClasses = tuple(view.__class__ for view in widget.currentAvailableViews())
        assert viewClasses == (
            DataViews._NxDataScatter3D,
            DataViews._NXdataCurveView,
            DataViews._Hdf5View,
        )

        scatterView = widget.currentAvailableViews()[0]
        sceneWindow = scatterView.getWidget()
        plotItems = sceneWindow.getSceneWidget().getItems()
        assert len(plotItems) == 1

        plotItem = plotItems[0]
        assert isinstance(plotItem, Scatter3D)
        numpy.testing.assert_equal(plotItem.getXData(), x)
        assert len(plotItem.getSymbolSize()) == len(x)
