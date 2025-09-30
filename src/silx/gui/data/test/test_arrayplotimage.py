import h5py
import numpy

from silx.gui.data.NXdataWidgets import ArrayImagePlot
from silx.gui.plot.items import ImageDataAggregated, ImageRgba


def test_image(qWidgetFactory, tmp_path):
    widget: ArrayImagePlot = qWidgetFactory(ArrayImagePlot)
    with h5py.File(tmp_path / "test.h5", "w") as h5file:
        signal = h5file.create_dataset(
            name="signal", data=numpy.random.random((100, 100))
        )

        widget.setImageData(signals=[signal], signals_names=["signal"])

    assert isinstance(widget.getPlot().getImage("signal"), ImageDataAggregated)


def test_rgb_image(qWidgetFactory, tmp_path):
    widget: ArrayImagePlot = qWidgetFactory(ArrayImagePlot)
    with h5py.File(tmp_path / "test.h5", "w") as h5file:
        signal = h5file.create_dataset(
            name="rgb", data=numpy.random.random((100, 100, 3))
        )

        widget.setImageData(signals=[signal], signals_names=["rgb"], isRgba=True)

    assert isinstance(widget.getPlot().getImage("rgb"), ImageRgba)
