"""Tests for PlotWidget.saveGraph and PlotWindow.saveGraph"""

from io import BytesIO, StringIO

import numpy
import pytest

from silx.gui import qt
from silx.gui.plot import PlotWidget, PlotWindow


@pytest.fixture(params=[PlotWidget, PlotWindow])
def plotWithImage(qWidgetFactory, request):
    plot = qWidgetFactory(request.param)
    plot.addImage(numpy.arange(100).reshape(10, 10))
    yield plot


@pytest.mark.parametrize("fileFormat", ["png", None])
def testSaveGraphPngToFile(plotWithImage, tmp_path, fileFormat):
    """Test saveGraph to a png file path, with explicit or deduced format"""
    filename = str(tmp_path / "plot.png")
    assert plotWithImage.saveGraph(filename, fileFormat=fileFormat)
    assert not qt.QImage(filename, "png").isNull()


@pytest.mark.skipif(not qt.HAS_SVG, reason="QtSvg not available")
@pytest.mark.parametrize("fileFormat", ["svg", None])
def testSaveGraphSvgToFile(plotWithImage, tmp_path, fileFormat):
    """Test saveGraph to a svg file path, with explicit or deduced format"""
    filename = str(tmp_path / "plot.svg")
    assert plotWithImage.saveGraph(filename, fileFormat=fileFormat)
    assert qt.QSvgRenderer(filename).isValid()


@pytest.mark.parametrize("dpi", [None, 150])
def testSaveGraphPngToBytesIO(plotWithImage, dpi):
    """Test saveGraph to a BytesIO in png format"""
    output = BytesIO()
    assert plotWithImage.saveGraph(output, fileFormat="png", dpi=dpi)
    assert not qt.QImage.fromData(output.getvalue(), "png").isNull()


@pytest.mark.skipif(not qt.HAS_SVG, reason="QtSvg not available")
def testSaveGraphSvgToStringIO(plotWithImage):
    """Test saveGraph to a StringIO in svg format"""
    output = StringIO()
    assert plotWithImage.saveGraph(output, fileFormat="svg")
    svgData = qt.QByteArray(output.getvalue().encode("utf-8"))
    assert qt.QSvgRenderer(svgData).isValid()


def testSaveGraphFileLikeWithoutFormat(plotWithImage):
    """Test saveGraph fails when format cannot be deduced"""
    assert not plotWithImage.saveGraph(BytesIO())


def testSaveGraphUnsupportedFormat(plotWithImage, tmp_path):
    """Test saveGraph fails with an unsupported format"""
    assert not plotWithImage.saveGraph(str(tmp_path / "plot.xyz"))


def testCopyAction(qWidgetFactory):
    """Test copying a PlotWindow to the clipboard"""
    plot = qWidgetFactory(PlotWindow)
    plot.addImage(numpy.arange(100).reshape(10, 10))
    clipboard = qt.QApplication.clipboard()
    clipboard.clear()

    plot.getCopyAction().trigger()

    assert not clipboard.image().isNull()
