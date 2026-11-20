import numpy

from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.PrintPreviewToolButton import PrintPreviewToolButton


def test_print_preview_svg_renderer(qapp, tmp_path):
    """The plot is rendered as a valid SVG in the print preview"""
    plot = PlotWidget()
    plot.addCurve(numpy.arange(10), numpy.arange(10))
    button = PrintPreviewToolButton(parent=plot, plot=plot)
    printer = qt.QPrinter(qt.QPrinter.HighResolution)
    printer.setOutputFileName(str(tmp_path / "preview.pdf"))
    button.printPreviewDialog.printer = printer

    svgRenderer, _ = button._getSvgRendererAndViewbox()

    assert svgRenderer.isValid()
    plot.close()
