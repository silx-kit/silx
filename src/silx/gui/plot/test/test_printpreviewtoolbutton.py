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


def test_print_geometry_too_big_is_refused(qapp, qapp_utils, tmp_path):
    """A print size larger than the page is refused and the previous geometry is kept"""
    plot = PlotWidget()
    plot.addCurve(numpy.arange(10), numpy.arange(10))
    button = PrintPreviewToolButton(parent=plot, plot=plot)
    printer = qt.QPrinter(qt.QPrinter.HighResolution)
    printer.setOutputFileName(str(tmp_path / "preview.pdf"))
    button.printPreviewDialog.printer = printer
    oldGeometry = dict(button._printGeometry)

    def setBigWidthAndAccept():
        dialog = button._printConfigurationDialog
        dialog.setPrintGeometry({"width": 100000, "units": "page"})
        qt.QTimer.singleShot(0, closeWarning)
        dialog.okButton.click()

    def closeWarning():
        qt.QApplication.activeModalWidget().close()

    qt.QTimer.singleShot(0, setBigWidthAndAccept)
    button._setPrintConfiguration()

    assert button._printGeometry == oldGeometry
    button._getViewBox(button._printGeometry)  # does not raise
    plot.close()
