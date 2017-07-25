# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""
Print preview action, to send the content of a plot to a print preview page.
The plot content can then be moved on the page and resized prior to printing.

Classes
-------

- :class:`PrintPreviewAction`
- :class:`SingletonPrintPreviewAction`

Examples
--------

Simple example
++++++++++++++

.. code-block:: python

    from silx.gui import qt
    from silx.gui.plot import PlotWidget
    from silx.gui.plot.actions import printpreview
    import numpy

    app = qt.QApplication([])

    pw = PlotWidget()
    toolbar = qt.QToolBar()
    action = printpreview.PrintPreviewAction(plot=pw)
    pw.addToolBar(toolbar)
    toolbar.addAction(action)
    pw.show()

    x = numpy.arange(1000)
    y = x / numpy.sin(x)
    pw.addCurve(x, y)

    app.exec_()

Singleton example
+++++++++++++++++

This example illustrates how to print the content of several different
plots on the same page. The plots all instantiate a
:class:`SingletonPrintPreviewAction`, which relies on a singleton widget
:class:`silx.gui.widgets.PrintPreview.SingletonPrintPreviewDialog`.

.. image:: img/printPreviewMultiPlot.png

.. code-block:: python

    from silx.gui import qt
    from silx.gui.plot import PlotWidget
    from silx.gui.plot.actions import printpreview
    import numpy

    app = qt.QApplication([])

    plot_widgets = []

    for i in range(3):
        pw = PlotWidget()
        toolbar = qt.QToolBar()
        action = printpreview.SingletonPrintPreviewAction(plot=pw,
                                                          parent=pw)
        pw.addToolBar(toolbar)
        toolbar.addAction(action)
        pw.show()
        plot_widgets.append(pw)

    x = numpy.arange(1000)

    plot_widgets[0].addCurve(x, numpy.sin(x * 2 * numpy.pi / 1000))
    plot_widgets[1].addCurve(x, numpy.cos(x * 2 * numpy.pi / 1000))
    plot_widgets[2].addCurve(x, numpy.tan(x * 2 * numpy.pi / 1000))

    app.exec_()

"""
from __future__ import absolute_import

import logging
from io import StringIO

from . import PlotAction
from ... import qt
from ...widgets.PrintPreview import PrintPreviewDialog, SingletonPrintPreviewDialog
from ...widgets.PrintGeometryDialog import PrintGeometryDialog

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "18/07/2017"

_logger = logging.getLogger(__name__)
# _logger.setLevel(logging.DEBUG)


class PrintPreviewAction(PlotAction):
    """QAction to open a :class:`PrintPreviewDialog` (if not already open)
    and add the current plot to its page to be printed.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """
    def __init__(self, plot, parent=None):
        super(PrintPreviewAction, self).__init__(
            plot, icon='document-print', text='Print preview',
            tooltip='Send plot data to a print preview dialog',
            triggered=self._plotToPrintPreview, parent=parent)
        self._printPreviewDialog = None
        self.printConfigurationDialog = None

        self._printConfiguration = {"xOffset": 0.1,
                                    "yOffset": 0.1,
                                    "width": 0.9,
                                    "height": 0.9,
                                    "units": "page",
                                    "keepAspectRatio": True}

    @property
    def printPreviewDialog(self):
        """Lazy loaded :class:`PrintPreviewDialog`"""
        if self._printPreviewDialog is None:
            self._printPreviewDialog = PrintPreviewDialog(self.parent())
            self._printPreviewDialog.sigSetupButtonClicked.connect(
                    self._setPrintConfiguration)
        return self._printPreviewDialog

    def _plotToPrintPreview(self):
        """Grab the plot widget and send it to the print preview dialog.
        Make sure the print preview dialog is shown and raised."""
        self.printPreviewDialog.ensurePrinterIsSet()

        if qt.HAS_SVG:
            svgRenderer, viewBox = self._getSvgRendererAndViewBox()
            self.printPreviewDialog.addSvgItem(svgRenderer,
                                               viewBox=viewBox)
        else:
            _logger.warning("Missing QtSvg library, using a raster image")
            if qt.BINDING in ["PyQt4", "PySide"]:
                pixmap = qt.QPixmap.grabWidget(self.plot.centralWidget())
            else:
                # PyQt5 and hopefully PyQt6+
                pixmap = self.plot.centralWidget().grab()
            self.printPreviewDialog.addPixmap(pixmap)
        self.printPreviewDialog.show()
        self.printPreviewDialog.raise_()

    def _getSvgRendererAndViewBox(self):
        """Return a SVG renderer displaying the plot and its viewbox
        (interactively specified by the user the first time this is called).

        The size of the renderer is adjusted to the printer configuration
        and to the geometry configuration (width, height, ratio) specified
        by the user."""
        imgData = StringIO()
        assert self.plot.saveGraph(imgData, fileFormat="svg"), \
            "Unable to save graph"
        imgData.flush()
        imgData.seek(0)
        svgData = imgData.read()

        svgRenderer = qt.QSvgRenderer()

        defaultViewBox = self.printPreviewDialog.getDefaultViewBox()
        if defaultViewBox is None:
            # opens a dialog and updates print configuration
            self._setPrintConfiguration()

        # Now it should be available
        defaultViewBox = self.printPreviewDialog.getDefaultViewBox()

        svgRenderer.setViewBox(defaultViewBox)

        xml_stream = qt.QXmlStreamReader(svgData.encode(errors="replace"))

        # This is for PyMca compatibility, to share a print preview with PyMca plots
        svgRenderer._viewBox = defaultViewBox
        svgRenderer._svgRawData = svgData.encode(errors="replace")
        svgRenderer._svgRendererData = xml_stream

        if not svgRenderer.load(xml_stream):
            raise RuntimeError("Cannot interpret svg data")

        return svgRenderer, defaultViewBox

    def _setPrintConfiguration(self):
        """Open a dialog to prompt the user to adjust print
        geometry parameters."""
        if self.printConfigurationDialog is None:
            self.printConfigurationDialog = PrintGeometryDialog(self.parent())

        self.printConfigurationDialog.setPrintGeometry(self._printConfiguration)
        if self.printConfigurationDialog.exec_():
            self._printConfiguration = self.printConfigurationDialog.getPrintGeometry()

            defaultPrintGeom = self._printConfiguration.copy()
            if self._printConfiguration["keepAspectRatio"]:
                defaultPrintGeom["aspectRatio"] = self._getPlotAspectRatio()
            else:
                defaultPrintGeom["aspectRatio"] = None
            del defaultPrintGeom["keepAspectRatio"]
            self.printPreviewDialog.setDefaultPrintGeometry(defaultPrintGeom)

    def _getPlotAspectRatio(self):
        widget = self.plot.centralWidget()
        graphWidth = float(widget.width())
        graphHeight = float(widget.height())
        return graphHeight / graphWidth


class SingletonPrintPreviewAction(PrintPreviewAction):
    """This class is similar to its parent class :class:`PrintPreviewAction`
    but it uses a singleton print preview widget.

    This allows for several plots to send their content to the
    same print page, and for users to arrange them."""
    @property
    def printPreviewDialog(self):
        if self._printPreviewDialog is None:
            self._printPreviewDialog = SingletonPrintPreviewDialog(self.parent())
        return self._printPreviewDialog


if __name__ == '__main__':
    from silx.gui.plot import PlotWidget
    import numpy
    app = qt.QApplication([])

    pw = PlotWidget()
    toolbar = qt.QToolBar()
    action = PrintPreviewAction(plot=pw)
    pw.addToolBar(toolbar)
    toolbar.addAction(action)
    pw.show()

    x = numpy.arange(1000)
    y = x / numpy.sin(x)
    pw.addCurve(x, y)

    app.exec_()
