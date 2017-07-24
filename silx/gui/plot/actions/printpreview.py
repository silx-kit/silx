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
plots on the same page. The plot instantiate a
:class:`SingletonPrintPreviewAction`, which relies on a singleton widget
:class:`SingletonPrintPreviewDialog`.

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
        if self._printPreviewDialog is None:
            self._printPreviewDialog = PrintPreviewDialog(self.parent())
        return self._printPreviewDialog

    def _plotToPrintPreview(self):
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
        """Return a SVG renderer displaying the plot.
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

        printer = self.printPreviewDialog.printer

        self._getPrintConfiguration()     # opens a dialog and updates _printConfiguration
        config = self._printConfiguration
        width = config['width']
        height = config['height']
        xOffset = config['xOffset']
        yOffset = config['yOffset']
        units = config['units']
        keepAspectRatio = config['keepAspectRatio']
        _logger.debug("Requested print configuration %s",
                      config)

        dpix = printer.logicalDpiX()
        dpiy = printer.logicalDpiY()

        availableWidth = printer.width()
        availableHeight = printer.height()

        _logger.debug("Printer parameters: width %f; height %f; " +
                      "logicalDpiX: %f; logicalDpiY: %f",
                      availableWidth, availableHeight, dpix, dpiy)

        # convert the offsets to dpi
        if units.lower() in ['inch', 'inches']:
            xOffset = xOffset * dpix
            yOffset = yOffset * dpiy
            if width is not None:
                width = width * dpix
            if height is not None:
                height = height * dpiy
        elif units.lower() in ['cm', 'centimeters']:
            xOffset = (xOffset / 2.54) * dpix
            yOffset = (yOffset / 2.54) * dpiy
            if width is not None:
                width = (width / 2.54) * dpix
            if height is not None:
                height = (height / 2.54) * dpiy
        else:
            # page units
            xOffset = availableWidth * xOffset
            yOffset = availableHeight * yOffset
            if width is not None:
                width = availableWidth * width
            if height is not None:
                height = availableHeight * height

        _logger.debug("Parameters in dots (dpi): width %f; height %f; " +
                      "xOffset: %f; yOffset: %f",
                      width, height, xOffset, yOffset)

        availableWidth -= xOffset
        availableHeight -= yOffset

        if width is not None:
            if (availableWidth + 0.1) < width:
                txt = "Available width  %f is less than requested width %f" % \
                              (availableWidth, width)
                raise ValueError(txt)
        if height is not None:
            if (availableHeight + 0.1) < height:
                txt = "Available height  %f is less than requested height %f" % \
                              (availableHeight, height)
                raise ValueError(txt)

        if keepAspectRatio:
            # get the aspect ratio
            _logger.debug("Preserving aspect ratio")
            widget = self.plot.centralWidget()
            graphWidth = float(widget.width())
            graphHeight = float(widget.height())
            graphRatio = graphHeight / graphWidth

            bodyWidth = width or availableWidth
            bodyHeight = bodyWidth * graphRatio
            _logger.debug("Calculated bodyWidth and bodyHeight: %f, %f",
                          bodyWidth, bodyHeight)

            if bodyHeight > availableHeight:
                bodyHeight = availableHeight
                bodyWidth = bodyHeight / graphRatio

        else:
            bodyWidth = width or availableWidth
            bodyHeight = height or availableHeight

        _logger.debug("Final parameters after taking available space"
                      " into accout: bodyWidth %f; bodyWidth %f; "
                      "xOffset %f; yOffset %f",
                      bodyWidth, bodyHeight, xOffset, yOffset)

        body = qt.QRectF(xOffset,
                         yOffset,
                         bodyWidth,
                         bodyHeight)

        svgRenderer.setViewBox(body)
        # FIXME: this info svgRenderer.viewBox seems to be lost, that's why we also
        # need to return body and pass it to PrintPreviewDialog.addSvgItem. Why?

        xml_stream = qt.QXmlStreamReader(svgData.encode(errors="replace"))

        # This is for PyMca compatibility, to share a print preview with PyMca plots
        svgRenderer._viewBox = body
        svgRenderer._svgRawData = svgData.encode(errors="replace")
        svgRenderer._svgRendererData = xml_stream

        if not svgRenderer.load(xml_stream):
            raise RuntimeError("Cannot interpret svg data")

        return svgRenderer, body

    def _getPrintConfiguration(self):
        """Open a dialog to prompt the user to adjust print parameters."""
        if self.printConfigurationDialog is None:
            self.printConfigurationDialog = PrintGeometryDialog(self.parent())

        self.printConfigurationDialog.setPrintGeometry(self._printConfiguration)
        if self.printConfigurationDialog.exec_():
            self._printConfiguration = self.printConfigurationDialog.getPrintGeometry()


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
