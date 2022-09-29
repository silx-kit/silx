# /*##########################################################################
#
# Copyright (c) 2017-2021 European Synchrotron Radiation Facility
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
This modules provides tool buttons to send the content of a plot to a
print preview page.
The plot content can then be moved on the page and resized prior to printing.

Classes
-------

- :class:`PrintPreviewToolButton`
- :class:`SingletonPrintPreviewToolButton`

Examples
--------

Simple example
++++++++++++++

.. code-block:: python

    from silx.gui import qt
    from silx.gui.plot import PlotWidget
    from silx.gui.plot.PrintPreviewToolButton import PrintPreviewToolButton
    import numpy

    app = qt.QApplication([])

    pw = PlotWidget()
    toolbar = qt.QToolBar(pw)
    toolbutton = PrintPreviewToolButton(parent=toolbar, plot=pw)
    pw.addToolBar(toolbar)
    toolbar.addWidget(toolbutton)
    pw.show()

    x = numpy.arange(1000)
    y = x / numpy.sin(x)
    pw.addCurve(x, y)

    app.exec()

Singleton example
+++++++++++++++++

This example illustrates how to print the content of several different
plots on the same page. The plots all instantiate a
:class:`SingletonPrintPreviewToolButton`, which relies on a singleton widget
(:class:`silx.gui.widgets.PrintPreview.SingletonPrintPreviewDialog`).

.. image:: img/printPreviewMultiPlot.png

.. code-block:: python

    from silx.gui import qt
    from silx.gui.plot import PlotWidget
    from silx.gui.plot.PrintPreviewToolButton import SingletonPrintPreviewToolButton
    import numpy

    app = qt.QApplication([])

    plot_widgets = []

    for i in range(3):
        pw = PlotWidget()
        toolbar = qt.QToolBar(pw)
        toolbutton = SingletonPrintPreviewToolButton(parent=toolbar,
                                                     plot=pw)
        pw.addToolBar(toolbar)
        toolbar.addWidget(toolbutton)
        pw.show()
        plot_widgets.append(pw)

    x = numpy.arange(1000)

    plot_widgets[0].addCurve(x, numpy.sin(x * 2 * numpy.pi / 1000))
    plot_widgets[1].addCurve(x, numpy.cos(x * 2 * numpy.pi / 1000))
    plot_widgets[2].addCurve(x, numpy.tan(x * 2 * numpy.pi / 1000))

    app.exec()

"""

import logging
from io import StringIO

from .. import qt
from .. import icons
from . import PlotWidget
from ..widgets.PrintPreview import PrintPreviewDialog, SingletonPrintPreviewDialog
from ..widgets.PrintGeometryDialog import PrintGeometryDialog
from silx.utils.deprecation import deprecated

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "20/12/2018"

_logger = logging.getLogger(__name__)
# _logger.setLevel(logging.DEBUG)


class PrintPreviewToolButton(qt.QToolButton):
    """QToolButton to open a :class:`PrintPreviewDialog` (if not already open)
    and add the current plot to its page to be printed.

    :param parent: See :class:`QAction`
    :param plot: :class:`.PlotWidget` instance on which to operate
    """
    def __init__(self, parent=None, plot=None):
        super(PrintPreviewToolButton, self).__init__(parent)

        if not isinstance(plot, PlotWidget):
            raise TypeError("plot parameter must be a PlotWidget")
        self._plot = plot

        self.setIcon(icons.getQIcon('document-print'))

        printGeomAction = qt.QAction("Print geometry", self)
        printGeomAction.setToolTip("Define a print geometry prior to sending "
                                   "the plot to the print preview dialog")
        printGeomAction.setIcon(icons.getQIcon('shape-rectangle'))
        printGeomAction.triggered.connect(self._setPrintConfiguration)

        printPreviewAction = qt.QAction("Print preview", self)
        printPreviewAction.setToolTip("Send plot to the print preview dialog")
        printPreviewAction.setIcon(icons.getQIcon('document-print'))
        printPreviewAction.triggered.connect(self._plotToPrintPreview)

        menu = qt.QMenu(self)
        menu.addAction(printGeomAction)
        menu.addAction(printPreviewAction)
        self.setMenu(menu)
        self.setPopupMode(qt.QToolButton.InstantPopup)

        self._printPreviewDialog = None
        self._printConfigurationDialog = None

        self._printGeometry = {"xOffset": 0.1,
                               "yOffset": 0.1,
                               "width": 0.9,
                               "height": 0.9,
                               "units": "page",
                               "keepAspectRatio": True}

    @property
    def printPreviewDialog(self):
        """Lazy loaded :class:`PrintPreviewDialog`"""
        # if changes are made here, don't forget making them in
        # SingletonPrintPreviewToolButton.printPreviewDialog as well
        if self._printPreviewDialog is None:
            self._printPreviewDialog = PrintPreviewDialog(self.parent())
        return self._printPreviewDialog

    def getTitle(self):
        """Implement this method to fetch the title in the plot.

        :return: Title to be printed above the plot, or None (no title added)
        :rtype: str or None
        """
        return None

    def getCommentAndPosition(self):
        """Implement this method to fetch the legend to be printed below the
        figure and its position.

        :return: Legend to be printed below the figure and its position:
            "CENTER", "LEFT" or "RIGHT"
        :rtype: (str, str) or (None, None)
        """
        return None, None

    @property
    @deprecated(since_version="0.10",
                replacement="getPlot()")
    def plot(self):
        return self._plot

    def getPlot(self):
        """Return the :class:`.PlotWidget` associated with this tool button.

        :rtype: :class:`.PlotWidget`
        """
        return self._plot

    def _plotToPrintPreview(self):
        """Grab the plot widget and send it to the print preview dialog.
        Make sure the print preview dialog is shown and raised."""
        if not self.printPreviewDialog.ensurePrinterIsSet():
            return

        comment, commentPosition = self.getCommentAndPosition()

        if qt.HAS_SVG:
            svgRenderer, viewBox = self._getSvgRendererAndViewbox()
            self.printPreviewDialog.addSvgItem(svgRenderer,
                                               title=self.getTitle(),
                                               comment=comment,
                                               commentPosition=commentPosition,
                                               viewBox=viewBox,
                                               keepRatio=self._printGeometry["keepAspectRatio"])
        else:
            _logger.warning("Missing QtSvg library, using a raster image")
            pixmap = self._plot.centralWidget().grab()
            self.printPreviewDialog.addPixmap(pixmap,
                                              title=self.getTitle(),
                                              comment=comment,
                                              commentPosition=commentPosition)
        self.printPreviewDialog.show()
        self.printPreviewDialog.raise_()

    def _getSvgRendererAndViewbox(self):
        """Return a SVG renderer displaying the plot and its viewbox
        (interactively specified by the user the first time this is called).

        The size of the renderer is adjusted to the printer configuration
        and to the geometry configuration (width, height, ratio) specified
        by the user."""
        imgData = StringIO()
        assert self._plot.saveGraph(imgData, fileFormat="svg"), \
            "Unable to save graph"
        imgData.flush()
        imgData.seek(0)
        svgData = imgData.read()

        svgRenderer = qt.QSvgRenderer()

        viewbox = self._getViewBox()

        svgRenderer.setViewBox(viewbox)

        xml_stream = qt.QXmlStreamReader(svgData.encode(errors="replace"))

        # This is for PyMca compatibility, to share a print preview with PyMca plots
        svgRenderer._viewBox = viewbox
        svgRenderer._svgRawData = svgData.encode(errors="replace")
        svgRenderer._svgRendererData = xml_stream

        if not svgRenderer.load(xml_stream):
            raise RuntimeError("Cannot interpret svg data")

        return svgRenderer, viewbox

    def _getViewBox(self):
        """
        """
        printer = self.printPreviewDialog.printer
        dpix = printer.logicalDpiX()
        dpiy = printer.logicalDpiY()
        availableWidth = printer.width()
        availableHeight = printer.height()

        config = self._printGeometry
        width = config['width']
        height = config['height']
        xOffset = config['xOffset']
        yOffset = config['yOffset']
        units = config['units']
        keepAspectRatio = config['keepAspectRatio']
        aspectRatio = self._getPlotAspectRatio()

        # convert the offsets to dots
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
            bodyWidth = width or availableWidth
            bodyHeight = bodyWidth * aspectRatio

            if bodyHeight > availableHeight:
                bodyHeight = availableHeight
                bodyWidth = bodyHeight / aspectRatio

        else:
            bodyWidth = width or availableWidth
            bodyHeight = height or availableHeight

        return qt.QRectF(xOffset,
                         yOffset,
                         bodyWidth,
                         bodyHeight)

    def _setPrintConfiguration(self):
        """Open a dialog to prompt the user to adjust print
        geometry parameters."""
        self.printPreviewDialog.ensurePrinterIsSet()
        if self._printConfigurationDialog is None:
            self._printConfigurationDialog = PrintGeometryDialog(self.parent())

        self._printConfigurationDialog.setPrintGeometry(self._printGeometry)
        if self._printConfigurationDialog.exec():
            self._printGeometry = self._printConfigurationDialog.getPrintGeometry()

    def _getPlotAspectRatio(self):
        widget = self._plot.centralWidget()
        graphWidth = float(widget.width())
        graphHeight = float(widget.height())
        return graphHeight / graphWidth


class SingletonPrintPreviewToolButton(PrintPreviewToolButton):
    """This class is similar to its parent class :class:`PrintPreviewToolButton`
    but it uses a singleton print preview widget.

    This allows for several plots to send their content to the
    same print page, and for users to arrange them."""
    def __init__(self, parent=None, plot=None):
        PrintPreviewToolButton.__init__(self, parent, plot)

    @property
    def printPreviewDialog(self):
        if self._printPreviewDialog is None:
            self._printPreviewDialog = SingletonPrintPreviewDialog(self.parent())
        return self._printPreviewDialog


if __name__ == '__main__':
    import numpy
    app = qt.QApplication([])

    pw = PlotWidget()
    toolbar = qt.QToolBar(pw)
    toolbutton = PrintPreviewToolButton(parent=toolbar,
                                        plot=pw)
    pw.addToolBar(toolbar)
    toolbar.addWidget(toolbutton)
    pw.show()

    x = numpy.arange(1000)
    y = x / numpy.sin(x)
    pw.addCurve(x, y)

    app.exec()
