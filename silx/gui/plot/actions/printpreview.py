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

- :class:`PrintPreviewAction`
"""
from __future__ import absolute_import

from io import StringIO

from . import PlotAction
from ...widgets.PrintPreview import PrintPreviewDialog
from ... import qt


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "18/07/2017"


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
            triggered=self._plotToPrintPreview,
            checkable=False, parent=parent)
        self.printPreviewDialog = None
        self.printConfigurationDialog = None

        self._printConfiguration = {"xOffset": 0.1,
                                    "yOffset": 0.1,
                                    "width": 0.9,
                                    "height": 0.9,
                                    "units": "page",
                                    "keepAspectRatio": True}

    def _plotToPrintPreview(self):
        if self.printPreviewDialog is None:
            self.printPreviewDialog = PrintPreviewDialog(self.parent())
        self.printPreviewDialog.show()

        if qt.HAS_SVG:
            svgRenderer = self._getSvgRenderer()
            self.printPreviewDialog.addSvgItem(svgRenderer)
        else:
            if qt.BINDING in ["PyQt4", "PySide"]:
                pixmap = qt.QPixmap.grabWidget(self.plot.centralWidget())
            else:
                # PyQt5 and hopefully PyQt6+
                pixmap = self.plot.centralWidget().grab()
            self.printPreviewDialog.addPixmap(pixmap)
        self.printPreviewDialog.raise_()

    def _getSvgRenderer(self):
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
        # if printer is None:   # Fixme: this probably can't happen
        #     # printer was not selected, don't adjust the viewbox
        #     if not svgRenderer.load(qt.QXmlStreamReader(svgData.encode())):
        #         raise RuntimeError("Cannot interpret svg data")
        #     return svgRenderer

        self._printConfigurationDialog()     # opens a dialog and updates _printConfiguration
        config = self._printConfiguration
        width = config['width']
        height = config['height']
        xOffset = config['xOffset']
        yOffset = config['yOffset']
        units = config['units']
        keepAspectRatio = config['keepAspectRatio']

        dpix = printer.logicalDpiX()
        dpiy = printer.logicalDpiY()

        availableWidth = printer.width()
        availableHeight = printer.height()

        # convert the offsets to dpi
        if units.lower() in ['inch', 'inches']:
            xOffset = xOffset * dpix
            yOffset = yOffset * dpiy
            if width is not None:
                width = width * dpix
            if height is not None:
                height = height * dpiy
        elif units.lower() in ['cm', 'centimeters']:
            xOffset = (xOffset/2.54) * dpix
            yOffset = (yOffset/2.54) * dpiy
            if width is not None:
                width = (width/2.54) * dpix
            if height is not None:
                height = (height/2.54) * dpiy
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
            # get the aspect ratio
            widget = self.plot.centralWidget()
            graphWidth = float(widget.width())
            graphHeight = float(widget.height())
            graphRatio = graphHeight / graphWidth

            bodyWidth = width or availableWidth
            bodyHeight = bodyWidth * graphRatio

            if bodyHeight > availableHeight:
                bodyHeight = availableHeight
                bodyWidth = bodyHeight / graphRatio
        else:
            bodyWidth = width or availableWidth
            bodyHeight = height or availableHeight

        body = qt.QRectF(xOffset,
                         yOffset,
                         bodyWidth,
                         bodyHeight)

        svgRenderer.setViewBox(body)

        xml_stream = qt.QXmlStreamReader(svgData.encode(errors="replace"))

        if not svgRenderer.load(xml_stream):
            raise RuntimeError("Cannot interpret svg data")

        return svgRenderer

    def _printConfigurationDialog(self):
        """Open a dialog to prompt the user to adjust print parameters."""
        # if self.printConfigurationDialog is None:
        #     self.printConfigurationDialog = \
        #                         ObjectPrintConfigurationDialog(self)
        #
        # self._printConfigurationDialog.setPrintConfiguration(self._printConfiguration)
        # if self._printConfigurationDialog.exec_():
        #     self._printConfiguration = self._printConfigurationDialog._printConfigurationDialog()
        pass


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
