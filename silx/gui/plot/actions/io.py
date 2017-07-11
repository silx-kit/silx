# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
:mod:`silx.gui.plot.actions.io` provides a set of QAction relative of inputs
and outputs for a :class:`.PlotWidget`.

The following QAction are available:

- :class:`CopyAction`
- :class:`PrintAction`
- :class:`SaveAction`
"""

from __future__ import division

__authors__ = ["V.A. Sole", "T. Vincent", "P. Knobel"]
__license__ = "MIT"
__date__ = "27/06/2017"

from . import PlotAction
from silx.io.utils import save1D, savespec
import logging
import sys
from collections import OrderedDict
import traceback
import numpy
from silx.gui import qt
from silx.third_party.EdfFile import EdfFile
from silx.third_party.TiffIO import TiffIO
from silx.gui._utils import convertArrayToQImage
if sys.version_info[0] == 3:
    from io import BytesIO
else:
    import cStringIO as _StringIO
    BytesIO = _StringIO.StringIO

_logger = logging.getLogger(__name__)


class SaveAction(PlotAction):
    """QAction for saving Plot content.

    It opens a Save as... dialog.

    :param plot: :class:`.PlotWidget` instance on which to operate.
    :param parent: See :class:`QAction`.
    """
    # TODO find a way to make the filter list selectable and extensible

    SNAPSHOT_FILTER_SVG = 'Plot Snapshot as SVG (*.svg)'
    SNAPSHOT_FILTER_PNG = 'Plot Snapshot as PNG (*.png)'

    SNAPSHOT_FILTERS = (SNAPSHOT_FILTER_PNG, SNAPSHOT_FILTER_SVG)

    # Dict of curve filters with CSV-like format
    # Using ordered dict to guarantee filters order
    # Note: '%.18e' is numpy.savetxt default format
    CURVE_FILTERS_TXT = OrderedDict((
        ('Curve as Raw ASCII (*.txt)',
         {'fmt': '%.18e', 'delimiter': ' ', 'header': False}),
        ('Curve as ";"-separated CSV (*.csv)',
         {'fmt': '%.18e', 'delimiter': ';', 'header': True}),
        ('Curve as ","-separated CSV (*.csv)',
         {'fmt': '%.18e', 'delimiter': ',', 'header': True}),
        ('Curve as tab-separated CSV (*.csv)',
         {'fmt': '%.18e', 'delimiter': '\t', 'header': True}),
        ('Curve as OMNIC CSV (*.csv)',
         {'fmt': '%.7E', 'delimiter': ',', 'header': False}),
        ('Curve as SpecFile (*.dat)',
         {'fmt': '%.7g', 'delimiter': '', 'header': False})
    ))

    CURVE_FILTER_NPY = 'Curve as NumPy binary file (*.npy)'

    CURVE_FILTERS = list(CURVE_FILTERS_TXT.keys()) + [CURVE_FILTER_NPY]

    ALL_CURVES_FILTERS = ("All curves as SpecFile (*.dat)", )

    IMAGE_FILTER_EDF = 'Image data as EDF (*.edf)'
    IMAGE_FILTER_TIFF = 'Image data as TIFF (*.tif)'
    IMAGE_FILTER_NUMPY = 'Image data as NumPy binary file (*.npy)'
    IMAGE_FILTER_ASCII = 'Image data as ASCII (*.dat)'
    IMAGE_FILTER_CSV_COMMA = 'Image data as ,-separated CSV (*.csv)'
    IMAGE_FILTER_CSV_SEMICOLON = 'Image data as ;-separated CSV (*.csv)'
    IMAGE_FILTER_CSV_TAB = 'Image data as tab-separated CSV (*.csv)'
    IMAGE_FILTER_RGB_PNG = 'Image as PNG (*.png)'
    IMAGE_FILTER_RGB_TIFF = 'Image as TIFF (*.tif)'
    IMAGE_FILTERS = (IMAGE_FILTER_EDF,
                     IMAGE_FILTER_TIFF,
                     IMAGE_FILTER_NUMPY,
                     IMAGE_FILTER_ASCII,
                     IMAGE_FILTER_CSV_COMMA,
                     IMAGE_FILTER_CSV_SEMICOLON,
                     IMAGE_FILTER_CSV_TAB,
                     IMAGE_FILTER_RGB_PNG,
                     IMAGE_FILTER_RGB_TIFF)

    def __init__(self, plot, parent=None):
        super(SaveAction, self).__init__(
            plot, icon='document-save', text='Save as...',
            tooltip='Save curve/image/plot snapshot dialog',
            triggered=self._actionTriggered,
            checkable=False, parent=parent)
        self.setShortcut(qt.QKeySequence.Save)
        self.setShortcutContext(qt.Qt.WidgetShortcut)

    def _errorMessage(self, informativeText=''):
        """Display an error message."""
        # TODO issue with QMessageBox size fixed and too small
        msg = qt.QMessageBox(self.plot)
        msg.setIcon(qt.QMessageBox.Critical)
        msg.setInformativeText(informativeText + ' ' + str(sys.exc_info()[1]))
        msg.setDetailedText(traceback.format_exc())
        msg.exec_()

    def _saveSnapshot(self, filename, nameFilter):
        """Save a snapshot of the :class:`PlotWindow` widget.

        :param str filename: The name of the file to write
        :param str nameFilter: The selected name filter
        :return: False if format is not supported or save failed,
                 True otherwise.
        """
        if nameFilter == self.SNAPSHOT_FILTER_PNG:
            fileFormat = 'png'
        elif nameFilter == self.SNAPSHOT_FILTER_SVG:
            fileFormat = 'svg'
        else:  # Format not supported
            _logger.error(
                'Saving plot snapshot failed: format not supported')
            return False

        self.plot.saveGraph(filename, fileFormat=fileFormat)
        return True

    def _saveCurve(self, filename, nameFilter):
        """Save a curve from the plot.

        :param str filename: The name of the file to write
        :param str nameFilter: The selected name filter
        :return: False if format is not supported or save failed,
                 True otherwise.
        """
        if nameFilter not in self.CURVE_FILTERS:
            return False

        # Check if a curve is to be saved
        curve = self.plot.getActiveCurve()
        # before calling _saveCurve, if there is no selected curve, we
        # make sure there is only one curve on the graph
        if curve is None:
            curves = self.plot.getAllCurves()
            if not curves:
                self._errorMessage("No curve to be saved")
                return False
            curve = curves[0]

        if nameFilter in self.CURVE_FILTERS_TXT:
            filter_ = self.CURVE_FILTERS_TXT[nameFilter]
            fmt = filter_['fmt']
            csvdelim = filter_['delimiter']
            autoheader = filter_['header']
        else:
            # .npy
            fmt, csvdelim, autoheader = ("", "", False)

        # If curve has no associated label, get the default from the plot
        xlabel = curve.getXLabel()
        if xlabel is None:
            xlabel = self.plot.getXAxis().getLabel()
        ylabel = curve.getYLabel()
        if ylabel is None:
            ylabel = self.plot.getYAxis().getLabel()

        try:
            save1D(filename,
                   curve.getXData(copy=False),
                   curve.getYData(copy=False),
                   xlabel, [ylabel],
                   fmt=fmt, csvdelim=csvdelim,
                   autoheader=autoheader)
        except IOError:
            self._errorMessage('Save failed\n')
            return False

        return True

    def _saveCurves(self, filename, nameFilter):
        """Save all curves from the plot.

        :param str filename: The name of the file to write
        :param str nameFilter: The selected name filter
        :return: False if format is not supported or save failed,
                 True otherwise.
        """
        if nameFilter not in self.ALL_CURVES_FILTERS:
            return False

        curves = self.plot.getAllCurves()
        if not curves:
            self._errorMessage("No curves to be saved")
            return False

        curve = curves[0]
        scanno = 1
        try:
            specfile = savespec(filename,
                                curve.getXData(copy=False),
                                curve.getYData(copy=False),
                                curve.getXLabel(),
                                curve.getYLabel(),
                                fmt="%.7g", scan_number=1, mode="w",
                                write_file_header=True,
                                close_file=False)
        except IOError:
            self._errorMessage('Save failed\n')
            return False

        for curve in curves[1:]:
            try:
                scanno += 1
                specfile = savespec(specfile,
                                    curve.getXData(copy=False),
                                    curve.getYData(copy=False),
                                    curve.getXLabel(),
                                    curve.getYLabel(),
                                    fmt="%.7g", scan_number=scanno, mode="w",
                                    write_file_header=False,
                                    close_file=False)
            except IOError:
                self._errorMessage('Save failed\n')
                return False
        specfile.close()

        return True

    def _saveImage(self, filename, nameFilter):
        """Save an image from the plot.

        :param str filename: The name of the file to write
        :param str nameFilter: The selected name filter
        :return: False if format is not supported or save failed,
                 True otherwise.
        """
        if nameFilter not in self.IMAGE_FILTERS:
            return False

        image = self.plot.getActiveImage()
        if image is None:
            qt.QMessageBox.warning(
                self.plot, "No Data", "No image to be saved")
            return False

        data = image.getData(copy=False)

        # TODO Use silx.io for writing files
        if nameFilter == self.IMAGE_FILTER_EDF:
            edfFile = EdfFile(filename, access="w+")
            edfFile.WriteImage({}, data, Append=0)
            return True

        elif nameFilter == self.IMAGE_FILTER_TIFF:
            tiffFile = TiffIO(filename, mode='w')
            tiffFile.writeImage(data, software='silx')
            return True

        elif nameFilter == self.IMAGE_FILTER_NUMPY:
            try:
                numpy.save(filename, data)
            except IOError:
                self._errorMessage('Save failed\n')
                return False
            return True

        elif nameFilter in (self.IMAGE_FILTER_ASCII,
                            self.IMAGE_FILTER_CSV_COMMA,
                            self.IMAGE_FILTER_CSV_SEMICOLON,
                            self.IMAGE_FILTER_CSV_TAB):
            csvdelim, filetype = {
                self.IMAGE_FILTER_ASCII: (' ', 'txt'),
                self.IMAGE_FILTER_CSV_COMMA: (',', 'csv'),
                self.IMAGE_FILTER_CSV_SEMICOLON: (';', 'csv'),
                self.IMAGE_FILTER_CSV_TAB: ('\t', 'csv'),
                }[nameFilter]

            height, width = data.shape
            rows, cols = numpy.mgrid[0:height, 0:width]
            try:
                save1D(filename, rows.ravel(), (cols.ravel(), data.ravel()),
                       filetype=filetype,
                       xlabel='row',
                       ylabels=['column', 'value'],
                       csvdelim=csvdelim,
                       autoheader=True)

            except IOError:
                self._errorMessage('Save failed\n')
                return False
            return True

        elif nameFilter in (self.IMAGE_FILTER_RGB_PNG,
                            self.IMAGE_FILTER_RGB_TIFF):
            # Get displayed image
            rgbaImage = image.getRgbaImageData(copy=False)
            # Convert RGB QImage
            qimage = convertArrayToQImage(rgbaImage[:, :, :3])

            if nameFilter == self.IMAGE_FILTER_RGB_PNG:
                fileFormat = 'PNG'
            else:
                fileFormat = 'TIFF'

            if qimage.save(filename, fileFormat):
                return True
            else:
                _logger.error('Failed to save image as %s', filename)
                qt.QMessageBox.critical(
                    self.parent(),
                    'Save image as',
                    'Failed to save image')

        return False

    def _actionTriggered(self, checked=False):
        """Handle save action."""
        # Set-up filters
        filters = []

        # Add image filters if there is an active image
        if self.plot.getActiveImage() is not None:
            filters.extend(self.IMAGE_FILTERS)

        # Add curve filters if there is a curve to save
        if (self.plot.getActiveCurve() is not None or
                len(self.plot.getAllCurves()) == 1):
            filters.extend(self.CURVE_FILTERS)
        if len(self.plot.getAllCurves()) > 1:
            filters.extend(self.ALL_CURVES_FILTERS)

        filters.extend(self.SNAPSHOT_FILTERS)

        # Create and run File dialog
        dialog = qt.QFileDialog(self.plot)
        dialog.setWindowTitle("Output File Selection")
        dialog.setModal(1)
        dialog.setNameFilters(filters)

        dialog.setFileMode(dialog.AnyFile)
        dialog.setAcceptMode(dialog.AcceptSave)

        if not dialog.exec_():
            return False

        nameFilter = dialog.selectedNameFilter()
        filename = dialog.selectedFiles()[0]
        dialog.close()

        # Forces the filename extension to match the chosen filter
        extension = nameFilter.split()[-1][2:-1]
        if (len(filename) <= len(extension) or
                filename[-len(extension):].lower() != extension.lower()):
            filename += extension

        # Handle save
        if nameFilter in self.SNAPSHOT_FILTERS:
            return self._saveSnapshot(filename, nameFilter)
        elif nameFilter in self.CURVE_FILTERS:
            return self._saveCurve(filename, nameFilter)
        elif nameFilter in self.ALL_CURVES_FILTERS:
            return self._saveCurves(filename, nameFilter)
        elif nameFilter in self.IMAGE_FILTERS:
            return self._saveImage(filename, nameFilter)
        else:
            _logger.warning('Unsupported file filter: %s', nameFilter)
            return False


def _plotAsPNG(plot):
    """Save a :class:`Plot` as PNG and return the payload.

    :param plot: The :class:`Plot` to save
    """
    pngFile = BytesIO()
    plot.saveGraph(pngFile, fileFormat='png')
    pngFile.flush()
    pngFile.seek(0)
    data = pngFile.read()
    pngFile.close()
    return data


class PrintAction(PlotAction):
    """QAction for printing the plot.

    It opens a Print dialog.

    Current implementation print a bitmap of the plot area and not vector
    graphics, so printing quality is not great.

    :param plot: :class:`.PlotWidget` instance on which to operate.
    :param parent: See :class:`QAction`.
    """

    # Share QPrinter instance to propose latest used as default
    _printer = None

    def __init__(self, plot, parent=None):
        super(PrintAction, self).__init__(
            plot, icon='document-print', text='Print...',
            tooltip='Open print dialog',
            triggered=self.printPlot,
            checkable=False, parent=parent)
        self.setShortcut(qt.QKeySequence.Print)
        self.setShortcutContext(qt.Qt.WidgetShortcut)

    @property
    def printer(self):
        """The QPrinter instance used by the actions.

        This is shared accross all instances of PrintAct
        """
        if self._printer is None:
            PrintAction._printer = qt.QPrinter()
        return self._printer

    def printPlotAsWidget(self):
        """Open the print dialog and print the plot.

        Use :meth:`QWidget.render` to print the plot

        :return: True if successful
        """
        dialog = qt.QPrintDialog(self.printer, self.plot)
        dialog.setWindowTitle('Print Plot')
        if not dialog.exec_():
            return False

        # Print a snapshot of the plot widget at the top of the page
        widget = self.plot.centralWidget()

        painter = qt.QPainter()
        if not painter.begin(self.printer):
            return False

        pageRect = self.printer.pageRect()
        xScale = pageRect.width() / widget.width()
        yScale = pageRect.height() / widget.height()
        scale = min(xScale, yScale)

        painter.translate(pageRect.width() / 2., 0.)
        painter.scale(scale, scale)
        painter.translate(-widget.width() / 2., 0.)
        widget.render(painter)
        painter.end()

        return True

    def printPlot(self):
        """Open the print dialog and print the plot.

        Use :meth:`Plot.saveGraph` to print the plot.

        :return: True if successful
        """
        # Init printer and start printer dialog
        dialog = qt.QPrintDialog(self.printer, self.plot)
        dialog.setWindowTitle('Print Plot')
        if not dialog.exec_():
            return False

        # Save Plot as PNG and make a pixmap from it with default dpi
        pngData = _plotAsPNG(self.plot)

        pixmap = qt.QPixmap()
        pixmap.loadFromData(pngData, 'png')

        xScale = self.printer.pageRect().width() / pixmap.width()
        yScale = self.printer.pageRect().height() / pixmap.height()
        scale = min(xScale, yScale)

        # Draw pixmap with painter
        painter = qt.QPainter()
        if not painter.begin(self.printer):
            return False

        painter.drawPixmap(0, 0,
                           pixmap.width() * scale,
                           pixmap.height() * scale,
                           pixmap)
        painter.end()

        return True


class CopyAction(PlotAction):
    """QAction to copy :class:`.PlotWidget` content to clipboard.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(CopyAction, self).__init__(
            plot, icon='edit-copy', text='Copy plot',
            tooltip='Copy a snapshot of the plot into the clipboard',
            triggered=self.copyPlot,
            checkable=False, parent=parent)
        self.setShortcut(qt.QKeySequence.Copy)
        self.setShortcutContext(qt.Qt.WidgetShortcut)

    def copyPlot(self):
        """Copy plot content to the clipboard as a bitmap."""
        # Save Plot as PNG and make a QImage from it with default dpi
        pngData = _plotAsPNG(self.plot)
        image = qt.QImage.fromData(pngData, 'png')
        qt.QApplication.clipboard().setImage(image)
