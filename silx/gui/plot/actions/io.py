# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
__date__ = "12/07/2018"

from . import PlotAction
from silx.io.utils import save1D, savespec, NEXUS_HDF5_EXT
from silx.io.nxdata import save_NXdata
import logging
import sys
import os.path
from collections import OrderedDict
import traceback
import numpy
from silx.utils.deprecation import deprecated
from silx.gui import qt, printer
from silx.gui.dialog.GroupDialog import GroupDialog
from silx.third_party.EdfFile import EdfFile
from silx.third_party.TiffIO import TiffIO
from ...utils.image import convertArrayToQImage
if sys.version_info[0] == 3:
    from io import BytesIO
else:
    import cStringIO as _StringIO
    BytesIO = _StringIO.StringIO

_logger = logging.getLogger(__name__)

_NEXUS_HDF5_EXT_STR = ' '.join(['*' + ext for ext in NEXUS_HDF5_EXT])


def selectOutputGroup(h5filename):
    """Open a dialog to prompt the user to select a group in
    which to output data.

    :param str h5filename: name of an existing HDF5 file
    :rtype: str
    :return: Name of output group, or None if the dialog was cancelled
    """
    dialog = GroupDialog()
    dialog.addFile(h5filename)
    dialog.setWindowTitle("Select an output group")
    if not dialog.exec_():
        return None
    return dialog.getSelectedDataUrl().data_path()


class SaveAction(PlotAction):
    """QAction for saving Plot content.

    It opens a Save as... dialog.

    :param plot: :class:`.PlotWidget` instance on which to operate.
    :param parent: See :class:`QAction`.
    """

    SNAPSHOT_FILTER_SVG = 'Plot Snapshot as SVG (*.svg)'
    SNAPSHOT_FILTER_PNG = 'Plot Snapshot as PNG (*.png)'

    DEFAULT_ALL_FILTERS = (SNAPSHOT_FILTER_PNG, SNAPSHOT_FILTER_SVG)

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
         {'fmt': '%.10g', 'delimiter': '', 'header': False})
    ))

    CURVE_FILTER_NPY = 'Curve as NumPy binary file (*.npy)'

    CURVE_FILTER_NXDATA = 'Curve as NXdata (%s)' % _NEXUS_HDF5_EXT_STR

    DEFAULT_CURVE_FILTERS = list(CURVE_FILTERS_TXT.keys()) + [
        CURVE_FILTER_NPY, CURVE_FILTER_NXDATA]

    DEFAULT_ALL_CURVES_FILTERS = ("All curves as SpecFile (*.dat)",)

    IMAGE_FILTER_EDF = 'Image data as EDF (*.edf)'
    IMAGE_FILTER_TIFF = 'Image data as TIFF (*.tif)'
    IMAGE_FILTER_NUMPY = 'Image data as NumPy binary file (*.npy)'
    IMAGE_FILTER_ASCII = 'Image data as ASCII (*.dat)'
    IMAGE_FILTER_CSV_COMMA = 'Image data as ,-separated CSV (*.csv)'
    IMAGE_FILTER_CSV_SEMICOLON = 'Image data as ;-separated CSV (*.csv)'
    IMAGE_FILTER_CSV_TAB = 'Image data as tab-separated CSV (*.csv)'
    IMAGE_FILTER_RGB_PNG = 'Image as PNG (*.png)'
    IMAGE_FILTER_NXDATA = 'Image as NXdata (%s)' % _NEXUS_HDF5_EXT_STR
    DEFAULT_IMAGE_FILTERS = (IMAGE_FILTER_EDF,
                             IMAGE_FILTER_TIFF,
                             IMAGE_FILTER_NUMPY,
                             IMAGE_FILTER_ASCII,
                             IMAGE_FILTER_CSV_COMMA,
                             IMAGE_FILTER_CSV_SEMICOLON,
                             IMAGE_FILTER_CSV_TAB,
                             IMAGE_FILTER_RGB_PNG,
                             IMAGE_FILTER_NXDATA)

    SCATTER_FILTER_NXDATA = 'Scatter as NXdata (%s)' % _NEXUS_HDF5_EXT_STR
    DEFAULT_SCATTER_FILTERS = (SCATTER_FILTER_NXDATA,)

    # filters for which we don't want an "overwrite existing file" warning
    DEFAULT_APPEND_FILTERS = (CURVE_FILTER_NXDATA, IMAGE_FILTER_NXDATA,
                              SCATTER_FILTER_NXDATA)

    def __init__(self, plot, parent=None):
        self._filters = {
            'all': OrderedDict(),
            'curve': OrderedDict(),
            'curves': OrderedDict(),
            'image': OrderedDict(),
            'scatter': OrderedDict()}

        # Initialize filters
        for nameFilter in self.DEFAULT_ALL_FILTERS:
            self.setFileFilter(
                dataKind='all', nameFilter=nameFilter, func=self._saveSnapshot)

        for nameFilter in self.DEFAULT_CURVE_FILTERS:
            self.setFileFilter(
                dataKind='curve', nameFilter=nameFilter, func=self._saveCurve)

        for nameFilter in self.DEFAULT_ALL_CURVES_FILTERS:
            self.setFileFilter(
                dataKind='curves', nameFilter=nameFilter, func=self._saveCurves)

        for nameFilter in self.DEFAULT_IMAGE_FILTERS:
            self.setFileFilter(
                dataKind='image', nameFilter=nameFilter, func=self._saveImage)

        for nameFilter in self.DEFAULT_SCATTER_FILTERS:
            self.setFileFilter(
                dataKind='scatter', nameFilter=nameFilter, func=self._saveScatter)

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

    def _saveSnapshot(self, plot, filename, nameFilter):
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

        plot.saveGraph(filename, fileFormat=fileFormat)
        return True

    def _getAxesLabels(self, item):
        # If curve has no associated label, get the default from the plot
        xlabel = item.getXLabel() or self.plot.getXAxis().getLabel()
        ylabel = item.getYLabel() or self.plot.getYAxis().getLabel()
        return xlabel, ylabel

    def _selectWriteableOutputGroup(self, filename):
        if os.path.exists(filename) and os.path.isfile(filename) \
                and os.access(filename, os.W_OK):
            entryPath = selectOutputGroup(filename)
            if entryPath is None:
                _logger.info("Save operation cancelled")
                return None
            return entryPath
        elif not os.path.exists(filename):
            # create new entry in new file
            return "/entry"
        else:
            self._errorMessage('Save failed (file access issue)\n')
            return None

    def _saveCurveAsNXdata(self, curve, filename):
        entryPath = self._selectWriteableOutputGroup(filename)
        if entryPath is None:
            return False

        xlabel, ylabel = self._getAxesLabels(curve)

        return save_NXdata(
            filename,
            nxentry_name=entryPath,
            signal=curve.getYData(copy=False),
            axes=[curve.getXData(copy=False)],
            signal_name="y",
            axes_names=["x"],
            signal_long_name=ylabel,
            axes_long_names=[xlabel],
            signal_errors=curve.getYErrorData(copy=False),
            axes_errors=[curve.getXErrorData(copy=True)],
            title=self.plot.getGraphTitle())

    def _saveCurve(self, plot, filename, nameFilter):
        """Save a curve from the plot.

        :param str filename: The name of the file to write
        :param str nameFilter: The selected name filter
        :return: False if format is not supported or save failed,
                 True otherwise.
        """
        if nameFilter not in self.DEFAULT_CURVE_FILTERS:
            return False

        # Check if a curve is to be saved
        curve = plot.getActiveCurve()
        # before calling _saveCurve, if there is no selected curve, we
        # make sure there is only one curve on the graph
        if curve is None:
            curves = plot.getAllCurves()
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
            # .npy or nxdata
            fmt, csvdelim, autoheader = ("", "", False)

        xlabel, ylabel = self._getAxesLabels(curve)

        if nameFilter == self.CURVE_FILTER_NXDATA:
            return self._saveCurveAsNXdata(curve, filename)

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

    def _saveCurves(self, plot, filename, nameFilter):
        """Save all curves from the plot.

        :param str filename: The name of the file to write
        :param str nameFilter: The selected name filter
        :return: False if format is not supported or save failed,
                 True otherwise.
        """
        if nameFilter not in self.DEFAULT_ALL_CURVES_FILTERS:
            return False

        curves = plot.getAllCurves()
        if not curves:
            self._errorMessage("No curves to be saved")
            return False

        curve = curves[0]
        scanno = 1
        try:
            xlabel = curve.getXLabel() or plot.getGraphXLabel()
            ylabel = curve.getYLabel() or plot.getGraphYLabel(curve.getYAxis())
            specfile = savespec(filename,
                                curve.getXData(copy=False),
                                curve.getYData(copy=False),
                                xlabel,
                                ylabel,
                                fmt="%.7g", scan_number=1, mode="w",
                                write_file_header=True,
                                close_file=False)
        except IOError:
            self._errorMessage('Save failed\n')
            return False

        for curve in curves[1:]:
            try:
                scanno += 1
                xlabel = curve.getXLabel() or plot.getGraphXLabel()
                ylabel = curve.getYLabel() or plot.getGraphYLabel(curve.getYAxis())
                specfile = savespec(specfile,
                                    curve.getXData(copy=False),
                                    curve.getYData(copy=False),
                                    xlabel,
                                    ylabel,
                                    fmt="%.7g", scan_number=scanno,
                                    write_file_header=False,
                                    close_file=False)
            except IOError:
                self._errorMessage('Save failed\n')
                return False
        specfile.close()

        return True

    def _saveImage(self, plot, filename, nameFilter):
        """Save an image from the plot.

        :param str filename: The name of the file to write
        :param str nameFilter: The selected name filter
        :return: False if format is not supported or save failed,
                 True otherwise.
        """
        if nameFilter not in self.DEFAULT_IMAGE_FILTERS:
            return False

        image = plot.getActiveImage()
        if image is None:
            qt.QMessageBox.warning(
                plot, "No Data", "No image to be saved")
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

        elif nameFilter == self.IMAGE_FILTER_NXDATA:
            entryPath = self._selectWriteableOutputGroup(filename)
            if entryPath is None:
                return False
            xorigin, yorigin = image.getOrigin()
            xscale, yscale = image.getScale()
            xaxis = xorigin + xscale * numpy.arange(data.shape[1])
            yaxis = yorigin + yscale * numpy.arange(data.shape[0])
            xlabel, ylabel = self._getAxesLabels(image)
            interpretation = "image" if len(data.shape) == 2 else "rgba-image"

            return save_NXdata(filename,
                               nxentry_name=entryPath,
                               signal=data,
                               axes=[yaxis, xaxis],
                               signal_name="image",
                               axes_names=["y", "x"],
                               axes_long_names=[ylabel, xlabel],
                               title=plot.getGraphTitle(),
                               interpretation=interpretation)

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

        elif nameFilter == self.IMAGE_FILTER_RGB_PNG:
            # Get displayed image
            rgbaImage = image.getRgbaImageData(copy=False)
            # Convert RGB QImage
            qimage = convertArrayToQImage(rgbaImage[:, :, :3])

            if qimage.save(filename, 'PNG'):
                return True
            else:
                _logger.error('Failed to save image as %s', filename)
                qt.QMessageBox.critical(
                    self.parent(),
                    'Save image as',
                    'Failed to save image')

        return False

    def _saveScatter(self, plot, filename, nameFilter):
        """Save an image from the plot.

        :param str filename: The name of the file to write
        :param str nameFilter: The selected name filter
        :return: False if format is not supported or save failed,
                 True otherwise.
        """
        if nameFilter not in self.DEFAULT_SCATTER_FILTERS:
            return False

        if nameFilter == self.SCATTER_FILTER_NXDATA:
            entryPath = self._selectWriteableOutputGroup(filename)
            if entryPath is None:
                return False
            scatter = plot.getScatter()

            x = scatter.getXData(copy=False)
            y = scatter.getYData(copy=False)
            z = scatter.getValueData(copy=False)

            xerror = scatter.getXErrorData(copy=False)
            if isinstance(xerror, float):
                xerror = xerror * numpy.ones(x.shape, dtype=numpy.float32)

            yerror = scatter.getYErrorData(copy=False)
            if isinstance(yerror, float):
                yerror = yerror * numpy.ones(x.shape, dtype=numpy.float32)

            xlabel = plot.getGraphXLabel()
            ylabel = plot.getGraphYLabel()

            return save_NXdata(
                filename,
                nxentry_name=entryPath,
                signal=z,
                axes=[x, y],
                signal_name="values",
                axes_names=["x", "y"],
                axes_long_names=[xlabel, ylabel],
                axes_errors=[xerror, yerror],
                title=plot.getGraphTitle())

    def setFileFilter(self, dataKind, nameFilter, func):
        """Set a name filter to add/replace a file format support

        :param str dataKind:
            The kind of data for which the provided filter is valid.
            One of: 'all', 'curve', 'curves', 'image', 'scatter'
        :param str nameFilter: The name filter in the QFileDialog.
            See :meth:`QFileDialog.setNameFilters`.
        :param callable func: The function to call to perform saving.
            Expected signature is:
            bool func(PlotWidget plot, str filename, str nameFilter)
        """
        assert dataKind in ('all', 'curve', 'curves', 'image', 'scatter')

        self._filters[dataKind][nameFilter] = func

    def getFileFilters(self, dataKind):
        """Returns the nameFilter and associated function for a kind of data.

        :param str dataKind:
            The kind of data for which the provided filter is valid.
            On of: 'all', 'curve', 'curves', 'image', 'scatter'
        :return: {nameFilter: function} associations.
        :rtype: collections.OrderedDict
        """
        assert dataKind in ('all', 'curve', 'curves', 'image', 'scatter')

        return self._filters[dataKind].copy()

    def _actionTriggered(self, checked=False):
        """Handle save action."""
        # Set-up filters
        filters = OrderedDict()

        # Add image filters if there is an active image
        if self.plot.getActiveImage() is not None:
            filters.update(self._filters['image'].items())

        # Add curve filters if there is a curve to save
        if (self.plot.getActiveCurve() is not None or
                len(self.plot.getAllCurves()) == 1):
            filters.update(self._filters['curve'].items())
        if len(self.plot.getAllCurves()) >= 1:
            filters.update(self._filters['curves'].items())

        # Add scatter filters if there is a scatter
        # todo: CSV
        if self.plot.getScatter() is not None:
            filters.update(self._filters['scatter'].items())

        filters.update(self._filters['all'].items())

        # Create and run File dialog
        dialog = qt.QFileDialog(self.plot)
        dialog.setOption(dialog.DontUseNativeDialog)
        dialog.setWindowTitle("Output File Selection")
        dialog.setModal(1)
        dialog.setNameFilters(list(filters.keys()))

        dialog.setFileMode(dialog.AnyFile)
        dialog.setAcceptMode(dialog.AcceptSave)

        def onFilterSelection(filt_):
            # disable overwrite confirmation for NXdata types,
            # because we append the data to existing files
            if filt_ in self.DEFAULT_APPEND_FILTERS:
                dialog.setOption(dialog.DontConfirmOverwrite)
            else:
                dialog.setOption(dialog.DontConfirmOverwrite, False)

        dialog.filterSelected.connect(onFilterSelection)

        if not dialog.exec_():
            return False

        nameFilter = dialog.selectedNameFilter()
        filename = dialog.selectedFiles()[0]
        dialog.close()

        if '(' in nameFilter and ')' == nameFilter.strip()[-1]:
            # Check for correct file extension
            # Extract file extensions as .something
            extensions = [ext[ext.find('.'):] for ext in
                          nameFilter[nameFilter.find('(')+1:-1].split()]
            for ext in extensions:
                if (len(filename) > len(ext) and
                        filename[-len(ext):].lower() == ext.lower()):
                    break
            else:  # filename has no extension supported in nameFilter, add one
                if len(extensions) >= 1:
                    filename += extensions[0]

        # Handle save
        func = filters.get(nameFilter, None)
        if func is not None:
            return func(self.plot, filename, nameFilter)
        else:
            _logger.error('Unsupported file filter: %s', nameFilter)
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

    def __init__(self, plot, parent=None):
        super(PrintAction, self).__init__(
            plot, icon='document-print', text='Print...',
            tooltip='Open print dialog',
            triggered=self.printPlot,
            checkable=False, parent=parent)
        self.setShortcut(qt.QKeySequence.Print)
        self.setShortcutContext(qt.Qt.WidgetShortcut)

    def getPrinter(self):
        """The QPrinter instance used by the PrintAction.

        :rtype: QPrinter
        """
        return printer.getDefaultPrinter()

    @property
    @deprecated(replacement="getPrinter()", since_version="0.8.0")
    def printer(self):
        return self.getPrinter()

    def printPlotAsWidget(self):
        """Open the print dialog and print the plot.

        Use :meth:`QWidget.render` to print the plot

        :return: True if successful
        """
        dialog = qt.QPrintDialog(self.getPrinter(), self.plot)
        dialog.setWindowTitle('Print Plot')
        if not dialog.exec_():
            return False

        # Print a snapshot of the plot widget at the top of the page
        widget = self.plot.centralWidget()

        painter = qt.QPainter()
        if not painter.begin(self.getPrinter()):
            return False

        pageRect = self.getPrinter().pageRect()
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
        dialog = qt.QPrintDialog(self.getPrinter(), self.plot)
        dialog.setWindowTitle('Print Plot')
        if not dialog.exec_():
            return False

        # Save Plot as PNG and make a pixmap from it with default dpi
        pngData = _plotAsPNG(self.plot)

        pixmap = qt.QPixmap()
        pixmap.loadFromData(pngData, 'png')

        xScale = self.getPrinter().pageRect().width() / pixmap.width()
        yScale = self.getPrinter().pageRect().height() / pixmap.height()
        scale = min(xScale, yScale)

        # Draw pixmap with painter
        painter = qt.QPainter()
        if not painter.begin(self.getPrinter()):
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
