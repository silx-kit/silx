# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""This module provides a set of QAction to use with :class:`.PlotWidget`.

The following QAction are available:

- :class:`ColormapAction`
- :class:`CopyAction`
- :class:`CurveStyleAction`
- :class:`GridAction`
- :class:`KeepAspectRatioAction`
- :class:`PrintAction`
- :class:`ResetZoomAction`
- :class:`SaveAction`
- :class:`XAxisLogarithmicAction`
- :class:`XAxisAutoScaleAction`
- :class:`YAxisInvertedAction`
- :class:`YAxisLogarithmicAction`
- :class:`YAxisAutoScaleAction`
"""

from __future__ import division


__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "12/04/2016"


from collections import OrderedDict
import logging
import sys
import traceback
import weakref

if sys.version_info[0] == 3:
    from io import BytesIO
else:
    import cStringIO as _StringIO
    BytesIO = _StringIO.StringIO

import numpy

from .. import icons
from .. import qt
from .ColormapDialog import ColormapDialog
from silx.third_party.EdfFile import EdfFile
from silx.third_party.TiffIO import TiffIO

from silx.io.utils import save1D, savespec


_logger = logging.getLogger(__name__)


class _PlotAction(qt.QAction):
    """Base class for QAction that operates on a PlotWidget.

    :param plot: :class:`.PlotWidget` instance on which to operate.
    :param icon: QIcon or str name of icon to use
    :param str text: The name of this action to be used for menu label
    :param str tooltip: The text of the tooltip
    :param triggered: The callback to connect to the action's triggered
                      signal or None for no callback.
    :param bool checkable: True for checkable action, False otherwise (default)
    :param parent: See :class:`QAction`.
    """

    def __init__(self, plot, icon, text, tooltip=None,
                 triggered=None, checkable=False, parent=None):
        assert plot is not None
        self._plotRef = weakref.ref(plot)

        if not isinstance(icon, qt.QIcon):
            # Try with icon as a string and load corresponding icon
            icon = icons.getQIcon(icon)

        super(_PlotAction, self).__init__(icon, text, None)

        if tooltip is not None:
            self.setToolTip(tooltip)

        self.setCheckable(checkable)

        if triggered is not None:
            self.triggered.connect(triggered)

    @property
    def plot(self):
        """The :class:`.PlotWidget` this action group is controlling."""
        return self._plotRef()


class ResetZoomAction(_PlotAction):
    """QAction controlling reset zoom on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(ResetZoomAction, self).__init__(
            plot,  icon='zoom-original', text='Reset Zoom',
            tooltip='Auto-Scale the Graph',
            triggered=self._actionTriggered,
            checkable=False, parent=parent)

    def _actionTriggered(self, checked=False):
        self.plot.resetZoom()


class XAxisAutoScaleAction(_PlotAction):
    """QAction controlling X axis autoscale on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(XAxisAutoScaleAction, self).__init__(
            plot, icon='plot-xauto', text='X Autoscale',
            tooltip='Enable X Axis Autoscale when checked',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.setChecked(plot.isXAxisAutoScale())
        plot.sigSetXAxisAutoScale.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.setXAxisAutoScale(checked)


class YAxisAutoScaleAction(_PlotAction):
    """QAction controlling Y axis autoscale on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(YAxisAutoScaleAction, self).__init__(
            plot, icon='plot-yauto', text='Y Autoscale',
            tooltip='Enable Y Axis Autoscale when checked',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.setChecked(plot.isXAxisAutoScale())
        plot.sigSetYAxisAutoScale.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.setYAxisAutoScale(checked)


class XAxisLogarithmicAction(_PlotAction):
    """QAction controlling X axis log scale on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(XAxisLogarithmicAction, self).__init__(
            plot, icon='plot-xlog', text='X Log. scale',
            tooltip='Logarithmic X Axis when checked',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.setChecked(plot.isXAxisLogarithmic())
        plot.sigSetXAxisLogarithmic.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.setXAxisLogarithmic(checked)


class YAxisLogarithmicAction(_PlotAction):
    """QAction controlling Y axis log scale on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(YAxisLogarithmicAction, self).__init__(
            plot, icon='plot-ylog', text='Y Log. scale',
            tooltip='Logarithmic Y Axis when checked',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.setChecked(plot.isYAxisLogarithmic())
        plot.sigSetYAxisLogarithmic.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.setYAxisLogarithmic(checked)


class GridAction(_PlotAction):
    """QAction controlling grid mode on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param str gridMode: The grid mode to use in 'both', 'major'.
                         See :meth:`.PlotWidget.setGraphGrid`
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, gridMode='both', parent=None):
        assert gridMode in ('both', 'major')
        self._gridMode = gridMode

        super(GridAction, self).__init__(
            plot, icon='plot-grid', text='Grid',
            tooltip='Toggle grid (On/Off)',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.setChecked(plot.getGraphGrid() is not None)
        plot.sigSetGraphGrid.connect(self._gridChanged)

    def _gridChanged(self, which):
        """Slot listening for PlotWidget grid mode change."""
        self.setChecked(which != 'None')

    def _actionTriggered(self, checked=False):
        self.plot.setGraphGrid(self._gridMode if checked else None)


class CurveStyleAction(_PlotAction):
    """QAction controlling curve style on a :class:`.PlotWidget`.

    It changes the default line and markers style which updates all
    curves on the plot.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(CurveStyleAction, self).__init__(
            plot, icon='plot-toggle-points', text='Curve style',
            tooltip='Change curve line and markers style',
            triggered=self._actionTriggered,
            checkable=False, parent=parent)

    def _actionTriggered(self, checked=False):
        currentState = (self.plot.isDefaultPlotLines(),
                        self.plot.isDefaultPlotPoints())

        # line only, line and symbol, symbol only
        states = (True, False), (True, True), (False, True)
        newState = states[(states.index(currentState) + 1) % 3]

        self.plot.setDefaultPlotLines(newState[0])
        self.plot.setDefaultPlotPoints(newState[1])


class ColormapAction(_PlotAction):
    """QAction opening a ColormapDialog to update the colormap.

    Both the active image colormap and the default colormap are updated.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """
    def __init__(self, plot, parent=None):
        self._dialog = None  # To store an instance of ColormapDialog
        super(ColormapAction, self).__init__(
            plot, icon='colormap', text='Colormap',
            tooltip="Change colormap",
            triggered=self._actionTriggered,
            checkable=False, parent=parent)

    def _actionTriggered(self, checked=False):
        """Create a cmap dialog and update active image and default cmap."""
        # Create the dialog if not already existing
        if self._dialog is None:
            self._dialog = ColormapDialog()

        image = self.plot.getActiveImage()
        if image is None:
            # No active image, set dialog from default info
            colormap = self.plot.getDefaultColormap()

            self._dialog.setHistogram()  # Reset histogram and range if any

        else:
            # Set dialog from active image
            colormap = image[4]['colormap']

            data = image[0]

            goodData = data[numpy.isfinite(data)]
            if goodData.size > 0:
                dataMin = goodData.min()
                dataMax = goodData.max()
            else:
                qt.QMessageBox.warning(
                    self, "No Data",
                    "Image data does not contain any real value")
                dataMin, dataMax = 1., 10.

            self._dialog.setHistogram()  # Reset histogram if any
            self._dialog.setDataRange(dataMin, dataMax)
            # The histogram should be done in a worker thread
            # hist, bin_edges = numpy.histogram(goodData, bins=256)
            # self._dialog.setHistogram(hist, bin_edges)

        self._dialog.setColormap(**colormap)

        # Run the dialog listening to colormap change
        self._dialog.sigColormapChanged.connect(self._colormapChanged)
        result = self._dialog.exec_()
        self._dialog.sigColormapChanged.disconnect(self._colormapChanged)

        if not result:  # Restore the previous colormap
            self._colormapChanged(colormap)

    def _colormapChanged(self, colormap):
        # Update default colormap
        self.plot.setDefaultColormap(colormap)

        # Update active image
        image = self.plot.getActiveImage()
        if image is not None:
            # Update image: This do not preserve pixmap
            params = image[4].copy()
            params['colormap'] = colormap
            self.plot.addImage(image[0],
                               legend=image[1],
                               replace=False,
                               resetzoom=False,
                               **params)


class KeepAspectRatioAction(_PlotAction):
    """QAction controlling aspect ratio on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        # Uses two images for checked/unchecked states
        self._states = {
            False: (icons.getQIcon('shape-circle-solid'),
                    "Keep data aspect ratio"),
            True: (icons.getQIcon('shape-ellipse-solid'),
                   "Do no keep data aspect ratio")
        }

        icon, tooltip = self._states[plot.isKeepDataAspectRatio()]
        super(KeepAspectRatioAction, self).__init__(
            plot,
            icon=icon,
            text='Toggle keep aspect ratio',
            tooltip=tooltip,
            triggered=self._actionTriggered,
            checkable=False,
            parent=parent)
        plot.sigSetKeepDataAspectRatio.connect(
            self._keepDataAspectRatioChanged)

    def _keepDataAspectRatioChanged(self, aspectRatio):
        """Handle Plot set keep aspect ratio signal"""
        icon, tooltip = self._states[aspectRatio]
        self.setIcon(icon)
        self.setToolTip(tooltip)

    def _actionTriggered(self, checked=False):
        # This will trigger _keepDataAspectRatioChanged
        self.plot.setKeepDataAspectRatio(not self.plot.isKeepDataAspectRatio())


class YAxisInvertedAction(_PlotAction):
    """QAction controlling Y orientation on a :class:`.PlotWidget`.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        # Uses two images for checked/unchecked states
        self._states = {
            False: (icons.getQIcon('plot-ydown'),
                    "Orient Y axis downward"),
            True: (icons.getQIcon('plot-yup'),
                   "Orient Y axis upward"),
        }

        icon, tooltip = self._states[plot.isYAxisInverted()]
        super(YAxisInvertedAction, self).__init__(
            plot,
            icon=icon,
            text='Invert Y Axis',
            tooltip=tooltip,
            triggered=self._actionTriggered,
            checkable=False,
            parent=parent)
        plot.sigSetYAxisInverted.connect(self._yAxisInvertedChanged)

    def _yAxisInvertedChanged(self, inverted):
        """Handle Plot set y axis inverted signal"""
        icon, tooltip = self._states[inverted]
        self.setIcon(icon)
        self.setToolTip(tooltip)

    def _actionTriggered(self, checked=False):
        # This will trigger _yAxisInvertedChanged
        self.plot.setYAxisInverted(not self.plot.isYAxisInverted())


class SaveAction(_PlotAction):
    """QAction for saving Plot content.

    It opens a Save as... dialog.

    :param plot: :class:`.PlotWidget` instance on which to operate.
    :param parent: See :class:`QAction`.
    """
    # TODO find a way to make the filter list selectable and extensible

    SNAPSHOT_FILTERS = ('Plot Snapshot PNG *.png', 'Plot Snapshot JPEG *.jpg')

    # Dict of curve filters with CSV-like format
    # Using ordered dict to guarantee filters order
    # Note: '%.18e' is numpy.savetxt default format
    CURVE_FILTERS_TXT = OrderedDict((
        ('Curve as Raw ASCII *.txt',
         {'fmt': '%.18e', 'delimiter': ' ', 'header': False}),
        ('Curve as ";"-separated CSV *.csv',
         {'fmt': '%.18e', 'delimiter': ';', 'header': True}),
        ('Curve as ","-separated CSV *.csv',
         {'fmt': '%.18e', 'delimiter': ',', 'header': True}),
        ('Curve as tab-separated CSV *.csv',
         {'fmt': '%.18e', 'delimiter': '\t', 'header': True}),
        ('Curve as OMNIC CSV *.csv',
         {'fmt': '%.7E', 'delimiter': ',', 'header': False}),
        ('Curve as SpecFile *.dat',
         {'fmt': '%.7g', 'delimiter': '', 'header': False})
    ))

    CURVE_FILTER_NPY = 'Curve as NumPy binary file *.npy'

    CURVE_FILTERS = list(CURVE_FILTERS_TXT.keys()) + [CURVE_FILTER_NPY]

    ALL_CURVES_FILTERS = ("All curves as SpecFile *.dat", )

    IMAGE_FILTER_EDF = 'Image as EDF *.edf'
    IMAGE_FILTER_TIFF = 'Image as TIFF *.tif'
    IMAGE_FILTER_NUMPY = 'Image as NumPy binary file *.npy'
    IMAGE_FILTERS = (IMAGE_FILTER_EDF, IMAGE_FILTER_TIFF, IMAGE_FILTER_NUMPY)

    def __init__(self, plot, parent=None):
        super(SaveAction, self).__init__(
            plot, icon='document-save', text='Save as...',
            tooltip='Save Curve/Image/Plot Snapshot Dialog',
            triggered=self._actionTriggered,
            checkable=False, parent=parent)
        self.setShortcut(qt.QKeySequence.Save)

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
        pixmap = qt.QPixmap.grabWidget(self.plot.centralWidget())
        if not pixmap.save(filename):
            self._errorMessage()
            return False
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
            fmt=filter_['fmt']
            csvdelim=filter_['delimiter']
            autoheader=filter_['header']
        else:
            # .npy
            fmt, csvdelim, autoheader = ("", "", False)

        try:
            save1D(filename, curve[0], curve[1],
                   curve[4]['xlabel'], [curve[4]['ylabel']],
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
            specfile = savespec(filename, curve[0], curve[1],
                                curve[4]['xlabel'], curve[4]['ylabel'],
                                fmt="%.7g", scan_number=1, mode="w",
                                write_file_header=True,
                                close_file = False)
        except IOError:
            self._errorMessage('Save failed\n')
            return False

        for curve in curves[1:]:
            try:
                scanno += 1
                specfile = savespec(specfile, curve[0], curve[1],
                                    curve[4]['xlabel'], curve[4]['ylabel'],
                                    fmt="%.7g", scan_number=scanno, mode="w",
                                    write_file_header=False,
                                    close_file = False)
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

        data = image[0]

        # TODO Use silx.io for writing files
        if nameFilter == self.IMAGE_FILTER_EDF:
            edfFile = EdfFile(filename, access="w+")
            edfFile.WriteImage({}, data, Append=0)

        elif nameFilter == self.IMAGE_FILTER_TIFF:
            tiffFile = TiffIO(filename, mode='w')
            tiffFile.writeImage(data, software='silx')

        elif nameFilter == self.IMAGE_FILTER_NUMPY:
            try:
                numpy.save(filename, data)
            except IOError:
                self._errorMessage('Save failed\n')
                return False
            return True

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
        extension = nameFilter.split()[-1][1:]
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


class PrintAction(_PlotAction):
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
            tooltip='Open Print Dialog',
            triggered=self.printPlot,
            checkable=False, parent=parent)
        self.setShortcut(qt.QKeySequence.Print)

    def printPlotAsWidget(self):
        """Open the print dialog and print the plot.

        Use :meth:`QWidget.render` to print the plot

        :return: True if successful
        """
        printer = qt.QPrinter()
        dialog = qt.QPrintDialog(printer, self.plot)
        dialog.setWindowTitle('Print Plot')
        if not dialog.exec_():
            return False

        # Print a snapshot of the plot widget at the top of the page
        widget = self.plot.centralWidget()

        painter = qt.QPainter()
        if not painter.begin(printer):
            return False

        pageRect = printer.pageRect()
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
        printer = qt.QPrinter()
        dialog = qt.QPrintDialog(printer, self.plot)
        dialog.setWindowTitle('Print Plot')
        if not dialog.exec_():
            return False

        # Save Plot as PNG and make a pixmap from it with default dpi
        pngData = _plotAsPNG(self.plot)

        pixmap = qt.QPixmap()
        pixmap.loadFromData(pngData, 'png')

        xScale = printer.pageRect().width() / pixmap.width()
        yScale = printer.pageRect().height() / pixmap.height()
        scale = min(xScale, yScale)

        # Draw pixmap with painter
        painter = qt.QPainter()
        if not painter.begin(printer):
            return False

        painter.drawPixmap(0, 0,
                           pixmap.width() * scale,
                           pixmap.height() * scale,
                           pixmap)
        painter.end()

        return True


class CopyAction(_PlotAction):
    """QAction to copy :class:`.PlotWidget` content to clipboard.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(CopyAction, self).__init__(
            plot, icon='edit-copy', text='Copy plot',
            tooltip='Copy a snapshot of the plot the clipboard',
            triggered=self.copyPlot,
            checkable=False, parent=parent)
        self.setShortcut(qt.QKeySequence.Copy)

    def copyPlot(self):
        """Copy plot content to the clipboard as a bitmap."""
        # Save Plot as PNG and make a QImage from it with default dpi
        pngData = _plotAsPNG(self.plot)
        image = qt.QImage.fromData(pngData, 'png')
        qt.QApplication.clipboard().setImage(image)
