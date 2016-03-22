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
"""A :class:`PlotWidget` with additionnal toolbars.

The :class:`PlotWindow` is a subclass of :class:`PlotWidget`.

Actions
-------

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
__date__ = "07/03/2016"


from collections import OrderedDict
import logging
import os.path
import sys
import traceback
import weakref

if sys.version_info[0] == 3:
    from io import BytesIO
else:
    import cStringIO as _StringIO
    BytesIO = _StringIO.StringIO

import numpy

from . import PlotWidget

from .. import icons
from .. import qt


_logger = logging.getLogger(__name__)


class _PlotAction(qt.QAction):
    """Base class for QAction that operates on a PlotWindow.

    :param plot: :class:`PlotWidget` instance on which to operate.
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
        """The :class:`PlotWidget` this action group is controlling."""
        return self._plotRef()  # TODO handle dead PlotWidget?


class ResetZoomAction(_PlotAction):
    """QAction controlling reset zoom on a PlotWidget.

    :param PlotWidget plot: instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(ResetZoomAction, self).__init__(
            plot,  icon='zoomreset', text='Reset Zoom',
            tooltip='Auto-Scale the Graph',
            triggered=self._actionTriggered,
            checkable=False, parent=parent)

    def _actionTriggered(self, checked=False):
        self.plot.resetZoom()


class XAxisAutoScaleAction(_PlotAction):
    """QAction controlling X axis autoscale on a PlotWidget.

    :param PlotWidget plot: instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(XAxisAutoScaleAction, self).__init__(
            plot, icon='xauto', text='X Autoscale',
            tooltip='Enable X Axis Autoscale when checked',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.setChecked(plot.isXAxisAutoScale())
        plot.sigSetXAxisAutoScale.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.setXAxisAutoScale(checked)


class YAxisAutoScaleAction(_PlotAction):
    """QAction controlling Y axis autoscale on a PlotWidget.

    :param PlotWidget plot: instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(YAxisAutoScaleAction, self).__init__(
            plot, icon='yauto', text='Y Autoscale',
            tooltip='Enable Y Axis Autoscale when checked',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.setChecked(plot.isXAxisLogarithmic())
        plot.sigSetYAxisAutoScale.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.setYAxisAutoScale(checked)


class XAxisLogarithmicAction(_PlotAction):
    """QAction controlling X axis log scale on a PlotWidget.

    :param PlotWidget plot: instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(XAxisLogarithmicAction, self).__init__(
            plot, icon='logx', text='X Log. scale',
            tooltip='Logarithmic X Axis when checked',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.setChecked(plot.isXAxisLogarithmic())
        plot.sigSetXAxisLogarithmic.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.setXAxisLogarithmic(checked)


class YAxisLogarithmicAction(_PlotAction):
    """QAction controlling Y axis log scale on a PlotWidget.

    :param PlotWidget plot: instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(YAxisLogarithmicAction, self).__init__(
            plot, icon='logy', text='Y Log. scale',
            tooltip='Logarithmic Y Axis when checked',
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.setChecked(plot.isYAxisLogarithmic())
        plot.sigSetYAxisLogarithmic.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.setYAxisLogarithmic(checked)


class GridAction(_PlotAction):
    """QAction controlling grid mode on a PlotWidget.

    :param PlotWidget plot: instance on which to operate
    :param str gridMode: The grid mode to use in 'both', 'major'.
                         See :meth:`PlotWidget.setGraphGrid`
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, gridMode='both', parent=None):
        assert gridMode in ('both', 'major')
        self._gridMode = gridMode

        super(GridAction, self).__init__(
            plot, icon='grid16', text='Grid',
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
    """QAction controlling curve style on a PlotWidget.

    It changes the default line and markers style which updates all
    curves on the plot.

    :param PlotWidget plot: instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(CurveStyleAction, self).__init__(
            plot, icon='togglepoints', text='Curve style',
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


class KeepAspectRatioAction(_PlotAction):
    """QAction controlling aspect ratio on a PlotWidget.

    :param PlotWidget plot: instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        # Icon uses two images for checked/unchecked states
        icon = icons.getQIcon('solid_ellipse16')
        icon.addPixmap(icons.getQPixmap('solid_circle16'), state=qt.QIcon.On)
        super(KeepAspectRatioAction, self).__init__(
            plot, icon=icon, text='Keep aspect ratio',
            tooltip="""Change keep data aspect ratio:
            Keep aspect ratio when checked""",
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.setChecked(self.plot.isKeepDataAspectRatio())
        plot.sigSetKeepDataAspectRatio.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.keepDataAspectRatio(checked)


class YAxisInvertedAction(_PlotAction):
    """QAction controlling Y orientation on a PlotWidget.

    :param PlotWidget plot: instance on which to operate
    :param parent: See :class:`QAction`
    """

    def __init__(self, plot, parent=None):
        super(YAxisInvertedAction, self).__init__(
            plot, icon='gioconda16mirror', text='Invert Y Axis',
            tooltip="""Change Y Axis orientation:
            - upward when unchecked,
            - downward when checked""",
            triggered=self._actionTriggered,
            checkable=True, parent=parent)
        self.setChecked(plot.isYAxisInverted())
        plot.sigSetYAxisInverted.connect(self.setChecked)

    def _actionTriggered(self, checked=False):
        self.plot.invertYAxis(checked)


class SaveAction(_PlotAction):
    """QAction for saving Plot content.

    It opens a Save as... dialog.

    :param plot: :class:`PlotWidget` instance on which to operate.
    :param parent: See :class:`QAction`.
    """
    # TODO find a way to make the filter list selectable and extensible

    SNAPSHOT_FILTERS = ('Plot Snapshot PNG *.png', 'Plot Snapshot JPEG *.jpg')

    # Dict of curve filters with CSV-like format
    CURVE_OMNIC_FILTER = 'Curve as OMNIC CSV *.csv'

    # Using ordered dict to guarantee filters order
    # Note: '%.18e' is numpy.savetxt default format
    CURVE_FILTERS_TXT = OrderedDict((
        ('Curve as Raw ASCII *.txt',
         {'fmt': '%.18e', 'delimiter': ' ', 'header': False}),
        ('Curve as ","-separated CSV *.csv',
         {'fmt': '%.18e', 'delimiter': ',', 'header': True}),
        ('Curve as ";"-separated CSV *.csv',
         {'fmt': '%.18e', 'delimiter': ';', 'header': True}),
        ('Curve as tab-separated CSV *.csv',
         {'fmt': '%.18e', 'delimiter': '\t', 'header': True}),
        (CURVE_OMNIC_FILTER,
         {'fmt': '%.7E', 'delimiter': ',', 'header': False})
    ))

    CURVE_FILTERS = list(CURVE_FILTERS_TXT.keys())

    IMAGE_FILTER_NUMPY = 'Image as NumPy binary file *.npy'
    IMAGE_FILTERS = (IMAGE_FILTER_NUMPY,)

    def __init__(self, plot, parent=None):
        super(SaveAction, self).__init__(
            plot, icon='filesave', text='Save as...',
            tooltip='Save Curve/Image/Plot Snapshot Dialog',
            triggered=self._actionTriggered,
            checkable=False, parent=parent)

    @staticmethod
    def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header=''):
        """numpy.savetxt backport of header argument from numpy=1.7.0.

        For Debian 7 compatibility, replace by numpy.savetxt when dropping
        support of numpy < 1.7.0

        See numpy.savetxt for details.
        """
        # Open the file in text mode with \n newline on all OS
        if sys.version_info[0] >= 3:
            ffile = open(fname, 'w', newline='\n')
        else:
            ffile = open(fname,'wb')

        if header:
            ffile.write(header + '\n')

        numpy.savetxt(ffile, X, fmt, delimiter, newline)

        ffile.close()

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
        if curve is None:
            curves = self.plot.getAllCurves()
            if not curves:
                self._errorMessage("No curve to be saved")
                return False
            curve = curves[0]  # TODO why not the last one?

        # TODO Use silx.io for writing files
        if nameFilter in self.CURVE_FILTERS_TXT:
            filter_ = self.CURVE_FILTERS_TXT[nameFilter]
            if filter_['header']:
                header = '"%s"%s"%s"' % (curve[4]['xlabel'],
                                         delimiter,
                                         curve[4]['ylabel'])
            else:
                header = ''

            # For numpy<1.7.0 compatibility
            # replace with numpy.savetxt when dropping Debian 7 support
            try:
                self.savetxt(filename,
                             numpy.array((curve[0], curve[1])).T,
                             fmt=filter_['fmt'],
                             delimiter=filter_['delimiter'],
                             header=header)
            except IOError:
                self._errorMessage('Save failed\n')
                return False
            return True

        return False

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
        if nameFilter == self.IMAGE_FILTER_NUMPY:
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
                self.plot.getAllCurves()):
            filters.extend(self.CURVE_FILTERS)

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
        elif nameFilter in self.IMAGE_FILTERS:
            return self._saveImage(filename, nameFilter)
        else:
            _logger.warning('Unsupported file filter: %s', nameFilter)
            return False


class PrintAction(_PlotAction):
    """QAction for printing the plot.

    It opens a Print dialog.

    Current implementation print a bitmap of the plot area and not vector
    graphics, so printing quality is not great.

    :param plot: :class:`PlotWidget` instance on which to operate.
    :param parent: See :class:`QAction`.
    """

    def __init__(self, plot, parent=None):
        super(PrintAction, self).__init__(
            plot, icon='fileprint', text='Print...',
            tooltip='Open Print Dialog',
            triggered=self.printPlot,
            checkable=False, parent=parent)

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
        pngFile = BytesIO()
        self.plot.saveGraph(pngFile, fileFormat='png')
        pngFile.flush()
        pngFile.seek(0)
        pngData = pngFile.read()
        pngFile.close()

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


class _PlotActionGroup(qt.QActionGroup):
    """Base class for QActionGroup to attach to a PlotWindow.

    :param plot: :class:`PlotWidget` instance on which to operate.
    :param str title: The title to use when creating menus and toolbars.
    :param parent: See :class:`QActionGroup`.
    """

    def __init__(self, plot, title, parent=None):
        super(_PlotActionGroup, self).__init__(parent)
        assert plot is not None
        self._plotRef = weakref.ref(plot)

        self.title = title
        """Title to give to menu and toolbar created from this action group."""

    @property
    def plot(self):
        """The :class:`PlotWidget` this action group is controlling."""
        return self._plotRef()  # TODO handle dead PlotWidget?

    def toolBar(self, parent=None):
        """Return a QToolBar from the QAction in this group.

        :param parent: See :class:`QToolBar`
        """
        toolbar = qt.QToolBar(self.title, parent)
        for action in self.actions():
            toolbar.addAction(action)
        toolbar.actionGroup = self  # Toolbar keeps a reference to actionGroup
        return toolbar

    def menu(self, parent=None):
        """Return a QMenu from the QAction in this group.

        :param parent: See :class:`QMenu`
        """
        menu = qt.QMenu(self.title, parent)
        for action in self.actions():
            menu.addAction(action)
        menu.actionGroup = self  # Menu keeps a reference to actionGroup
        return menu


class PlotActionGroup(_PlotActionGroup):
    """QActionGroup with tools to control plot area.
    
    This QActionGroup includes the following QAction:

    :var:`resetZoomAction`: Reset zoom
    :var:`xAxisAutoScaleAction`: Toggle X axis autoscale
    :var:`yAxisAutoScaleAction`: Toggle Y axis autoscale
    :var:`xAxisLogarithmicAction`: Toggle X axis log scale
    :var:`yAxisLogarithmicAction`: Toggle Y axis log scale
    :var:`gridAction`: Toggle plot grid
    :var:`curveStyleAction`: Change curve line and markers style
    :var:`keepDataAspectRatioAction`: Toggle keep aspect ratio
    :var:`yAxisInvertedAction`: Toggle Y Axis direction
    :var:`saveAction`: Save plot
    :var:`printAction`: Print plot

    :param plot: :class:`PlotWidget` instance on which to operate.
    :param str title: See :class:`QToolBar`.
    :param parent: See :class:`QToolBar`.
    :param bool resetZoom: Toggle visibility of reset zoom action.
    :param bool autoScale: Toggle visibility of axes autoscale actions.
    :param bool logScale: Toggle visibility of axes log scale actions.
    :param bool grid: Toggle visibility of grid mode action.
    :param bool curveStyle: Toggle visibility of curve style action.
    :param bool yInverted: Toggle visibility of Y axis direction action.
    :param bool save: Toggle visibility of save action.
    :param bool print_: Toggle visibility of print action.
    """

    def __init__(self, plot, title='Plot', parent=None,
                 resetZoom=True, autoScale=True, logScale=True, grid=True,
                 curveStyle=True,
                 aspectRatio=True, yInverted=True, save=True, print_=True):
        super(PlotActionGroup, self).__init__(plot, title, parent)

        self.resetZoomAction = self.addAction(ResetZoomAction(plot))
        if not resetZoom:
            self.resetZoomAction.setVisible(False)

        self.xAxisAutoScaleAction = self.addAction(
            XAxisAutoScaleAction(plot))
        if not autoScale:
            self.xAxisAutoScaleAction.setVisible(False)

        self.yAxisAutoScaleAction = self.addAction(
            YAxisAutoScaleAction(plot))
        if not autoScale:
            self.yAxisAutoScaleAction.setVisible(False)

        self.xAxisLogarithmicAction = self.addAction(
            XAxisLogarithmicAction(plot))
        if not logScale:
            self.xAxisLogarithmicAction.setVisible(False)

        self.yAxisLogarithmicAction = self.addAction(
            YAxisLogarithmicAction(plot))
        if not logScale:
            self.yAxisLogarithmicAction.setVisible(False)

        self.gridAction = self.addAction(GridAction(plot, gridMode='both'))
        if not grid:
            self.gridAction.setVisible(False)

        self.curveStyleAction = self.addAction(CurveStyleAction(plot))
        if not curveStyle:
            self.curveStyleAction.setVisible(False)

        # colormap TODO need a dialog

        # zoom mode TODO need sync with other toolbars

        # Make icon with on and 
        self.keepDataAspectRatioAction = self.addAction(
            KeepAspectRatioAction(plot))
        if not aspectRatio:
            self.keepDataAspectRatioAction.setVisible(False)

        self.yAxisInvertedAction = self.addAction(
            YAxisInvertedAction(plot))
        if not yInverted:
            self.yAxisInvertedAction.setVisible(False)

        # Plugin TODO outside here

        separator = self.addAction('separator')
        separator.setSeparator(True)

        # TODO Copy to clipboard?

        self.saveAction = self.addAction(SaveAction(self.plot))
        if not save:
            self.saveAction.setVisible(False)

        self.printAction = self.addAction(PrintAction(self.plot))
        if not print_:
            self.print_Action.setVisible(False)

        self.setExclusive(False)


# TODO synchro between toolbars
# TODO mask toolbar
# TODO profile toolbar

class PlotWindow(PlotWidget):
    """Qt Widget providing a 1D/2D plot area and additional tools.

    :param parent: The parent of this widget or None.
    :param backend: The backend to use for the plot.
                    The default is to use matplotlib.
    :type backend: str or :class:`BackendBase.BackendBase`
    """

    def __init__(self, parent=None, backend=None):
        super(PlotWindow, self).__init__(parent=parent, backend=backend)
        self._plotGroup = PlotActionGroup(self)
        self.addToolBar(self._plotGroup.toolBar())
        self._plotMenu = self._plotGroup.menu()
        self.menuBar().addMenu(self._plotMenu)
