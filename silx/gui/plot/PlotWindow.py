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
"""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "07/03/2016"


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

from . import PlotWidget

from .. import icons
from .. import qt

# TODO move actions to PlotWindow and override plot setters to sync the widgets

_logger = logging.getLogger(__name__)


class _PlotAction(qt.QAction):
    """Base class for QAction to attache to a PlotWindow.

    :param plotWindow: :class:`PlotWindow` instance on which to operate.
    :param icon: QIcon or str name of icon to use
    :param str text: The name of this action to be used for menu label
    :param str tooltip: The text of the tooltip
    :param triggered: The callback to connect to the action's triggered
                      signal or None for no callback.
    :param bool checkable: True for checkable action, False otherwise (default)
    :param parent: See :class:`QActionGroup`.
    """

    def __init__(self, plotWindow, icon, text, tooltip=None,
                 triggered=None, checkable=False, parent=None):
        assert plotWindow is not None
        self._plotWindowRef = weakref.ref(plotWindow)

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
    def plotWindow(self):
        """The :class:`PlotWindow` this action group is controlling."""
        return self._plotWindowRef()  # TODO handle dead PlotWindow?

    def sync(self):
        """Synchronize action with current state of the PlotWindow.

        This might be necessary as it is not possible to listen all PlotWindow
        state changes.
        """
        pass


class _PlotActionGroup(qt.QActionGroup):
    """Base class for QActionGroup to attach to a PlotWindow.

    :param plotWindow: :class:`PlotWindow` instance on which to operate.
    :param str title: The title to use when creating menus and toolbars.
    :param parent: See :class:`QActionGroup`.
    """

    def __init__(self, plotWindow, title, parent=None):
        super(_PlotActionGroup, self).__init__(parent)
        assert plotWindow is not None
        self._plotWindowRef = weakref.ref(plotWindow)

        self.title = title
        """Title to give to menu and toolbar created from this action group."""

    @property
    def plotWindow(self):
        """The :class:`PlotWindow` this action group is controlling."""
        return self._plotWindowRef()  # TODO handle dead PlotWindow?

    def _addAction(self, icon, text, tooltip=None, triggered=None, checkable=False):
        """Convenient method to create a QAction and add it to the action group.

        :param icon: QIcon or str name of icon to use
        :param str text: The name of this action to be used for menu label
        :param str tooltip: The text of the tooltip
        :param triggered: The callback to connect to the action's triggered
                          signal or None for no callback.
        :param bool checkable: True for checkable action, False otherwise (default)
        :return: The added QAction
        """
        action = _PlotAction(
            self.plotWindow, icon, text, tooltip, triggered, checkable)
        return self.addAction(action)

    def sync(self):
        """Synchronize action group with current state of the PlotWindow.

        This might be necessary as it is not possible to listen all PlotWindow
        state changes.
        """
        for action in self.actions():
            if hasattr(action, 'sync'):
                action.sync()

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


class SaveAction(_PlotAction):

    SNAPSHOT_FILTERS = ('Plot Snapshot PNG *.png', 'Plot Snapshot JPEG *.jpg')

    CURVE_FILTER_ASCII = 'Curve Raw ASCII *.txt'
    CURVE_FILTERS = (CURVE_FILTER_ASCII,)

    IMAGE_FILTER_ASCII = 'Image Raw ASCII *.dat'
    IMAGE_FILTERS = (IMAGE_FILTER_ASCII,)

    def __init__(self, plotWindow, parent=None):
        super(SaveAction, self).__init__(
            plotWindow, icon='filesave', text='Save as...',
            tooltip='Open Save Plot Snapshot/Curve/Image Dialog',
            checkable=False, parent=parent)
        self.triggered.connect(self._saveActionTriggered)

    def _errorMessage(self, informativeText=''):
        """Display an error message."""
        # TODO issue with QMessageBox size fixed and too small
        msg = qt.QMessageBox(self.plotWindow)
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
        pixmap = qt.QPixmap.grabWidget(self.plotWindow.centralWidget())
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
        curve = self.plotWindow.getActiveCurve()
        if curve is None:
            curves = self.plotWindow.getAllCurves()
            if not curves:
                self._errorMessage("No curve to be saved")
                return False
            curve = curves[0]  # TODO why not the last one?

        # TODO Use silx.io for writing files
        if nameFilter == self.CURVE_FILTER_ASCII:  # Save curve as raw ASCII
            try:
                if sys.version_info.major >= 3:
                    ffile = open(filename, 'w', newline='\n')
                else:
                    ffile = open(filename,'wb')
            except IOError:
                self._errorMessage("Cannot open file.")
                return False
            else:
                for x, y in zip(curve['x'], curve['y']):
                    ffile.write("%.7g  %.7g\n" % (x, y))
                ffile.close()
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

        image = self.plotWindow.getActiveImage()
        if image is None:
            qt.QMessageBox.warning(
                self.plotWindow, "No Data", "No image to be saved")
            return False

        # TODO Use silx.io for writing files
        if nameFilter == self.IMAGE_FILTER_ASCII:  # Save image as raw ASCII
            try:
                if sys.version_info.major >= 3:
                    ffile = open(filename, 'w', newline='\n')
                else:
                    ffile = open(filename,'wb')
            except IOError:
                self._errorMessage("Cannot open file.")
            else:
                ffile.write("row column value\n")
                for row in range(image.shape[0]):
                    for column in range(image.shape[1]):
                        ffile.write(
                            "%d  %d  %g\n" % (row, column, image[row, column]))
                ffile.close()
        return True

    def _saveActionTriggered(self, checked=False):
        """Handle save action."""
        # Set-up filters
        filters = []

        # Add image filters if there is an active image
        if self.plotWindow.getActiveImage() is not None:
            filters.extend(self.IMAGE_FILTERS)

        # Add curve filters if there is a curve to save
        if (self.plotWindow.getActiveCurve() is not None or
                self.plotWindow.getAllCurves()):
            filters.extend(self.CURVE_FILTERS)

        filters.extend(self.SNAPSHOT_FILTERS)

        # Create and run File dialog
        dialog = qt.QFileDialog(self.plotWindow)
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
    # TODO issue with QtSvg which do not clip content out of plot area

    def __init__(self, plotWindow, parent=None):
        super(PrintAction, self).__init__(
            plotWindow, 'fileprint', 'Print...', parent)
        self.setToolTip('Open Print Dialog')
        self.setCheckable(False)
        self.triggered.connect(self.printPlot)

    def _getSvgRenderer(self):
        if not qt.HAS_SVG:
            raise RuntimeError(
                "QtSvg module missing. Please compile Qt with SVG support")

        # Save plot as svg
        imgData = BytesIO()
        self.plotWindow.saveGraph(imgData, fileFormat='svg')
        imgData.flush()
        imgData.seek(0)

        # Give it to QtSVG
        svgRawData = imgData.read()
        svgRendererData = qt.QXmlStreamReader(svgRawData)
        svgRenderer = qt.QSvgRenderer(svgRendererData)
        # TODO is this useful? does it really needs to keep references
        svgRenderer._svgRawData = svgRawData
        svgRenderer._svgRendererData = svgRendererData
        return svgRenderer

    def printPlot(self):
        # TODO make this settable through a second icon
        width, height = None, None
        xOffset, yOffset = 0., 0.
        units = 'inches'
        keepAspectRatio = True

        printer = qt.QPrinter()

        # allow printer selection/configuration
        printDialog = qt.QPrintDialog(printer, self.plotWindow)
        if not printDialog.exec_():
            return

        try:
            painter = qt.QPainter()
            if not(painter.begin(printer)):
                return 0
            dpix = printer.logicalDpiX()
            dpiy = printer.logicalDpiY()

            # margin = int((2/2.54) * dpiy)  # 2cm margin
            availableWidth = printer.width()  # - 1 * margin
            availableHeight = printer.height()  # - 2 * margin

            # get the available space
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
                    txt = ("Available width  %f is less than " +
                           "requested width %f" % (availableWidth, width))
                    raise ValueError(txt)
                availableWidth = width
            if height is not None:
                if (availableHeight + 0.1) < height:
                    txt = ("Available height %f is less than " +
                           "requested height %f" %
                           (availableHeight, height))
                    raise ValueError(txt)
                availableHeight = height

            if keepAspectRatio:
                # get the aspect ratio
                widget = self.plotWindow.centralWidget()
                if widget is None:
                    # does this make sense?
                    graphWidth = availableWidth
                    graphHeight = availableHeight
                else:
                    graphWidth = float(widget.width())
                    graphHeight = float(widget.height())

                graphRatio = graphHeight / graphWidth
                # that ratio has to be respected

                bodyWidth = availableWidth
                bodyHeight = availableWidth * graphRatio

                if bodyHeight > availableHeight:
                    bodyHeight = availableHeight
                    bodyWidth = bodyHeight / graphRatio
            else:
                bodyWidth = availableWidth
                bodyHeight = availableHeight

            body = qt.QRectF(
                xOffset, yOffset, bodyWidth, bodyHeight)
            svgRenderer = self._getSvgRenderer()
            svgRenderer.render(painter, body)
        finally:
            painter.end()


class PlotActionGroup(_PlotActionGroup):
    """QActionGroup with tools to control plot area.
    
    Provided functionalities:

    - Reset zoom
    - Toggle axes autoscale
    - Toggle axes log scale
    - Toggle plot grid
    - Toggle Y Axis direction
    - Save plot
    - Print plot

    :param plotWindow: :class:`PlotWindow` instance on which to operate.
    :param str title: See :class:`QToolBar`.
    :param parent: See :class:`QToolBar`.
    """

    def __init__(self, plotWindow, title='Plot', parent=None,
                 resetZoom=True, autoScale=True, logScale=True, grid=True,
                 curveStyle=True,
                 aspectRatio=True, yInverted=True, save=True, print_=True):
        super(PlotActionGroup, self).__init__(plotWindow, title, parent)

        self.resetZoomAction = self._addAction(
            icon='zoomreset',
            text='Reset Zoom',
            tooltip='Auto-Scale the Graph',
            triggered=self._resetZoomActionTriggered,
            checkable=False)
        if not resetZoom:
            self.resetZoomAction.setVisible(False)

        self.xAxisAutoScaleAction = self._addAction(
            icon='xauto',
            text='X Autoscale',
            tooltip='Enable X Axis Autoscale when checked',
            triggered=self._xAxisAutoScaleActionTriggered,
            checkable=True)
        if not autoScale:
            self.xAxisAutoScaleAction.setVisible(False)

        self.yAxisAutoScaleAction = self._addAction(
            icon='yauto',
            text='Y Autoscale',
            tooltip='Enable Y Axis Autoscale when checked',
            triggered=self._yAxisAutoScaleActionTriggered,
            checkable=True)
        if not autoScale:
            self.yAxisAutoScaleAction.setVisible(False)

        self.xAxisLogarithmicAction = self._addAction(
            icon='logx',
            text='X Log. scale',
            tooltip='Logarithmic X Axis when checked',
            triggered=self._xAxisLogarithmicActionTriggered,
            checkable=True)
        if not logScale:
            self.xAxisLogarithmicAction.setVisible(False)

        self.yAxisLogarithmicAction = self._addAction(
            icon='logy',
            text='Y Log. scale',
            tooltip='Logarithmic Y Axis when checked',
            triggered=self._yAxisLogarithmicActionTriggered,
            checkable=True)
        if not logScale:
            self.yAxisLogarithmicAction.setVisible(False)

        self.gridAction = self._addAction(
            icon='grid16',
            text='Grid',
            tooltip='Toggle grid (On/Off)',
            triggered=self._gridActionTriggered,
            checkable=True)
        if not grid:
            self.gridAction.setVisible(False)

        self.curveStyleAction = self._addAction(
            icon='togglepoints',
            text='Curve style',
            tooltip='Change curve line and markers style',
            triggered=self._curveStyleActionTriggered,
            checkable=False)
        if not curveStyle:
            self.curveStyleAction.setVisible(False)

        # colormap TODO need a dialog

        # zoom mode TODO need sync with other toolbars

        # Make icon with on and 
        icon = icons.getQIcon('solid_ellipse16')
        icon.addPixmap(icons.getQPixmap('solid_circle16'), state=qt.QIcon.On)
        self.keepDataAspectRatioAction = self._addAction(
            icon=icon,
            text='Keep aspect ratio',
            tooltip="""Change keep data aspect ratio:
            Keep aspect ratio when checked""",
            triggered=self._keepDataAspectRatioActionTriggered,
            checkable=True)
        if not aspectRatio:
            self.keepDataAspectRatioAction.setVisible(False)

        self.yAxisInvertedAction = self._addAction(
            icon='gioconda16mirror',
            text='Invert Y Axis',
            tooltip="""Change Y Axis orientation:
            - upward when unchecked,
            - downward when checked""",
            triggered=self._yAxisInvertedActionTriggered,
            checkable=True)
        if not yInverted:
            self.yAxisInvertedAction.setVisible(False)

        # Plugin TODO outside here

        separator = self.addAction('separator')
        separator.setSeparator(True)

        # TODO Copy to clipboard?

        self.saveAction = self.addAction(SaveAction(self.plotWindow))
        if not save:
            self.saveAction.setVisible(False)

        self.printAction = self.addAction(PrintAction(self.plotWindow))
        if not print_:
            self.print_Action.setVisible(False)


        self.setExclusive(False)
        self.sync()

    def _resetZoomActionTriggered(self, checked=False):
        self.plotWindow.resetZoom()

    def _xAxisAutoScaleActionTriggered(self, checked=False):
        self.plotWindow.setXAxisAutoScale(checked)

    def _yAxisAutoScaleActionTriggered(self, checked=False):
        self.plotWindow.setYAxisAutoScale(checked)

    def _xAxisLogarithmicActionTriggered(self, checked=False):
        self.plotWindow.setXAxisLogarithmic(checked)

    def _yAxisLogarithmicActionTriggered(self, checked=False):
        self.plotWindow.setYAxisLogarithmic(checked)

    def _gridActionTriggered(self, checked=False):
        self.plotWindow.setGraphGrid('both' if checked else None)

    def _curveStyleActionTriggered(self, checked=False):
        currentState = (self.plotWindow.isDefaultPlotLines(),
                        self.plotWindow.isDefaultPlotPoints())

        # line only, line and symbol, symbol only
        states = (True, False), (True, True), (False, True)
        newState = states[(states.index(currentState) + 1) % 3]

        self.plotWindow.setDefaultPlotLines(newState[0])
        self.plotWindow.setDefaultPlotPoints(newState[1])

    def _keepDataAspectRatioActionTriggered(self, checked=False):
        self.plotWindow.keepDataAspectRatio(checked)

    def _yAxisInvertedActionTriggered(self, checked=False):
        self.plotWindow.invertYAxis(checked)
        self.plotWindow.replot()

    def sync(self):
        self.xAxisAutoScaleAction.setChecked(
            self.plotWindow.isXAxisAutoScale())
        self.yAxisAutoScaleAction.setChecked(
            self.plotWindow.isYAxisAutoScale())
        self.xAxisLogarithmicAction.setChecked(
            self.plotWindow.isXAxisLogarithmic())
        self.yAxisLogarithmicAction.setChecked(
            self.plotWindow.isXAxisLogarithmic())
        self.gridAction.setChecked(
            self.plotWindow.getGraphGrid() is not None)
        self.yAxisInvertedAction.setChecked(
            self.plotWindow.isYAxisInverted())
        self.keepDataAspectRatioAction.setChecked(
            self.plotWindow.isKeepDataAspectRatio())


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

    def sync(self):
        """Synchronize attached tools (i.e. toolbar and menus)."""
        self._plotGroup.sync()
