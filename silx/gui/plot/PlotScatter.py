# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""A widget dedicated to display scatter plots

It is based on a :class:`!silx.gui.plot.PlotWidget` with additional tools
for scatter plots..
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/03/2018"

import logging
import weakref

from . import PlotWidget
from . import tools
from .ColorBar import ColorBarWidget
from .ScatterMaskToolsWidget import ScatterMaskToolsWidget

from ..widgets.BoxLayoutDockWidget import BoxLayoutDockWidget
from .. import qt, icons


_logger = logging.getLogger(__name__)


class PlotScatter(qt.QMainWindow):
    """Main window with a PlotWidget and tools specific for scatter plots.

    :param parent: The parent of this widget
    :param backend: The backend to use for the plot (default: matplotlib).
                    See :class:`~silx.gui.plot.PlotWidget` for the list of supported backend.
    :type backend: Union[str, silx.gui.plot.backends.BackendBase.BackendBase]
    """

    def __init__(self, parent=None, backend=None):
        super(PlotScatter, self).__init__(parent=parent)
        if parent is None:
            self.setWindowTitle('PlotScatter')

        # Create plot widget
        plot = PlotWidget(parent=self, backend=backend)
        self._plot = weakref.ref(plot)

        # Create colorbar widget with white background
        self._colorbar = ColorBarWidget(parent=self, plot=plot)
        self._colorbar.setAutoFillBackground(True)
        palette = self._colorbar.palette()
        palette.setColor(qt.QPalette.Background, qt.Qt.white)
        palette.setColor(qt.QPalette.Window, qt.Qt.white)
        self._colorbar.setPalette(palette)

        # Combine plot and colorbar into central widget
        gridLayout = qt.QGridLayout()
        gridLayout.setSpacing(0)
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.addWidget(plot, 0, 0)
        gridLayout.addWidget(self._colorbar, 0, 1)
        gridLayout.setRowStretch(0, 1)
        gridLayout.setColumnStretch(0, 1)
        centralWidget = qt.QWidget()
        centralWidget.setLayout(gridLayout)
        self.setCentralWidget(centralWidget)

        # Create mask tool dock widget
        self._maskToolsWidget = ScatterMaskToolsWidget(parent=self, plot=plot)
        self._maskDock = BoxLayoutDockWidget()
        self._maskDock.setWidget(self._maskToolsWidget)
        self._maskDock.setVisible(False)
        self.addDockWidget(qt.Qt.BottomDockWidgetArea, self._maskDock)

        self._maskAction = self._maskDock.toggleViewAction()
        self._maskAction.setIcon(icons.getQIcon('image-mask'))
        self._maskAction.setToolTip("Display/hide mask tools")

        # Create toolbars
        self._interactiveModeToolBar = tools.InteractiveModeToolBar(
            parent=self, plot=plot)

        self._scatterToolBar = tools.ScatterToolBar(
            parent=self, plot=plot)
        self._scatterToolBar.addAction(self._maskAction)

        self._outputToolBar = tools.OutputToolBar(parent=self, plot=plot)

        # Activate shortcuts in PlotWindow widget:
        for toolbar in (self._interactiveModeToolBar,
                        self._scatterToolBar,
                        self._outputToolBar):
            self.addToolBar(toolbar)
            for action in toolbar.actions():
                self.addAction(action)

    def getPlotWidget(self):
        """Returns the :class:`~silx.gui.plot.PlotWidget` this window is based on.

        :rtype: silx.gui.plot.PlotWidget
        """
        return self._plot()

    def getMaskToolsWidget(self):
        """Returns the widget controlling mask drawing

        :rtype: silx.gui.plot.ScatterMaskToolsWidget
        """
        return self._maskToolsWidget

    def getInteractiveModeToolBar(self):
        """Returns QToolBar controlling interactive mode.

        :rtype: silx.gui.plot.tools.InteractiveModeToolBar
        """
        return self._interactiveModeToolBar

    def getScatterToolBar(self):
        """Returns QToolBar providing scatter plot tools.

        :rtype: silx.gui.plot.tools.ScatterToolBar
        """
        return self._scatterToolBar

    def getOutputToolBar(self):
        """Returns QToolBar containing save, copy and print actions

        :rtype: silx.gui.plot.tools.OutputToolBar
        """
        return self._outputToolBar

    # Convenient proxies

    def addScatter(self, *args, **kwargs):
        return self.getPlotWidget().addScatter(*args, **kwargs)

    addScatter.__doc__ = PlotWidget.addScatter.__doc__

    def clear(self, *args, **kwargs):
        return self.getPlotWidget().clear(*args, **kwargs)

    clear.__doc__ = PlotWidget.clear.__doc__

    def setSelectionMask(self, *args, **kwargs):
        return self._maskToolsWidget.setSelectionMask(*args, **kwargs)

    setSelectionMask.__doc__ = ScatterMaskToolsWidget.setSelectionMask.__doc__

    def getSelectionMask(self, *args, **kwargs):
        return self._maskToolsWidget.getSelectionMask(*args, **kwargs)

    getSelectionMask.__doc__ = ScatterMaskToolsWidget.getSelectionMask.__doc__
