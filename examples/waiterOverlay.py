#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2023 European Synchrotron Radiation Facility
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
Demonstration window that displays a wait icon until the plot is updated
"""

import numpy.random
from silx.gui import qt
from silx.gui.widgets.WaitingOverlay import WaitingOverlay
from silx.gui.plot import Plot2D


class MyMainWindow(qt.QMainWindow):
    WAITING_TIME = 2000  # ms

    def __init__(self, parent=None):
        super().__init__(parent)

        # central plot
        self._plot = Plot2D()
        self._waitingOverlay = WaitingOverlay(self._plot)
        self.setCentralWidget(self._plot)

        # button to trigger image generation
        self._rightPanel = qt.QWidget(self)
        self._rightPanel.setLayout(qt.QVBoxLayout())
        self._button = qt.QPushButton("generate image", self)
        self._rightPanel.layout().addWidget(self._button)

        self._dockWidget = qt.QDockWidget()
        self._dockWidget.setWidget(self._rightPanel)
        self.addDockWidget(qt.Qt.RightDockWidgetArea, self._dockWidget)

        # set up
        self._waitingOverlay.hide()
        self._waitingOverlay.setIconSize(qt.QSize(60, 60))
        # connect signal / slot
        self._button.clicked.connect(self._triggerImageCalculation)

    def _generateRandomData(self):
        self.setData(numpy.random.random(1000 * 500).reshape((1000, 500)))
        self._button.setEnabled(True)

    def setData(self, data):
        self._plot.addImage(data)
        self._waitingOverlay.hide()

    def _triggerImageCalculation(self):
        self._plot.clear()
        self._button.setEnabled(False)
        self._waitingOverlay.show()
        qt.QTimer.singleShot(self.WAITING_TIME, self._generateRandomData)


qapp = qt.QApplication([])
window = MyMainWindow()
window.show()
qapp.exec_()
