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
"""This script shows a gallery of simple widgets provided by silx.

It shows the following widgets:

- :class:WaitingPushButton: A button with a progress-like waiting animated icon
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "01/09/2016"

import sys
from silx.gui import qt
from silx.gui.widgets.WaitingPushButton import WaitingPushButton


class SimpleWidgetExample(qt.QMainWindow):
    """This windows shows simple widget provided by silx."""

    def __init__(self):
        """Constructor"""
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("Silx simple widget example")

        main_panel = qt.QWidget(self)
        main_panel.setLayout(qt.QVBoxLayout())
        main_panel.layout().addWidget(self.createWaitingPushButton())
        main_panel.layout().addWidget(self.createWaitingPushButton2())

        self.setCentralWidget(main_panel)

    def createWaitingPushButton(self):
        widget = WaitingPushButton("Push me and wait for ever")
        widget.clicked.connect(widget.swapWaiting)
        return widget

    def createWaitingPushButton2(self):
        widget = WaitingPushButton("Push me")
        widget.setDisabledWhenWaiting(False)
        widget.clicked.connect(widget.swapWaiting)
        return widget


def main():
    """
    Main function
    """
    app = qt.QApplication([])
    window = SimpleWidgetExample()
    window.show()
    result = app.exec_()
    # remove ending warnings relative to QTimer
    app.deleteLater()
    sys.exit(result)


main()
