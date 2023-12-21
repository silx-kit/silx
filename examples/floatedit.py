#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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
This script illustrates the use of :class:`silx.gui.widgets.FloatEdit.FloatEdit`.
"""

from silx.gui import qt
from silx.gui.widgets.FloatEdit import FloatEdit


class Dialog(qt.QMainWindow):
    def __init__(self, *args, **kwargs):
        qt.QMainWindow.__init__(self, *args, **kwargs)

        widget = qt.QWidget(self)
        layout = qt.QHBoxLayout(widget)

        f1 = FloatEdit(self)
        layout.addWidget(f1)

        f2 = FloatEdit(self)
        layout.addWidget(f2)

        f3 = FloatEdit(self)
        f3.setWidgetResizable(True)
        layout.addWidget(f3)

        b = qt.QPushButton(self)
        b.setText("f3=100")
        b.clicked.connect(lambda: f3.setValue(100))
        layout.addWidget(b)

        self.setCentralWidget(widget)


if __name__ == "__main__":
    app = qt.QApplication([])
    window = Dialog()
    window.setVisible(True)
    app.exec()
