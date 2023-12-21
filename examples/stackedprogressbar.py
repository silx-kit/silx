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
This script illustrates the use of :class:`silx.gui.widgets.StackedProgressBar.StackedProgressBar`.
"""

from silx.gui import qt
from silx.gui.widgets.StackedProgressBar import StackedProgressBar


class Dialog(qt.QMainWindow):
    def __init__(self, *args, **kwargs):
        qt.QMainWindow.__init__(self, *args, **kwargs)

        widget = qt.QWidget(self)
        layout = qt.QVBoxLayout(widget)

        f1 = StackedProgressBar(self)
        f1.setRange(0, 100)
        f1.setProgressItem("foo1", value=50, color=qt.QColor("#800000"))
        f1.setProgressItem("foo2", value=20, color=qt.QColor("#008000"))
        f1.setProgressItem("foo3", value=10, color=qt.QColor("#000080"))
        layout.addWidget(f1)

        f2 = StackedProgressBar(self)
        f2.setRange(0, 100)
        f2.setProgressItem("foo1", value=50, color=qt.QColor("#800000"), striped=True)
        f2.setProgressItem("foo2", value=20, color=qt.QColor("#008000"), striped=True)
        f2.setProgressItem("foo3", value=10, color=qt.QColor("#000080"), striped=True)
        layout.addWidget(f2)

        f2_2 = StackedProgressBar(self)
        f2_2.setRange(0, 100)
        f2_2.setProgressItem("foo1", value=50, color=qt.QColor("#FF8080"), striped=True)
        f2_2.setProgressItem("foo2", value=20, color=qt.QColor("#80FF80"), striped=True)
        f2_2.setProgressItem("foo3", value=10, color=qt.QColor("#8080FF"), striped=True)
        layout.addWidget(f2_2)

        f3 = StackedProgressBar(self)
        f3.setRange(0, 100)
        f3.setSpacing(1)
        f3.setProgressItem("foo1", value=50, color=qt.QColor("#800000"), striped=True, animated=True, toolTip="That's foo1")
        f3.setProgressItem("foo2", value=20, color=qt.QColor("#008000"), striped=True, animated=True, toolTip="That's foo2")
        f3.setProgressItem("foo3", value=10, color=qt.QColor("#000080"), striped=True, animated=True, toolTip="That's foo3")
        layout.addWidget(f3)

        b = qt.QPushButton(self)
        b.setText("foo1=20")
        b.clicked.connect(lambda: f3.setProgressItem("foo1", value=20))
        layout.addWidget(b)

        b = qt.QPushButton(self)
        b.setText("foo1=50")
        b.clicked.connect(lambda: f3.setProgressItem("foo1", value=50))
        layout.addWidget(b)

        self.setCentralWidget(widget)


if __name__ == "__main__":
    app = qt.QApplication([])
    window = Dialog()
    window.setVisible(True)
    app.exec()
