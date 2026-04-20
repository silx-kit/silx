#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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
Example to show the use of :class:`~silx.gui.items.group.Group`.
"""

__license__ = "MIT"

import logging
import numpy
from silx.gui.plot import PlotWindow
from silx.gui import qt
from silx.gui.plot.items.group import Group
from silx.gui.plot.items.curve import Curve


logging.basicConfig()
logger = logging.getLogger(__name__)


class MyGroup(Group):

    def __init__(self):
        super(MyGroup, self).__init__()
        self._cursor = 0
        self._tick = 0

    def update(self):
        x = numpy.linspace(self._cursor, self._cursor + 100, 50)
        y = numpy.random.poisson((numpy.sin(x / 100) + 1) * 100)
        self._cursor += 100
        self._tick += 1

        curve = Curve()
        curve.setColor("C0")
        curve.setSymbol("")
        curve.setLineStyle("-")
        curve.setData(x, y, copy=False)
        curve.setVisible(self._tick % 7 != 0)
        self.addItem(curve)

        if len(self.getItems()) > 10:
            old = self.getItems()[0]
            self.removeItem(old)


def main(argv=None):
    global app  # QApplication must be global to avoid seg fault on quit
    app = qt.QApplication([])
    sys.excepthook = qt.exceptionHandler

    win = qt.QWidget()
    win.setAttribute(qt.Qt.WA_DeleteOnClose)
    layout = qt.QVBoxLayout(win)

    plot = PlotWindow()
    layout.addWidget(plot)

    item = MyGroup()
    plot.addItem(item)

    def update():
        item.update()
        plot.resetZoom()

    t = qt.QTimer()
    t.timeout.connect(update)
    t.start(1000)

    show = qt.QPushButton()
    show.setText("Show")
    layout.addWidget(show)
    show.clicked.connect(lambda: item.setVisible(True))

    hide = qt.QPushButton()
    hide.setText("Hide")
    layout.addWidget(hide)
    hide.clicked.connect(lambda: item.setVisible(False))

    win.show()
    return app.exec()


if __name__ == "__main__":
    import sys
    sys.exit(main(argv=sys.argv[1:]))
