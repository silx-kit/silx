#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""Qt data view example
"""

import logging
import sys

logging.basicConfig()
_logger = logging.getLogger("customDataView")
"""Module logger"""

from silx.gui import qt
from silx.gui.data.DataViewerFrame import DataViewerFrame
from silx.gui.data.DataViews import DataView
from silx.third_party import enum


class Color(enum.Enum):
    RED = 1
    BLUE = 2
    GREEN = 3


class MyColorView(DataView):

    def __init__(self, parent):
        DataView.__init__(self, parent)

    def label(self):
        return "Color"

    def icon(self):
        pixmap = qt.QPixmap(2, 2)
        painter = qt.QPainter(pixmap)
        painter.setPen(qt.QColor(255, 0, 0))
        painter.drawPoint(qt.QPoint(0, 0))
        painter.setPen(qt.QColor(255, 255, 0))
        painter.drawPoint(qt.QPoint(1, 0))
        painter.setPen(qt.QColor(0, 255, 0))
        painter.drawPoint(qt.QPoint(0, 1))
        painter.setPen(qt.QColor(0, 255, 255))
        painter.drawPoint(qt.QPoint(1, 1))
        painter.end()
        pixmap = pixmap.scaled(32, 32, qt.Qt.IgnoreAspectRatio, qt.Qt.FastTransformation)
        return qt.QIcon(pixmap)

    def setData(self, data):
        widget = self.getWidget()
        colors = {Color.RED: "#FF0000",
                  Color.GREEN: "#00FF00",
                  Color.BLUE: "#0000FF"}
        color = colors.get(data, "#000000")
        text = "<span style='color:%s'>%s</span>" % (color, str(data))
        widget.setText(text)

    def axesNames(self, data, info):
        return None

    def createWidget(self, parent):
        return qt.QLabel(parent)

    def getDataPriority(self, data, info):
        if isinstance(data, Color):
            return 100
        return self.UNSUPPORTED


def main():
    app = qt.QApplication([])

    widget = DataViewerFrame()
    widget.addView(MyColorView(widget))
    widget.setData(Color.GREEN)
    widget.show()
    result = app.exec_()
    # remove ending warnings relative to QTimer
    app.deleteLater()
    sys.exit(result)


if __name__ == "__main__":
    main()
