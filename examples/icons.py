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
"""
Display available project icons using Qt.
"""
from silx.gui import qt
import silx.gui.icons
import pkg_resources
import functools


class IconPreview(qt.QMainWindow):

    def __init__(self, *args, **kwargs):
        qt.QMainWindow.__init__(self, *args, **kwargs)

        widget = qt.QWidget(self)
        self.iconPanel = self.createIconPanel(widget)
        self.sizePanel = self.createSizePanel(widget)

        layout = qt.QVBoxLayout()
        widget.setLayout(layout)
        # layout.setSizeConstraint(qt.QLayout.SetMinAndMaxSize)
        layout.addWidget(self.sizePanel)
        layout.addWidget(self.iconPanel)
        layout.addStretch()
        self.setCentralWidget(widget)

    def createSizePanel(self, parent):
        group = qt.QButtonGroup()
        group.setExclusive(True)
        panel = qt.QWidget(parent)
        panel.setLayout(qt.QHBoxLayout())

        for size in [16, 24, 32]:
            button = qt.QPushButton("%spx" % size, panel)
            button.clicked.connect(functools.partial(self.setIconSize, size))
            button.setCheckable(True)
            panel.layout().addWidget(button)
            group.addButton(button)

        self.__sizeGroup = group
        button.setChecked(True)
        return panel

    def createIconPanel(self, parent):
        panel = qt.QWidget(parent)
        layout = qt.QGridLayout()
        # layout.setSizeConstraint(qt.QLayout.SetMinAndMaxSize)
        panel.setLayout(layout)

        self.tools = []

        icons = pkg_resources.resource_listdir("silx.resources", "gui/icons")
        # filter out sub-directories
        icons = filter(lambda x: not pkg_resources.resource_isdir("silx.resources", "gui/icons/" + x), icons)
        # remove extension
        icons = [i.split(".")[0] for i in icons]
        # remove duplicated names
        icons = set(icons)
        # sort by names
        icons = sorted(icons)

        for i, icon_name in enumerate(icons):
            col, line = i / 10, i % 10
            icon = silx.gui.icons.getQIcon(icon_name)
            tool = qt.QToolButton(panel)
            tool.setIcon(icon)
            tool.setIconSize(qt.QSize(32, 32))
            tool.setToolTip(icon_name)
            layout.addWidget(tool, col, line)
            self.tools.append(tool)

        return panel

    def setIconSize(self, size):
        for tool in self.tools:
            tool.setIconSize(qt.QSize(size, size))


if __name__ == "__main__":
    app = qt.QApplication([])
    window = IconPreview()
    window.setVisible(True)
    app.exec_()
