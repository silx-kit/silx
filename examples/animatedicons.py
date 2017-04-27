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
import functools


class AnimatedToolButton(qt.QToolButton):
    """ToolButton which support animated icons"""

    def __init__(self, parent=None):
        super(AnimatedToolButton, self).__init__(parent)
        self.__animatedIcon = None

    def setIcon(self, icon):
        if isinstance(icon, silx.gui.icons.AbstractAnimatedIcon):
            self._setAnimatedIcon(icon)
        else:
            self._setAnimatedIcon(None)
            super(AnimatedToolButton, self).setIcon(icon)

    def _setAnimatedIcon(self, icon):
        if self.__animatedIcon is not None:
            self.__animatedIcon.unregister(self)
            self.__animatedIcon.iconChanged.disconnect(self.__updateIcon)
        self.__animatedIcon = icon
        if self.__animatedIcon is not None:
            self.__animatedIcon.register(self)
            self.__animatedIcon.iconChanged.connect(self.__updateIcon)
            i = self.__animatedIcon.currentIcon()
        else:
            i = qt.QIcon()
        super(AnimatedToolButton, self).setIcon(i)

    def __updateIcon(self, icon):
        super(AnimatedToolButton, self).setIcon(icon)

    def icon(self):
        if self.__animatedIcon is not None:
            return self.__animatedIcon
        else:
            return super(AnimatedToolButton, self).icon()


class AnimatedIconPreview(qt.QMainWindow):

    def __init__(self, *args, **kwargs):
        qt.QMainWindow.__init__(self, *args, **kwargs)

        widget = qt.QWidget(self)
        self.iconPanel = self.createIconPanel(widget)
        self.sizePanel = self.createSizePanel(widget)

        layout = qt.QVBoxLayout(widget)
        layout.addWidget(self.sizePanel)
        layout.addWidget(self.iconPanel)
        layout.addStretch()
        self.setCentralWidget(widget)

    def createSizePanel(self, parent):
        group = qt.QButtonGroup()
        group.setExclusive(True)
        panel = qt.QWidget(parent)
        panel.setLayout(qt.QHBoxLayout())

        buttons = {}
        for size in [16, 24, 32]:
            button = qt.QPushButton("%spx" % size, panel)
            button.clicked.connect(functools.partial(self.setIconSize, size))
            button.setCheckable(True)
            panel.layout().addWidget(button)
            group.addButton(button)
            buttons[size] = button

        self.__sizeGroup = group
        buttons[24].setChecked(True)
        return panel

    def createIconPanel(self, parent):
        panel = qt.QWidget(parent)
        layout = qt.QVBoxLayout()
        panel.setLayout(layout)

        self.tools = []

        # wait icon
        icon = silx.gui.icons.getWaitIcon()
        tool = AnimatedToolButton(panel)
        tool.setIcon(icon)
        tool.setText("getWaitIcon")
        tool.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
        self.tools.append(tool)

        icon = silx.gui.icons.getAnimatedIcon("process-working")
        tool = AnimatedToolButton(panel)
        tool.setIcon(icon)
        tool.setText("getAnimatedIcon")
        tool.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
        self.tools.append(tool)

        icon = silx.gui.icons.MovieAnimatedIcon("process-working", self)
        tool = AnimatedToolButton(panel)
        tool.setIcon(icon)
        tool.setText("MovieAnimatedIcon")
        tool.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
        self.tools.append(tool)

        icon = silx.gui.icons.MultiImageAnimatedIcon("process-working", self)
        tool = AnimatedToolButton(panel)
        tool.setIcon(icon)
        tool.setText("MultiImageAnimatedIcon")
        tool.setToolButtonStyle(qt.Qt.ToolButtonTextBesideIcon)
        self.tools.append(tool)

        for t in self.tools:
            layout.addWidget(t)

        return panel

    def setIconSize(self, size):
        for tool in self.tools:
            tool.setIconSize(qt.QSize(size, size))


if __name__ == "__main__":
    app = qt.QApplication([])
    window = AnimatedIconPreview()
    window.setVisible(True)
    app.exec_()
