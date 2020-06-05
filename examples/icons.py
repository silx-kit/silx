#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
Display icons available in silx.
"""

import functools
import os.path

from silx.gui import qt
import silx.gui.icons
import silx.resources


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

    def getAllAvailableIcons(self):
        def isAnIcon(name):
            if silx.resources.is_dir("gui/icons/" + name):
                return False
            _, ext = os.path.splitext(name)
            return ext in [".svg", ".png"]
        icons = silx.resources.list_dir("gui/icons")
        # filter out sub-directories
        icons = filter(isAnIcon, icons)
        # remove extension
        icons = [i.split(".")[0] for i in icons]
        # remove duplicated names
        icons = set(icons)
        # sort by names
        return icons

    def getAllAvailableAnimatedIcons(self):
        icons = silx.resources.list_dir("gui/icons")
        icons = filter(lambda x: silx.resources.exists("gui/icons/%s/00.png" % x), icons)
        icons = filter(lambda x: not silx.resources.is_dir("gui/icons/%s/00.png" % x), icons)
        return icons

    def createIconPanel(self, parent):
        panel = qt.QWidget(parent)
        layout = qt.QGridLayout()
        # layout.setSizeConstraint(qt.QLayout.SetMinAndMaxSize)
        panel.setLayout(layout)

        self.tools = []

        # Sort together animated and non animated icons
        fix_icons = self.getAllAvailableIcons()
        animated_icons = self.getAllAvailableAnimatedIcons()
        icons = []
        icons.extend([(i, "_") for i in fix_icons])
        icons.extend([(i, "anim") for i in animated_icons])
        icons = sorted(icons)

        for i, icon_info in enumerate(icons):
            icon_name, icon_kind = icon_info
            col, line = i // 10, i % 10
            if icon_kind == "anim":
                tool = AnimatedToolButton(panel)
                try:
                    icon = silx.gui.icons.getAnimatedIcon(icon_name)
                except ValueError:
                    icon = qt.QIcon()
                tool.setToolTip("Animated icon '%s'" % icon_name)
            else:
                tool = qt.QToolButton(panel)
                try:
                    icon = silx.gui.icons.getQIcon(icon_name)
                except ValueError:
                    icon = qt.QIcon()
                tool.setToolTip("Icon '%s'" % icon_name)
            tool.setIcon(icon)
            tool.setIconSize(qt.QSize(32, 32))
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
