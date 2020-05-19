# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
"""Action to hold many mode actions, usually for a tool bar.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__data__ = "22/04/2020"


from silx.gui import qt


class MultiModeAction(qt.QWidgetAction):
    """This action provides a default checkable action from a list of checkable
    actions.

    The default action can be selected from a drop down list. The last one used
    became the default one.

    The default action is directly usable without using the drop down list.
    """

    def __init__(self, parent=None):
        assert isinstance(parent, qt.QWidget)
        qt.QWidgetAction.__init__(self, parent)
        button = qt.QToolButton(parent)
        button.setPopupMode(qt.QToolButton.MenuButtonPopup)
        self.setDefaultWidget(button)
        self.__button = button

    def getMenu(self):
        """Returns the menu.

        :rtype: qt.QMenu
        """
        button = self.__button
        menu = button.menu()
        if menu is None:
            menu = qt.QMenu(button)
            button.setMenu(menu)
        return menu

    def addAction(self, action):
        """Add a new action to the list.

        :param qt.QAction action: New action
        """
        menu = self.getMenu()
        button = self.__button
        menu.addAction(action)
        if button.defaultAction() is None:
            button.setDefaultAction(action)
        if action.isCheckable():
            action.toggled.connect(self._toggled)

    def _toggled(self, checked):
        if checked:
            action = self.sender()
            button = self.__button
            button.setDefaultAction(action)
