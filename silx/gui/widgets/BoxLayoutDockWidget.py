# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""A QDockWidget that update the layout direction of its widget
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "06/03/2018"


from .. import qt


class BoxLayoutDockWidget(qt.QDockWidget):
    """QDockWidget adjusting its child widget QBoxLayout direction.

    The child widget layout direction is set according to dock widget area.
    The child widget MUST use a QBoxLayout

    :param parent: See :class:`QDockWidget`
    :param flags: See :class:`QDockWidget`
    """

    def __init__(self, parent=None, flags=qt.Qt.Widget):
        super(BoxLayoutDockWidget, self).__init__(parent, flags)
        self._currentArea = qt.Qt.NoDockWidgetArea
        self.dockLocationChanged.connect(self._dockLocationChanged)
        self.topLevelChanged.connect(self._topLevelChanged)

    def setWidget(self, widget):
        """Set the widget of this QDockWidget

        See :meth:`QDockWidget.setWidget`
        """
        super(BoxLayoutDockWidget, self).setWidget(widget)
        # Update widget's layout direction
        self._dockLocationChanged(self._currentArea)

    def _dockLocationChanged(self, area):
        self._currentArea = area

        widget = self.widget()
        if widget is not None:
            layout = widget.layout()
            if isinstance(layout, qt.QBoxLayout):
                if area in (qt.Qt.LeftDockWidgetArea, qt.Qt.RightDockWidgetArea):
                    direction = qt.QBoxLayout.TopToBottom
                else:
                    direction = qt.QBoxLayout.LeftToRight
                layout.setDirection(direction)
                self.resize(widget.minimumSize())
                self.adjustSize()

    def _topLevelChanged(self, topLevel):
        widget = self.widget()
        if widget is not None and topLevel:
            layout = widget.layout()
            if isinstance(layout, qt.QBoxLayout):
                layout.setDirection(qt.QBoxLayout.LeftToRight)
                self.resize(widget.minimumSize())
                self.adjustSize()

    def showEvent(self, event):
        """Make sure this dock widget is raised when it is shown.

        This is useful for tabbed dock widgets.
        """
        self.raise_()
