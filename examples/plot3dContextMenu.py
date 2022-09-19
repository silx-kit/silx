# /*##########################################################################
#
# Copyright (c) 2017-2021 European Synchrotron Radiation Facility
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
This script adds a context menu to a :class:`silx.gui.plot3d.ScalarFieldView`.

This is done by adding a custom context menu to the :class:`Plot3DWidget`:

- set the context menu policy to Qt.CustomContextMenu.
- connect to the customContextMenuRequested signal.

For more information on context menus, see Qt documentation.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/10/2017"


import logging

import numpy

from silx.gui import qt

from silx.gui.plot3d.ScalarFieldView import ScalarFieldView
from silx.gui.plot3d import actions

logging.basicConfig()

_logger = logging.getLogger(__name__)


class ScalarFieldViewWithContextMenu(ScalarFieldView):
    """Subclass ScalarFieldView to add a custom context menu to its 3D area."""

    def __init__(self, parent=None):
        super(ScalarFieldViewWithContextMenu, self).__init__(parent)
        self.setWindowTitle("Right-click to open the context menu")

        # Set Plot3DWidget custom context menu
        self.getPlot3DWidget().setContextMenuPolicy(qt.Qt.CustomContextMenu)
        self.getPlot3DWidget().customContextMenuRequested.connect(
            self._contextMenu)

    def _contextMenu(self, pos):
        """Handle plot area customContextMenuRequested signal.

        :param QPoint pos: Mouse position relative to plot area
        """
        # Create the context menu
        menu = qt.QMenu(self)
        menu.addAction(actions.mode.PanAction(
            parent=menu, plot3d=self.getPlot3DWidget()))
        menu.addAction(actions.mode.RotateArcballAction(
            parent=menu, plot3d=self.getPlot3DWidget()))
        menu.addSeparator()
        menu.addAction(actions.io.CopyAction(
            parent=menu, plot3d=self.getPlot3DWidget()))

        # Displaying the context menu at the mouse position requires
        # a global position.
        # The position received as argument is relative to Plot3DWidget
        # and needs to be converted.
        globalPosition = self.getPlot3DWidget().mapToGlobal(pos)
        menu.exec(globalPosition)


# Start Qt QApplication
app = qt.QApplication([])

# Create the viewer main window
window = ScalarFieldViewWithContextMenu()

# Create dummy data
coords = numpy.linspace(-10, 10, 64)
z = coords.reshape(-1, 1, 1)
y = coords.reshape(1, -1, 1)
x = coords.reshape(1, 1, -1)
data = numpy.sin(x * y * z) / (x * y * z)

# Set ScalarFieldView data
window.setData(data)

# Add an iso-surface
window.addIsosurface(0.2, '#FF0000FF')

window.show()
app.exec()
