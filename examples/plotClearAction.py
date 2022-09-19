# /*##########################################################################
#
# Copyright (c) 2018-2021 European Synchrotron Radiation Facility
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
"""This script shows how to create a minimalistic
:class:`~silx.gui.plot.actions.PlotAction` that clear the plot.

This illustrates how to add more buttons in a plot widget toolbar.
"""

__authors__ = ["T. VINCENT"]
__license__ = "MIT"
__date__ = "14/02/2018"


from silx.gui.plot.actions import PlotAction


class ClearPlotAction(PlotAction):
    """A QAction that can be added to PlotWidget toolbar to clear the plot"""

    def __init__(self, plot, parent=None):
        super(ClearPlotAction, self).__init__(
            plot,
            icon='close',
            text='Clear',
            tooltip='Clear the plot',
            triggered=self._clear,
            parent=parent)

    def _clear(self):
        """Handle action triggered and clear the plot"""
        self.plot.clear()


if __name__ == '__main__':
    from silx.gui import qt
    from silx.gui.plot import Plot1D

    app = qt.QApplication([])  # First create QApplication

    plot = Plot1D()  # Create plotting widget

    # Create a toolbar and add it to the plot widget
    toolbar = qt.QToolBar()
    plot.addToolBar(toolbar)

    # Create clear action and add it to the toolbar
    action = ClearPlotAction(plot, parent=plot)
    toolbar.addAction(action)

    plot.addCurve((0, 1, 2, 3, 4), (0, 1, 1.5, 1, 0))  # Add a curve to the plot

    plot.show()  # Show the plot widget
    app.exec()  # Start Qt application
