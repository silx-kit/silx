# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""This script is a simple (trivial) example of how to create a PlotWindow,
create a custom :class:`PlotAction` and add it to the toolbar.

The action simply shifts the selected curve up by 1 unit by adding 1 to each
value of y.
"""
from silx.gui import qt
from silx.gui.plot import PlotWindow
from silx.gui.plot.PlotActions import PlotAction

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "20/09/2016"


class ShiftUpAction(PlotAction):
    """QAction shifting up a curve by one unit

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """
    def __init__(self, plot, parent=None):
        PlotAction.__init__(self,
                            plot,
                            icon='shape-circle',
                            text='Shift up',
                            tooltip='Shift active curve up by one unit',
                            triggered=self.shiftActiveCurveUp,
                            checkable=False, parent=parent)

    def shiftActiveCurveUp(self):
        # By inheriting from PlotAction, we get access to attribute self.plot
        # which is a reference to the PlotWindow
        activeCurve = self.plot.getActiveCurve()

        if activeCurve is not None:
            # unpack curve data
            x0, y0, legend, info, params = activeCurve

            # add 1 to all values in the y array
            # and assign the result to a new array y1
            # (IMPORTANT: do not modify y0 directly)
            y1 = y0 + 1.0

            # By re-usinq the same legend, this causes the original curve
            # to be replaced
            self.plot.addCurve(x0, y1, legend=legend,
                               info=info)


app = qt.QApplication([])

# create a PlotWindow with a new toolbar
pw = PlotWindow()
toolbar = qt.QToolBar("My toolbar")
pw.addToolBar(toolbar)

# Create our action and add it to the toolbar.
# Note that we provide our PlotWindow (pw) as a parameter.
myaction = ShiftUpAction(pw)
toolbar.addAction(myaction)

# Plot a curve with synthetic data
x = [0, 1, 2, 3, 4, 5, 6]
y = [0, 1, 0, 1, 0, 1, 0]
pw.addCurve(x, y, legend="triangle shaped curve")

pw.show()
app.exec_()
