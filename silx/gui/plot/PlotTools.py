# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Set of widgets to associate with a :class:'PlotWidget'.
"""

__authors__ = ["V.A. Sole", "T. Vincent"]
__license__ = "MIT"
__date__ = "28/04/2016"

import logging
import numbers
import traceback

from .. import qt


_logger = logging.getLogger(__name__)


class PositionInfo(qt.QWidget):
    """QWidget displaying coords converted from data coords of the mouse.

    Provide this widget with a list of couple:

    - A name to display before the data
    - A function that takes (x, y) as arguments and returns something that
      gets converted to a string.
      If the result is a float it is converted with '%.7g' format.

    To run the following sample code, a QApplication must be initialized.
    First, create a PlotWindow and add a QToolBar where to place the
    PositionInfo widget.

    >>> from silx.gui.plot import PlotWindow
    >>> from silx.gui import qt

    >>> plot = PlotWindow()  # Create a PlotWindow to add the widget to
    >>> toolBar = qt.QToolBar()  # Create a toolbar to place the widget in
    >>> plot.addToolBar(qt.Qt.BottomToolBarArea, toolBar)  # Add it to plot

    Then, create the PositionInfo widget and add it to the toolbar.
    The PositionInfo widget is created with a list of converters, here
    to display polar coordinates of the mouse position.

    >>> import math
    >>> from silx.gui.plot.PlotTools import PositionInfo

    >>> position = PositionInfo(plot, converters=[
    ...     ('Radius', lambda x, y: math.sqrt(x*x + y*y)),
    ...     ('Angle', lambda x, y: math.degrees(math.atan2(y, x)))])

    >>> toolBar.addWidget(position)  # Add the widget to the toolbar
    <...>

    >>> plot.show()  # To display the PlotWindow with the position widget

    :param plot: The PlotWidget this widget is displaying data coords from.
    :param converters: List of name to display and conversion function from
                       (x, y) in data coords to displayed value.
                       If None, the default, it displays X and Y.
    :type converters: Iterable of 2-tuple (str, function)
    :param parent: Parent widget
    """

    def __init__(self, plot, converters=None, parent=None):
        super(PositionInfo, self).__init__(parent)

        if converters is None:
            converters = (('X', lambda x, y: x), ('Y', lambda x, y: y))

        self._fields = []  # To store (QLineEdit, name, function (x, y)->v)

        # Create a new layout with new widgets
        layout = qt.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # layout.setSpacing(0)

        # Create all LineEdit and store them with the corresponding converter
        for name, func in converters:
            layout.addWidget(qt.QLabel('<b>' + name + ':</b>'))

            lineEdit = qt.QLineEdit()
            lineEdit.setText('------')
            lineEdit.setReadOnly(1)
            lineEdit.setFixedWidth(
                lineEdit.fontMetrics().width('##############'))
            layout.addWidget(lineEdit)
            self._fields.append((lineEdit, name, func))

        layout.addStretch(1)
        self.setLayout(layout)

        # Connect to Plot events
        plot.sigPlotSignal.connect(self._plotEvent)

    def getConverters(self):
        """Return the list of converters as 2-tuple (name, function)."""
        return [(name, func) for lineEdit, name, func in self._fields]

    def _plotEvent(self, event):
        """Handle events from the Plot.

        :param dict event: Plot event
        """
        if event['event'] == 'mouseMoved':
            x, y = event['x'], event['y']

            for lineEdit, name, func in self._fields:
                try:
                    value = func(x, y)
                except:
                    lineEdit.setText('Error')
                    _logger.error(
                        "Error while converting coordinates (%f, %f)"
                        "with converter '%s'" % (x, y, name))
                    _logger.error(traceback.format_exc())
                else:
                    if isinstance(value, numbers.Real):
                        value = '%.7g' % value  # Use this for floats and int
                    else:
                        value = str(value)  # Fallback for other types
                    lineEdit.setText(value)
