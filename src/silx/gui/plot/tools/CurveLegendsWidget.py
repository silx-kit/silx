# /*##########################################################################
#
# Copyright (c) 2018-2020 European Synchrotron Radiation Facility
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
"""This module provides a widget to display :class:`PlotWidget` curve legends.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "20/07/2018"


import logging
import weakref


from ... import qt
from ...widgets.FlowLayout import FlowLayout as _FlowLayout
from ..LegendSelector import LegendIcon as _LegendIcon
from .. import items


_logger = logging.getLogger(__name__)


class _LegendWidget(qt.QWidget):
    """Widget displaying curve style and its legend

    :param QWidget parent: See :class:`QWidget`
    :param ~silx.gui.plot.items.Curve curve: Associated curve
    """

    def __init__(self, parent, curve):
        super(_LegendWidget, self).__init__(parent)
        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)

        curve.sigItemChanged.connect(self._curveChanged)

        icon = _LegendIcon(curve=curve)
        layout.addWidget(icon)

        label = qt.QLabel(curve.getName())
        label.setAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
        layout.addWidget(label)

        self._update()

    def getCurve(self):
        """Returns curve associated to this widget

        :rtype: Union[~silx.gui.plot.items.Curve,None]
        """
        icon = self.findChild(_LegendIcon)
        return icon.getCurve()

    def _update(self):
        """Update widget according to current curve state.
        """
        curve = self.getCurve()
        if curve is None:
            _logger.error('Curve no more exists')
            self.setVisible(False)
            return

        self.setEnabled(curve.isVisible())

        label = self.findChild(qt.QLabel)
        if curve.isHighlighted():
            label.setStyleSheet("border: 1px solid black")
        else:
            label.setStyleSheet("")

    def _curveChanged(self, event):
        """Handle update of curve item

        :param event: Kind of change
        """
        if event in (items.ItemChangedType.VISIBLE,
                     items.ItemChangedType.HIGHLIGHTED,
                     items.ItemChangedType.HIGHLIGHTED_STYLE):
            self._update()


class CurveLegendsWidget(qt.QWidget):
    """Widget displaying curves legends in a plot

    :param QWidget parent: See :class:`QWidget`
    """

    sigCurveClicked = qt.Signal(object)
    """Signal emitted when the legend of a curve is clicked

    It provides the corresponding curve.
    """

    def __init__(self, parent=None):
        super(CurveLegendsWidget, self).__init__(parent)
        self._clicked = None
        self._legends = {}
        self._plotRef = None

    def layout(self):
        layout = super(CurveLegendsWidget, self).layout()
        if layout is None:
            # Lazy layout initialization to allow overloading
            layout = _FlowLayout()
            layout.setHorizontalSpacing(0)
            self.setLayout(layout)
        return layout

    def getPlotWidget(self):
        """Returns the associated :class:`PlotWidget`

        :rtype: Union[~silx.gui.plot.PlotWidget,None]
        """
        return None if self._plotRef is None else self._plotRef()

    def setPlotWidget(self, plot):
        """Set the associated :class:`PlotWidget`

        :param ~silx.gui.plot.PlotWidget plot: Plot widget to attach
        """
        previousPlot = self.getPlotWidget()
        if previousPlot is not None:
            previousPlot.sigItemAdded.disconnect( self._itemAdded)
            previousPlot.sigItemAboutToBeRemoved.disconnect(self._itemRemoved)
            for legend in list(self._legends.keys()):
                self._removeLegend(legend)

        self._plotRef = None if plot is None else weakref.ref(plot)

        if plot is not None:
            plot.sigItemAdded.connect(self._itemAdded)
            plot.sigItemAboutToBeRemoved.connect(self._itemRemoved)

            for legend in plot.getAllCurves(just_legend=True):
                self._addLegend(legend)

    def curveAt(self, *args):
        """Returns the curve object represented at the given position

        Either takes a QPoint or x and y as input in widget coordinates.

        :rtype: Union[~silx.gui.plot.items.Curve,None]
        """
        if len(args) == 1:
            point = args[0]
        elif len(args) == 2:
            point = qt.QPoint(*args)
        else:
            raise ValueError('Unsupported arguments')
        assert isinstance(point, qt.QPoint)

        widget = self.childAt(point)
        while widget not in (self, None):
            if isinstance(widget, _LegendWidget):
                return widget.getCurve()
            widget = widget.parent()
        return None  # No widget or not in _LegendWidget

    def _itemAdded(self, item):
        """Handle item added to the plot content"""
        if isinstance(item, items.Curve):
            self._addLegend(item.getName())

    def _itemRemoved(self, item):
        """Handle item removed from the plot content"""
        if isinstance(item, items.Curve):
            self._removeLegend(item.getName())

    def _addLegend(self, legend):
        """Add a curve to the legends

        :param str legend: Curve's legend
        """
        if legend in self._legends:
            return  # Can happen when changing curve's y axis

        plot = self.getPlotWidget()
        if plot is None:
            return None

        curve = plot.getCurve(legend)
        if curve is None:
            _logger.error('Curve not found: %s' % legend)
            return

        widget = _LegendWidget(parent=self, curve=curve)
        self.layout().addWidget(widget)
        self._legends[legend] = widget

    def _removeLegend(self, legend):
        """Remove a curve from the legends if it exists

        :param str legend: The curve's legend
        """
        widget = self._legends.pop(legend, None)
        if widget is None:
            _logger.warning('Unknown legend: %s' % legend)
        else:
            self.layout().removeWidget(widget)
            widget.setParent(None)

    def mousePressEvent(self, event):
        if event.button() == qt.Qt.LeftButton:
            self._clicked = event.pos()

    _CLICK_THRESHOLD = 5
    """Threshold for clicks"""

    def mouseMoveEvent(self, event):
        if self._clicked is not None:
            dx = abs(self._clicked.x() - event.pos().x())
            dy = abs(self._clicked.y() - event.pos().y())
            if dx > self._CLICK_THRESHOLD or dy > self._CLICK_THRESHOLD:
                self._clicked = None  # Click is cancelled

    def mouseReleaseEvent(self, event):
        if event.button() == qt.Qt.LeftButton and self._clicked is not None:
            curve = self.curveAt(event.pos())
            if curve is not None:
                self.sigCurveClicked.emit(curve)

            self._clicked = None
