
.. currentmodule:: silx.gui.plot

:mod:`PlotWidget`: Base class for plotting widgets
==================================================

.. module:: silx.gui.plot.PlotWidget

.. currentmodule:: silx.gui.plot.PlotWidget

The :class:`PlotWidget` is a Qt widget providing the plot API initially
provided in `PyMca <http://pymca.sourceforge.net/>`_.
It is the basis of other plot widget, thus all plot widgets share the same API.

For an introduction and examples of the plot API, see :doc:`getting_started`.

:class:`PlotWidget`
-------------------

.. currentmodule:: silx.gui.plot.PlotWidget

.. autoclass:: PlotWidget
   :show-inheritance:

Plot data
.........

Those methods allow to add and update plotted data:

.. automethod:: PlotWidget.addCurve
.. automethod:: PlotWidget.addImage
.. automethod:: PlotWidget.addScatter
.. automethod:: PlotWidget.addHistogram

Get data
........

Those methods return objects providing access to plotted data:

.. automethod:: PlotWidget.getItems

.. automethod:: PlotWidget.getCurve
.. automethod:: PlotWidget.getImage
.. automethod:: PlotWidget.getScatter
.. automethod:: PlotWidget.getHistogram

Plot markers
............

It is also possible to add point or line markers to the plot:

.. automethod:: PlotWidget.addMarker
.. automethod:: PlotWidget.addXMarker
.. automethod:: PlotWidget.addYMarker

Remove data from the plot
.........................

.. automethod:: PlotWidget.clear
.. automethod:: PlotWidget.remove

Title
.....

Those methods handle the plot title:

.. automethod:: PlotWidget.getGraphTitle
.. automethod:: PlotWidget.setGraphTitle

Axes
....

Those two methods give access to :class:`.items.Axis` which handle the limits, scales and labels of axis:

.. automethod:: PlotWidget.getXAxis
.. automethod:: PlotWidget.getYAxis

The following methods handle plot limits, aspect ratio, grid and axes display:

.. automethod:: PlotWidget.setLimits
.. automethod:: PlotWidget.isKeepDataAspectRatio
.. automethod:: PlotWidget.setKeepDataAspectRatio
.. automethod:: PlotWidget.getGraphGrid
.. automethod:: PlotWidget.setGraphGrid
.. automethod:: PlotWidget.setAxesDisplayed

Reset zoom
..........

.. automethod:: PlotWidget.resetZoom

The following methods allow to add margins around the data when performing a zoom reset:

.. automethod:: PlotWidget.getDataMargins
.. automethod:: PlotWidget.setDataMargins

Defaults
........

Those methods set-up default values for :meth:`PlotWidget.addCurve` and
:meth:`PlotWidget.addImage`:

.. automethod:: PlotWidget.getDefaultColormap
.. automethod:: PlotWidget.setDefaultColormap
.. automethod:: PlotWidget.getSupportedColormaps
.. automethod:: PlotWidget.setDefaultPlotPoints
.. automethod:: PlotWidget.setDefaultPlotLines

Interaction
...........

Those methods allow to change the interaction mode (e.g., drawing mode)
of the plot and to toggle the use of a crosshair cursor:

.. automethod:: PlotWidget.getInteractiveMode
.. automethod:: PlotWidget.setInteractiveMode

.. automethod:: PlotWidget.getGraphCursor
.. automethod:: PlotWidget.setGraphCursor

.. automethod:: PlotWidget.isPanWithArrowKeys
.. automethod:: PlotWidget.setPanWithArrowKeys

Coordinates conversion
......................

.. automethod:: PlotWidget.getDataRange
.. automethod:: PlotWidget.getPlotBoundsInPixels
.. automethod:: PlotWidget.dataToPixel
.. automethod:: PlotWidget.pixelToData

Active Item
...........

.. automethod:: PlotWidget.setActiveCurveSelectionMode
.. automethod:: PlotWidget.getActiveCurveSelectionMode
.. automethod:: PlotWidget.getActiveCurveStyle
.. automethod:: PlotWidget.setActiveCurveStyle
.. automethod:: PlotWidget.getActiveCurve
.. automethod:: PlotWidget.setActiveCurve
.. automethod:: PlotWidget.getActiveImage
.. automethod:: PlotWidget.setActiveImage

Misc.
.....

.. automethod:: PlotWidget.getWidgetHandle
.. automethod:: PlotWidget.saveGraph

Signals
.......

The :class:`PlotWidget` provides the following Qt signals:

.. autoattribute:: PlotWidget.sigPlotSignal
.. autoattribute:: PlotWidget.sigSetKeepDataAspectRatio
.. autoattribute:: PlotWidget.sigSetGraphGrid
.. autoattribute:: PlotWidget.sigSetGraphCursor
.. autoattribute:: PlotWidget.sigSetPanWithArrowKeys
.. autoattribute:: PlotWidget.sigItemAdded
.. autoattribute:: PlotWidget.sigItemAboutToBeRemoved
.. autoattribute:: PlotWidget.sigContentChanged
.. autoattribute:: PlotWidget.sigActiveCurveChanged
.. autoattribute:: PlotWidget.sigActiveImageChanged
.. autoattribute:: PlotWidget.sigActiveScatterChanged
.. autoattribute:: PlotWidget.sigInteractiveModeChanged

.. toctree::
   :hidden:

   plotsignal.rst

.. PlotWidget public API that is not documented:
   Could be added:
   - addItem
   - pan
   - getLimitsHistory
   - isDefaultPlotPoints
   - isDefaultPlotLines
   - setGraphCursorShape
   - getAutoReplot, setAutoReplot, replot
   Should not be added:
   * Should be private:
     - notify, setCallback, graphCallback
   * Use remove instead:
     - removeCurve, removeImage, removeItem, removeMarker
     - clearCurves, clearImages, clearItems, clearMarkers
   * Use items instead:
     - isCurveHidden, hideCurve
   * Use items.axis instead:
     - getGraphXLimits, setGraphXLimits
     - getGraphYLimits, setGraphYLimits
     - getGraphXLabel, setGraphXLabel
     - getGraphYLabel, setGraphYLabel
     - isXAxisLogarithmic, setXAxisLogarithmic
     - isYAxisLogarithmic, setXAxisLogarithmic
     - isXAxisAutoScale, setXAxisAutoScale
     - isYAxisAutoScale, setYAxisAutoScale
     - setYAxisInverted, isYAxisInverted
