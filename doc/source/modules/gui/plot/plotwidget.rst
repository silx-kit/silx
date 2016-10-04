
.. currentmodule:: silx.gui.plot

:mod:`PlotWidget`: Base class for plotting widgets
==================================================

.. module:: silx.gui.plot.PlotWidget

.. currentmodule:: silx.gui.plot.PlotWidget

The :class:`PlotWidget` is a Qt widget providing the plot API initially
provided in `PyMca <http://pymca.sourceforge.net/>`_.
It is the basis of other plot widget, thus all plot widgets share the same API.

For an introduction and examples of the plot API, see :doc:`getting_started`.

Plot API
--------

.. currentmodule:: silx.gui.plot.Plot

This is a choosen subset of the complete plot API, the full API is
documented in :class:`silx.gui.plot.Plot`.

.. currentmodule:: silx.gui.plot.PlotWidget

.. autoclass:: PlotWidget
   :show-inheritance:

Plot data
.........

Those methods allow to add and update plotted data:

.. automethod:: PlotWidget.addCurve
.. automethod:: PlotWidget.addImage

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

Title and labels
................

Those methods handle the text labels of the axes and the plot title:

.. automethod:: PlotWidget.getGraphTitle
.. automethod:: PlotWidget.setGraphTitle
.. automethod:: PlotWidget.getGraphXLabel
.. automethod:: PlotWidget.setGraphXLabel
.. automethod:: PlotWidget.getGraphYLabel
.. automethod:: PlotWidget.setGraphYLabel

Axes limits
...........

Those methods change the range of data values displayed on each axis.

.. automethod:: PlotWidget.getGraphXLimits
.. automethod:: PlotWidget.setGraphXLimits
.. automethod:: PlotWidget.getGraphYLimits
.. automethod:: PlotWidget.setGraphYLimits
.. automethod:: PlotWidget.setLimits

Axes
....

The following methods handle the display properties of the axes:

.. automethod:: PlotWidget.isXAxisLogarithmic
.. automethod:: PlotWidget.setXAxisLogarithmic
.. automethod:: PlotWidget.isYAxisLogarithmic
.. automethod:: PlotWidget.setYAxisLogarithmic

.. automethod:: PlotWidget.isYAxisInverted
.. automethod:: PlotWidget.setYAxisInverted
.. automethod:: PlotWidget.isKeepDataAspectRatio
.. automethod:: PlotWidget.setKeepDataAspectRatio
.. automethod:: PlotWidget.getGraphGrid
.. automethod:: PlotWidget.setGraphGrid

Reset zoom
..........

.. automethod:: PlotWidget.resetZoom

Those methods change the behavior of :meth:`PlotWidget.resetZoom`.

.. automethod:: PlotWidget.getDataMargins
.. automethod:: PlotWidget.setDataMargins
.. automethod:: PlotWidget.isXAxisAutoScale
.. automethod:: PlotWidget.setXAxisAutoScale
.. automethod:: PlotWidget.isYAxisAutoScale
.. automethod:: PlotWidget.setYAxisAutoScale

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

Misc.
.....

.. automethod:: PlotWidget.saveGraph

Signals
.......

The :class:`PlotWidget` provides the following Qt signals:

.. autoattribute:: PlotWidget.sigPlotSignal
.. autoattribute:: PlotWidget.sigSetYAxisInverted
.. autoattribute:: PlotWidget.sigSetXAxisLogarithmic
.. autoattribute:: PlotWidget.sigSetYAxisLogarithmic
.. autoattribute:: PlotWidget.sigSetXAxisAutoScale
.. autoattribute:: PlotWidget.sigSetYAxisAutoScale
.. autoattribute:: PlotWidget.sigSetKeepDataAspectRatio
.. autoattribute:: PlotWidget.sigSetGraphGrid
.. autoattribute:: PlotWidget.sigSetGraphCursor
.. autoattribute:: PlotWidget.sigSetPanWithArrowKeys
.. autoattribute:: PlotWidget.sigContentChanged
.. autoattribute:: PlotWidget.sigActiveCurveChanged
.. autoattribute:: PlotWidget.sigActiveImageChanged
.. autoattribute:: PlotWidget.sigInteractiveModeChanged

.. Not documented:
   addItem, removeItem, clearItems
   isActiveCurveHandling, enableActiveCurveHandling,
   getActiveCurveColor, setActiveCurveColor,
   getActiveCurve, setActiveCurve,
   isCurveHidden, hideCurve,
   getActiveImage, setActiveImage,
   getAllCurves, getCurve, getMonotonicCurves, getImage,
   setDefaultPlotPoints, setDefaultPlotLines,
   getWidgetHandle, notify, setCallback, graphCallback,
   dataToPixel, pixelToData, getPlotBoundsInPixels,
   setGraphCursorShape, pickMarker, moveMarker, pickImageOrCurve, moveImage,
   onMousePress, onMouseMove, onMouseRelease, onMouseWheel,
   isDrawModeEnabled, setDrawModeEnabled, getDrawMode,
   isZoomModeEnabled, setZoomModeEnabled,
