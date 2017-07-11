
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

.. automethod:: PlotWidget.getCurve
.. automethod:: PlotWidget.getImage
.. automethod:: PlotWidget.getScatter
.. automethod:: PlotWidget.getHistogram

.. automethod:: PlotWidget.getAllCurves
.. automethod:: PlotWidget.getAllImages


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

Those methods handle the plot title:

.. automethod:: PlotWidget.getGraphTitle
.. automethod:: PlotWidget.setGraphTitle

Axes
....

Those two methods give access to :class:`.items.Axis` which handle the limits, scales and labels of axis:

.. automethod:: PlotWidget.getXAxis
.. automethod:: PlotWidget.getYAxis

The following methods handle plot limits, aspect ratio and grid:

.. automethod:: PlotWidget.setLimits
.. automethod:: PlotWidget.isKeepDataAspectRatio
.. automethod:: PlotWidget.setKeepDataAspectRatio
.. automethod:: PlotWidget.getGraphGrid
.. automethod:: PlotWidget.setGraphGrid

Reset zoom
..........

.. automethod:: PlotWidget.resetZoom

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
   setDefaultPlotPoints, setDefaultPlotLines,
   getWidgetHandle, notify, setCallback, graphCallback,
   dataToPixel, pixelToData, getPlotBoundsInPixels,
   setGraphCursorShape, pickMarker, moveMarker, pickImageOrCurve, moveImage,
   onMousePress, onMouseMove, onMouseRelease, onMouseWheel,
   isDrawModeEnabled, setDrawModeEnabled, getDrawMode,
   isZoomModeEnabled, setZoomModeEnabled,
