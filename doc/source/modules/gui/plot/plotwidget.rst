
.. currentmodule:: silx.gui.plot

:mod:`PlotWidget`
=================

.. module:: silx.gui.plot.PlotWidget

.. currentmodule:: silx.gui.plot.PlotWidget

The :class:`PlotWidget` is a Qt widget providing the plot API initially
provided in `PyMca <http://pymca.sourceforge.net/>`_.

Examples
--------

As this widget is a Qt widget, a Qt application must be running in order to
use this widget.
The following sample code be included in a script that already created a
Qt application or from IPython.

To use this widget from `IPython <http://ipython.org/>`_,
IPython (and matplotlib) need to integrate with Qt.
You need to either start IPython with the following option
``ipython --pylab=qt`` or use the ``%pylab qt`` magic from IPython prompt.

.. code-block:: python

   %pylab qt

Basics
......

Displaying a curve:

.. code-block:: python

   from silx.gui.plot import PlotWidget

   plot = PlotWidget()  # Create the plot widget
   plot.addCurve(x=(1, 2, 3), y=(3, 2, 1))  # Add a curve with default style
   plot.show()  # Make the PlotWidget visible

Updating a curve:

.. code-block:: python

   plot.addCurve(x=(1, 2, 3), y=(1, 2, 3))  # Replace the existing curve

Displaying an image:

.. code-block:: python

   import numpy
   from silx.gui.plot import PlotWidget

   data = numpy.random.random(512 * 512).reshape(512, -1)  # Create 2D data

   plot = PlotWidget()  # Create the plot widget
   plot.addImage(data)  # Add a 2D data set with default colormap
   plot.show()  # Make the PlotWidget visible

API
---

.. currentmodule:: silx.gui.plot.Plot

This is a choosen subset of the complete plot API, the full API is
documented in :class:`silx.gui.plot.Plot`.

.. currentmodule:: silx.gui.plot.PlotWidget

.. autoclass:: PlotWidget
   :show-inheritance:
   :members: resetZoom, replot, remove, clear, saveGraph, sigPlotSignal

Curves
......

.. automethod:: PlotWidget.addCurve
.. automethod:: PlotWidget.removeCurve
.. automethod:: PlotWidget.clearCurves

Images
......

.. automethod:: PlotWidget.addImage
.. automethod:: PlotWidget.removeImage
.. automethod:: PlotWidget.clearImages

Markers
.......

.. automethod:: PlotWidget.addMarker
.. automethod:: PlotWidget.addXMarker
.. automethod:: PlotWidget.addYMarker
.. automethod:: PlotWidget.removeMarker
.. automethod:: PlotWidget.clearMarkers

Title and labels
................

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

Reset zoom settings
...................

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
:meth:`PlotWidget.addImage`.

.. automethod:: PlotWidget.getDefaultColormap
.. automethod:: PlotWidget.setDefaultColormap
.. automethod:: PlotWidget.getSupportedColormaps
.. automethod:: PlotWidget.setDefaultPlotPoints
.. automethod:: PlotWidget.setDefaultPlotLines

Interaction
...........

Those methods allow to change the interaction mode (e.g., drawing mode)
of the plot and to toggle the use of a crosshair cursor.

.. automethod:: PlotWidget.getInteractiveMode
.. automethod:: PlotWidget.setInteractiveMode

.. automethod:: PlotWidget.getGraphCursor
.. automethod:: PlotWidget.setGraphCursor


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
