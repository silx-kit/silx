
.. currentmodule:: silx.gui.plot.PlotWidget

.. _plot_signal:

Plot signal
-----------

The :class:`PlotWidget` sends events through its :attr:`PlotWidget.sigPlotSignal`
signal.
Those events are sent as a dictionary with a key 'event' describing the kind
of event.

.. note::

    These dictionary events will be replaced by objects in the future.


Drawing events
..............

'drawingProgress' and 'drawingFinished' events are sent during drawing
interaction (See :meth:`PlotWidget.setInteractiveMode`).

- 'event': 'drawingProgress' or 'drawingFinished'
- 'parameters': dict of parameters used by the drawing mode.
                It has the following keys: 'shape', 'label', 'color'.
                See :meth:`PlotWidget.setInteractiveMode`.
- 'points': Points (x, y) in data coordinates of the drawn shape.
            For 'hline' and 'vline', it is the 2 points defining the line.
            For 'line' and 'rectangle', it is the coordinates of the start
            drawing point and the latest drawing point.
            For 'polygon', it is the coordinates of all points of the shape.
- 'type': The type of drawing in 'line', 'hline', 'polygon', 'rectangle',
          'vline'.
- 'xdata' and 'ydata': X coords and Y coords of shape points in data
                       coordinates (as in 'points').

When the type is 'rectangle', the following additional keys are provided:

- 'x' and 'y': The origin of the rectangle in data coordinates
- 'widht' and 'height': The size of the rectangle in data coordinates


Mouse events
............

'mouseMoved', 'mouseClicked' and 'mouseDoubleClicked' events are sent for
mouse events.

They provide the following keys:

- 'event': 'mouseMoved', 'mouseClicked' or 'mouseDoubleClicked'
- 'button': the mouse button that was pressed in 'left', 'middle', 'right'
- 'x' and 'y': The mouse position in data coordinates
- 'xpixel' and 'ypixel': The mouse position in pixels


Marker events
.............

'hover', 'markerClicked', 'markerMoving' and 'markerMoved' events are
sent during interaction with markers.

'hover' is sent when the mouse cursor is over a marker.
'markerClicker' is sent when the user click on a selectable marker.
'markerMoving' and 'markerMoved' are sent when a draggable marker is moved.

They provide the following keys:

- 'event': 'hover', 'markerClicked', 'markerMoving' or 'markerMoved'
- 'button': the mouse button that is pressed in 'left', 'middle', 'right'
- 'draggable': True if the marker is draggable, False otherwise
- 'label': The legend associated with the clicked image or curve
- 'selectable': True if the marker is selectable, False otherwise
- 'type': 'marker'
- 'x' and 'y': The mouse position in data coordinates
- 'xdata' and 'ydata': The marker position in data coordinates

'markerClicked' and 'markerMoving' events have a 'xpixel' and a 'ypixel'
additional keys, that provide the mouse position in pixels.


Image and curve events
......................

'curveClicked' and 'imageClicked' events are sent when a selectable curve
or image is clicked.

Both share the following keys:

- 'event': 'curveClicked' or 'imageClicked'
- 'button': the mouse button that was pressed in 'left', 'middle', 'right'
- 'label': The legend associated with the clicked image or curve
- 'type': The type of item in 'curve', 'image'
- 'x' and 'y': The clicked position in data coordinates
- 'xpixel' and 'ypixel': The clicked position in pixels

'curveClicked' events have a 'xdata' and a 'ydata' additional keys, that
provide the coordinates of the picked points of the curve.
There can be more than one point of the curve being picked, and if a line of
the curve is picked, only the first point of the line is included in the list.

'imageClicked' have a 'col' and a 'row' additional keys, that provide
the column and row index in the image array that was clicked.


Limits changed events
.....................

.. warning::

    This event is deprecated. Use :attr:`silx.gui.plot.items.axis.Axis.sigLimitsChanged`
    instead. See :meth:`PlotWidget.getXAxis` and :meth:`PlotWidget.getYAxis`.

'limitsChanged' events are sent when the limits of the plot are changed.
This can results from user interaction or API calls.

It provides the following keys:

- 'event': 'limitsChanged'
- 'source': id of the widget that emitted this event.
- 'xdata': Range of X in graph coordinates: (xMin, xMax).
- 'ydata': Range of Y in graph coordinates: (yMin, yMax).
- 'y2data': Range of right axis in graph coordinates (y2Min, y2Max) or None.

Plot state change events
........................

.. warning::

    These events are deprecated.  
    Use :attr:`PlotWidget.sigSetKeepDataAspectRatio`,
    :attr:`PlotWidget.sigSetGraphGrid`, :attr:`PlotWidget.sigSetGraphCursor`,
    :attr:`PlotWidget.sigItemAdded`,:attr:`PlotWidget.sigItemAboutToBeRemoved`,
    :attr:`PlotWidget.sigContentChanged`, :attr:`PlotWidget.sigActiveCurveChanged`,
    :attr:`PlotWidget.sigActiveImageChanged` and
    :attr:`PlotWidget.sigInteractiveModeChanged` instead.


The following events are emitted when the plot is modified.
They provide the new state:

- 'setGraphCursor' event with a 'state' key (bool)
- 'setGraphGrid' event with a 'which' key (str), see :meth:`setGraphGrid`
- 'setKeepDataAspectRatio' event with a 'state' key (bool)

A 'contentChanged' event is triggered when the content of the plot is updated.
It provides the following keys:

- 'action': The change of the plot: 'add' or 'remove'
- 'kind': The kind of primitive changed: 'curve', 'image', 'item' or 'marker'
- 'legend': The legend of the primitive changed.

'activeCurveChanged' and 'activeImageChanged' events with the following keys:

- 'legend': Name (str) of the current active item or None if no active item.
- 'previous': Name (str) of the previous active item or None if no item was
              active. It is the same as 'legend' if 'updated' == True
- 'updated': (bool) True if active item name did not changed,
             but active item data or style was updated.

'interactiveModeChanged' event with a 'source' key identifying the object
setting the interactive mode.

'defaultColormapChanged' event is triggered when the default colormap of
the plot is updated.
