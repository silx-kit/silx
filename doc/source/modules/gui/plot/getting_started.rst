
.. role:: python(code)
    :language: python

.. currentmodule:: silx.gui

Getting started with plot widgets
=================================

This introduction to :mod:`silx.gui.plot` covers the following topics:

- `Use silx.gui.plot from (I)Python console`_
- `Use silx.gui.plot from a script`_
- `Plot curves in a widget`_
- `Plot images in a widget`_
- `Configure plot axes`_

For a complete description of the API, see :mod:`silx.gui.plot`.

Use :mod:`silx.gui.plot` from (I)Python console
-----------------------------------------------

We recommend to use (I)Python 3.x and PyQt5.

From a Python or IPython interpreter, the simplest way is to import the :mod:`silx.sx` module:

>>> from silx import sx

The :mod:`silx.sx` module initialises Qt and provides access to :mod:`silx.gui.plot` widgets and extra plot functions.

.. note:: The :mod:`silx.sx` module does NOT initialise Qt and does NOT expose silx widget in a notebook.

An alternative to run :mod:`silx.gui` widgets from `IPython <http://ipython.org/>`_,
is to set IPython to use Qt(5), e.g., with the `--gui` option::

  ipython --gui=qt5


Compatibility with IPython
++++++++++++++++++++++++++

silx widgets require Qt to be initialized.
If Qt is not yet loaded, silx tries to load PyQt5 first before trying other supported bindings.

With versions of IPython lower than 3.0 (e.g., on Debian 8), there is an incompatibility between
the way silx loads Qt and the way IPython is doing it through the ``--gui`` option,
`%gui <http://ipython.org/ipython-doc/stable/interactive/magics.html#magic-gui>`_ or
`%pylab <http://ipython.org/ipython-doc/stable/interactive/magics.html#magic-pylab>`_ magics.
In this case, IPython magics that initialize Qt might not work after importing modules from silx.gui.

When using Python2.7 and PyQt4, there is another incompatibility to deal with as
silx requires PyQt4 API version 2 (See note below for explanation).
In this case, start IPython with the ``QT_API`` environment variable set to ``pyqt``.

On Linux and MacOS X, run from the command line::

  QT_API=pyqt ipython

On Windows, run from the command line::

  set QT_API=pyqt&&ipython


.. note:: PyQt4 used from Python 2.x provides 2 incompatible versions of QString and QVariant:

   - version 1, the legacy version which is also the default, and
   - version 2, a more pythonic one, which is the only one supported by *silx*.

   All other configurations (i.e., PyQt4 on Python 3.x, PySide2, PyQt5, IPython QtConsole widget) uses version 2.

   For more information, see `IPython, PyQt and PySide <http://ipython.org/ipython-doc/stable/interactive/reference.html#pyqt-and-pyside>`_.


Plot functions
++++++++++++++

The :mod:`silx.sx` module provides functions to plot curves and images with :mod:`silx.gui.plot` widgets:

- :func:`~silx.sx.plot` for curves, e.g., :python:`sx.plot(y)` or :python:`sx.plot(x, y)`
- :func:`~silx.sx.imshow` for images, e.g., :python:`sx.imshow(image)`

See :mod:`silx.sx` for documentation and how to use it.

For more features, use widgets directly (see `Plot curves in a widget`_ and `Plot images in a widget`_).


Use :mod:`silx.gui.plot` from a script
--------------------------------------

A Qt GUI script must have a QApplication initialised before creating widgets:

.. code-block:: python

   from silx.gui import qt

   [...]

   qapp = qt.QApplication([])

   [...] # Widgets initialisation

   if __name__ == '__main__':
       [...]
       qapp.exec_()

Unless a Qt binding has already been loaded, :mod:`silx.gui.qt` uses one of the supported Qt bindings (PyQt5, PyQt4, PySide2).
If you prefer to choose the Qt binding yourself, import it before importing
a module from :mod:`silx.gui`:

.. code-block:: python

   import PyQt5.QtCore  # Importing PyQt5 will force silx to use it
   from silx.gui import qt


Plot curves in a widget
-----------------------

The :class:`~silx.gui.plot.PlotWindow.Plot1D` widget provides a plotting area and a toolbar with tools useful for curves such as setting a logarithmic scale or defining a region of interest.

First, create a :class:`~silx.gui.plot.PlotWindow.Plot1D` widget:

.. code-block:: python

   from silx.gui.plot import Plot1D

   plot = Plot1D()  # Create the plot widget
   plot.show()  # Make the plot widget visible


One curve
+++++++++

To display a single curve, use the :meth:`.PlotWidget.addCurve` method:

.. code-block:: python

   plot.addCurve(x=(1, 2, 3), y=(3, 2, 1), legend='curve')  # Add a curve named 'curve'

When you need to update this curve, first get the curve invoking :meth:`.PlotWidget.getCurve` and
update its points invoking the curve's :meth:`~silx.gui.plot.items.Curve.setData` method:

.. code-block:: python

   mycurve = plot.getCurve('curve')  # Retrieve the curve
   mycurve.setData(x=(1, 2, 3), y=(1, 2, 3))  # Update its data

To clear the plot, call :meth:`.PlotWidget.clear`:

.. code-block:: python

   plot.clear()


Multiple curves
+++++++++++++++

In order to display multiple curves in a frame, you need to provide a different ``legend`` string for each of them:

.. code-block:: python

   import numpy

   x = numpy.linspace(-numpy.pi, numpy.pi, 1000)
   plot.addCurve(x, numpy.sin(x), legend='sinus')
   plot.addCurve(x, numpy.cos(x), legend='cosinus')
   plot.addCurve(x, numpy.random.random(len(x)), legend='random')


To update a curve, call :meth:`.PlotWidget.getCurve` with the ``legend`` of the curve you want to update,
and update its data through :meth:`~silx.gui.plot.items.Curve.setData`:

.. code-block:: python

   curve = plot.getCurve('random')
   curve.setData(x, numpy.random.random(len(x)) - 1.)

To remove a curve from the plot, call :meth:`.PlotWidget.remove` with the ``legend`` of the curve you want to remove:

.. code-block:: python

   plot.remove('random')

To clear the plotting area, call :meth:`.PlotWidget.clear`:

.. code-block:: python

   plot.clear()

Curve style
+++++++++++

By default, different curves will automatically be displayed using different styles, and keep the same style when updating the plot.

It is possible to specify the ``color`` of the curve, its ``linewidth`` and ``linestyle`` as well as the ``symbol`` to use as marker for data points (See :meth:`.PlotWidget.addCurve` for more details):

.. code-block:: python

   import numpy

   x = numpy.linspace(-numpy.pi, numpy.pi, 100)

   # Curve with a thick dashed line
   plot.addCurve(x, numpy.sin(x), legend='sinus',
                 linewidth=3, linestyle='--')

   # Curve with pink markers only
   plot.addCurve(x, numpy.cos(x), legend='cosinus',
                 color='pink', linestyle=' ', symbol='o')

   # Curve with green line with square markers
   plot.addCurve(x, numpy.random.random(len(x)), legend='random',
                 color='green', linestyle='-', symbol='s')



Histogram
+++++++++

To display histograms, use :meth:`.PlotWidget.addHistogram`:

.. code-block:: python

    import numpy
    values = numpy.arange(20)  # Values of the histogram
    edges = numpy.arange(21)  # Edges of the bins (number of values + 1)
    plot.addHistogram(values, edges, legend='histo1', fill=True, color='green')

Alternatively, :meth:`.PlotWidget.addCurve` can be used to display histograms with the ``histogram`` argument.
(See :meth:`.PlotWidget.addCurve` for more details).

.. code-block:: python
  
    import numpy
    x = numpy.arange(0, 20, 1)
    plot.addCurve(x, x+1, legend='histo2', histogram='center', fill=False, color='black')

Histogram bins can be centred on x values or set on the left hand side or the right hand side of the given x values.

Plot images in a widget
-----------------------

The :class:`~silx.gui.plot.PlotWindow.Plot2D` widget provides a plotting area and a toolbar with tools useful for images, such as keeping the aspect ratio, changing the colormap or defining a mask.

First, create a :class:`~silx.gui.plot.PlotWindow.Plot2D` widget:

.. code-block:: python

   from silx.gui.plot import Plot2D

   plot = Plot2D()  # Create the plot widget
   plot.show()  # Make the plot widget visible

One image
+++++++++

To display a single image, use the :meth:`.PlotWidget.addImage` method:

.. code-block:: python

   import numpy

   data = numpy.random.random(512 * 512).reshape(512, -1)  # Create 2D image
   plot.addImage(data, legend='image')  # Plot the 2D data set with default colormap

To update this image, call :meth:`.PlotWidget.getImage` with its ``legend`` and
update its data with :meth:`~silx.gui.plot.items.Image.setData`:

.. code-block:: python

   data2 = numpy.arange(512*512).reshape(512, 512)

   image = plot.getImage('image')  # Retrieve the image
   image.setData(data2)  # Update the displayed data

:meth:`.PlotWidget.addImage` supports both 2D arrays of data displayed with a colormap and RGB(A) images as 3D arrays of shape (height, width, color channels).

To clear the plot area, call :meth:`.PlotWidget.clear`:

.. code-block:: python

   plot.clear()

Origin and scale
++++++++++++++++


When displaying an image, it is possible to define the ``origin`` and the ``scale`` of the image array in the plot area coordinates:

.. code-block:: python

   data = numpy.random.random(512 * 512).reshape(512, -1)
   plot.addImage(data, legend='image', origin=(100, 100), scale=(0.1, 0.1))

Colormap
++++++++

A ``colormap`` is described with a :class:`~silx.gui.colors.Colormap` class as follows:

.. code-block:: python

   from silx.gui.colors import Colormap

   colormap = Colormap(name='gray',             # Name of the colormap
                       normalization='linear',  # Either 'linear' or 'log'
                       vmin=0.0,                # If not autoscale, data value to bind to min of colormap
                       vmax=1.0                 # If not autoscale, data value to bind to max of colormap
               )


The following colormap names are guaranteed to be available:

- gray
- reversed gray
- temperature
- red
- green
- blue
- viridis
- magma
- inferno
- plasma

Yet, any colormap name from `matplotlib <http://matplotlib.org/>`_ (see `Choosing Colormaps <http://matplotlib.org/users/colormaps.html>`_) should work.

It is possible to change the default colormap of the plot widget by :meth:`.PlotWidget.setDefaultColormap` (and to get it with :meth:`.PlotWidget.getDefaultColormap`):

.. code-block:: python

   from silx.gui.colors import Colormap

   colormap = Colormap(name='viridis',
                       normalization='linear',
                       vmin=0.0,
                       vmax=10000.0)
   plot.setDefaultColormap(colormap)

   data = numpy.arange(512 * 512.).reshape(512, -1)
   plot.addImage(data)  # Rendered with the default colormap set before

It is also possible to provide a :class:`~silx.gui.colors.Colormap` to :meth:`.PlotWidget.addImage` to override this default for an image:

.. code-block:: python

   colormap = Colormap(name='magma',
                       normalization='log',
                       vmin=1.8,
                       vmax=2.2)
   data = numpy.random.random(512 * 512).reshape(512, -1) + 1.
   plot.addImage(data, colormap=colormap)

The colormap can be changed by the user from the widget's toolbar.


Multiple images
+++++++++++++++

In order to display multiple images in a frame, you need to provide a different ``legend`` string for each of them and to set the ``replace`` argument to ``False``:

.. code-block:: python

   data = numpy.random.random(512 * 512).reshape(512, -1)
   plot.addImage(data, legend='random', replace=False)

   data = numpy.arange(512 * 512.).reshape(512, -1)
   plot.addImage(data, legend='arange', replace=False, origin=(512, 512))


To update an image, call :meth:`.PlotWidget.getImage` with the ``legend`` to get the corresponding curve.
Update its data values using :meth:`~silx.gui.plot.items.setData`.

.. code-block:: python

   data = (512 * 512. - numpy.arange(512 * 512.)).reshape(512, -1)
   arange_image = plot.getImage('arange')
   arange_image.setData(data)

To remove an image from a plot, call :meth:`.PlotWidget.remove` with the ``legend`` of the image you want to remove:

.. code-block:: python

   plot.remove('random')


Configure plot axes
-------------------

The following examples illustrate the API to configure the plot axes.
:meth:`.PlotWidget.getXAxis` and :meth:`.PlotWidget.getYAxis` give access to each plot axis (:class:`.items.Axis`) in order to configure them.

Labels and title
++++++++++++++++

Use :meth:`.PlotWidget.setGraphTitle` to set the plot main title.
Use :meth:`.PlotWidget.getXAxis` and :meth:`.PlotWidget.getYAxis` to get the axes and set their text label with :meth:`.items.Axis.setLabel`:

.. code-block:: python

   plot.setGraphTitle('My plot')
   plot.getXAxis().setLabel('X')
   plot.getYAxis().setLabel('Y')


Axes limits
+++++++++++

Different methods allow to retrieve and set the data limits displayed on each axis.

The following code moves the visible plot area to the right:

.. code-block:: python

    xmin, xmax = plot.getXAxis().getLimits()
    offset = 0.1 * (xmax - xmin)
    plot.getXAxis().setLimits(xmin + offset, xmax + offset)

:meth:`.PlotWidget.resetZoom` set the plot limits to the upper and lower bounds of the data:

.. code-block:: python

   plot.resetZoom()

See :meth:`.PlotWidget.resetZoom`, :meth:`.PlotWidget.setLimits`, :meth:`.PlotWidget.getXAxis`, :meth:`.PlotWidget.getYAxis` and :class:`.items.Axis` for details.


Axes
++++

The axes of a plot can be modified via different methods:

.. code-block:: python

   plot.getYAxis().setInverted(True)  # Makes the Y axis pointing downward
   plot.setKeepDataAspectRatio(True)  # To keep aspect ratio between X and Y axes

See :meth:`.PlotWidget.getYAxis`, :meth:`.PlotWidget.setKeepDataAspectRatio` for details.

.. code-block:: python

   plot.setGraphGrid(which='both')  # To show a grid for both minor and major axes ticks

   # Use logarithmic axes
   plot.getXAxis().setScale("log")
   plot.getYAxis().setScale("log")

See :meth:`.PlotWidget.setGraphGrid`, :meth:`.PlotWidget.getXAxis`, :meth:`.PlotWidget.getXAxis` and :class:`.items.Axis` for details.
