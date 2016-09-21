.. currentmodule:: silx.gui

Getting started with plot widgets
=================================

This introduction to :mod:`silx.gui.plot` covers the following topics:

- `Use silx.gui.plot from the console`_
- `Use silx.gui.plot from a script`_
- `Plot curves in a widget`_
- `Plot images in a widget`_
- `Control plot axes`_
- `Add plot actions`_

For a complete description of the API, see :mod:`silx.gui.plot`.

Use :mod:`silx.gui.plot` from the console
-----------------------------------------

From IPython
++++++++++++

To run :mod:`silx.gui.plot` widgets from `IPython <http://ipython.org/>`_, IPython must be set to use Qt (and in case of using PyQt4 and Python 2.7, PyQt must be set ti use API version 2, see Explanation_ below).

As *silx* is performing some configuration of the Qt binding and `matplotlib <http://matplotlib.org/>`_, the safest way to use *silx* from IPython is to import :mod:`silx.gui.plot` first and then run either `%gui <http://ipython.org/ipython-doc/stable/interactive/magics.html#magic-gui>`_ qt  or `%pylab <http://ipython.org/ipython-doc/stable/interactive/magics.html#magic-pylab>`_ qt::

  In [1]: from silx.gui.plot import *
  In [2]: %pylab qt

Alternatively, when using Python 2.7 and PyQt4, you can start IPython with the ``QT_API`` environment variable set to ``pyqt``.

On Linux and MacOS X, run::

  QT_API=pyqt ipython

On Windows, run from the command line::

  set QT_API=pyqt&&ipython


Explanation
...........

PyQt4 used from Python 2.x provides 2 incompatible versions of QString and QVariant:

- version 1, the legacy which is the default, and
- version 2, a more pythonic one, which is the only one supported by *silx*.

All other configurations (i.e., PyQt4 on Python 3.x, PySide, PyQt5, IPython QtConsole widget) uses version 2 only or as the default.

For more information, see `IPython, PyQt and PySide <http://ipython.org/ipython-doc/stable/interactive/reference.html#pyqt-and-pyside>`_.


From Python
+++++++++++

:mod:`silx.gui.plot` widgets are Qt widgets, a QApplication needs to be started before using those widgets.
To start a QApplication, run:

>>> from silx.gui import qt  # Import Qt binding and do some set-up
>>> qapp = qt.QApplication([])

>>> from silx.gui.plot import *  # Import plot widgets and set-up matplotlib

.. currentmodule:: silx.gui.plot

Plot functions
++++++++++++++

:mod:`silx.gui.plot` package provides 2 functions to plot curves and images from the (I)Python console in a widget with a set of tools:

- :func:`plot1D`, and
- :func:`plot2D`.

For more features, use widgets directly (see `Plot curves in a widget`_ and `Plot images in a widget`_).


Curve: :func:`plot1D`
.....................

The following examples must run with a Qt QApplication initialized (see `Use silx.gui.plot from the console`_).

First import :func:`plot1D` function:

>>> from silx.gui.plot import plot1D
>>> import numpy

Plot a single curve given some values:

>>> values = numpy.random.random(100)
>>> plot_1curve = plot1D(values, title='Random data')

Plot a single curve given the x and y values:

>>> angles = numpy.linspace(0, numpy.pi, 100)
>>> sin_a = numpy.sin(angles)
>>> plot_sinus = plot1D(angles, sin_a,
...                     xlabel='angle (radian)', ylabel='sin(a)')

Plot many curves by giving a 2D array:

>>> curves = numpy.random.random(10 * 100).reshape(10, 100)
>>> plot_curves = plot1D(curves)

Plot many curves sharing the same x values:

>>> angles = numpy.linspace(0, numpy.pi, 100)
>>> values = (numpy.sin(angles), numpy.cos(angles))
>>> plot = plot1D(angles, values)

See :func:`plot1D` for details.


Image: :func:`plot2D`
.....................

This example plot a single image.

This example must run with a Qt QApplication initialized (see `Use silx.gui.plot from the console`_).

First, import :func:`plot2D`:

>>> from silx.gui.plot import plot2D
>>> import numpy

Then plot it:

>>> data = numpy.random.random(1024 * 1024).reshape(1024, 1024)
>>> plot = plot2D(data, title='Random data')

See :func:`plot2D` for more details.


Use :mod:`silx.gui.plot` from a script
--------------------------------------

A Qt GUI script must have a QApplication initialized before creating widgets:

.. code-block:: python

   from silx.gui import qt

   [...]

   qapp = qt.QApplication([])

   [...] # Widgets initialisation

   if __name__ == '__main__':
       [...]
       qapp.exec_()

Unless a Qt binding has already been loaded, :mod:`silx.gui.qt` uses the first Qt binding it founds by probing in the following order: PyQt5, PyQt4 and finally PySide.
If you prefer to choose the Qt binding yourself, import it before importing
a module from :mod:`silx.gui`:

.. code-block:: python

   import PySide  # Importing PySide will force silx to use it
   from silx.gui import qt


.. warning::
   :mod:`silx.gui.plot` widgets are not thread-safe.
   All calls to :mod:`silx.gui.plot` widgets must be made from the main thread.

Plot curves in a widget
-----------------------

The :class:`Plot1D` widget provides a plotting area and a toolbar with tools useful for curves such as setting logarithmic scale or defining region of interest.

First, create a :class:`Plot1D` widget:

.. code-block:: python

   from silx.gui.plot import Plot1D

   plot = Plot1D()  # Create the plot widget
   plot.show()  # Make the plot widget visible


One curve
+++++++++

To display a single curve, use the :meth:`.PlotWidget.addCurve` method:

.. code-block:: python

   plot.addCurve(x=(1, 2, 3), y=(3, 2, 1))  # Add a curve with default style

When you need to update this curve, call :meth:`.PlotWidget.addCurve` again with the new values to display:

.. code-block:: python

   plot.addCurve(x=(1, 2, 3), y=(1, 2, 3))  # Replace the existing curve

To clear the plotting area, call :meth:`.PlotWidget.clear`:

.. code-block:: python

   plot.clear()


Multiple curves
+++++++++++++++

In order to display multiple curves at the same time, you need to provide a different ``legend`` string for each of them:

.. code-block:: python

   import numpy

   x = numpy.linspace(-numpy.pi, numpy.pi, 1000)
   plot.addCurve(x, numpy.sin(x), legend='sinus')
   plot.addCurve(x, numpy.cos(x), legend='cosinus')
   plot.addCurve(x, numpy.random.random(len(x)), legend='random')


To update a curve, call :meth:`.PlotWidget.addCurve` with the ``legend`` of the curve you want to udpdate.
By default, the new curve will keep the same color (and style) as the curve it is updating:

.. code-block:: python

   plot.addCurve(x, numpy.random.random(len(x)) - 1., legend='random')

To remove a curve from the plot, call :meth:`.PlotWidget.remove` with the ``legend`` of the curve you want to remove from the plot:

.. code-block:: python

   plot.remove('random')

To clear the plotting area, call :meth:`.PlotWidget.clear`:

.. code-block:: python

   plot.clear()

Curve style
+++++++++++

By default, different curves will automatically use different styles to render, and keep the same style when updated.

It is possible to specify the ``color`` of the curve, its ``linewidth`` and ``linestyle`` as well as the ``symbol`` to use as markers for data points (See :meth:`.PlotWidget.addCurve` for more details):

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


Plot images in a widget
-----------------------

The :class:`Plot2D` widget provides a plotting area and a toolbar with tools useful for images, such as keeping aspect ratio, changing the colormap or defining a mask.

First, create a :class:`Plot2D` widget:

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
   plot.addImage(data)  # Plot the 2D data set with default colormap


To update this image, call :meth:`.PlotWidget.addImage` again with the new image to display:

.. code-block:: python

   # Create a RGB image
   rgb_image = (numpy.random.random(512*512*3) * 255).astype(numpy.uint8)
   rgb_image.shape = 512, 512, 3

   plot.addImage(rgb_image)  # Plot the RGB image instead of the previous data


To clear the plotting area, call :meth:`.PlotWidget.clear`:

.. code-block:: python

   plot.clear()


Origin and scale
++++++++++++++++

:meth:`.PlotWidget.addImage` supports both 2D arrays of data displayed with a colormap and RGB(A) images as 3D arrays of shape (height, width, color channels).

When displaying an image, it is possible to specify the ``origin`` and the ``scale`` of the image array in the plot area coordinates:

.. code-block:: python

   data = numpy.random.random(512 * 512).reshape(512, -1)
   plot.addImage(data, origin=(100, 100), scale=(0.1, 0.1))

When updating an image, if ``origin`` and ``scale`` are not provided, the previous values will be used:

.. code-block:: python

   data = numpy.random.random(512 * 512).reshape(512, -1)
   plot.addImage(data)  # Keep previous origin and scale


Colormap
++++++++

A ``colormap`` is described with a :class:`dict` as follows (See :mod:`silx.gui.plot.Plot` for full documentation of the colormap):

.. code-block:: python

   colormap = {
       'name': 'gray',             # Name of the colormap
       'normalization': 'linear',  # Either 'linear' or 'log'
       'autoscale': True,          # True to autoscale colormap to data range, False to use [vmin, vmax]
       'vmin': 0.0,                # If not autoscale, data value to bind to min of colormap
       'vmax': 1.0                 # If not autoscale, data value to bind to max of colormap
    }


At least the following colormap names are guaranteed to be available, but any colormap name from `matplotlib <http://matplotlib.org/>`_ (see `Choosing Colormaps <http://matplotlib.org/users/colormaps.html>`_) should work:

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

It is possible to change the default colormap of :meth:`.PlotWidget.addImage` for the plot widget with :meth:`.PlotWidget.setDefaultColormap` (and to get it with :meth:`.PlotWidget.getDefaultColormap`):

.. code-block:: python

   colormap = {'name': 'viridis', 'normalization': 'linear',
               'autoscale': True, 'vmin': 0.0, 'vmax': 1.0}
   plot.setDefaultColormap(colormap)

   data = numpy.arange(512 * 512.).reshape(512, -1)
   plot.addImage(data)  # Rendered with the default colormap set before

It is also possible to provide a ``colormap`` to :meth:`.PlotWidget.addImage` to override this default for an image:

.. code-block:: python

   colormap = {'name': 'magma', 'normalization': 'log',
               'autoscale': False, 'vmin': 1.2, 'vmax': 1.8}
   data = numpy.random.random(512 * 512).reshape(512, -1) + 1.
   plot.addImage(data, colormap=colormap)

As for `Origin and scale`_, when updating an image, if ``colormap`` is not provided, the previous colormap will be used:

.. code-block:: python

   data = numpy.random.random(512 * 512).reshape(512, -1) + 1.
   plot.addImage(data)  # Keep previous colormap

The colormap can be changed by the user from the widget's toolbar.


Multiple images
+++++++++++++++

In order to display multiple images at the same time, you need to provide a different ``legend`` string for each of them and to set the ``replace`` argument to ``False``:

.. code-block:: python

   data = numpy.random.random(512 * 512).reshape(512, -1)
   plot.addImage(data, legend='random', replace=False)

   data = numpy.arange(512 * 512.).reshape(512, -1)
   plot.addImage(data, legend='arange', replace=False, origin=(512, 512))


To update an image, call :meth:`.PlotWidget.addImage` with the ``legend`` of the curve you want to udpdate.
By default, the new image will keep the same colormap, origin and scale as the image it is updating:

.. code-block:: python

   data = (512 * 512. - numpy.arange(512 * 512.)).reshape(512, -1)
   plot.addImage(data, legend='arange', replace=False)  # Beware of replace=False


To remove an image from the plot, call :meth:`.PlotWidget.remove` with the ``legend`` of the image you want to remove:

.. code-block:: python

   plot.remove('random')


Control plot axes
-----------------

The following examples illustrate the API to control the plot axes.

Labels and title
++++++++++++++++

Use :meth:`.PlotWidget.setGraphTitle` to set the plot main title.
Use :meth:`.PlotWidget.setGraphXLabel` and :meth:`.PlotWidget.setGraphYLabel` to set the axes text labels:

.. code-block:: python

   plot.setGraphTitle('My plot')
   plot.setGraphXLabel('X')
   plot.setGraphYLabel('Y')


Axes limits
+++++++++++

Different methods allows to get and set the data limits displayed on each axis.

The following code moves the visible plot area to the right:

.. code-block:: python

    xmin, xmax = plot.getGraphXLimits()
    offset = 0.1 * (xmax - xmin)
    plot.setGraphXLimits(xmin + offset, xmax + offset)

:meth:`.PlotWidget.resetZoom` set the plot limits to the bounds of the data:

.. code-block:: python

   plot.resetZoom()

See :meth:`.PlotWidget.resetZoom`, :meth:`.PlotWidget.setLimits`, :meth:`.PlotWidget.getGraphXLimits`, :meth:`.PlotWidget.setGraphXLimits`, :meth:`.PlotWidget.getGraphYLimits`, :meth:`.PlotWidget.setGraphYLimits` for details.


Axes
++++

Different methods allow plot axes modifications:

.. code-block:: python

   plot.setYAxisInverted(True)  # Makes the Y axis pointing downward
   plot.setKeepDataAspectRatio(True)  # To keep aspect ratio between X and Y axes

See :meth:`.PlotWidget.setYAxisInverted`, :meth:`.PlotWidget.setKeepDataAspectRatio` for details.

.. code-block:: python

   plot.setGraphGrid(which='both')  # To show a grid for both minor and major axes ticks

   # Use logarithmic axes
   plot.setXAxisLogarithmic(True)
   plot.setYAxisLogarithmic(True)

See :meth:`.PlotWidget.setGraphGrid`, :meth:`.PlotWidget.setXAxisLogarithmic`, :meth:`.PlotWidget.setYAxisLogarithmic` for details.


Add plot actions
----------------

Additional plot actions can be added as toolbar icons or as menu entries, to interact with a plot.

Simple example
++++++++++++++

The following script is a simplistic example to show which basic steps are required:

 - create a new class inheriting from :class:`silx.gui.plot.PlotActions.PlotAction`
 - define basic parameters such as the icon, the tooltip...
 - write a method that will be triggered by the action
 - initialize the new plot action by passing a reference to a plot window
 - add the action to a toolbar or a menu

The method implemented in this action interacts with the plot in a basic way. It gets the active curve,
then it creates a new data array based on the curve data, and finally it replaces the original curve
with a new one using the modified data array.

.. code-block:: python

   from silx.gui import qt
   from silx.gui.plot import PlotWindow
   from silx.gui.plot.PlotActions import PlotAction


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
                               parent=parent)

       def shiftActiveCurveUp(self):
           """Get the active curve, add 1 to all y values, use this new y
           array to replace the original curve
           """
           activeCurve = self.plot.getActiveCurve()
           if activeCurve is not None:
               # Unpack curve data.
               # Each curve is represented by a tuple of 5 variables:
               #  - x and y are the array of abscissa and ordinate values
               #  - legend is a unique text identifying a curve
               #  - info and params are dictionaries of additional data
               #    (user defined, curve style and color...)
               x0, y0, legend, info, params = activeCurve

               # add 1 to all values in the y array
               # and assign the result to a new array y1
               # (IMPORTANT: do not modify y0 directly)
               y1 = y0 + 1.0

               # Re-using the same legend causes the original curve
               # to be replaced
               self.plot.addCurve(x0, y1, legend=legend,
                                  info=info)

   # creating QApplication is mandatory in order to use qt widget
   app = qt.QApplication([])

   # create a PlotWindow
   plotwin = PlotWindow()
   # Add a new toolbar
   toolbar = qt.QToolBar("My toolbar")
   plotwin.addToolBar(toolbar)
   # Get a reference to the PlotWindow's menu bar, add a menu
   menubar = plotwin.menuBar()
   actions_menu = menubar.addMenu("Custom actions")

   # Initialize our action, give it plotwin as a parameter
   myaction = ShiftUpAction(plotwin)
   # Add action to the menubar and toolbar
   toolbar.addAction(myaction)
   actions_menu.addAction(myaction)

   # Plot a couple of curves with synthetic data
   x = [0, 1, 2, 3, 4, 5, 6]
   y1 = [0, 1, 0, 1, 0, 1, 0]
   y2 = [0, 1, 2, 3, 4, 5, 6]
   plotwin.addCurve(x, y1, legend="triangle shaped curve")
   plotwin.addCurve(x, y2, legend="oblique line")

   plotwin.show()
   app.exec_()

.. |imgShiftAction0| image:: img/shiftAction0.png
   :height: 300px
   :align: middle

.. |imgShiftAction3| image:: img/shiftAction3.png
   :height: 300px
   :align: middle

.. list-table::
   :widths: 1 2

   * - |imgShiftAction0|
     - Initial state
   * - |imgShiftAction3|
     - After triggering the action 3 times, the selected triangle shaped curve
       is shifted up by 3 units

Advanced example
++++++++++++++++

This more advanced example shows additional ways of interacting with the plot, by changing
labels, storing additional data array along with the curve data.

This action is *checkable*, meaning that is has two stable states. When clicking the toolbar icon
or the menu item, it remains in a *pushed* state until it is clicked again.

In one state (*un-checked*), the original data is displayed. In the other state, the amplitude
spectrum of the original signal is displayed. When the state is changed, the triggered action
computes either the Fast Fourier Transform (FFT), or the reverse FFT.

.. code-block:: python

   import numpy
   import os

   from silx.gui import qt
   from silx.gui.plot import PlotWindow
   from silx.gui.plot.PlotActions import PlotAction

   # Custom icon
   # make sure there is a "fft.png" file saved in the same folder as this script
   scriptdir = os.path.dirname(os.path.realpath(__file__))
   my_icon = os.path.join(scriptdir, "fft.png")


   class FftAction(PlotAction):
       """QAction performing a Fourier transform on all curves when checked,
       and reverse transform when unchecked.

       :param plot: PlotWindow on which to operate
       :param parent: See documentation of :class:`QAction`
       """
       def __init__(self, plot, parent=None):
           PlotAction.__init__(
                   self,
                   plot,
                   icon=qt.QIcon(my_icon),
                   text='FFT',
                   tooltip='Perform Fast Fourier Transform on all curves',
                   triggered=self.fftAllCurves,
                   checkable=True,
                   parent=parent)

       def _rememberGraphLabels(self):
           """Store labels and title as attributes"""
           self.original_title = self.plot.getGraphTitle()
           self.original_xlabel = self.plot.getGraphXLabel()
           self.original_ylabel = self.plot.getGraphYLabel()

       def fftAllCurves(self, checked=False):
           """Get all curves from our PlotWindow, compute the amplitude spectrum
           using a Fast Fourier Transform, replace all curves with their
           amplitude spectra.

           When un-checking the button, do the reverse transform.

           :param checked: Boolean parameter signaling whether the action
               has been checked or unchecked.
           """
           allCurves = self.plot.getAllCurves(withhidden=True)

           if checked:
               # remember original labels
               self._rememberGraphLabels()
               # change them
               self.plot.setGraphTitle("Amplitude spectrum")
               self.plot.setGraphXLabel("Frequency")
               self.plot.setGraphYLabel("Amplitude")
           else:
               # restore original labels
               self.plot.setGraphTitle(self.original_title)
               self.plot.setGraphXLabel(self.original_xlabel)
               self.plot.setGraphYLabel(self.original_ylabel)

           self.plot.clearCurves()

           for i, curve in enumerate(allCurves):
               x, y, legend, info, params = curve

               if checked:
                   # FAST FOURIER TRANSFORM
                   fft_y = numpy.fft.fft(y)
                   # amplitude spectrum
                   A = numpy.abs(fft_y)

                   # sampling frequency (samples per X unit)
                   Fs = len(x) / (max(x) - min(x))
                   # frequency array (abscissa of new curve)
                   F = [k * Fs / len(x) for k in range(len(A))]

                   # we need to store  the complete transform (complex data) to be
                   # able to perform the reverse transform.
                   info["complex fft"] = fft_y
                   info["original x"] = x

                   # plot the amplitude spectrum
                   self.plot.addCurve(F, A, legend="FFT of " + legend,
                                      info=info)

               else:
                   # INVERSE FFT
                   fft_y = info["complex fft"]
                   # we keep only the real part because we know the imaginary
                   # part is 0 (our original data was real numbers)
                   y1 = numpy.real(numpy.fft.ifft(fft_y))

                   # recover original info
                   x1 = info["original x"]
                   legend1 = legend[7:]    # remove "FFT of "

                   # remove restored data from info dict
                   for key in ["complex fft", "original x"]:
                       del info[key]

                   # plot the original data
                   self.plot.addCurve(x1, y1, legend=legend1,
                                      info=info)

           self.plot.resetZoom()


   app = qt.QApplication([])

   plotwin = PlotWindow(control=True)
   toolbar = qt.QToolBar("My toolbar")
   plotwin.addToolBar(toolbar)

   myaction = FftAction(plotwin)
   toolbar.addAction(myaction)

   # x range: 0 -- 10 (1000 points)
   x = numpy.arange(1000) * 0.01

   twopi = 2 * numpy.pi
   # Sum of sine functions with frequencies 3, 20 and 42 Hz
   y1 = numpy.sin(twopi * 3 * x) + 1.5 * numpy.sin(twopi * 20 * x) + \
        2 * numpy.sin(twopi * 42 * x)
   # Cosine with frequency 7 Hz and phase pi / 3
   y2 = numpy.cos(twopi * 7 * (x - numpy.pi / 3))
   # 5 periods of square wave, amplitude 2
   y3 = numpy.zeros_like(x)
   for i in [0, 2, 4, 6, 8]:
       y3[i * len(x) / 10:(i + 1) * len(x) / 10] = 2

   plotwin.addCurve(x, y1, legend="sin")
   plotwin.addCurve(x, y2, legend="cos")
   plotwin.addCurve(x, y3, legend="square wave")

   plotwin.setGraphTitle("Original data")
   plotwin.setGraphYLabel("amplitude")
   plotwin.setGraphXLabel("time")

   plotwin.show()
   app.exec_()

.. |imgFftAction0| image:: img/fftAction0.png
   :height: 300px
   :align: middle

.. |imgFftAction1| image:: img/fftAction1.png
   :height: 300px
   :align: middle

.. list-table::
   :widths: 1 2

   * - |imgFftAction0|
     - Original signals (zoom applied). In red, a cosine wave at 7 Hz.
       In black, a sum of sines with frequencies of 3, 20 and 42 Hz.
       In green, a square wave with a fundamental frequency of 0.5 Hz
       (wavelength of 2 seconds).
   * - |imgFftAction1|
     - Amplitude spectra (zoom applied), with peaks visible at
       the expected frequencies of 3, 7, 20 and 42 Hz for the sine and cosine
       signals. In green, we see the complete series of peaks related to the square wave,
       with a fundamental frequency at 0.5 Hz and harmonic frequencies at every
       odd multiple of the fundamental.
