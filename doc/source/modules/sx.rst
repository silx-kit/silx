
:mod:`silx.sx`: Using silx from Python Interpreter
==================================================

.. currentmodule:: silx.sx

.. automodule:: silx.sx

Plot functions
--------------

.. currentmodule:: silx.sx._plot

The following functions plot curves and images with silx widgets:

- :func:`plot` for curves
- :func:`imshow` for images
- :func:`scatter` for scatter plot

The :func:`ginput` function handles user selection on those widgets.


.. note:: Those functions are not available from a notebook.

:func:`plot`
++++++++++++

.. autofunction:: plot

:func:`imshow`
++++++++++++++

.. autofunction:: imshow

:func:`scatter`
+++++++++++++++

.. autofunction:: scatter


:func:`ginput`
++++++++++++++

.. autofunction:: ginput

3D plot functions
-----------------

.. currentmodule:: silx.sx._plot3d

The following functions plot 3D data with silx widgets (it requires OpenGL):

- :func:`contour3d` for isosurfaces (and cut plane) in a 3D scalar field
- :func:`points3d` for 2D/3D scatter plots

.. note:: Those functions are not available from a notebook.

:func:`contour3d`
+++++++++++++++++

.. autofunction:: contour3d

:func:`points3d`
++++++++++++++++

.. autofunction:: points3d

Widgets
-------

The widgets of the :mod:`silx.gui.plot` package are also exposed in this package.
See :mod:`silx.gui.plot` for documentation.

Input/Output
------------

The content of the :mod:`silx.io` package is also exposed in this package.
See :mod:`silx.io` for documentation.

Math
----

The following classes from :mod:`silx.math` are exposed in this package:

- :class:`~silx.math.histogram.Histogramnd`
- :class:`~silx.math.histogram.HistogramndLut`
- :class:`~silx.math.fit.leastsq`
