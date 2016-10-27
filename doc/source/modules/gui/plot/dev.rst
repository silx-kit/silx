Package structure
=================

The :mod:`silx.gui.hdf5` package provides HDF5 widgets.
This package is structured as follows.

.. currentmodule:: silx.gui.plot

:mod:`.PlotWidget` and :mod:`.PlotWindow` provides the user API.
:class:`PlotWidget` is a Qt widget (actually a :class:`QMainWindow`) displaying a 1D, 2D plot area.
It provides different interaction modes.
:class:`PlotWindow` is a Qt widget (actually a :class:`QMainWindow`) which adds a set of toolbar buttons and associated functionalities to :class:`PlotWidget`.
The toolbar QActions are implemented in :mod:`.PlotActions`.

:mod:`.Plot`, :mod:`.PlotEvents` and :mod:`.PlotInteraction` implement the plotting API regardless of the rendering backend and regardless of its integration in Qt.
The plotting API in defined in :mod:`.Plot`.
The different interaction modes (zoom, drawing, pan) are implemented in :mod:`.PlotInteraction`.
Each interaction mode is implemented with a state machine structure (implemented in :mod:`.Interaction`).
The different events emitted by :class:`Plot` and by the interaction modes are created with helper functions defined in :mod:`.PlotEvents`.

The :class:`PlotWindow` uses additional widgets:

- :mod:`.ColormapDialog` to change colormap settings.
- :mod:`.CurvesROIWidget` to create regions of interest for curves
- :mod:`.LegendSelector` to display a list of curves legends which provides some control on the curves (e.g., select, delete).
- :mod:`.MaskToolsWidget` to provide tools to draw a mask on an image.
- The :mod:`.PlotTools` module provides a set of additional widgets:

  - :class:`.PlotTools.PositionInfo` to display information related to mouse coordinates in the plot and related information.
  - :class:`.PlotTools.LimitsToolBar` to display and set the limits of the plot area.
  - :class:`.PlotTools.ProfileToolBar` to provide tools to extract a profile along a line on an image.

- :mod:`silx.gui.console` to provide an IPython console which can control the plot area content.

The widgets also use the following miscellaneous modules:

- :mod:`.Colors` to convert colors from name to RGB(A)
- :mod:`.ModestImage` to provide a faster matplotlib AxesImage class using nearest values.
- :mod:`.MPLColormap` to embed recent matplotlib colormaps: 'magma', 'inferno', 'plasma' and 'viridis'.
- :mod:`._utils`: utility functions

:mod:`.BackendBase` defines the API any plot backend should provide in :class:`BackendBase`.
:mod:`.BackendMatplotlib` implements a `matplotlib <http://matplotlib.org/>`_ backend.
It is splitted in two classes:

.. currentmodule:: silx.gui.plot.BackendMatplotlib

- :class:`BackendMatplotlib` that provides a matplotlib backend without a specific canvas.
- :class:`BackendMatplotlibQt` which inherits from :class:`BackendMatplotlib` and adds a Qt canvas, and Qt specific functionalities.

Modules
=======

.. currentmodule:: silx.gui.plot

For :mod:`.PlotWidget` and :mod:`.Plot` modules, see their respective documentations: :mod:`.PlotWidget`, :mod:`.Plot`.

The following modules are the modules used internally by the plot package.

:mod:`BackendBase`
++++++++++++++++++

.. currentmodule:: silx.gui.plot.BackendBase

.. automodule:: silx.gui.plot.BackendBase
   :members:

:mod:`BackendMatplotlib`
++++++++++++++++++++++++

.. currentmodule:: silx.gui.plot.BackendMatplotlib

.. automodule:: silx.gui.plot.BackendMatplotlib
   :members:

:mod:`ColormapDialog`
+++++++++++++++++++++

.. currentmodule:: silx.gui.plot.ColormapDialog

.. automodule:: silx.gui.plot.ColormapDialog
   :members:

:mod:`Colors`
+++++++++++++

.. currentmodule:: silx.gui.plot.Colors

.. automodule:: silx.gui.plot.Colors
   :members: rgba

:mod:`CurvesROIWidget`
++++++++++++++++++++++

.. currentmodule:: silx.gui.plot.CurvesROIWidget

.. automodule:: silx.gui.plot.CurvesROIWidget
   :members:

:mod:`Interaction`
++++++++++++++++++

.. currentmodule:: silx.gui.plot.Interaction

.. automodule:: silx.gui.plot.Interaction
   :members:

:mod:`LegendSelector`
+++++++++++++++++++++

.. currentmodule:: silx.gui.plot.LegendSelector

.. automodule:: silx.gui.plot.LegendSelector
   :members:

:mod:`MaskToolsWidget`
++++++++++++++++++++++

.. currentmodule:: silx.gui.plot.MaskToolsWidget

.. automodule:: silx.gui.plot.MaskToolsWidget
   :members:

:mod:`ModestImage`
++++++++++++++++++

.. currentmodule:: silx.gui.plot.ModestImage

.. automodule:: silx.gui.plot.ModestImage
   :members:
   :undoc-members:

:mod:`MPLColormap`
++++++++++++++++++

.. currentmodule:: silx.gui.plot.MPLColormap

.. automodule:: silx.gui.plot.MPLColormap
   :members:

:mod:`PlotEvents`
+++++++++++++++++

.. currentmodule:: silx.gui.plot.PlotEvents

.. automodule:: silx.gui.plot.PlotEvents
   :members:
   :undoc-members:

:mod:`PlotInteraction`
++++++++++++++++++++++

.. currentmodule:: silx.gui.plot.PlotInteraction

.. automodule:: silx.gui.plot.PlotInteraction
   :members:

:mod:`PlotTools`
++++++++++++++++

.. currentmodule:: silx.gui.plot.PlotTools

.. automodule:: silx.gui.plot.PlotTools
   :members:

:mod:`_utils`
+++++++++++++

.. currentmodule:: silx.gui.plot._utils

.. automodule:: silx.gui.plot._utils
   :members:
