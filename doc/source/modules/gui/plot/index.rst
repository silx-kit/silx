
.. currentmodule:: silx.gui

:mod:`plot`: 1D and 2D Plot widgets
===================================

.. toctree::
   :hidden:

   getting_started.rst

.. currentmodule:: silx.gui.plot

.. automodule:: silx.gui.plot

For an introduction to the widgets of this package, see :doc:`getting_started`.

For examples of custom plot actions, see :doc:`plotactions_examples`.

Widgets gallery
---------------

.. |imgPlotWidget| image:: img/PlotWidget.png
   :height: 150px
   :align: middle

.. |imgPlotWindow| image:: img/PlotWindow.png
   :height: 150px
   :align: middle

.. |imgPlot1D| image:: img/Plot1D.png
   :height: 150px
   :align: middle

.. |imgPlot2D| image:: img/Plot2D.png
   :height: 150px
   :align: middle

.. |imgImageView| image:: img/ImageView.png
   :height: 150px
   :align: middle

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - |imgPlotWidget|
     - :class:`PlotWidget` is the base Qt widget providing a plot area.
       Other plot widgets are based on this one and provides the same API.
   * - |imgPlotWindow|
     - :class:`PlotWindow` adds a toolbar to :class:`PlotWidget`.
       The content of this toolbar can be configured from the
       :class:`PlotWindow` constructor or by hiding its content afterward.
   * - |imgPlot1D|
     - :class:`.Plot1D` is a :class:`PlotWindow` configured with tools useful
       for curves.
   * - |imgPlot2D|
     - :class:`.Plot2D` is a :class:`PlotWindow` configured with tools useful
       for images.
   * - |imgImageView|
     - :class:`ImageView` adds side histograms to a :class:`.Plot2D` widget.


Public modules
--------------

.. toctree::
   :maxdepth: 2

   plotwidget.rst
   plotwindow.rst
   imageview.rst
   plot.rst
   plotactions.rst

Internals
---------

.. toctree::
   :maxdepth: 2

   dev.rst
