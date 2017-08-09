
.. currentmodule:: silx.gui

:mod:`console`
---------------

.. currentmodule:: silx.gui.console

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: img/IPythonWidget.png
         :height: 150px
         :align: center
     - :class:`IPythonWidget` is an interactive console widget.
   * - .. image:: img/IPythonDockWidget.png
         :height: 150px
         :align: center
     - :class:`IPythonDockWidget` is an :class:`IPythonWidget` embedded in
       a :class:`QDockWidget`.


:mod:`data`
---------------

.. currentmodule:: silx.gui.data

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: data/img/ArrayTableWidget.png
         :height: 150px
         :align: center
     - :class:`ArrayTableWidget`
   * - .. image:: data/img/DataViewer.png
         :height: 150px
         :align: center
     - :class:`DataViewer`
   * - .. image:: data/img/DataViewerFrame.png
         :height: 150px
         :align: center
     - :class:`DataViewerFrame`
   * - .. image:: data/img/NumpyAxesSelector.png
         :height: 50px
         :align: center
     - :class:`NumpyAxesSelector`


:mod:`fit`
---------------

.. currentmodule:: silx.gui.fit

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: fit/img/FitWidget.png
         :height: 150px
         :align: center
     - :class:`FitWidget`
   * - .. image:: fit/img/BackgroundDialog.png
         :height: 150px
         :align: center
     - :class:`BackgroundWidget.BackgroundDialog` (dialog embedding a :class:`BackgroundWidget.BackgroundWidget`)


:mod:`hdf5`
---------------

.. currentmodule:: silx.gui.hdf5

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: hdf5/img/Hdf5TreeView.png
         :height: 150px
         :align: center
     - :class:`Hdf5TreeView`


:mod:`plot`
------------

.. currentmodule:: silx.gui.plot

.. todo: include "Widgets gallery" section from ./plot/index.rst,
   and remove duplicated code


.. |imgPlotWindow| image:: plot/img/PlotWindow.png
   :height: 150px
   :align: middle

.. |imgPlot1D| image:: plot/img/Plot1D.png
   :height: 150px
   :align: middle

.. |imgPlot2D| image:: plot/img/Plot2D.png
   :height: 150px
   :align: middle

.. |imgImageView| image:: plot/img/ImageView.png
   :height: 150px
   :align: middle

.. |imgStackView| image:: plot/img/StackView.png
   :height: 150px
   :align: middle

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: plot/img/PlotWidget.png
          :height: 150px
          :align: center
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
   * - |imgStackView|
     - :class:`StackView` is a widget designed to display an image from a
       stack of images in a :class:`PlotWindow` widget, with a frame browser
       to navigate in the stack. The profile tool can do a 2D profile on the
       stack of images.


.. todo: plot3d


:mod:`widgets`
---------------

.. currentmodule:: silx.gui.widgets

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: widgets/img/HorizontalSliderWithBrowser.png
         :width: 150px
         :align: center
     - :class:`FrameBrowser.HorizontalSliderWithBrowser`
   * - .. image:: widgets/img/FrameBrowser.png
         :width: 110px
         :align: center
     - :class:`FrameBrowser.FrameBrowser`
   * - .. image:: widgets/img/PeriodicCombo.png
         :width: 150px
         :align: center
     - :class:`PeriodicTable.PeriodicCombo`
   * - .. image:: widgets/img/PeriodicList.png
         :height: 150px
         :align: center
     - :class:`PeriodicTable.PeriodicList`
   * - .. image:: widgets/img/PeriodicTable.png
         :height: 150px
         :align: center
     - :class:`PeriodicTable.PeriodicTable`
   * - .. image:: widgets/img/TableWidget.png
         :height: 150px
         :align: center
     - :class:`TableWidget.TableWidget` and :class:`TableWidget.TableView`
   * - .. image:: widgets/img/ThreadPoolPushButton.png
         :width: 100px
         :align: center
     - :class:`ThreadPoolPushButton`
   * - .. image:: widgets/img/WaitingPushButton.png
         :width: 100px
         :align: center
     - :class:`WaitingPushButton`
