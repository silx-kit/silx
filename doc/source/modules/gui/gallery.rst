
Widgets gallery
===============


:mod:`silx.gui.console` Widgets
+++++++++++++++++++++++++++++++

.. currentmodule:: silx.gui.console

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: img/IPythonWidget.png
         :height: 150px
         :align: center
     - :class:`IPythonWidget` is an interactive console widget running a
       :class`QtInProcessKernelManager`. This allows to push variables to the
       interactive console, and interact with your application (e.g. adding
       curves to a plot)
   * - .. image:: img/IPythonDockWidget.png
         :height: 150px
         :align: center
     - :class:`IPythonDockWidget` is an :class:`IPythonWidget` embedded in
       a :class:`QDockWidget`.


:mod:`silx.gui.data` Widgets
++++++++++++++++++++++++++++

.. currentmodule:: silx.gui.data

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: data/img/ArrayTableWidget.png
         :height: 150px
         :align: center
     - :class:`ArrayTableWidget` is a table widget with browsers designed to
       display the content of multi-dimensional data arrays.
   * - .. image:: data/img/DataViewer.png
         :height: 150px
         :align: center
     - :class:`DataViewer` is a widget designed to display data using the most
       adapted view.
   * - .. image:: data/img/DataViewerFrame.png
         :height: 150px
         :align: center
     - :class:`DataViewerFrame` is a :class:`DataViewer` with a view selector
       that lets you view the data using any compatible view.
   * - .. image:: data/img/NumpyAxesSelector.png
         :height: 50px
         :align: center
     - :class:`NumpyAxesSelector` is a widget designed to select a subarray in a
       n-dimensional array, by fixing the index on some of the dimensions.


:mod:`silx.gui.dialog` Widgets
++++++++++++++++++++++++++++++

.. currentmodule:: silx.gui.dialog

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: dialog/img/datafiledialog.png
         :height: 150px
         :align: center
     - :class:`DataFileDialog` is a dialog that allows users to select
       any datasets or groups from any HDF5-like file. It features a file
       browser that can also browse the content of HDF5 file as if they were
       directories.
   * - .. image:: dialog/img/imagefiledialog_h5.png
         :height: 150px
         :align: center
     - :class:`ImageFileDialog` is a dialog that allows users to select
       an image from any HDF5-like file.
   * - .. image:: dialog/img/groupdialog.png
         :height: 150px
         :align: center
     - :class:`GroupDialog` is a dialog that allows users to select
       a group from one or several specified HDF5-like files.


:mod:`silx.gui.fit` Widgets
+++++++++++++++++++++++++++

.. currentmodule:: silx.gui.fit

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: fit/img/FitWidget.png
         :height: 150px
         :align: center
     - :class:`FitWidget` is a widget designed to configure and run a fitting process,
       with constraints on parameters.
   * - .. image:: fit/img/BackgroundDialog.png
         :height: 150px
         :align: center
     - :class:`BackgroundWidget.BackgroundDialog` is a widget designed to adjust
       the parameters and preview the results of a *snip* or *strip* background
       filter.


:mod:`silx.gui.hdf5` Widgets
++++++++++++++++++++++++++++

.. currentmodule:: silx.gui.hdf5

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: hdf5/img/Hdf5TreeView.png
         :height: 150px
         :align: center
     - :class:`Hdf5TreeView` is a tree view desiged to browse an HDF5
       file structure.

.. _plot-gallery:

:mod:`silx.gui.plot` Widgets
++++++++++++++++++++++++++++

.. currentmodule:: silx.gui.plot

Plotting widgets:

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
   * - .. image:: plot/img/PlotWindow.png
          :height: 150px
          :align: center
     - :class:`PlotWindow` adds a toolbar to :class:`PlotWidget`.
       The content of this toolbar can be configured from the
       :class:`PlotWindow` constructor or by hiding its content afterward.
   * - .. image:: plot/img/Plot1D.png
          :height: 150px
          :align: center
     - :class:`.Plot1D` is a :class:`PlotWindow` configured with tools useful
       for curves.
   * - .. image:: plot/img/Plot2D.png
          :height: 150px
          :align: center
     - :class:`.Plot2D` is a :class:`PlotWindow` configured with tools useful
       for images.
   * - .. image:: plot/img/ImageView.png
          :height: 150px
          :align: center
     - :class:`ImageView` adds side histograms to a :class:`.Plot2D` widget.
   * - .. image:: plot/img/StackView.png
          :height: 150px
          :align: center
     - :class:`StackView` is a widget designed to display an image from a
       stack of images in a :class:`PlotWindow` widget, with a frame browser
       to navigate in the stack. The profile tool can do a 2D profile on the
       stack of images.
   * - .. image:: plot/img/ComplexImageView.png
          :height: 150px
          :align: center
     - :class:`ComplexImageView` is a widget dedicated to visualize a single
       2D dataset of complex data.
       It allows to switch between viewing amplitude, phase, real, imaginary,
       colored phase with amplitude or log10(amplitude) as brightness.
   * - .. image:: plot/img/ScatterView.png
          :height: 150px
          :align: center
     - :class:`ScatterView` is a widget dedicated to visualize a scatter plot.

Additional widgets:

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: plot/img/PositionInfo.png
          :width: 300px
          :align: center
     - :class:`.PlotTools.PositionInfo` is a widget displaying mouse position and
       information of a :class:`PlotWidget` associated to the mouse position.
   * - .. image:: plot/img/LimitsToolBar.png
          :width: 300px
          :align: center
     - :class:`.PlotTools.LimitsToolBar` is a QToolBar displaying and
       controlling the limits of a :class:`PlotWidget`.
   * - .. image:: plot/img/logColorbar.png
          :height: 150px
          :align: center
     - :class:`.ColorBar.ColorBarWidget` display colormap gradient and can be linked with a plot
       to display the colormap
   * - .. image:: plot/img/statsWidget.png
          :height: 150px
          :align: center
     - :class:`.statsWidget.StatsWidget` display statistics on plot's items (curve, images...)

.. _plot3d-gallery:

:mod:`silx.gui.plot3d` Widgets
++++++++++++++++++++++++++++++

.. currentmodule:: silx.gui.plot3d

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: plot3d/img/SceneWindow.png
          :height: 150px
          :align: center
     - :class:`SceneWindow` is a :class:`QMainWindow` embedding a 3D data visualization :class:`SceneWidget`
       and associated toolbars.
       It can display 2D images, 2D scatter data, 3D scatter data and 3D volumes with different visualizations.
       See ``plot3dSceneWindow.py`` in :ref:`plot3d-sample-code`.
   * - .. image:: plot3d/img/SceneWidget.png
          :height: 150px
          :align: center
     - :class:`SceneWidget` is a :class:`Plot3DWidget` providing a 3D scene for visualizing different kind of data.
       It can display 2D images, 2D scatter data, 3D scatter data and 3D volumes with different visualizations.
       See ``plot3dSceneWindow.py`` in :ref:`plot3d-sample-code`.
   * - .. image:: plot3d/img/ScalarFieldView.png
          :height: 150px
          :align: center
     - :class:`ScalarFieldView` is a :class:`Plot3DWindow` dedicated to display 3D scalar field.
       It can display iso-surfaces and an interactive cutting plane.
   * - .. image:: plot3d/img/Plot3DWindow.png
          :height: 150px
          :align: center
     - :class:`Plot3DWindow` is a :class:`QMainWindow` with a :class:`Plot3DWidget` as central widget
       and toolbars.
   * - .. image:: plot3d/img/Plot3DWidget.png
          :height: 150px
          :align: center
     - :class:`Plot3DWidget` is the base Qt widget providing an OpenGL 3D scene.
       Other widgets are using this widget as the OpenGL scene canvas.
   * - .. image:: plot3d/img/SFViewParamTree.png
         :height: 150px
         :align: center
     - :class:`SFViewParamTree` is a :class:`QTreeView` widget that can be attached to a :class:`ScalarFieldView`.
       It displays current parameters of the :class:`ScalarFieldView` and allows to modify it.
       See :ref:`plot3d-sample-code`.


:mod:`silx.gui.widgets` Widgets
+++++++++++++++++++++++++++++++

.. currentmodule:: silx.gui.widgets

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Widget
     - Description
   * - .. image:: widgets/img/FrameBrowser.png
         :width: 110px
         :align: center
     - :class:`FrameBrowser.FrameBrowser` is a browser widget designed to
       browse through a sequence of integers (e.g. the indices of an array)
   * - .. image:: widgets/img/HorizontalSliderWithBrowser.png
         :width: 150px
         :align: center
     - :class:`FrameBrowser.HorizontalSliderWithBrowser` is a :class:`FrameBrowser`
       with an additional slider.
   * - .. image:: widgets/img/PeriodicCombo.png
         :width: 150px
         :align: center
     - :class:`PeriodicTable.PeriodicCombo` is a :class:`QComboBox` widget designed to
       select a single atomic element.
   * - .. image:: widgets/img/PeriodicList.png
         :height: 150px
         :align: center
     - :class:`PeriodicTable.PeriodicList` is a :class:`QTreeWidget` designed to select one
       or more atomic elements.
   * - .. image:: widgets/img/PeriodicTable.png
         :height: 150px
         :align: center
     - :class:`PeriodicTable.PeriodicTable` is a periodic table widget designed to select one
       or more atomic elements.
   * - .. image:: widgets/img/TableWidget.png
         :height: 150px
         :align: center
     - :class:`TableWidget.TableWidget` and :class:`TableWidget.TableView` inherit respectively
       :class:`QTableWidget` and :class:`QTableView`, and add a context menu with *cut/copy/paste*
       actions.
   * - .. image:: widgets/img/WaitingPushButton.png
         :width: 60px
         :align: center
     - :class:`WaitingPushButton` is a :class:`QPushButton` that can be graphically disabled,
       for example to wait for a callback function to finish computing.
   * - .. image:: widgets/img/ThreadPoolPushButton.png
         :width: 100px
         :align: center
     - :class:`ThreadPoolPushButton` is a :class:`WaitingPushButton` that executes a
       callback in a thread.
