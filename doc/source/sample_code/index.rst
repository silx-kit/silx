.. _sample-code:

Sample Code
===========

All sample codes can be downloaded as a zip file: |sample_code_archive|.

.. |sample_code_archive| archive:: ../../../examples/
   :filename: silx_examples.zip
   :basedir: silx_examples
   :filter: *.py *.png

:mod:`silx.gui` sample code
+++++++++++++++++++++++++++

:mod:`silx.gui.icons`
.....................

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`icons.py <../../../examples/icons.py>`
     - .. image:: img/icons.png
         :height: 150px
         :align: center
     - Display icons and animated icons provided by silx.

:mod:`silx.gui.data` and :mod:`silx.gui.hdf5`
.............................................

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`customHdf5TreeModel.py <../../../examples/customHdf5TreeModel.py>`
     - .. image:: img/customHdf5TreeModel.png
         :height: 150px
         :align: center
     - Qt Hdf5 widget examples
   * - :download:`customDataView.py <../../../examples/customDataView.py>`
     - .. image:: img/customDataView.png
         :height: 150px
         :align: center
     - Qt data view example
   * - :download:`hdf5widget.py <../../../examples/hdf5widget.py>`
     - .. image:: img/hdf5widget.png
         :height: 150px
         :align: center
     - Qt Hdf5 widget examples

       .. note:: This module has a dependency on the `h5py <http://www.h5py.org/>`_
           library, which is not a mandatory dependency for `silx`. You might need
           to install it if you don't already have it.

:mod:`silx.gui.dialog`
......................

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`fileDialog.py <../../../examples/fileDialog.py>`
     - .. image:: img/fileDialog.png
         :height: 150px
         :align: center
     - Example for the use of the ImageFileDialog.

:mod:`silx.gui.widgets`
.......................

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`periodicTable.py <../../../examples/periodicTable.py>`
     - .. image:: img/periodicTable.png
         :height: 150px
         :align: center
     - This script is a simple example of how to use the periodic table widgets,
       select elements and connect signals.
   * - :download:`simplewidget.py <../../../examples/simplewidget.py>`
     - .. image:: img/simplewidget.png
         :height: 150px
         :align: center
     - This script shows a gallery of simple widgets provided by silx.

       It shows the following widgets:

       - :class:`~silx.gui.widgets.WaitingPushButton`:
         A button with a progress-like waiting animated icon.

:mod:`silx.gui.plot` sample code
++++++++++++++++++++++++++++++++

Widgets
.......

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`imageview.py <../../../examples/imageview.py>`
     - .. image:: img/imageview.png
         :height: 150px
         :align: center
     - Example to show the use of :mod:`~silx.gui.plot.ImageView` widget.

       It can be used to open an EDF or TIFF file from the shell command line.

       To view an image file with the current installed silx library:
       ``python examples/imageview.py <file to open>``
       To get help:
       ``python examples/imageview.py -h``

       For developers with a git clone you can use it with the bootstrap
       To view an image file with the current installed silx library:

       ``./bootstrap.py python examples/imageview.py <file to open>``
   * - :download:`stackView.py <../../../examples/stackView.py>`
     - .. image:: img/stackView.png
         :height: 150px
         :align: center
     - This script is a simple example to illustrate how to use the
       :mod:`~silx.gui.plot.StackView` widget.
   * - :download:`colormapDialog.py <../../../examples/colormapDialog.py>`
     - .. image:: img/colormapDialog.png
         :height: 150px
         :align: center
     - This script shows the features of a :mod:`~silx.gui.dialog.ColormapDialog`.

:class:`silx.gui.plot.actions.PlotAction`
.........................................

Sample code that adds buttons to the toolbar of a silx plot widget.

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`plotClearAction.py <../../../examples/plotClearAction.py>`
     - .. image:: img/plotClearAction.png
         :height: 150px
         :align: center
     - This script shows how to create a minimalistic
       :class:`~silx.gui.plot.actions.PlotAction` that clear the plot.

       This illustrates how to add more buttons in a plot widget toolbar.
   * - :download:`shiftPlotAction.py <../../../examples/shiftPlotAction.py>`
     - .. image:: img/shiftPlotAction.png
         :height: 150px
         :align: center
     - This script is a simple (trivial) example of how to create a :class:`~silx.gui.plot.PlotWindow`,
       create a custom :class:`~silx.gui.plot.actions.PlotAction` and add it to the toolbar.

       The action simply shifts the selected curve up by 1 unit by adding 1 to each
       value of y.
   * - :download:`fftPlotAction.py <../../../examples/fftPlotAction.py>`,
       :download:`fft.png <../../../examples/fft.png>`
     - .. image:: img/fftPlotAction.png
         :height: 150px
         :align: center
     - This script is a simple example of how to create a :class:`~silx.gui.plot.PlotWindow`
       with a custom :class:`~silx.gui.plot.actions.PlotAction` added to the toolbar.

       The action computes the FFT of all curves and plots their amplitude spectrum.
       It also performs the reverse transform.

       This example illustrates:
          - how to create a checkable action
          - how to store user info with a curve in a PlotWindow
          - how to modify the graph title and axes labels
          - how to add your own icon as a PNG file

       See shiftPlotAction.py for a simpler example with more basic comments.

Add features to :class:`~silx.gui.plot.PlotWidget`
..................................................

Sample code that adds specific tools or functions to plot widgets.

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`plotWidget.py <../../../examples/plotWidget.py>`
     - .. image:: img/plotWidget.png
         :height: 150px
         :align: center
     - This script shows how to create a custom window around a PlotWidget.

       It subclasses :class:`QMainWindow`, uses a :class:`~silx.gui.plot.PlotWidget`
       as its central widget and adds toolbars and a colorbar by using pluggable widgets:

       - :class:`~silx.gui.plot.PlotWidget` from :mod:`silx.gui.plot`
       - QToolBar from :mod:`silx.gui.plot.tools`
       - QAction from :mod:`silx.gui.plot.actions`
       - QToolButton from :mod:`silx.gui.plot.PlotToolButtons`
       - :class:`silx.gui.plot.ColorBar.ColorBarWidget`
   * - :download:`plotContextMenu.py <../../../examples/plotContextMenu.py>`
     - .. image:: img/plotContextMenu.png
         :height: 150px
         :align: center
     - This script illustrates the addition of a context menu to a
       :class:`~silx.gui.plot.PlotWidget`.

       This is done by adding a custom context menu to the plot area of PlotWidget:
       - set the context menu policy of the plot area to Qt.CustomContextMenu.
       - connect to the plot area customContextMenuRequested signal.

       The same method works with :class:`~silx.gui.plot.PlotWindow.PlotWindow`,
       :class:`~silx.gui.plot.PlotWindow.Plot1D` and
       :class:`~silx.gui.plot.PlotWindow.Plot2D` widgets as they
       inherit from :class:`~silx.gui.plot.PlotWidget`.

       For more information on context menus, see Qt documentation.
   * - :download:`plotItemsSelector.py <../../../examples/plotItemsSelector.py>`
     - .. image:: img/plotItemsSelector.png
         :height: 150px
         :align: center
     - This example illustrates how to use a :class:`ItemsSelectionDialog` widget
       associated with a :class:`~silx.gui.plot.PlotWidget`
   * - :download:`plotLimits.py <../../../examples/plotLimits.py>`
     - .. image:: img/plotLimits.png
         :height: 150px
         :align: center
     - This script is an example to illustrate how to use axis synchronization
       tool.
   * - :download:`plotUpdateCurveFromThread.py <../../../examples/plotUpdateCurveFromThread.py>`
     - .. image:: img/plotUpdateCurveFromThread.png
         :height: 150px
         :align: center
     - This script illustrates the update of a :mod:`silx.gui.plot` widget from a thread.

       The problem is that plot and GUI methods should be called from the main thread.
       To safely update the plot from another thread, one need to execute the update
       asynchronously in the main thread.
       In this example, this is achieved with
       :func:`~silx.gui.utils.concurrent.submitToQtMainThread`.

       In this example a thread calls submitToQtMainThread to update the curve
       of a plot.
   * - :download:`plotUpdateImageFromThread.py <../../../examples/plotUpdateImageFromThread.py>`
     - .. image:: img/plotUpdateImageFromThread.png
         :height: 150px
         :align: center
     - This script illustrates the update of a :mod:`silx.gui.plot` widget from a thread.

       The problem is that plot and GUI methods should be called from the main thread.
       To safely update the plot from another thread, one need to execute the update
       asynchronously in the main thread.
       In this example, this is achieved with
       :func:`~silx.gui.utils.concurrent.submitToQtMainThread`.

       In this example a thread calls submitToQtMainThread to update the curve
       of a plot.
   * - :download:`plotInteractiveImageROI.py <../../../examples/plotInteractiveImageROI.py>`
     - .. image:: img/plotInteractiveImageROI.png
         :height: 150px
         :align: center
     - This script illustrates image ROI selection in a :class:`~silx.gui.plot.PlotWidget`

       It uses :class:`~silx.gui.plot.tools.roi.RegionOfInterestManager` and
       :class:`~silx.gui.plot.tools.roi.RegionOfInterestTableWidget` to handle the
       interactive selection and to display the list of selected ROIs.
   * - :download:`printPreview.py <../../../examples/printPreview.py>`
     - .. image:: img/printPreview.png
         :height: 150px
         :align: center
     - This script illustrates how to add a print preview tool button to any plot
       widget inheriting :class:`~silx.gui.plot.PlotWidget`.

       Three plot widgets are instantiated. One of them uses a standalone
       :class:`~silx.gui.plot.PrintPreviewToolButton.PrintPreviewToolButton`,
       while the other two use a
       :class:`~silx.gui.plot.PrintPreviewToolButton.SingletonPrintPreviewToolButton`
       which allows them to send their content to the same print preview page.
   * - :download:`scatterMask.py <../../../examples/scatterMask.py>`
     - .. image:: img/scatterMask.png
         :height: 150px
         :align: center
     - This example demonstrates how to use ScatterMaskToolsWidget
       and NamedScatterAlphaSlider with a PlotWidget.
   * - :download:`syncaxis.py <../../../examples/syncaxis.py>`
     - .. image:: img/syncaxis.png
         :height: 150px
         :align: center
     - This script is an example to illustrate how to use axis synchronization
       tool.

.. _plot3d-sample-code:

:mod:`silx.gui.plot3d` sample code
++++++++++++++++++++++++++++++++++

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`plot3dContextMenu.py <../../../examples/plot3dContextMenu.py>`
     - .. image:: img/plot3dContextMenu.png
         :height: 150px
         :align: center
     - This script adds a context menu to a :class:`silx.gui.plot3d.ScalarFieldView`.

       This is done by adding a custom context menu to the :class:`Plot3DWidget`:

       - set the context menu policy to Qt.CustomContextMenu.
       - connect to the customContextMenuRequested signal.

       For more information on context menus, see Qt documentation.
   * - :download:`viewer3DVolume.py <../../../examples/viewer3DVolume.py>`
     - .. image:: img/viewer3DVolume.png
         :height: 150px
         :align: center
     - This script illustrates the use of :class:`silx.gui.plot3d.ScalarFieldView`.

       It loads a 3D scalar data set from a file and displays iso-surfaces and
       an interactive cutting plane.
       It can also be started without providing a file.
   * - :download:`plot3dSceneWindow.py <../../../examples/plot3dSceneWindow.py>`
     - .. image:: img/plot3dSceneWindow.png
         :height: 150px
         :align: center
     - This script displays the different items of :class:`~silx.gui.plot3d.SceneWindow`.

       It shows the different visualizations of :class:`~silx.gui.plot3d.SceneWindow`
       and :class:`~silx.gui.plot3d.SceneWidget`.
       It illustrates the API to set those items.

       It features:

       - 2D images: data and RGBA images
       - 2D scatter data, displayed either as markers, wireframe or surface.
       - 3D scatter plot
       - 3D scalar field with iso-surface and cutting plane.
       - A clipping plane.

:mod:`silx.io` sample code
++++++++++++++++++++++++++

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`writetoh5.py <../../../examples/writetoh5.py>`
     -
     - This script is an example of how to use the :mod:`silx.io.convert` module.
       See the following tutorial for more information: :doc:`../Tutorials/convert`
