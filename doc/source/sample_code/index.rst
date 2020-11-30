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
         :width: 150px
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
         :width: 150px
     - Qt Hdf5 widget examples
   * - :download:`customDataView.py <../../../examples/customDataView.py>`
     - .. image:: img/customDataView.png
         :width: 150px
     - Qt data view example
   * - :download:`hdf5widget.py <../../../examples/hdf5widget.py>`
     - .. image:: img/hdf5widget.png
         :width: 150px
     - Qt Hdf5 widget examples

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
         :width: 150px
     - Example for the use of the ImageFileDialog.
   * - :download:`colormapDialog.py <../../../examples/colormapDialog.py>`
     - .. image:: img/colormapDialog.png
         :width: 150px
     - This script shows the features of a :mod:`~silx.gui.dialog.ColormapDialog`.

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
         :width: 150px
         :align: center
     - This script is a simple example of how to use the periodic table widgets,
       select elements and connect signals.
   * - :download:`simplewidget.py <../../../examples/simplewidget.py>`
     - .. image:: img/simplewidget.png
         :width: 150px
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
         :width: 150px
     - Example to show the use of :mod:`~silx.gui.plot.ImageView` widget.

       It can be used to open an EDF or TIFF file from the shell command line.

       To view an image file with the current installed silx library:
       ``python examples/imageview.py <file to open>``
       To get help:
       ``python examples/imageview.py -h``
   * - :download:`stackView.py <../../../examples/stackView.py>`
     - .. image:: img/stackView.png
         :width: 150px
     - This script is a simple example to illustrate how to use the
       :mod:`~silx.gui.plot.StackView` widget.
   * - :download:`scatterview.py <../../../examples/scatterview.py>`
     - .. image:: img/scatterview.png
         :width: 150px
     - Example to show the use of :class:`~silx.gui.plot.ScatterView.ScatterView` widget
   * - :download:`compareImages.py <../../../examples/compareImages.py>`
     - .. image:: img/compareImages.png
          :width: 150px
     - usage: compareImages.py [-h] [--debug] [--testdata] [--use-opengl-plot]
                               [files [files ...]]

       Example demonstrating the use of the widget CompareImages

       positional arguments:
         files              Image data to compare (HDF5 file with path, EDF files,
                            JPEG/PNG image files). Data from HDF5 files can be
                            accessed using dataset path and slicing as an URL:
                            silx:../my_file.h5?path=/entry/data&slice=10 EDF file
                            frames also can can be accessed using URL:
                            fabio:../my_file.edf?slice=10 Using URL in command like
                            usually have to be quoted: "URL".

       optional arguments:
         -h, --help         show this help message and exit
         --debug            Set logging system in debug mode
         --testdata         Use synthetic images to test the application
         --use-opengl-plot  Use OpenGL for plots (instead of matplotlib)
   * - :download:`imageStack.py <../../../examples/imageStack.py>`
     - .. image:: img/imageStack.png
         :width: 150px
     - Simple example for using the ImageStack.

       In this example we want to display images from different source: .h5, .edf
       and .npy files.

       To do so we simple reimplement the thread managing the loading of data.


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
         :width: 150px
     - This script shows how to create a minimalistic
       :class:`~silx.gui.plot.actions.PlotAction` that clear the plot.

       This illustrates how to add more buttons in a plot widget toolbar.
   * - :download:`shiftPlotAction.py <../../../examples/shiftPlotAction.py>`
     - .. image:: img/shiftPlotAction.png
         :width: 150px
     - This script is a simple (trivial) example of how to create a :class:`~silx.gui.plot.PlotWindow`,
       create a custom :class:`~silx.gui.plot.actions.PlotAction` and add it to the toolbar.

       The action simply shifts the selected curve up by 1 unit by adding 1 to each
       value of y.
   * - :download:`fftPlotAction.py <../../../examples/fftPlotAction.py>`,
       :download:`fft.png <../../../examples/fft.png>`
     - .. image:: img/fftPlotAction.png
         :width: 150px
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

Sample code that adds specific tools or functions to :class:`~silx.gui.plot.PlotWidget`.

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`plotWidget.py <../../../examples/plotWidget.py>`
     - .. image:: img/plotWidget.png
         :width: 150px
     - This script shows how to create a custom window around a PlotWidget.

       It subclasses :class:`QMainWindow`, uses a :class:`~silx.gui.plot.PlotWidget`
       as its central widget and adds toolbars and a colorbar by using pluggable widgets:

       - :class:`~silx.gui.plot.PlotWidget` from :mod:`silx.gui.plot`
       - QToolBar from :mod:`silx.gui.plot.tools`
       - QAction from :mod:`silx.gui.plot.actions`
       - QToolButton from :mod:`silx.gui.plot.PlotToolButtons`
       - :class:`silx.gui.plot.ColorBar.ColorBarWidget`
   * - :download:`plotItemsSelector.py <../../../examples/plotItemsSelector.py>`
     - .. image:: img/plotItemsSelector.png
         :width: 150px
     - This example illustrates how to use a :class:`ItemsSelectionDialog` widget
       associated with a :class:`~silx.gui.plot.PlotWidget`
   * - :download:`plotInteractiveImageROI.py <../../../examples/plotInteractiveImageROI.py>`
     - .. image:: img/plotInteractiveImageROI.png
         :width: 150px
     - This script illustrates image ROI selection in a :class:`~silx.gui.plot.PlotWidget`

       It uses :class:`~silx.gui.plot.tools.roi.RegionOfInterestManager` and
       :class:`~silx.gui.plot.tools.roi.RegionOfInterestTableWidget` to handle the
       interactive selection and to display the list of selected ROIs.
   * - :download:`printPreview.py <../../../examples/printPreview.py>`
     - .. image:: img/printPreview.png
         :width: 150px
     - This script illustrates how to add a print preview tool button to any plot
       widget inheriting :class:`~silx.gui.plot.PlotWidget`.

       Three plot widgets are instantiated. One of them uses a standalone
       :class:`~silx.gui.plot.PrintPreviewToolButton.PrintPreviewToolButton`,
       while the other two use a
       :class:`~silx.gui.plot.PrintPreviewToolButton.SingletonPrintPreviewToolButton`
       which allows them to send their content to the same print preview page.
   * - :download:`scatterMask.py <../../../examples/scatterMask.py>`
     - .. image:: img/scatterMask.png
         :width: 150px
     - This example demonstrates how to use ScatterMaskToolsWidget
       and NamedScatterAlphaSlider with a PlotWidget.
   * - :download:`plotCurveLegendWidget.py <../../../examples/plotCurveLegendWidget.py>`
     - .. image:: img/plotCurveLegendWidget.png
         :width: 150px
     - This example illustrates the use of :class:`CurveLegendsWidget`.

       :class:`CurveLegendsWidget` display curves style and legend currently visible
       in a :class:`~silx.gui.plot.PlotWidget`
   * - :download:`plotStats.py <../../../examples/plotStats.py>`
     - .. image:: img/plotStats.png
         :width: 150px
     - This script is a simple example of how to add your own statistic to a
       :class:`~silx.gui.plot.statsWidget.StatsWidget` from customs
       :class:`~silx.gui.plot.stats.Stats` and display it.

       On this example we will:

          - show sum of values for each type
          - compute curve integrals (only for 'curve').
          - compute center of mass for all possible items

       .. note:: for now the possible types manged by the Stats are ('curve', 'image',
                 'scatter' and 'histogram')
   * - :download:`plotROIStats.py <../../../examples/plotROIStats.py>`
     - .. image:: img/plotROIStats.png
         :width: 150px
     - This script is a simple example of how to display statistics on a specific
       region of interest.

       An example on how to define your own statistic is given in the 'plotStats.py'
       script.
   * - :download:`plotProfile.py <../../../examples/plotProfile.py>`
     - .. image:: img/plotProfile.png
         :width: 150px
     - Example illustrating the different profile tools.


:class:`~silx.gui.plot.PlotWidget` features
...........................................

Sample code that illustrates some functionalities of :class:`~silx.gui.plot.PlotWidget`.

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`plotContextMenu.py <../../../examples/plotContextMenu.py>`
     - .. image:: img/plotContextMenu.png
         :width: 150px
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
   * - :download:`plotLimits.py <../../../examples/plotLimits.py>`
     - .. image:: img/plotLimits.png
         :width: 150px
     - This script is an example to illustrate how to use axis synchronization
       tool.
   * - :download:`plotUpdateCurveFromThread.py <../../../examples/plotUpdateCurveFromThread.py>`
     - .. image:: img/plotUpdateCurveFromThread.png
         :width: 150px
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
         :width: 150px
     - This script illustrates the update of a :mod:`silx.gui.plot` widget from a thread.

       The problem is that plot and GUI methods should be called from the main thread.
       To safely update the plot from another thread, one need to execute the update
       asynchronously in the main thread.
       In this example, this is achieved with
       :func:`~silx.gui.utils.concurrent.submitToQtMainThread`.

       In this example a thread calls submitToQtMainThread to update the curve
       of a plot.
   * - :download:`syncaxis.py <../../../examples/syncaxis.py>`
     - .. image:: img/syncaxis.png
         :width: 150px
     - This script is an example to illustrate how to use axis synchronization
       tool.
   * - :download:`compositeline.py <../../../examples/compositeline.py>`
     - .. image:: img/compositeline.png
         :width: 150px
     - Example to show the use of markers to draw head and tail of lines.
   * - :download:`dropZones.py <../../../examples/dropZones.py>`
     - .. image:: img/dropZones.png
         :width: 150px
     - Example of drop zone supporting application/x-silx-uri.

       This example illustrates the support of drag&drop of silx URLs.
       It provides 2 URLs (corresponding to 2 datasets) that can be dragged to
       either a :class:`PlotWidget` or a QLable displaying the URL information.
   * - :download:`exampleBaseline.py <../../../examples/exampleBaseline.py>`
     - .. image:: img/exampleBaseline.png
         :width: 150px
     - This example illustrates some usage possible with the baseline parameter
   * - :download:`syncPlotLocation.py <../../../examples/syncPlotLocation.py>`
     - .. image:: img/syncPlotLocation.png
         :width: 150px
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
   * - :download:`plot3dSceneWindow.py <../../../examples/plot3dSceneWindow.py>`
     - .. image:: img/plot3dSceneWindow.png
         :width: 150px
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
   * - :download:`plot3dUpdateScatterFromThread.py <../../../examples/plot3dUpdateScatterFromThread.py>`
     - .. image:: img/plot3dUpdateScatterFromThread.png
         :width: 150px
     - This script illustrates the update of a
       :class:`~silx.gui.plot3d.SceneWindow.SceneWindow` widget from a thread.

       The problem is that GUI methods should be called from the main thread.
       To safely update the scene from another thread, one need to execute the update
       asynchronously in the main thread.
       In this example, this is achieved with
       :func:`~silx.gui.utils.concurrent.submitToQtMainThread`.

       In this example a thread calls submitToQtMainThread to append data to a 3D scatter.
   * - :download:`plot3dContextMenu.py <../../../examples/plot3dContextMenu.py>`
     - .. image:: img/plot3dContextMenu.png
         :width: 150px
     - This script adds a context menu to a :class:`silx.gui.plot3d.ScalarFieldView`.

       This is done by adding a custom context menu to the :class:`Plot3DWidget`:

       - set the context menu policy to Qt.CustomContextMenu.
       - connect to the customContextMenuRequested signal.

       For more information on context menus, see Qt documentation.
   * - :download:`viewer3DVolume.py <../../../examples/viewer3DVolume.py>`
     - .. image:: img/viewer3DVolume.png
         :width: 150px
     - This script illustrates the use of :class:`silx.gui.plot3d.ScalarFieldView`.

       It loads a 3D scalar data set from a file and displays iso-surfaces and
       an interactive cutting plane.
       It can also be started without providing a file.


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


:mod:`silx.image` sample code
+++++++++++++++++++++++++++++

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`findContours.py <../../../examples/findContours.py>`
     - .. image:: img/findContours.png
         :width: 150px
     - Find contours examples

       .. note:: This module has an optional dependency with sci-kit image library.
          You might need to install it if you don't already have it.

:mod:`silx.app` sample code
+++++++++++++++++++++++++++

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`customSilxView.py <../../../examples/customSilxView.py>`
     - .. image:: img/customSilxView.png
         :width: 150px
     - Sample code illustrating how to custom silx view into another application.
