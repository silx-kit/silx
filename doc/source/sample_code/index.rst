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

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`animatedicons.py <../../../examples/animatedicons.py>`
     - .. image:: img/animatedicons.png
         :height: 150px
         :align: center
     - Display available project icons using Qt.
   * - :download:`customHdf5TreeModel.py <../../../examples/customHdf5TreeModel.py>`
     - .. image:: img/customHdf5TreeModel.png
         :height: 150px
         :align: center
     - Qt Hdf5 widget examples
   * - :download:`hdf5widget.py <../../../examples/hdf5widget.py>`
     - .. image:: img/hdf5widget.png
         :height: 150px
         :align: center
     - Qt Hdf5 widget examples

       .. note:: This module has a dependency on the `h5py <http://www.h5py.org/>`_
           library, which is not a mandatory dependency for `silx`. You might need
           to install it if you don't already have it.
   * - :download:`icons.py <../../../examples/icons.py>`
     - .. image:: img/icons.png
         :height: 150px
         :align: center
     - Display available project icons using Qt.
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

       - :class:WaitingPushButton: A button with a progress-like waiting animated icon

:mod:`silx.gui.plot` sample code
++++++++++++++++++++++++++++++++

.. list-table::
   :widths: 1 1 4
   :header-rows: 1

   * - Source
     - Screenshot
     - Description
   * - :download:`fftPlotAction.py <../../../examples/fftPlotAction.py>`,
       :download:`fft.png <../../../examples/fft.png>`
     - .. image:: img/fftPlotAction.png
         :height: 150px
         :align: center
     - This script is a simple example of how to create a PlotWindow with a custom
       PlotAction added to the toolbar.

       The action computes the FFT of all curves and plots their amplitude spectrum.
       It also performs the reverse transform.

       This example illustrates:
          - how to create a checkable action
          - how to store user info with a curve in a PlotWindow
          - how to modify the graph title and axes labels
          - how to add your own icon as a PNG file

       See shiftPlotAction.py for a simpler example with more basic comments.
   * - :download:`imageview.py <../../../examples/imageview.py>`
     - .. image:: img/imageview.png
         :height: 150px
         :align: center
     - Example to show the use of `ImageView` widget. It can be used to open an EDF
       or TIFF file from the shell command line.

       To view an image file with the current installed silx library:
       ``python examples/imageview.py <file to open>``
       To get help:
       ``python examples/imageview.py -h``

       For developers with a git clone you can use it with the bootstrap
       To view an image file with the current installed silx library:

       ``./bootstrap.py python examples/imageview.py <file to open>``
   * - :download:`plotContextMenu.py <../../../examples/plotContextMenu.py>`
     - .. image:: img/plotContextMenu.png
         :height: 150px
         :align: center
     - This script illustrates the addition of a context menu to a PlotWidget.

       This is done by adding a custom context menu to the plot area of PlotWidget:

       - set the context menu policy of the plot area to Qt.CustomContextMenu.
       - connect to the plot area customContextMenuRequested signal.

       The same method works with PlotWindow, Plot1D and Plot2D widgets as they
       inherit from PlotWidget.

       For more information on context menus, see Qt documentation.
   * - :download:`plotItemsSelector.py <../../../examples/plotItemsSelector.py>`
     - .. image:: img/plotItemsSelector.png
         :height: 150px
         :align: center
     - This example illustrates how to use a :class:`ItemsSelectionDialog` widget
       associated with a :class:`PlotWidget`.
   * - :download:`plotLimits.py <../../../examples/plotLimits.py>`
     - .. image:: img/plotLimits.png
         :height: 150px
         :align: center
     - This script is an example to illustrate how to use axis synchronization
       tool.
   * - :download:`plotUpdateFromThread.py <../../../examples/plotUpdateFromThread.py>`
     - .. image:: img/plotUpdateFromThread.png
         :height: 150px
         :align: center
     - This script illustrates the update of a silx.gui.plot widget from a thread.

       The problem is that plot and GUI methods should be called from the main thread.
       To safely update the plot from another thread, one need to make the update
       asynchronously from the main thread.
       In this example, this is achieved through a Qt signal.

       In this example we create a subclass of :class:`silx.gui.plot.Plot1D`
       that adds a thread-safe method to add curves:
       :meth:`ThreadSafePlot1D.addCurveThreadSafe`.
       This thread-safe method is then called from a thread to update the plot.
   * - :download:`plotWidget.py <../../../examples/plotWidget.py>`
     - .. image:: img/plotWidget.png
         :height: 150px
         :align: center
     - This script shows how to subclass :class:`PlotWidget` to tune its tools.

       It subclasses a :class:`silx.gui.plot.PlotWidget` and adds toolbars and
       a colorbar by using pluggable widgets:

       - QAction from :mod:`silx.gui.plot.actions`
       - QToolButton from :mod:`silx.gui.plot.PlotToolButtons`
       - QToolBar from :mod:`silx.gui.plot.PlotTools`
       - :class:`ColorBarWidget` from :mod:`silx.gui.plot.ColorBar`.
   * - :download:`printPreview.py <../../../examples/printPreview.py>`
     - .. image:: img/printPreview.png
         :height: 150px
         :align: center
     - This script illustrates how to add a print preview tool button to any plot
       widget inheriting :class:`PlotWidget`.

       Three plot widgets are instantiated. One of them uses a standalone
       :class:`PrintPreviewToolButton`, while the other two use a
       :class:`SingletonPrintPreviewToolButton` which allows them to send their content
       to the same print preview page.
   * - :download:`scatterMask.py <../../../examples/scatterMask.py>`
     - .. image:: img/scatterMask.png
         :height: 150px
         :align: center
     - This example demonstrates how to use ScatterMaskToolsWidget
       and NamedScatterAlphaSlider with a PlotWidget.
   * - :download:`shiftPlotAction.py <../../../examples/shiftPlotAction.py>`
     - .. image:: img/shiftPlotAction.png
         :height: 150px
         :align: center
     - This script is a simple (trivial) example of how to create a PlotWindow,
       create a custom :class:`PlotAction` and add it to the toolbar.

       The action simply shifts the selected curve up by 1 unit by adding 1 to each
       value of y.
   * - :download:`stackView.py <../../../examples/stackView.py>`
     - .. image:: img/stackView.png
         :height: 150px
         :align: center
     - This script is a simple example to illustrate how to use the StackView
       widget.
   * - :download:`syncaxis.py <../../../examples/syncaxis.py>`
     - .. image:: img/syncaxis.png
         :height: 150px
         :align: center
     - This script is an example to illustrate how to use axis synchronization
       tool.

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
     - This script converts a supported data file (SPEC, EDF...) to a HDF5 file.

       By default, it creates a new output file or fails if the output file given
       on the command line already exist, but the user can choose to overwrite
       an existing file, or append data to an existing HDF5 file.

       In case of appending data to HDF5 files, the user can choose between ignoring
       input data if a corresponding dataset already exists in the output file, or
       overwriting the existing dataset.

       By default, new scans are written to the root (/) of the HDF5 file, but it is
       possible to specify a different target path.
