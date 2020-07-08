Change Log
==========

0.13.0: 2020/06/23
------------------

This version drops the support of Python 2.7 and Python <= 3.4.

* silx view application:

  * Added support of compound data (PR #2948)
  * Added `Close All` menu (PR #2963)
  * Added default title to plots (PR #2979, #2999)
  * Added a button to enable/disable file content sorting (PR #3132)
  * Added support of a `SILX_style` HDF5 attribute to provide axes and colormap scale (PR #3092)
  * Improved `HDF5TableView` information table to make text selectable and ease copy (PR #2903)
  * Fixes (PR #2881, #2902, #3083)

* `silx.gui`:

  * `silx.gui.colors.Colormap`:

    * Added mean+/-3std autoscale mode (PR #2877, #2900)
    * Added sqrt, arcsinh and gamma correction colormap normalizations (PR #3010, #3054, #3057, #3066, #3070, #3133)
    * Limit number of threads used for computing the colormap (PR #3073)
    * Reordered colormaps (PR #3137)

  * `silx.gui.dialog.ColormapDialog`: Improved widget (PR #2874, #2915, #2924, #2954, #3136)
  * `silx.gui.plot`:

    * Major rework/extension of the regions of interest (ROI) (PR #3007, #3008, #3018, #3020, #3022, #3026, #3029, #3044, #3045, #3055, #3059, #3074, #3076, #3078, #3079, #3081, #3131)
    * Major rework/extension of the profile tools (PR #2933, #2980, #2988, #3004, #3011, #3037, #3048, #3058, #3084, #3088, #3095, #3097)
    * Added `silx.gui.plot.ImageStack` widget (PR #2480)
    * Added support of scatter in `PixelIntensitiesHistoAction` (PR #3089, #3107)
    * Added auto update of `FitAction` fitted data and range (PR #2960, #2961, #2969, #2981)
    * Improved mask tools (PR #2986)
    * Fixed `PlotWindow` (PR #2965) and `MaskToolsWidget` (PR #3125)

    * `silx.gui.plot.PlotWidget`:

      * Changed behaviour of `PlotWidget.addItem` and `PlotWidget.removeItem` to handle object items (previous behavior deprecated, not removed) and added `PlotWidget.addShape` method to add `Shape` items (PR #2873, #2904, #2919, #2925, #3120)
      * Added support of uint16 RGBA images (PR #2889)
      * Improved interaction (PR #2909, #3014, #3033)
      * Fixed `PlotWidget` (PR #2884, #2901, #2970, #3002)
      * Fixed and cleaned-up backends (PR #2887, #2910, #2913, #2957, #2964, #2984, #2991, #3023, #3064, #3135)

    * `silx.gui.plot.items`:

      * Added `sigDragStarted` and `sigDragFinished` signals to marker items and `sigEditingStarted` and `sigEditingFinished` signals to region of interest items (PR #2754)
      * Added `XAxisExtent` and `YAxisExtent` items in `silx.gui.plot.items` to control the plot data extent (PR #2932)
      * Added `ImageStack` item (PR #2994)
      * Added `Scatter` item histogram visualization mode (PR #2912, #2923)
      * Added `isDragged` method to marker items (PR #3000)
      * Improved performance of colormapped items by caching data min/max (PR #2876, #2886)
      * Improved `Scatter` item regular grid (PR #2918) and irregular grid (PR #3108) visualizations

  * `silx.gui.qt`:

    * Changed behavior of `QObject` multiple-inheritance (PR #3052)
    * Limit `silxGlobalThreadPool` function to use 4 threads maximum (PR #3072)

  * `silx.gui.utils.glutils`: Added `isOpenGLAvailable` to check the availability of OpenGL (PR #2878)
  * `silx.gui.widgets`:

    * Added `ElidedLabel` widget (PR #3110, #3111)
    * Fixed `LegendIconWidget` (PR #3112)

* `silx.io`:

  * Added support of signal dataset name-based errors to NXdata (PR #2976)
  * Added `dicttonx` function and support of HDF5 attibutes in `dicttoh5` function (PR #3013, #3017, #3031, #3093)
  * Fixed `url.DataUrl.path` (PR #2973)

* `silx.opencl`:

  * Fixed issue with Python 3.8 (PR #3036)
  * Disable textures for Nvidia Fermi GPUs for `convolution` (PR #3101)

* Miscellaneous:

  * Requires fabio >= 0.9 (PR #2937)
  * Fixed compatibility with h5py<v2.9 (PR #3024), cython 3 (PR #3034)
  * Avoid deprecation warnings (PR #3104) from Python 3.7 (PR #3012), Python 3.8 (PR #2891, #2934, #2989, #2993, #3127), h5py (PR #2854, #2893), matplotlib (PR #2890), fabio (PR #2930) and numpy (PR #3129)
  * Use `numpy.errstate` to ignore warnings rather than the `warnings` module (PR #2920)

* Build, documentation and tests:

  * Dropped Python2 support (PR #3119, #3140) and removed Python 2 tests and packaging (PR #2838, #2917)
  * Added debian 11/Ubuntu 20.04 packaging (PR #2875)
  * Improved test environment (PR #2870, #2949, #2995, #3009, #3061, #3086, #3087, #3122), documentation (PR #2872, #2894, #2937, #2987, #3042, #3053, #3068, #3091, #3103, #3115) and sample code (PR #2978, #3130, #3138)
  * Fixed Windows "fat binary" build (PR #2971)


0.12.0: 2020/01/09
------------------

Python 2.7 is no longer officially supported (even if tests pass and most of the library should work).

* silx view application:

  * Added: keep the same axes selection when changing dataset except for the stack view (PR #2701, #2780)
  * Added a Description column in the browsing tree to display NeXus title or name (PR #2804)
  * Added support of URL as filename (PR #2750)
  * Behavior changed: no longer lock HDF5 files by default, can be changed with `--hdf5-file-locking` option (PR #2861)

* `silx.gui`:

  * `silx.gui.plot`:

    * Added scatter plot regular and irregular grid visualization mode (PR #2810, #2815, #2820, #2824, #2831)
    * Added `baseline` argument to `PlotWidget` `addCurve` and `addHistogram` methods (PR #2715)
    * Added right axis support to `PlotWidget` marker items (PR #2744)
    * Added `BoundingRect` `PlotWidget` item (PR #2823)
    * Added more markers to `PlotWidget` items using symbols (PR #2792)
    * Improved and fixed `PlotWidget` and backends rendering and picking to guarantee rendering order of items (PR #2602, #2694, #2726, #2728, #2730, #2731, #2732, #2734, #2746, #2800, #2822, #2829, #2851, #2853)
    * Improved `RegionOfInterest`: Added `sigItemChanged` signal, renamed `get|setLabel` to `get|setName` (PR #2684, #2729, #2794, #2803, #2860)
    * Improved `StackView`: Allow to save dataset to HDF5 (PR #2813)

  * `silx.gui.plot3d`:

    * Added colormapped isosurface display to `ComplexField3D` (PR #2675)

  * Miscellaneous:

    * Added `cividis` colormap (PR #2763)
    * Added `silx.gui.widgets.ColormapNameComboBox` widget (PR #2814)
    * Added `silx.gui.widgets.LegendIconWidget` widget (PR #2783)
    * Added `silx.gui.utils.blockSignals` context manager (PR #2697, #2702)
    * Added `silx.gui.utils.qtutils.getQEventName` function (PR #2725)
    * Added `silx.gui.colors.asQColor` function (PR #2753)
    * Minor fixes (PR #2662, #2667, #2674, #2719, #2724, #2747, #2757, #2760, #2766, #2789, #2798, #2799, #2805, #2811, #2832, #2834, #2839, #2849, #2852, #2857, #2864, #2867)

* `silx.opencl`:

  * Added `silx.opencl.sparse.CSR` with support of different data types (PR #2671)
  * Improved support of different platforms like PoCL (PR #2669, #2698, #2806)
  * Moved non-OpenCL related utilities to `silx.opencl.utils` module (PR #2782)
  * Fixed `silx.opencl.sinofilter.SinoFilter` to avoid importing scikit-cuda (PR #2721)
  * Fixed kernel garbage collection (PR #2708)
  * Fixed `silx.opencl.convolution.Convolution` (PR #2781)

* `silx.math`/`silx.image`:

  * Added trilinear interpolator: `silx.math.interpolate.interp3d` (PR #2678)
  * Added `silx.image.utils.gaussian_kernel` function (PR #2782)
  * Improved `silx.image.shapes.Polygon` argument check (PR #2761)
  * Fixed and improved `silx.math.fft` with FFTW backend (PR #2751)
  * Fixed support of not finite data in fit manager (PR #2868)

* `silx.io`:

  * Added `asarray=True` argument to `silx.io.dictdump.h5todict` function (PR #2692, #2767)
  * Improved `silx.io.utils.DataUrl` (PR #2790)
  * Increased max number of motors in `specfile` (PR #2817)
  * Fixed data conversion when reading images with `fabio` (PR #2735)

* Build, documentation and tests:

  * Added `Cython` as a build dependency (PR #2795, #2807, #2808)
  * Added Debian 10 packaging (PR #2670, #2672, #2666, #2686, #2706)
  * Improved documentation (PR #2673, #2680, #2679, #2772, #2759, #2779, #2801, #2802, #2833, #2857, #2869)
  * Improved testing tools (PR #2704, #2796, #2818)
  * Improved `bootstrap.py` script (PR #2727, #2733)


0.11.0: 2019/07/03
------------------

 * Graphical user interface:

   * Plot:

     * Add sample code on how to update a plot3d widget from a thread
     * ScatterPlot: add the possibility to plot as a surface using Delaunay triangulation
     * ScatterView: add a tool button to change scatter visualization mode (ex. Solid)
     * (OpenGL backend) Fix memory leak when creating/deleting widgets in a loop


   * Plot3D:

     * Add an action to toggle plot3d's `PositionInfoWidget` picking.
     * Add a 3D complex field visualization: Complex3DField (also available from silx view)
     * Add a PositionInfoWidget and a tool button to toggle the picking mode to SceneWindow
     * Add the possibility to render the scene with linear fog.

   * `silx.gui.widgets`:

     * Fix ImageFileDialog selection for a cube with shape like `1,y,x`.

 * Miscellaneous:

    * Requires numpy version >= 1.12
    * HDF5 creator script
    * Support of Python 3.4 is dropped. Please upgrade to at least Python 3.5.
    * This is the last version to officially support Python 2.7.
    * The source code is Python 3.8 ready.
    * Improve PySide2 support. PyQt4 and PySide are deprecated.



0.10.0: 2019/02/19
------------------

 * Graphical user interface:

   * Plot:

    * Add support of foreground color
    * Fix plot background colors
    * Add tool to mask ellipse
    * StatsWidget:

     * Add support for plot3D widgets
     * Add a PyMca like widget

    * `Colormap`: Phase colormap is now editable
    * `ImageView`: Add ColorBarWidget
    * `PrintPreview`:

     * Add API to define 'comment' and 'title'
     * Fix resizing in PyQt5

    * Selection: Allow style definition
    * `ColormapDialog`: display 'values' plot in log if colormap uses log
    * Synchronize ColorBar with plot background colors
    * `CurvesROIWidget`: ROI is now an object.

   * Plot3D:

    * `SceneWidget`: add ColormapMesh item
    * Add compatibility with the StatsWidget to display statistic on 3D volumes.
    * Add `ScalarFieldView.get|setOuterScale`
    * Fix label update in param tree
    * Add `ColormapMesh` item to the `SceneWidget`

   * HDF5 tree:

    * Allow URI drop
    * Robustness of hdf5 tree with corrupted files

   * `silx.gui.widgets`:

    * Add URL selection table

 * Input/output:

   * Support compressed Fabio extensions
   * Add a function to create external dataset for .vol file

 * `silx view`:

    * Support 2D view for 3D NXData
    * Add a NXdata for complex images
    * Add a 3d scalar field view to the NXdata views zoo
    * Improve shortcuts, view loading
    * Improve silx view loading, shortcuts and sliders ergonomy
    * Support default attribute pointing to an NXdata at any group level

 * `silx convert`

    * Allow to use a filter id for compression

 * Math:

    * fft: multibackend fft

 * OpenCL:

    * Compute statistics on a numpy.ndarray
    * Backprojection:

     * Add sinogram filters (SinoFilter)
     * Input and/or output can be device arrays.

 * Miscellaneous:

    * End of PySide support (use PyQt5)
    * Last version supporting numpy 1.8.0. Next version will drop support for numpy < 1.12
    * Python 2.7 support will be dropped before end 2019. From version 0.11, a deprecation warning will be issued.
    * Remove some old deprecated methods/arguments
    * Set Cython language_level to 3


0.9.0: 2018/10/23
-----------------

 * Graphical user interface:

   * `silx.gui.widgets`:

     * Adds `RangeSlider` widget, a slider with 2 thumbs
     * Adds `CurveLegendsWidget` widget to display PlotWidget curve legends
       (as an alternative to `LegendSelector` widget)
     * Adds `FlowLayout` QWidget layout

   * Plot:

     * Adds `CompareImages` widget providing tools to compare 2 images
     * `ScatterView`: Adds alpha channel support
     * `MaskToolsWidget`: Adds load/save masks from/to HDF5 files

     * `PlotWidget`:

       * Adds `getItems` method, `sigItemAdded` and `sigItemAboutToBeRemoved` signals
       * Adds more options for active curve highlighting (see `get|setActiveCurveStyle` method)
       * Deprecates `get|setActiveCurveColor` methods
       * Adds `get|setActiveCurveSelectionMode` methods to change the behavior of active curve selection
       * Adds configurable line style and width to line markers
       * Fixes texture cache size in OpenGL backend

   * Plot3D:

     * Adds `SceneWidget.pickItems` method to retrieve the item and data at a given mouse position
     * Adds `PositionInfoWidget` widget to display data value at a given mouse position

   * `silx.gui.utils`:

     * Adds `image` module for QImage/numpy array conversion functions
     * Adds `testutils` helper module for writing Qt tests
       (previously available internally as `silx.gui.test.utils`)

   * Adds `silx.gui.qt.inspect` module to inspect Qt objects

 * Math:

   * Updates median filter with support for Not-a-Number and a 'constant' padding mode

 * `silx view`:

    * Fixes file synchronization
    * Adds a refresh button to synchronize file content

 * Dependencies:

   * Deprecates support of PySide Qt4 binding
     (We intend to drop official support of PySide in silx 0.10.0)
   * Deprecates support of PyQt4
   * Adds `h5py` and `fabio` as strong dependencies

 * Miscellaneous:

   * Adds `silx.examples` package to ship the example with the library

0.8.0: 2018/07/04
-----------------

 * Graphical user interface:

   * Plot:

     * Adds support of x-axis date/time ticks for time series display (see `silx.gui.plot.items.XAxis.setTickMode`)
     * Adds support of interactive authoring of regions of interest (see `silx.gui.plot.items.roi` and `silx.gui.plot.tools.roi`)
     * Adds `StatsWidget` widget for displaying statistics on data displayed in a `PlotWidget`
     * Adds `ScatterView` widget for displaying scatter plot with tools such as line profile and mask
     * Overcomes the limitation to float32 precision with the OpenGL backend
     * Splits plot toolbar is several reusable thematic toolbars

   * Plot3D: Adds `SceneWidget` items to display many cubes, cylinders or hexagonal prisms at once
   * Adds `silx.gui.utils` package with `submitToQtMainThread` for asynchronous execution of Qt-related functions
   * Adds Qt signals to `Hdf5TreeView` to manage HDF5 file life-cycle
   * Adds `GroupDialog` dialog to select a group in a HDF5 file
   * Improves colormap computation with a Cython/OpenMP implementation

   * Main API changes:

     * `Colormap` is now part of `silx.gui.colors`
     * `ColormapDialog` is now part of `silx.gui.dialogs`
     * `MaskToolsWidget.getSelectionMask` method now returns `None` if no image is selected
     * Clean-up `FrameBrowser` API

 * Image

   * Adds an optimized marching squares algorithm to compute many iso contours from the same image

 * Input/output:

   * Improves handling of empty Spec scans
   * Add an API to `NXdata` parser to get messages about malformed input data

 * `silx.sx`

   * Allows to use `silx.sx` in script as in Python interpreter
   * `sx.imshow` supports custom y-axis orientation using argument `origin=upper|lower`
   * Adds `sx.enable_gui()` to enable silx widgets in IPython notebooks

 * `silx convert`

   * Improves conversion from EDF file series to HDF5

 * `silx view`

   * Adds user preferences to restore colormap, plot backend, y-axis of plot image,...
   * Adds `--fresh` option to clean up user preferences at startup
   * Adds a widget to create custom viewable `NXdata` by combining different datasets
   * Supports `CTRL+C` shortcut in the terminal to close the application
   * Adds buttons to collapse/expand tree items
   * NXdata view now uses the `ScatterView` widget for scatters

 * Miscellaneous

   * Drops official support of Debian 7
   * Drops versions of IPython console widget before the `qtconsole` package
   * Fixes EDF file size written by `EdfFile` module with Python 3

0.7.0: 2018/02/27
-----------------

 * Input/output:

   * Priovides `silx.io.url.DataUrl` to parse supported links identifying
     group or dataset from files.
   * `silx.io.open` now supports h5pyd and silx custom URLs.
   * `silx.io.get_data` is provided to allow to reach a numpy array from silx.

 * OpenCL:

   * Provides an API to share memory between OpenCL tasks within the same device.
   * Provides CBF compression and decompression.
   * Simple processing on images (normalization, histogram).
   * Sift upgrade using memory sharing.

 * `silx.sx`:

   * Added `contour3d` function for displaying 3D isosurfaces.
   * Added `points3d` function for displaying  2D/3D scatter plots.
   * Added `ginput` function for interactive input of points on 1D/2D plots.

 * Graphic user interface:

   * Provides a file dialog to pick a dataset or a group from HDF5 files.
   * Provides a file dialog to pick an image from HDF5 files or multiframes formats.
   * The colormap dialog can now be used as non-modal.
   * `PlotWidget` can save the displayed data as a new `NXentry` of a HDF5 file.
   * `PlotWidget` exports displayed data as spec files using more digits.
   * Added new OpenGL-based 3D visualization widgets:

     * Supports 3D scalar field view 2D/3D scatter plots and images.
     * Provides an object oriented API similar to that of the 1D/2D plot.
     * Features a tree of parameters to edit visualized item's properties
       (e.g., transforms, colormap...)
     * Provides interactive panning of cut and clip planes.

   * Updates of `ScalarFieldView` widget:

     * Added support for a 3x3 transform matrix (to support non orthogonal axes)
     * Added support of an alternative interaction when `ctrl` is pressed
       (e.g., rotate by default and pan when ctrl/command key is pressed).
     * Added 2 sliders to control light direction in associated parameter tree view.

 * `silx view`:

   * Uses a single colormap to show any datasets.
   * The colormap dialog can stay opened while browsing the data.
   * The application is associated with some file types to be used to load files
     on Debian.
   * Provides a square amplitude display mode to visualize complex images.
   * Browsing an `NXentry` can display a default `NXdata`.
   * Added explanation when an `NXdata` is not displayable.
   * `NXdata` visualization can now show multiple curves (see `@auxiliary_signals`).
   * Supports older `NXdata` specification.

 * `silx convert`:

   * Added handling of file series as a single multiframe
   * Default behavior changes to avoid to add an extra group at the root,
     unless explicitly requested (see `--add-root-group`).
   * Writer uses now utf-8 text as default (NeXus specification).
   * EDF files containing MCA data are now interpreted as spectrum.

 * Miscellaneous:

   * Added `silx.utils.testutils` to share useful unittest functions with other
     projects.
   * Python 2 on Mac OS X is no longer tested.
   * Experimental support to PySide2.
   * If fabio is used, a version >= 0.6 is mandatory.

0.6.0: 2017/10/02
-----------------

 * OpenCl. Tomography. Implement a filtered back projection.
 * Add a *PrintPreview* widget and a *PrintPreviewToolButton* for *PlotWidget*.
 * Plot:

   * Add a context menu on right click.
   * Add a *ComplexImageView* widget.
   * Merged abstract *Plot* class with *PlotWidget* class.
   * Make colormap an object with signals (*sigChanged*)
   * Add a colorbar widget *silx.gui.plot.ColorBar*.
   * Make axis an object, allow axis synchronization between plots,
     allow adding constraints on axes limits.
   * Refactor plot actions, new sub-package *silx.gui.plot.actions*.
   * Add signals on *PlotWidget* items notifying updates.
   * Mask. Support loading of TIFF images.

 * Plot3d:

   * Rework toolbar and interaction to use only the left mouse button.
   * Support any colormap.

 * Hdf5TreeView:

   * Add an API to select a single tree node item (*setSelectedH5Node*)
   * Better support and display of types.
   * New column for displaying the kind of links.
   * Broken nodes are now selectable.

 * StackView. Add a *setTitleCallback* method.
 * Median filter. Add new modes (*reflect, mirror, shrink*) in addition to *nearest*.

 * IO:

   * Rename module *spectoh5* to *convert*. Add support for conversion of *fabio* formats.
   * Support NPZ format.
   * Support opening an URI (*silx.io.open(filename::path)*).
   * *Group* methods *.keys*, *.value* and *.items* now return lists in Python 2
     and iterators in Python 3.

 * Image. Add tomography utils: *phantomgenerator* to produce Shepp-Logan phantom, function to compute center of rotation (*calc_center_corr*, *calc_center_centroid*) and rescale the intensity of an image (*rescale_intensity*).

 * Commands:

   * *silx view*:

     * Add command line option *--use-opengl-plot*.
     * Add command line option *--debug*, to print dataset reading errors.
     * Support opening URI (*silx view filename::path*).

   * *silx convert*. New command line application to convert supported data files to HDF5.

 * Enable usage of *silx.resources* for other projects.
 * The *silx* license is now fully MIT.


0.5.0: 2017/05/12
-----------------

 * Adds OpenGL backend to 1D and 2D graphics
 * Adds Object Oriented plot API with Curve, Histogram, Image, ImageRgba and Scatter items.
 * Implements generic launcher (``silx view``)
 * NXdataViewer. Module providing NeXus NXdata support
 * Math/OpenCL. Implementation of median filter.
 * Plot. Implementation of ColorBar widget.
 * Plot. Visualization of complex data type.
 * Plot. Implementation of Scatter Plot Item supporting colormaps and masks.
 * Plot. StackView now supports axes calibration.
 * I/O. Supports SPEC files not having #F or #S as first line character.
 * I/O. Correctly exposes UB matrix when found in file.
 * ROIs. Simplification of API: setRois, getRois, calculateRois.
 * ROIs. Correction of calculation bug when the X-axis values were not ordered.
 * Sift. Moves package from ``silx.image`` to ``silx.opencl``.


0.4.0: 2017/02/01
-----------------

 * Adds plot3D package (include visualization of 3-dimensional scalar fields)
 * Adds data viewer (it can handle n-dimensional data)
 * Adds StackView (ex. Visualization of stack of images)
 * Adds depth profile calculation (ex. extract profile of a stack of images)
 * Adds periodic table widget
 * Adds ArrayTableWidget
 * Adds pixel intensity histogram action
 * Adds histogram parameter to addCurve
 * Refactoring. Create silx.gui.data (include widgets for data)
 * Refactoring. Rename utils.load as silx.io.open
 * Changes active curve behavior in Plot. No default active curve is set by default
 * Fit Action. Add polynomial functions and background customization
 * PlotWindow. Provide API to access toolbar actions
 * Handle SPEC, HDF5 and image formats through an unified API
 * hdf5widget example. Inspect and visualize any datasets
 * Improves mask tool
 * Deprecates PlotWindow dock widgets attributes in favor of getter methods


0.3.0: 2016/10/12
-----------------

 * Adds OpenCL management
 * Adds isosurface marching cubes
 * Adds sift algorithm for image alignement
 * Adds octaveh5 module to insure communication between octave and python using HDF5 file
 * Adds silx.utils module containing weakref and html-escape
 * Adds silx.sx for flat import (helper for interactive shell)
 * Adds HDF5 load API (supporting Spec files) to silx.io.utils module
 * Adds SpecFile support for multiple MCA headers
 * Adds HDF5 TreeView
 * Adds FitManager to silx.math.fit and FitWidget to silx.gui.fit
 * Adds ThreadPoolPushButton to silx.gui.widgets
 * Adds getDataRange function to plot widget
 * Adds loadUi, Slot and Property to qt.py
 * Adds SVG icons and support
 * Adds examples for plot actions, HDF5 widget, helper widgets, converter from Spec to HDF5
 * Adds tutorials for plot actions, spech5, spectoh5, sift and fitmanager
 * Improves right axis support for plot widget
 * Improves mask tool
 * Refactors widgets constructor: first argument is now the parent widget
 * Changes plot documentation and add missing module to the documentation


0.2.0: 2016/07/12
-----------------

 * Adds bilinear interpolator and line-profile for images to silx.image
 * Adds Levenberg-Marquardt least-square fitting algorithm to silx.math.fit
 * Histogramnd changed to become a class rather than a function, API and return values changed
 * Adds HistogramndLut, using a lookup table to bin data onto a regular grid for several sets of
   data sharing the same coordinates
 * Adds legend widget and bottom toolbar to PlotWindow
 * Adds a line-profile toolbar to PlotWindow
 * Adds ImageView widget with side histograms and profile toolbar
 * Adds IPython console widget, to be started from PlotWindow toolbar
 * Adds Plot1D widget for curves and Plot2D widget for images
 * Adds ROI widget for curves in PlotWindow
 * Adds a mask widget and toolbar to plot (2D)
 * Renames silx.io.dicttoh5 to silx.io.dictdump
 * Adds configuration dictionary dumping/loading to/from JSON and INI files in silx.io.configdict
 * Adds specfile wrapper API compatible with legacy wrapper: silx.io.specfilewrapper
 * Transposes scan data in specfile module to have detector as first index
 * Set up nigthly build for sources package, debian packages (http://www.silx.org/pub/debian/)
   and documentation (http://www.silx.org/doc/)


0.1.0: 2016/04/14
-----------------

 * Adds project build, documentation and test structure
 * Adds continuous integration set-up for Travis-CI and Appveyor
 * Adds Debian packaging support
 * Adds SPEC file reader, SPEC file conversion to HDF5 in silx.io
 * Adds histogramnd function in silx.math
 * Adds 1D, 2D plot widget with a toolbar, refactored from PyMca PlotWindow in silx.gui.plot
