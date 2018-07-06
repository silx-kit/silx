Change Log
==========

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
