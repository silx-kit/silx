Change Log
==========

0.6.1: 2017/11/10
-----------------

 * Plot:

    * Fix logarithmic axes, broken by new matplotlib version 2.1.0.
    * Fix axis limits changed signal, not emitted with matplotlib 1.5.x and 2.0.x
      when the widget is resized.
    * Support displaying float128 data with matplotlib < 2.1.0.
    * Fix minor bugs with the crosshair cursor and the legends widget.

 * Update documentation on Zenodo DOI and Debian packages installation.
 * Fix the Debian packaging script for Ubuntu distributions.


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
