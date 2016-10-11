Change Log
==========

Unreleased
----------

0.3.0: 2016/10/12
-----------------

 * Added OpenCL management
 * Added isosurface marching cubes
 * Added sift algorithm for image alignement
 * Added octaveh5 module to insure communication between octave and python using HDF5 file
 * Added silx.utils module containing weakref and html-escape
 * Added silx.sx for flat import (helper for interactive shell)
 * Added HDF5 load API (supporting Spec files) to silx.io.utils module
 * Added SpecFile support for multiple MCA headers
 * Added HDF5 TreeView
 * Added FitManager and FitWidget to the silx.math.fit module
 * Added ThreadPoolPushButton to silx.gui.widgets
 * Added getDataRange function to plot widget
 * Added loadUi, Slot and Property to qt.py
 * Added SVG icons and support
 * Added examples for plot actions, HDF5 widget, helper widgets, converter from Spec to HDF5
 * Added tutorials for plot actions, spech5, spectoh5, sift and fitmanager
 * Improve right axis support for plot widget
 * Improve mask tool
 * Refactoring widgets constructor: first argument is now the parent widget
 * Change plot documentation and add missing module to the documentation


0.2.0: 2016/07/12
-----------------

 * Added bilinear interpolator and line-profile for images to silx.image
 * Added Levenberg-Marquardt least-square fitting algorithm to silx.math.fit
 * Histogramnd changed to become a class rather than a function, API and return values changed
 * Added HistogramndLut, using a lookup table to bin data onto a regular grid for several sets of
   data sharing the same coordinates
 * Added legend widget and bottom toolbar to PlotWindow
 * Added a line-profile toolbar to PlotWindow
 * Added ImageView widget with side histograms and profile toolbar
 * Added IPython console widget, to be started from PlotWindow toolbar
 * Added Plot1D widget for curves and Plot2D widget for images
 * Added ROI widget for curves in PlotWindow
 * Added a mask widget and toolbar to plot (2D)
 * Renamed silx.io.dicttoh5 to silx.io.dictdump
 * Added configuration dictionary dumping/loading to/from JSON and INI files in silx.io.configdict
 * Added specfile wrapper API compatible with legacy wrapper: silx.io.specfilewrapper
 * Transposed scan data in specfile module to have detector as first index
 * Set up nigthly build for sources package, debian packages (http://www.silx.org/pub/debian/)
   and documentation (http://www.silx.org/doc/)


0.1.0: 2016/04/14
-----------------

 * Added project build, documentation and test structure
 * Added continuous integration set-up for Travis-CI and Appveyor
 * Added Debian packaging support
 * Added SPEC file reader, SPEC file conversion to HDF5 in silx.io
 * Added histogramnd function in silx.math
 * Added 1D, 2D plot widget with a toolbar, refactored from PyMca PlotWindow in silx.gui.plot
