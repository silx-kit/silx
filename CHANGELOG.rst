Change Log
==========

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
