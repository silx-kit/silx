Release Notes
=============

2.0.0: 2024/01/30
-----------------

This version of `silx` supports Python 3.7 to 3.12.
This is the last version of `silx` supporting Python 3.7.
The `silx.gui` package supports `PySide6`, `PyQt6` and `PyQt5` (`PySide2` is no longer supported).

**Breaking API change**: `silx.gui.plot.PlotWidget`'s add methods (i.e., `addCurve`, `addImage`, ...) returns the plot item object instance instead of its legend (https://github.com/silx-kit/silx/pull/3996).

silx applications
.................

* Added `silx compare`, a dedicated application to compare images (PR #3788, #3827, #3884, #3943, #3944)
* `silx view`:

  * Added `--slices` option (PR #3860)
  * Added supports for data URL containing "**" to match multiple sub groups (PR #3795)
  * Added keyboard shortcuts for open/close all (PR #3863)
  * Improved: Stopped displaying a message box for each error (PR #3955)
  * Improved: Use matplotlib if OpenGL is not available (PR #3905)
  * Fixed support of NXData image with 0-length axis (PR #3768)
  * Fixed setting focus at startup when opening a dataset (PR #3953)

* `silx.app.utils`: Refactored (PR #3811)

silx.io
.......

* `silx.io.dictdump`:

  * Removed `dicttoh5`'s `overwrite_data` argument (PR #3806)
  * Improved error message for `dicttoh5` with non-serializable data (PR #3937)
  * Fixed `h5todict` errors argument issue (PR #3749) and fixed-length string issue (PR #3748)

* `silx.io.h5py_utils`:

  * Fixed `retry` (PR #3775)
  * Fixed: Do not call multiprocessing module in frozen binaries (PR #3984)

* `silx.io.nxdata.parser`: Fixed `NXdata` validation (PR #3782)

* `silx.io.url`:

  * Added support of URLs with slices to `DataUrl` (PR #3821)
  * Added typings for `DataUrl` (PR #3968)
  * Fixed `DataUrl`: Allow `file_path` to be `None` (PR #4051)

* `silx.io.utils`:

  * Added support of bare file to `get_data` with `check_schemas=True` argument (PR #3859)
  * Improved: `open` do not lock the file (PR #3939)
  * Fixed small/big endian support in test (PR #3873)
  * Fixed `h5py_read_dataset` support of empty arrays (PR #4052)

silx.math
.........

* Fixed several function docstrings (PR #3774)
* `silx.math.colormap`: Added `normalize` function to normalize to `uint8` (PR #3785)
* `silx.math.fit`: Added split pseudo-voigt with split lorentzian fraction (PR #3902)

silx.opencl
...........

* Fixed regression with PoCL and order of floating point operations (PR #3935)
* Fixed: skip test on broken platform (PR #3809)
* Fixed: prevent crash at init when loading silx with PortableCL + Cuda devices (PR #3924)
* `silx.opencl.atomic`: Added new checking for atomic32 and atomic64 operation (PR #3855)
* `silx.opencl.backprojection`: Removed deprecated `fourier_filter` function (PR #3816)
* `silx.opencl.codec`:

  * Added `bitshuffle_lz4`` decompression (PR #3714)
  * Added support of pyopencl's `Buffer` and `Array` to `BitshuffleLz4.decompress` (PR #3787)

* `silx.opencl.common``:

  * Removed `OpenCL.create_context` `useFp64` argument (PR #3801)
  * Reworked initialisation of the module (PR #3903)
  * Updated: Defer to pyopencl the interpretation of `PYOPENCL_CTX` (PR #3933)

* `silx.opencl.convolution`: Removed Python 2 compatible code (PR #3818)

silx.third_party
................

* Removed copy of scipy's Delaunay from third_party (PR #3808)
* Deprecated `EdfFile` and `TiffIO` (PR #3841)

silx.gui
........

* `silx.gui`:

  * Added support for `QT_API` environment variable (PR #3981)
  * Added a warning about pyOpenGL and Qt compatibility (PR #3738)
  * Added some Python typing (PR #3957)
  * Removed support of  PySide6<6.4 (PR #3872)
  * Improved `qWidgetFactory` test fixture (PR #4009)
  * Fixed support of PySide 6.4 enums (PR #3737)
  * Fixed support of PyQt6 (PR #3960, #3966, #3989, #3999, #4003)
  * Fixed support of OpenGL with python3.12 and pyopengl <=3.1.7 (PR #3982)
  * Fixed OpenGL version parsing (PR #3733)

* `silx.gui.colors`:

  * Added indexed color names support to `rgba` (PR #3836, #3861)
  * Added typing (PR #3974)
  * `silx.gui.colors.rgba`: Changed from `AssertionError` to `ValueError` (PR #3864)
  * Improved: `Colormap.setVRange` raises an exception if the range is not finite (PR #3794)

* `silx.gui.constants`: Added: expose URI used to drag and drop `DataUrl` (PR #3796)

* `silx.gui.data`:

  * Fixed issue with hdf5 attributes string formatting (PR #3790)
  * `silx.gui.data.DataView`: Removed patch for pymca <v5.3.0 support (PR #3800)
  * `silx.gui.data.HDF5TableView`: Fixed virtual and external dataset information (PR #3717)
  * `silx.gui.data.RecordTableView`: Fixed issue with datasets with many rows failing to load due to incorrect variable type (PR #3926)

* `silx.gui.dialog`:

  * `silx.gui.dialog.ColormapDialog`:

    * Added `DisplayMode` to API by renaming `_DataInPlotMode` (PR #3964)
    * Fixed layout (PR #3792)
    * Fixed state when updating `Item` (PR #3833)
    * Fixed robustness of tools with item inheriting from `ImageBase` (PR #3858)

* `silx.gui.hdf5`:

  * Added `NXnote` to the list of describable classes (PR #3832)
  * Added tests for `H5Node` soft link to an external link issue (PR #3220)

* `silx.gui.qt`:

  * Updated PySide6 `loadUi` function (PR #3783)
  * Fixed Python>3.9 support (PR #3779)

* `silx.gui.plot`:

  * `silx.gui.plot.actions`: Added typings for `PlotAction` (PR #3941)
  * `silx.gui.plot.items`:

    * Added `Marker` item font configuration (PR #3956)
    * Added background color for markers and removed automatic background color (PR #4012)
    * Added `get|setLineGapColor` methods to `Curve` and `Histogram` (PR #3973)
    * Renamed `Shape.get|setLineBgColor` to `get|setLineGapColor` (PR #4001)
    * Deprecated `Curve` and `Image` sequence-like access (PR #3803)
    * Improved handling of data ndim and shape for image items (PR #3976)
    * Fixed: Removed `ImageDataAggregated` all-NaN warning (PR #3786)
    * Fixed `Shape` display with dashes and a background color (PR #3906)
    * `silx.gui.plot.items.roi`:

      * Added `RegionOfInterest`'s `getText` and `setText` methods (PR #3847)
      * Added `populateContextMenu` method to ROIs (PR #3891)
      * Added `ArcROI.getPositionRole` method (PR #3894)
      * Added ROIs base classes to documentation (PR #3839)
      * Removed deprecated methods `RegionOfInterest.get|setLabel` (PR #3810)
      * Improved `ArcROI``: Hide the handler instead of hidding the symbol (PR #3887)
      * Improved: highlighted RegionOfInterest takes priority for interactions (PR #3975)
      * Fixed ROI initialisation with parent (PR #4053)

  * `silx.gui.plot.ColorBar`: Fixed division by zero issue (PR #4013)
  * `silx.gui.plot.CompareImages`:

    * Added profile to compare image (PR #3845)
    * Improved consistency of autoscale (PR #3823)
    * Fixed the A-B visualization mode (PR #3856)

  * `silx.gui.plot.ImageStack`:

    * Added URL removal feature if the list is editable (PR #3913)
    * Fixed `ImageStack` handling of visible state (PR #3834)
    * Fixed issue (PR #4050)

  * `silx.gui.plot.ImageView`: Fixed histogram visibility (PR #3742)
  * `silx.gui.plot.PlotWidget`:

    * Breaking changes:

      * Changed `add*` methods return value to return the item instead of its legend (PR #3996)
      * Refactored management of items (PR #3986, #3988)

    * Added `margins` argument to `PlotWidget.setLimits` (PR #3828)
    * Added `Plotwidget.get|setDefaultColors` and updated default colors behavior (PR #3835)
    * Added `PlotWidget.sigBackendChanged` (PR #3890)
    * Added per-axis zoom (PR #3842, #3843)
    * Added support for 'other' kind of plot items (PR #3908)
    * Added support of matplotlib tight layout as an experimental feature (PR #3865)
    * Added support of line style defined as `(offset, (dash pattern))` (PR #4020)
    * Added support for indexed color names support (PR #3836)
    * Added sample script to check and compare backend features (PR #4031)
    * Changed curve default colors to matchthe one from matplotlib >=2.0 (PR #3853)
    * Changed curve highlighting to use by default a linewidth of 2 (PR #3854)
    * Changed plot axes tick labels behavior to use offsets (PR #4007)
    * Changed: use the default font from mpl (PR #4025)
    * Changed font management (PR #4047)
    * Improved rendering for OpenGL backend (PR #4002, #4015, #4023, #4034, #4038)
    * Fixed documentation (PR #3773)
    * Fixed mouse cursor update (PR #3904)
    * Fixed: do not reset zoom when changing axes scales (PR #3862, #3869)
    * Fixed: use `PlotWidget.get|setActiveScatter` instead of private method (PR #3987)
    * Fixed tick display of time series (PR #4000)
    * Fixed matplotlib marker without background (PR #4028)

  * `silx.gui.plot.PlotWindow`: Fixed display of zoom in/out actions (PR #3837)
  * `silx.gui.plot.RulerToolButton`: Added interactive plot measurement tool (PR #3959, #4005)
  * `silx.gui.plot.StackView`: Removed `setColormap` `autoscale` argument (PR #3805)

  * `silx.gui.plot.tools`:

    * `silx.gui.plot.tools.PositionInfo`: Fixed support of dark theme (PR #3965)
    * `silx.gui.plot.tools.profile`: Fixed concurrency issue with RGB profiles (PR #3846)
    * `silx.gui.plot.tools.roi.RegionOfInterestManager`:

      * Changed interaction mode for ROI creation (PR #3978)
      * Fixed display glitch (PR #3954)

* `silx.gui.plot3d`:

  * Updated font management (PR #4047)
  * Fixed deprecation warning (PR #4046)
  * `silx.gui.plot3d.ParamTreeView`:

    * Added typing and code cleanup (PR #3972)
    * Fixed Qt6 support (PR #3971)

* `silx.gui.utils.image`: Added support of `QImage.Format_Grayscale8` to `convertQImageToArray` (PR #3958)

* `silx.gui.widgets`:

  * `silx.gui.widgets.FloatEdit`:

    * Added `widgetResizable` feature (PR #4006)
    * Added typing and code cleanup (PR #3972)

  * `silx.gui.widgets.StackedProgressBar`: Added widget displaying more complex information progress information (PR #4008)
  * `silx.gui.plot.widgets.UrlList`: Added `UrlList` widget (PR #3913)
  * `silx.gui.widget.UrlSelectionTable`:

    * Improved look&feel and enabled drag&drop from `silx view` (PR #3797)
    * Updated: Split the URL column in 3 columns (PR #3822)
    * Fixed exception with interaction, renamed `get|setSelection` to `get|setUrlSelection` (PR #3791)

  * `silx.gui.widgets.WaiterOverlay`: Added a widget to display processing wheel on top of another widget (PR #3876)

* `silx.utils`:

  * `silx.utils.launcher`: Improved error message (PR #3793)
  * `silx.utils.retry`: Fixed: Lazy-loading of multiprocessing module (PR #3979)

Miscellaneous
.............

* Dependencies:

  * Removed support of Python 3.6 (PR #3712), `PySide2` (PR #3784) and `fabio`<0.9 (PR #3829)
  * Replaced `setuptools`'s `pkg_resources` with `packaging` as runtime dependency (PR #3910)
  * Fixed support of `pint` >= 0.20 (PR #3725), `cython` (PR #3770, #4033) and `PyInstaller` v6 (PR #4041)
  * Fixed deprecation warnings from `numpy`, `scipy`, `matplotlib` and `h5py` (PR #3741, #3777, #4045, #3980)

* Clean-up:

  * Removed features deprecated since <1.0.0 (PR #3798, #3799, #3802, #3804)
  * Removed remaining Python2 support (PR #3815, #3840, #3952)
  * Removed unused imports (PR #3814)
  * Replaced `OrderedDict` by `dict` (PR #3830)
  * Updated: Using `black` to format the code (PR #3991)
  * Fixed typo: 4 `"` quotes instead of 3. (PR #3838)

* Build:

  * Removed `setup.py` commands and options (PR #3831)
  * Removed constraint on `setuptools` version (PR #3909)
  * Updated build dependencies (PR #4035)
  * Fixed Windows fat binary filename and links (PR #4048)
  * Bump to 2.0.dev (PR #4014)

* Debian packaging:

  * Removed Debian 10 and 11 packaging (PR #4017)
  * Added Debian 12 packaging (PR #3812)
  * Added `pytest-mock` to Debian build dependencies (PR #3740)
  * Updated `build-deb.sh` (PR #4022, #3772) and `rules` (PR #3732)

* Updated documentation (PR #3765, #3899, #3970, #3994, #4037, #4036, #4039, #4042, #4055)
* Updated continuous integration (PR #3727, #3967, #3983)
* Fixed tests (PR #3722, #3723, #4043, #4044)

1.1.2: 2022/12/16
-----------------

This is a bug fix version:

* `silx.gui`:

  * Fixed support of `PySide` 6.4 enums (PR #3737, #3738)
  * Fixed OpenGL version parsing (PR #3733, #3738)

  * `silx.gui.plot`:

    * Fixed issue when `PlotWidget` has a size of 0 (PR #3736, #3738)
    * Fixed reset of interaction when closing mask tool (PR #3735, #3738)

* Miscellaneous: Updated Debian packaging (PR #3732, #3738)

1.1.1: 2022/11/30
-----------------

This is a bug fix version:

* Fixed support of `pint` >= 0.20 (PR #3725, #3728)
* Fixed continuous integration (PR #3727, #3728)
* Updated changelog (PR #3729)

1.1.0: 2022/10/27
-----------------

This is the last version of `silx` supporting Python 3.6 and `PySide2`.
Next version will require Python >= 3.7

This is the first version of `silx` supporting `PyQt6` (for `Qt6`).
Please note that `PyQt6` >= v6.3.0 is required.

* `silx view`:

  * Improved wildcard support in filename and data path (PR #3663)
  * Enabled plot grid by default for curve plots (PR #3667)
  * Fixed refresh for content opened as `file.h5::/path` (PR #3665)

* `silx.gui`:

  * Added support of `PyQt6` >= 6.3.0 (PR #3655)
  * Fixed `matplotlib`>=3.6.0 and `PySide6` support (PR #3639)
  * Fixed `PySide6` >=6.2.2 support (PR #3581)
  * Fixed Python 3.10 with `PyQt5` support (PR #3591)
  * Fixed crashes on exit when deriving `QApplication` (PR #3588)
  * Deprecated `PySide2` support (PR #3648)
  * Fixed: raise exception early when using a version of `PyQt5` incompatible with Python 3.10 (PR #3694)

  * `silx.gui.data`:

    * Updated: Do not keep aspect ratio in `NXdata` image views when axes `@units` are different (PR #3660)
    * `silx.gui.data.ArrayTableWidget`: Updated to edit without clearing previous data (PR #3686)
    * `silx.gui.data.DataViewer`: Added `selectionChanged` signal (PR #3646)
    * `silx.gui.data.Hdf5TableView`: Fixed for virtual datasets in the same file (PR #3572)

  * `silx.gui.dialog.ColormapDialog`: Updated layout and presentation of the features (PR #3671, #3609)

  * `silx.gui.hdf5`: Fixed issue with unsupported hdf5 entity (e.g. datatype) (PR #3643)

  * `silx.gui.plot`:

    * `silx.gui.plot.items`:

      * Added `BandROI` item (PR #3680, #3702, #3707)
      * Updated to take errorbars into account for item bounds (PR #3647)
      * Fixed `ArcROI` display (PR #3617)
      * Fixed error logs for scatter triangle visualisation with aligned points (PR #3644)

    * `silx.gui.plot.MaskToolsWidget`: Changed mask load/save default directory (PR #3704)

    * `silx.gui.plot.PlotWidget`:

      * Fixed time axis with values outside of supported range ]0, 10000[ years (PR 3597)
      * Fixed matplotlib backend replot failure under specific conditions (PR #3590)

      * `silx.gui.PlotWidget`'s OpenGL backend:

        * Added support of LaTex-like math syntax to text display (PR #3600)
        * Updated text label background to be less transparent (PR #3593)
        * Fixed dashed curve rendering (PR #3596)
        * Fixed image rendering of arcsinh colormap for uint8 and uint16 data (PR #3604)
        * Fixed rendering on some GPU (PR #3695)
        * Fixed empty text support (PR #3701)
	* Fixed: Avoid rendering when OpenGL version/extension check fails (PR #3707)

    * `silx.gui.plot.PlotWindow`: Fixed management of DockWidgets when showing/hiding the `PlotWindow` (PR #3631)
    * `silx.gui.plot.PositionInfo`: Improved picking (PR #3640)
    * `silx.gui.plot.StackView`: Updated toolbar implementation (PR #3697)

    * `silx.gui.plot.stats`: Fixed warnings when all data is outside the selected stats region (PR #3659)
    * `silx.gui.plot.tools`:

      * Added snapping to profile curve (PR #3640)
      * Fixed handling of `disconnect` exception (PR #3692)
      * Fixed label formatting for 2D profile tool (PR #3698)
      * Fixed computation of the slice profile (PR #3708)

  * `silx.gui.utils.glutils.isOpenGLAvailable`: Added possibility to check `AA_ShareOpenGLContexts` (PR #3688)
  * `silx.gui.widgets.ElidedLabel`: Fixed API inherited from `QLabel` (PR #3650, #3707)

* `silx.io`:

  * `silx.io.dictdump`:

    * Added "info" logs when an entity is not copied to the output HDF5 file `dicttoh5` (PR #3664)
    * Added support of `pint` in `dicttoh5` and `dicttonx` (PR #3683)

  * `silx.io.nxdata`:

    * Updated `get_default` to be more permissive and follow `@default` recursively (PR #3662)
    * Updated error dataset retrieval (PR #3657, #3672)

  * `silx.io.specfile`:

    * Fixed buffer overflow for too long motor or label (PR #3622)
    * Fixed missing data if there is a trailing space in the mca array (PR #3612)

  * `silx.io.utils.retry`: Added retry for generator functions (PR #3679)

* `silx.math`:

  * `silx.math.histogram`:

    * Added support of `uint16` weights for LUT histogram (PR #3670)
    * Fixed `Histogramnd` computation on arrays with more than 2**31-1 samples (PR #3599)

  * `silx.math.fft`:

    * Added `export_wisdom()` and `import_wisdom()` (PR #3623)
    * Fixed normalization modes, notably account for regression in `pyfftw` normalization (PR #3625)
    * Fixed avoid creating OpenCL/Cuda contexts when not needed (PR #3587)

  * `silx.math.fit`: Updated documentation (PR #3582)

* `silx.opencl`: Updated OpenCL profiling, fixed memory leak (PR #3690)

* `silx.utils.ExternalResources`: Stored downloaded data checksum (PR #3580)

* Miscellaneous:

  * Added `SILX_INSTALL_REQUIRES_STRIP` build configuration environment variable (PR #3602)
  * Added optional use of `sphinx_autodoc_typehints` to generate the documentation (PR #3668)
  * Updated build and development tools to remove dependency to `distutils` and `numpy.distutils` (PR #3583, #3585, #3613, #3649, #3651, #3653, #3658, #3661, #3678)
  * Updated Windows installer (PR #3642)
  * Updated documentation (PR #3699, #3709)
  * Updated after 1.0.0 release (PR #3560, #3569)
  * Fixed tests and continuous integration (PR #3632, #3637, #3639, #3685)
  * Fixed Debian/Ubuntu packaging (PR #3693)
  * Cleaned-up Python 2 compatibility code (PR #3673)

1.0.0: 2021/12/06
-----------------

This is the first version of `silx` supporting `PySide6` (for `Qt6`) and using `pytest` to run the tests.

* `silx view`:

  * Added Windows installer generation (PR #3548)
  * Updated 'About' dialog (#3547, #3475)
  * Fixed: Keep curve legend selection with changing dimensions (PR #3529)
  * Fixed: Increase max number of opened file at start-up (PR #3545)

* `silx.gui`:

  * Added PySide6 support (PR #3486, #3528, #3479, #3542, #3549, #3478, #3481):
  * Removed support of PyQt4 / Pyside (PR #3423, #3424, #3480, #3482)
  * `silx.gui.colors`:

    * Fixed duplicated logs when colormap vmin/vmax are not valid (PR #3471)

  * `silx.gui.plot`:

    * `silx.gui.plot.actions`:

      * `silx.gui.plot.actions.fit`:

        * Updated behaviour of fitted item auto update (PR #3532)

      * `silx.gui.plot.actions.histogram`:

        * Enhanced: Allow user to change histogram nbins and range (PR #3514, #3514)
        * Updated `PixelIntensitiesHistoAction` to use `PlotWidget.selection` (PR #3408)
        * Fixed issue when the whole image is masked (PR #3544)
        * Fixed error on macOS 11 with 3D display in `silx view` (PR #3544)

      * `silx.gui.plot.CompareImages`:

        * Fixed `colormap`: avoid forcing vmin and vmax when not in 'HORIZONTAL_LINE' or 'VERTICAL_LINE' mode (PR #3510)
		
      * `silx.gui.plot.items`:
		
        * Added 'image_aggregated.ImageDataAggregated': item allowing to aggregate image data before display (PR #3503)
        * Fixed `ArcROI.setGeometry` (fix #3492)

      * `silx.gui.plot.ImageStack`:

        * Enhanced management of the `animation thread` (PR #3440, PR #3441)

      * `silx.gui.plot.ImageView`:

        * Added action to show/hide the side histogram (PR #3488)
        * Added 'resetzoom' parameter to 'ImageView.setImage' (PR #3488)
        * Added empty array support to 'ImageView.setImage' (PR #3530)
        * Added aggregation mode action (PR #3536)
        * Added support of RGB and RGBA images (PR #3487)
        * Updated 'imageview' example with a '--live' option (PR #3488)
        * Fixed profile window, added `setProfileWindowBehavior` method (PR #3457)
        * Fixed issue with profile window size (PR #3455)

      * `silx.gui.plot.PlotWidget`:

        * Fixed update of `Scatter` item binned statistics visualization (PR #3452)
        * Fixed OpenGL backend memory leak (PR #3453)
        * Enhanced: Optimized scatter when rendered as regular grid with the OpenGL backend (PR #3447)
        * Enhanced axis limits management by the OpenGL backend (PR #3504)
        * Enhanced control of repaint (PR #3449)
	* Enhanced text label background rendering with OpenGL backend (PR #3565)

      * `silx.gui.plot.PlotWindow`:

        * Fixed returned action from 'getKeepDataAspectRatioAction' (PR #3500)

    * `silx.gui.plot3d`:

      * Fixed picking on highdpi screen (PR #3550)
      * Fixed issue in parameter tree (PR #3550)

* `silx.io`:

  * Added read support for FIO files (PR #3539) thanks to tifuchs contribution
  * `silx.io.dictdump`:

    * Fixed missing conversion of the key (PR #3505) thanks to rnwatanabe contribution
    * Extract update modes list to a constant global variable (PR #3460) thanks to jpcbertoldo
	
  * `silx.io.convert`:
	
    * Enhanced `write_to_h5`: `infile` parameter can now also be a HDF5 file as input (PR #3511)
	
  * `silx.io.h5py_utils`:

    * Added support of `locking` argument from the h5py.File when possible (PR #3554)
    * Added log a critical message for unsupported versions of libhdf5 (PR #3533)

  * `silx.io.spech5`:
	
    * Enhanced: Improve robustness (PR #3507, #3463)
	
  * `silx.io.url`:

    * Fixed `is_absolute` in the case the `file_path()` returns None (PR #3437)

  * `silx.io.utils`:

    * Added 'silx.io.utils.visitall': provides a visitor of all items including links that works for both `commonh5` and `h5py` (PR #3511)

* `silx.math`:

  * `silx.math.colormap`:

    * Added `apply_colormap` function (PR #3525)
    * Enhanced `cmap` error messages (PR #3522)

* `silx.opencl`:

  * Added description of compute capabilities for Ampere generation GPU from Nvidia (PR #3535)
  * Added doubleword OpenCL library (PR #3466, PR #3472)

* Miscellaneous:

  * Enhanced: Setup the project to use `pytest` (PR #3431, #3516, #3526)
  * Enhanced: Minor test clean up (PR #3515, #3508)
  * Updated project structure: move `silx` sources in `src/silx` (PR #3412)
  * Fixed 'run_test.py --qt-binding' option (PR #3527)
  * Fixed support of numpy 1.21rc1 (PR ##3476)
  * Removed `six` dependency (PR #3483)


0.15.2: 2021/06/21
------------------

Minor release:

* `silx.io`:

  * `silx.io.spech5`: Enhanced robustness for missing positioner values (PR #3477)
  * `silx.io.url`: Fixed `DataUrl.is_absolute` (PR #3467)

* `silx.gui`:

  * Fixed naming of some loggers (PR #3477)
  * Fixed assert on `ImageStack` when length of urls > 0 (PR #3491)
  * `silx.gui.plot`: Fixed `ArcROI.setGeometry` (PR #3493)

* `silx.opencl`: Expose the double-word library and include it in tests (PR #3466)
* Misc: Fixed support of `numpy` 1.21rc1 (PR #3477)

0.15.1: 2021/05/17
------------------

Minor release:

* silx.gui.plot.PlotWidget: Fixed `PlotWidget` OpenGL backend memory leak (PR #3448)
* silx.gui.plot.ImageView:

  * Fixed profile window default behavior (PR #3458)
  * Added `setProfileWindowBehavior` method (PR #3458)

0.15.0: 2021/03/18
------------------

Main new features are the `silx.io.h5py_utils` module which provides `h5py` concurrency helpers and image mask support by `silx.gui.plot.PlotWidget`'s tools.

* `silx view`:

  * Fixed zoom reseting when scrolling a NXdata 3D stack (PR #3351)
  * Fixed support of very large 1D datasets in "Raw" table view (PR #3418)

* `silx.io`:

  * Added `h5py_utils` helper module for concurrent HDF5 reading and writing without SWMR (PR #3368, #3426)
  * Enhanced `dictdump` module functions regarding overwriting existing files (PR #3376)

* `silx.gui`:

  * Added scale to visible or selected area buttons options to `silx.gui.dialog.ColormapDialog` (PR #3365)
  * Fixed and enhanced`silx.gui.utils.glutils.isOpenGLAvailable` (PR #3356, #3385)
  * Fixed `silx.gui.widgets.FlowLayout` (PR #3389)
  * Enhanced `silx.gui.data.ArrayTableWidget`: Added support of array clipping if data is too large (PR #3419)

  * `silx.gui.plot`:

    * Added mask support to Image items and use it in plot tools (histogram, profile, colormap) (PR #3369, #3381)
    * Added `ImageStack` methods to configure automatic reset zoom (PR #3373)
    * Added some statistic indicators in `PixelIntensitiesHistoAction` action (PR #3391)
    * Enhanced `silx.gui.plot.ImageView` integration of ROI profiles in side plots (PR #3380)
    * Enhanced `PositionInfo`: snapping to histogram (PR #3405) and information labels layout (PR #3399)
    * Fixed `LegendSelector` blinking when updated (PR #3346)
    * Fixed profile tool issue when closing profile window after attaced PlotWidget (PR #3375)
    * Fixed histogram action (PR #3396)
    * Fixed support of histogram plot items in `stats` module (PR #3398, #3407)
    * Fixed `ColorBar` when deleting attached PlotWidget (PR #3403)

    * `silx.gui.plot.PlotWidget`:

      * Added `getValueData` method to image items (PR #3378)
      * Added `discardItem` method (PR #3400)
      * Added unified `selection()` handler compatible with active item management (PR #3401)
      * Fixed `addCurve` documentation (PR #3371)
      * Fixed complex image first displayed mode (PR #3364)
      * Fixed curve and scatter items support of complex data input (PR #3384)
      * Fixed histogram picking (PR #3405)
      * Fixed rendering (PR #3416)

  * `silx.gui.plot3d`:

    * Added `HeightMapData` and `HeightMapRGBA` items (PR #3386, #3397)
    * Fixed support for RGB colored points in internal scene graph (PR #3374)
    * Fixed `ImageRgba` alpha channel display (PR #3414)

* `silx.image`:

  * Added mask support to `bilinear` interpolator (PR #3286)

* `silx.opencl`:

  * Added print statics of OpenCL kernel execution time (PR #3395)

* Miscellaneous:

  * Removed debian 9 packaging (PR #3383)
  * Enhanced test functions: `silx.test.run_tests` (PR #3331), `silx.utils.testutils.TestLogging` (PR #3393)
  * Continuous integration: Added github actions and removed travis-ci (PR #3353, #3359), fixed (PR #3361, #3366)
  * Updated documentation (PR #3383, #3387, #3409, #3416, #3427)
  * Fixed debian packaging (PR #3362)
  * Fixed `silx test` application on Windows (PR #3411)

0.14.1: 2021/04/30
------------------

This is a bug-fix version of silx.

* silx.gui.plot: Fixed `PlotWidget` OpenGL backend memory leak (PR #3445)
* silx.gui.utils.glutils: Fixed `isOpenGLAvailable` (PR #3356)

0.14.0: 2020/12/11
------------------

This is the first version of `silx` supporting `h5py` >= v3.0.

This is the last version of `silx` officially supporting Python 3.5.

* `silx.gui`:

  * Added support for HDF5 external data (virtual and raw) (PR #3222)
  * Added lazy update handling of OpenGL textures (PR #3205)
  * Deprecated `silx.gui.plot.matplotlib` module (use `silx.gui.utils.matplotlib` instead) (PR #3158)
  * Improved memory allocation by using already defined `fontMetrics` instread of creating a new one (PR #3239)
  * Make `TextFormatter` compatible with `h5py`>=3 (PR #3253)
  * Fixed `matplotlib` 3.3.0rc1 deprecation warnings (PR #3145)

  * `silx.gui.colors.Colormap`:

    * Added `Colormap.get|setNaNColor` to change color used for NaN, fix different NaN displays for matplotlib/openGL backends (PR #3143)
    * Refactored PlotWidget OpenGL backend to enable extensions (PR #3147)
    * Fixed use of `QThreadPool.tryTake` to be Qt5.7 compliant (PR #3250)

  * `silx.gui.plot`:

    * Added the feature to compute statistics inside a specific region of interest (PR #3056)
    * Added an action to switch on/off OpenGL rendering on a plot (PR #3261)
    * Added test for ROI interaction mode (PR #3283)
    * Added saving of error bars when saving a plot (PR #3199)
    * Added `ImageStack.clear` (PR #3167)
    * Improved image profile tool to support `PlotWidget` item extension (PR #3150)
    * Improved `Stackview`: replaced `setColormap` `autoscale` argument by `scaleColormapRangeToStack` method (PR #3279)
    * Updated `3 stddev` autoscale algorithm, clamp it with the minmax data in order to improve the contrast (PR #3284)
    * Updated ROI module: splitted into 3 modules base/common/arc_roi (PR #3283)
    * Fixed `ColormapDialog` custom range input (PR #3153)
    * Fixed issue when changing ROI mode while a ROI is being created (PR #3186)
    * Fixed `RegionOfInterest` refresh when highlighted (PR #3197)
    * Fixed arc roi shape: make sure start and end points are part of the shape (PR #3257)
    * Fixed issue in `Colormap` `3 stdev` autoscale mode and avoided warnings (PR #3295)

    * Major improvements of `PlotWidget`:

      * Added `get|setAxesMargins` methods to control margin ratios around plot area (PR #3196)
      * Added `PlotWidget.[get|set]Backend` enabling switching backend (PR #3255)
      * Added multi interaction mode for ROIs (can be switched with a single click on an handle, or the context menu) (PR #3260)
      * Added polar interaction mode for arc ROI (PR #3260)
      * Added `PlotWidget.sigDefaultContextMenu` to allow to feed the default context menu (PR #3260)
      * Added context menu to the selected ROI to remove it (PR #3260)
      * Added pan interaction to ROI authoring (`select-draw`) interaction mode (PR #3291)
      * Added support of right axis label with OpenGL backend (PR #3293)
      * Added item visible bounds feature to PlotWidget items (PR #3223)
      * Added a `DataItem` base class for items having a "data extent" in the plot (PR #3212)
      * Added support for float16 texture in OpenGL backend (PR #3194)
      * Improved support of high-DPI screen in OpenGL backend (PR #3203)
      * Updated: Use points rather than pixels for marker size and line width with OpenGL backend (PR #3203)
      * Updated: Expose `PlotWidget` colors as Qt properties (PR #3269)
      * Fixed time serie axis for range < 2.5 microseconds (PR #3195)
      * Fixed initial size of OpenGL backend (PR #3209)
      * Fixed `PlotWidget` image items displayed below the grid by default (PR #3235)
      * Fixed OpenGL backend image display with sqrt colormap normalization (PR #3248)
      * Fixed support of shapes with multiple polygons in the OpenGL backend (PR #3259)
      * Fixes duplicated callback on ROIs (there was one for each ROI managed created on the plot) (PR #3260)
      * Fixed RegionOfInterest `contains` methods (PR #3336)

  * `silx.gui.colors.plot3d`:

    * Improved scene rendering (PR #3149)
    * Fixed handling of transparency of cut plane (PR #3204)

* `silx.image`:

  * Fixed slow `image.tomography.get_next_power()` (PR #3168)

* `silx.io`:

  * Added support for HDF5 link preservation in `dictdump` (PR #3224)
  * Added support for numpy arrays of `numbers` (PR #3251)
  * Make `h5todict` resilient to issues in the HDF5 file (PR #3162)

* `silx.math`:

  * Improved colormap performances for small datasets (PR #3282)

* `silx.opencl`:

  * Added textures availability check (PR #3273)
  * Added a warning when there is an issue in the Ocl destruction (PR #3280)
  * Fixed Sift test on modern GPU (PR #3262)

* Miscellaneous:

  * Added HDF5 strings: handle `h5py` 2.x and 3.x (PR #3240)
  * Fixed `cython` 3 compatibility and deprecation warning (PR #3164, #3189)


0.13.2: 2020/09/15
------------------

Minor release:

* silx view application: Prevent collapsing browsing panel, Added `-f` command line option (PR #3176)

* `silx.gui`:

  * `silx.gui.data`: Fixed `DataViews.titleForSelection` method (PR #3171).
  * `silx.gui.plot.items`: Added `DATA_BOUNDS` visualization parameter for `Scatter` item histogram bounds (PR #3180)
  * `silx.gui.plot.PlotWidget`: Fixed support of curves with infinite data (PR #3175)
  * `silx.gui.utils.glutils`: Fixed `isOpenGLAvailable` function (PR #3184)

* Documentation:

  * Update silx view command line options documentation (PR #3173)
  * Update version number and changelog (PR #3190)


0.13.1: 2020/07/22
------------------

Bug fix release:

* `silx.gui.plot.dialog`: Fixed `ColormapDialog` custom range input (PR #3155)
* Build: Fixed cython 3 compatibility (PR #3163).
* Documentation: Update version number and changelog (PR #3156)


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
