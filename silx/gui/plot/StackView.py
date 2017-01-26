# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""QWidget displaying a 3D volume as a stack of 2D images.

The :class:`StackView` class implements this widget.

Basic usage of :class:`StackView` is through the following methods:

- :meth:`StackView.getColormap`, :meth:`StackView.setColormap` to update the
  default colormap to use and update the currently displayed image.
- :meth:`StackView.setStack` to update the displayed image.

The :class:`StackView` uses :class:`PlotWindow` and also
exposes a subset of the :class:`silx.gui.plot.Plot` API for further control
(plot title, axes labels, ...).

The :class:`StackViewMainWindow` class implements a widget that adds a status
bar displaying the 3D index and the value under the mouse cursor.

Example::

    import numpy
    import sys
    from silx.gui import qt
    from silx.gui.plot.StackView import StackViewMainWindow


    app = qt.QApplication(sys.argv[1:])

    # synthetic data, stack of 100 images of size 200x300
    mystack = numpy.fromfunction(
        lambda i, j, k: numpy.sin(i/15.) + numpy.cos(j/4.) + 2 * numpy.sin(k/6.),
        (100, 200, 300)
    )


    sv = StackViewMainWindow()
    sv.setColormap("jet", autoscale=True)
    sv.setStack(mystack)
    sv.setLabels(["1st dim (0-99)", "2nd dim (0-199)",
                  "3rd dim (0-299)"])
    sv.show()

    app.exec_()

"""

__authors__ = ["P. Knobel", "H. Payno"]
__license__ = "MIT"
__date__ = "20/01/2017"

import numpy

try:
    import h5py
except ImportError:
    h5py = None

from silx.gui import qt
from .. import icons
from . import PlotWindow
from . import PlotActions
from .Colors import cursorColorForColormap
from .PlotTools import LimitsToolBar
from .Profile import Profile3DToolBar
from ..widgets.FrameBrowser import HorizontalSliderWithBrowser

from silx.utils.array_like import DatasetView, ListOfImages


class StackView(qt.QMainWindow):
    """Stack view widget, to display and browse through stack of
    images.

    The profile tool can be switched to "3D" mode, to compute the profile
    on each image of the stack (not only the active image currently displayed)
    and display the result as a slice.

    :param QWidget parent: the Qt parent, or None
    :param backend: The backend to use for the plot.
                    The default is to use matplotlib.
    :type backend: str or :class:`BackendBase.BackendBase`
    :param bool resetzoom: Toggle visibility of reset zoom action.
    :param bool autoScale: Toggle visibility of axes autoscale actions.
    :param bool logScale: Toggle visibility of axes log scale actions.
    :param bool grid: Toggle visibility of grid mode action.
    :param bool colormap: Toggle visibility of colormap action.
    :param bool aspectRatio: Toggle visibility of aspect ratio button.
    :param bool yInverted: Toggle visibility of Y axis direction button.
    :param bool copy: Toggle visibility of copy action.
    :param bool save: Toggle visibility of save action.
    :param bool print_: Toggle visibility of print action.
    :param bool control: True to display an Options button with a sub-menu
                         to show legends, toggle crosshair and pan with arrows.
                         (Default: False)
    :param position: True to display widget with (x, y) mouse position
                     (Default: False).
                     It also supports a list of (name, funct(x, y)->value)
                     to customize the displayed values.
                     See :class:`silx.gui.plot.PlotTools.PositionInfo`.
    :param bool mask: Toggle visibilty of mask action.
    """
    # Qt signals
    valueChanged = qt.Signal(object, object, object)
    """Signals that the data value under the cursor has changed.

    It provides: row, column, data value.
    """

    sigPlaneSelectionChanged = qt.Signal(int)
    """Signal emitted when there is a change is perspective/displayed axes.

    It provides the perspective as an integer, with the following meaning:

        - 0: axis Y is the 2nd dimension, axis X is the 3rd dimension
        - 1: axis Y is the 1st dimension, axis X is the 3rd dimension
        - 2: axis Y is the 1st dimension, axis X is the 2nd dimension
    """

    sigStackChanged = qt.Signal(int)
    """Signal emitted when the stack is changed.
    This happens when a new volume is loaded, or when the current volume
    is transposed (change in perspective).

    The signal provides the size (number of pixels) of the stack.
    This will be 0 if the stack is cleared, else it will be a positive
    integer.
    """

    def __init__(self, parent=None, resetzoom=True, backend=None,
                 autoScale=False, logScale=False, grid=False,
                 colormap=True, aspectRatio=True, yinverted=True,
                 copy=True, save=True, print_=True, control=False,
                 position=None, mask=True):
        qt.QMainWindow.__init__(self, parent)
        if parent is not None:
            # behave as a widget
            self.setWindowFlags(qt.Qt.Widget)
        else:
            self.setWindowTitle('StackView')

        self._stack = None
        """Loaded stack, as a 3D array, a 3D dataset or a list of 2D arrays."""
        self.__transposed_view = None
        """View on :attr:`_stack` with the axes sorted, to have
        the orthogonal dimension first"""
        self._perspective = 0
        """Orthogonal dimension (depth) in :attr:`_stack`"""

        self.__imageLegend = '__StackView__image' + str(id(self))
        self.__autoscaleCmap = False
        """Flag to disable/enable colormap auto-scaling
        based on the min/max values of the entire 3D volume"""
        self.__dimensionsLabels = ["Dimension 0", "Dimension 1",
                                   "Dimension 2"]
        """These labels are displayed on the X and Y axes.
        :meth:`setLabels` updates this attribute."""

        central_widget = qt.QWidget(self)

        self._plot = PlotWindow(parent=central_widget, backend=backend,
                                resetzoom=resetzoom, autoScale=autoScale,
                                logScale=logScale, grid=grid,
                                curveStyle=False, colormap=colormap,
                                aspectRatio=aspectRatio, yInverted=yinverted,
                                copy=copy, save=save, print_=print_,
                                control=control, position=position,
                                roi=False, mask=mask)
        self.sigInteractiveModeChanged = self._plot.sigInteractiveModeChanged
        self.sigActiveImageChanged = self._plot.sigActiveImageChanged
        self.sigPlotSignal = self._plot.sigPlotSignal

        self._plot.profile = Profile3DToolBar(parent=self._plot,
                                              plot=self)
        self._plot.addToolBar(self._plot.profile)
        self._plot.setGraphXLabel('Columns')
        self._plot.setGraphYLabel('Rows')
        self._plot.sigPlotSignal.connect(self._plotCallback)

        self._browser = HorizontalSliderWithBrowser(central_widget)
        self._browser.valueChanged[int].connect(self.__updateFrameNumber)
        self._browser.setEnabled(False)

        self.__planeSelection = PlanesWidget(self._plot)
        self.__planeSelection.sigPlaneSelectionChanged.connect(self.__setPerspective)

        layout = qt.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot, 0, 0, 1, 2)
        layout.addWidget(self.__planeSelection, 1, 0)
        layout.addWidget(self._browser, 1, 1)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # clear profile lines when the profile changes (plane browsed changed)
        self.__planeSelection.sigPlaneSelectionChanged.connect(
            self._plot.profile.hideProfileWindow)

    def setOptionVisible(self, isVisible):
        """
        Set the visibility of the browsing options.

        :param bool isVisible: True to have the options visible, else False
        """
        self._browser.setVisible(isVisible)
        self.__planeSelection.setVisible(isVisible)

    def _plotCallback(self, eventDict):
        """Callback for plot events.

        Emit :attr:`valueChanged` signal, with (x, y, value) tuple of the
        cursor location in the plot."""
        if eventDict['event'] == 'mouseMoved':
            activeImage = self.getActiveImage()
            if activeImage is not None:
                data = activeImage[0]
                height, width = data.shape

                # Get corresponding coordinate in image
                origin = activeImage[4]['origin']
                scale = activeImage[4]['scale']
                x = int((eventDict['x'] - origin[0]) / scale[0])
                y = int((eventDict['y'] - origin[1]) / scale[1])

                if 0 <= x < width and 0 <= y < height:
                    self.valueChanged.emit(float(x), float(y),
                                           data[y][x])
                else:
                    self.valueChanged.emit(float(x), float(y),
                                           None)

    def __setPerspective(self, perspective):
        """Function called when the browsed/orthogonal dimension changes

        :param perspective: the new browsed dimension
        """
        if perspective == self._perspective:
            return
        else:
            if perspective > 2 or perspective < 0:
                raise ValueError(
                        "Perspective must be 0, 1 or 2, not %s" % perspective)

            self._perspective = perspective
            self.__createTransposedView()
            self.__updateFrameNumber(self._browser.value())
            self._plot.resetZoom()
            self.__updatePlotLabels()

            self.sigPlaneSelectionChanged.emit(perspective)
            self.sigStackChanged.emit(self._stack.size if
                                      self._stack is not None else 0)

    def __updatePlotLabels(self):
        """Update plot axes labels depending on perspective"""
        y, x = (1, 2) if self._perspective == 0 else \
            (0, 2) if self._perspective == 1 else (0, 1)
        self.setGraphXLabel(self.__dimensionsLabels[x])
        self.setGraphYLabel(self.__dimensionsLabels[y])

    def __createTransposedView(self):
        """Create the new view on the stack depending on the perspective
        (set orthogonal axis browsed on the viewer as first dimension)
        """
        assert self._stack is not None
        assert 0 <= self._perspective < 3

        # ensure we have the stack encapsulated in an array like object
        # having a transpose() method
        if isinstance(self._stack, numpy.ndarray):
            self.__transposed_view = self._stack

        elif h5py is not None and isinstance(self._stack, h5py.Dataset) or \
                isinstance(self._stack, DatasetView):
            self.__transposed_view = DatasetView(self._stack)

        elif isinstance(self._stack, ListOfImages):
            self.__transposed_view = ListOfImages(self._stack)

        # transpose the array like object if necessary
        if self._perspective == 1:
            self.__transposed_view = self.__transposed_view.transpose((1, 0, 2))
        elif self._perspective == 2:
            self.__transposed_view = self.__transposed_view.transpose((2, 0, 1))

        self._browser.setRange(0, self.__transposed_view.shape[0] - 1)
        self._browser.setValue(0)

    def setFrameNumber(self, number):
        """Set the frame selection to a specific value\

        :param int number: Number of the frame
        """
        self._browser.setValue(number)

    def __updateFrameNumber(self, index):
        """Update the current image.

        :param index: index of the frame to be displayed
        """
        assert self.__transposed_view is not None
        self._plot.addImage(self.__transposed_view[index, :, :],
                            legend=self.__imageLegend,
                            resetzoom=False)

    # public API
    def setStack(self, stack, perspective=0, reset=True):
        """Set the 3D stack.

        The perspective parameter is used to define which dimension of the 3D
        array is to be used as frame index. The lowest remaining dimension
        number is the row index of the displayed image (Y axis), and the highest
        remaining dimension is the column index (X axis).

        :param stack: 3D stack, or `None` to clear plot.
        :type stack: 3D numpy.ndarray, or 3D h5py.Dataset, or list/tuple of 2D
            numpy arrays, or None.
        :param int perspective: Dimension for the frame index: 0, 1 or 2.
            By default, the dimension for the image index is the first
            dimension of the 3D stack (``perspective=0``).
        :param bool reset: Whether to reset zoom or not.
        """
        if stack is None:
            self.clear()
            self.sigStackChanged.emit(0)
            return

        # stack as list of 2D arrays: must be converted into an array_like
        if not isinstance(stack, numpy.ndarray):
            if h5py is None or not isinstance(stack, h5py.Dataset):
                try:
                    assert hasattr(stack, "__len__")
                    for img in stack:
                        assert hasattr(img, "shape")
                        assert len(img.shape) == 2
                except AssertionError:
                    raise ValueError(
                        "Stack must be a 3D array/dataset or a list of " +
                        "2D arrays.")
                stack = ListOfImages(stack)

        assert len(stack.shape) == 3, "data must be 3D"

        self._stack = stack
        self.__createTransposedView()

        # This call to setColormap redefines the meaning of autoscale
        # for 3D volume: take global min/max rather than frame min/max
        if self.__autoscaleCmap:
            self.setColormap(autoscale=True)

        # init plot
        self._plot.addImage(self.__transposed_view[0, :, :],
                            legend=self.__imageLegend,
                            colormap=self.getColormap(),
                            resetzoom=False)
        self._plot.setActiveImage(self.__imageLegend)
        self.__updatePlotLabels()

        if reset:
            self._plot.resetZoom()

        # enable and init browser
        self._browser.setEnabled(True)

        if perspective != self._perspective:
            self.__setPerspective(perspective)

        self.sigStackChanged.emit(stack.size)

    def getStack(self, copy=True, returnNumpyArray=False):
        """Get the original stack, as a 3D array or dataset.

        The output has the form: [data, params]
        where params is a dictionary containing display parameters.

        :param bool copy: If True (default), then the object is copied
            and returned as a numpy array.
            Else, a reference to original data is returned, if possible.
            If the original data is not a numpy array and parameter
            returnNumpyArray is True, a copy will be made anyway.
        :param bool returnNumpyArray: If True, the returned object is
            guaranteed to be a numpy array.
        :return: 3D stack and parameters.
        :rtype: (numpy.ndarray, dict)
        """
        if self.getActiveImage() is None:
            return None
        _img, _legend, _info, _pixmap, params = self.getActiveImage()
        if returnNumpyArray or copy:
            return numpy.array(self._stack, copy=copy), params

        # if a list of 2D arrays was cast into a ListOfImages,
        # return the original list
        if isinstance(self._stack, ListOfImages):
            return self._stack.images, params

        return self._stack, params

    def getCurrentView(self, copy=True, returnNumpyArray=False):
        """Get the stack, as it is currently displayed.

        The first index of the returned stack is always the frame
        index. If the perspective has been changed in the widget since the
        data was first loaded, this will be reflected in the order of the
        dimensions of the returned object.

        The output has the form: [data, params]
        where params is a dictionary containing display parameters.

        :param bool copy: If True (default), then the object is copied
            and returned as a numpy array.
            Else, a reference to original data is returned, if possible.
            If the original data is not a numpy array and parameter
            `returnNumpyArray` is `True`, a copy will be made anyway.
        :param bool returnNumpyArray: If `True`, the returned object is
            guaranteed to be a numpy array.
        :return: 3D stack and parameters.
        :rtype: (numpy.ndarray, dict)
        """
        if self.getActiveImage() is None:
            return None
        _img, _legend, _info, _pixmap, params = self.getActiveImage()
        if returnNumpyArray or copy:
            return numpy.array(self.__transposed_view, copy=copy), params
        return self.__transposed_view, params

    def getActiveImage(self, just_legend=False):
        """Returns the currently active image.

        It returns None in case of not having an active image.

        Default output has the form: [data, legend, info, pixmap, params]
        where params is a dictionary containing image parameters.

        :param bool just_legend: True to get the legend of the image,
            False (the default) to get the image data and info.
            Note: :class:`StackView` uses the same legend for all frames.
        :return: legend or [data, legend, info, pixmap, params]
        :rtype: str or list
        """
        return self._plot.getActiveImage(just_legend=just_legend)

    def clear(self):
        """Clear the widget:

         - clear the plot
         - clear the loaded data volume
        """
        self._stack = None
        self.__transposed_view = None
        self._perspective = 0
        self._browser.setEnabled(False)
        self._plot.clear()

    def resetZoom(self):
        """Reset the plot limits to the bounds of the data and redraw the plot.
        """
        self._plot.resetZoom()

    def getGraphTitle(self):
        """Return the plot main title as a str."""
        return self._plot.getGraphTitle()

    def setGraphTitle(self, title=""):
        """Set the plot main title.

        :param str title: Main title of the plot (default: '')
        """
        return self._plot.setGraphTitle(title)

    def setLabels(self, labels=None):
        """Set the labels to be displayed on the plot axes.

        You must provide a sequence of 3 strings, corresponding to the 3
        dimensions of the original data volume.
        The proper label will automatically be selected for each plot axis
        when the volume is rotated (when different axes are selected as the
        X and Y axes).

        :param list(str) labels: 3 labels corresponding to the 3 dimensions
             of the data volumes.
        """
        if labels is None:
            labels = ["Dimension 0", "Dimension 1", "Dimension 2"]
        self.__dimensionsLabels = labels
        self.__updatePlotLabels()

    def getGraphXLabel(self):
        """Return the current horizontal axis label as a str."""
        return self._plot.getGraphXLabel()

    def setGraphXLabel(self, label=None):
        """Set the plot horizontal axis label.

        :param str label: The horizontal axis label
        """
        if label is None:
            label = self.__dimensionsLabels[1 if self._perspective == 2 else 2]
        self._plot.setGraphXLabel(label)

    def getGraphYLabel(self, axis='left'):
        """Return the current vertical axis label as a str.

        :param str axis: The Y axis for which to get the label (left or right)
        """
        return self._plot.getGraphYLabel(axis)

    def setGraphYLabel(self, label=None, axis='left'):
        """Set the vertical axis label on the plot.

        :param str label: The Y axis label
        :param str axis: The Y axis for which to set the label (left or right)
        """
        if label is None:
            label = self.__dimensionsLabels[1 if self._perspective == 0 else 0]
        self._plot.setGraphYLabel(label, axis)

    def setYAxisInverted(self, flag=True):
        """Set the Y axis orientation.

        :param bool flag: True for Y axis going from top to bottom,
                          False for Y axis going from bottom to top
        """
        self._plot.setYAxisInverted(flag)

    def isYAxisInverted(self):
        """Return True if Y axis goes from top to bottom, False otherwise."""
        return self._backend.isYAxisInverted()

    def getSupportedColormaps(self):
        """Get the supported colormap names as a tuple of str.

        The list should at least contain and start by:
        ('gray', 'reversed gray', 'temperature', 'red', 'green', 'blue')
        """
        return self._plot.getSupportedColormaps()

    def getColormap(self):
        """Get the current colormap description.

        :return: A description of the current colormap.
                 See :meth:`setColormap` for details.
        :rtype: dict
        """
        # "default" colormap used by addImage when image is added without
        # specifying a special colormap
        return self._plot.getDefaultColormap()

    def setColormap(self, colormap=None, normalization=None,
                    autoscale=None, vmin=None, vmax=None, colors=None):
        """Set the colormap and update active image.

        Parameters that are not provided are taken from the current colormap.

        The colormap parameter can also be a dict with the following keys:

        - *name*: string. The colormap to use:
          'gray', 'reversed gray', 'temperature', 'red', 'green', 'blue'.
        - *normalization*: string. The mapping to use for the colormap:
          either 'linear' or 'log'.
        - *autoscale*: bool. Whether to use autoscale (True) or range
          provided by keys
          'vmin' and 'vmax' (False).
        - *vmin*: float. The minimum value of the range to use if 'autoscale'
          is False.
        - *vmax*: float. The maximum value of the range to use if 'autoscale'
          is False.
        - *colors*: optional. Nx3 or Nx4 array of float in [0, 1] or uint8.
                    List of RGB or RGBA colors to use (only if name is None)

        :param colormap: Name of the colormap in
            'gray', 'reversed gray', 'temperature', 'red', 'green', 'blue'.
            Or the description of the colormap as a dict.
        :type colormap: dict or str.
        :param str normalization: Colormap mapping: 'linear' or 'log'.
        :param bool autoscale: Whether to use autoscale or [vmin, vmax] range.
            Default value of autoscale is True if data is a numpy array,
            False if data is a h5py dataset.
        :param float vmin: The minimum value of the range to use if
                           'autoscale' is False.
        :param float vmax: The maximum value of the range to use if
                           'autoscale' is False.
        :param numpy.ndarray colors: Only used if name is None.
            Custom colormap colors as Nx3 or Nx4 RGB or RGBA arrays
        """
        cmapDict = self.getColormap()

        if isinstance(colormap, dict):
            # Support colormap parameter as a dict
            errmsg = "If colormap is provided as a dict, all other parameters"
            errmsg += " must not be specified when calling setColormap"
            assert normalization is None, errmsg
            assert autoscale is None, errmsg
            assert vmin is None, errmsg
            assert vmax is None, errmsg
            assert colors is None, errmsg
            cmapDict.update(colormap)

        else:
            if colormap is not None:
                cmapDict['name'] = colormap
            if normalization is not None:
                cmapDict['normalization'] = normalization
            if colors is not None:
                cmapDict['colors'] = colors

            # Default meaning of autoscale is to reset min and max
            # each time a new image is added to the plot.
            # We want to use min and max of global volume,
            # and not change them when browsing slides
            cmapDict['autoscale'] = False

            if autoscale is None:
                # set default
                autoscale = False
                # TODO: assess cost of computing min/max for large 3D array
                # if isinstance(self._stack, numpy.ndarray):
                #     autoscale = True
                # else:                    # h5py.Dataset
                #     autoscale = False
            elif autoscale and isinstance(self._stack, h5py.Dataset):
                # h5py dataset has no min()/max() methods
                raise RuntimeError(
                        "Cannot auto-scale colormap for a h5py dataset")
            else:
                autoscale = autoscale
            self.__autoscaleCmap = autoscale
            if autoscale and (self._stack is not None):
                cmapDict['vmin'] = self._stack.min()
                cmapDict['vmax'] = self._stack.max()
            else:
                if vmin is not None:
                    cmapDict['vmin'] = vmin
                if vmax is not None:
                    cmapDict['vmax'] = vmax

        cursorColor = cursorColorForColormap(cmapDict['name'])
        self._plot.setInteractiveMode('zoom', color=cursorColor)

        self._plot.setDefaultColormap(cmapDict)

        # Refresh image with new colormap
        activeImage = self._plot.getActiveImage()
        if activeImage is not None:
            data, legend, info, _pixmap = activeImage[0:4]
            self._plot.addImage(data, legend=legend, info=info,
                                colormap=self.getColormap(),
                                resetzoom=False)

    def isKeepDataAspectRatio(self):
        """Returns whether the plot is keeping data aspect ratio or not."""
        return self._plot.isKeepDataAspectRatio()

    def setKeepDataAspectRatio(self, flag=True):
        """Set whether the plot keeps data aspect ratio or not.

        :param bool flag: True to respect data aspect ratio
        """
        self._plot.setKeepDataAspectRatio(flag)

    def getProfileToolbar(self):
        """Profile tools attached to this plot

        See :class:`silx.gui.plot.Profile.Profile3DToolBar`
        """
        return self._plot.profile

    def getProfileWindow1D(self):
        """Plot window used to display 1D profile curve.

        :return: :class:`Plot1D`
        """
        return self._plot.profile.getProfileWindow1D()

    def getProfileWindow2D(self):
        """Plot window used to display 2D profile image.

        :return: :class:`Plot2D`
        """
        return self._plot.profile.getProfileWindow2D()

    # kind of private methods, but needed by Profile
    def remove(self, legend=None,
               kind=('curve', 'image', 'item', 'marker')):
        """See :meth:`Plot.Plot.remove`"""
        self._plot.remove(legend, kind)

    def setInteractiveMode(self, *args, **kwargs):
        """
        See :meth:`Plot.Plot.setInteractiveMode`
        """
        self._plot.setInteractiveMode(*args, **kwargs)

    def addItem(self, *args, **kwargs):
        """
        See :meth:`Plot.Plot.addItem`
        """
        self._plot.addItem(*args, **kwargs)


class PlanesWidget(qt.QWidget):
    """Widget for the plane/perspective selection

    :param parent: the parent QWidget
    """
    sigPlaneSelectionChanged = qt.Signal(int)

    def __init__(self, parent):
        super(PlanesWidget, self).__init__(parent)

        self.setSizePolicy(qt.QSizePolicy.Minimum, qt.QSizePolicy.Minimum)
        layout0 = qt.QHBoxLayout()
        self.setLayout(layout0)
        layout0.setContentsMargins(0, 0, 0, 0)

        layout0.addWidget(qt.QLabel("Axes selection:"))

        # By default, the first dimension (dim0) is the frame index/depth/z,
        # the second dimension is the image row number/y axis
        # and the third dimension is the image column index/x axis

        # 1
        # | 0
        # |/__2

        self.qcbAxisSelection = qt.QComboBox(self)
        self.qcbAxisSelection.addItem(icons.getQIcon("cube-front"),
                                      'Dim1-Dim2')
        self.qcbAxisSelection.addItem(icons.getQIcon("cube-bottom"),
                                      'Dim0-Dim2')
        self.qcbAxisSelection.addItem(icons.getQIcon("cube-left"),
                                      'Dim0-Dim1')
        self.qcbAxisSelection.currentIndexChanged[int].connect(
                self.__planeSelectionChanged)

        layout0.addWidget(self.qcbAxisSelection)

    def __planeSelectionChanged(self, idx):
        """Callback function when the combobox selection changes

        idx is the dimension number orthogonal to the slice plane,
        following the convention:

          - slice plane Dim1-Dim2: perspective 0
          - slice plane Dim0-Dim2: perspective 1
          - slice plane Dim0-Dim1: perspective 2
        """
        self.sigPlaneSelectionChanged.emit(idx)


class StackViewMainWindow(StackView):
    """This class is a :class:`StackView` with a menu, an additional toolbar
    to set the plot limits, and a status bar to display the value and 3D
    index of the data samples hovered by the mouse cursor.

    :param QWidget parent: Parent widget, or None
    """
    def __init__(self, parent=None):
        self._dataInfo = None
        super(StackViewMainWindow, self).__init__(parent)
        self.setWindowFlags(qt.Qt.Window)

        # Add toolbars and status bar
        self.addToolBar(qt.Qt.BottomToolBarArea,
                        LimitsToolBar(plot=self._plot))

        self.statusBar()

        menu = self.menuBar().addMenu('File')
        menu.addAction(self._plot.saveAction)
        menu.addAction(self._plot.printAction)
        menu.addSeparator()
        action = menu.addAction('Quit')
        action.triggered[bool].connect(qt.QApplication.instance().quit)

        menu = self.menuBar().addMenu('Edit')
        menu.addAction(self._plot.copyAction)
        menu.addSeparator()
        menu.addAction(self._plot.resetZoomAction)
        menu.addAction(self._plot.colormapAction)
        menu.addAction(PlotActions.KeepAspectRatioAction(self._plot, self))
        menu.addAction(PlotActions.YAxisInvertedAction(self._plot, self))

        menu = self.menuBar().addMenu('Profile')
        menu.addAction(self._plot.profile.browseAction)
        menu.addAction(self._plot.profile.hLineAction)
        menu.addAction(self._plot.profile.vLineAction)
        menu.addAction(self._plot.profile.lineAction)
        menu.addSeparator()
        menu.addAction(self._plot.profile.clearAction)
        self._plot.profile.profile3dAction.computeProfileIn2D()
        menu.addMenu(self._plot.profile.profile3dAction.menu())

        # Connect to StackView's signal
        self.valueChanged.connect(self._statusBarSlot)

    def _statusBarSlot(self, x, y, value):
        """Update status bar with coordinates/value from plots."""
        # todo (after implementing calibration):
        #  - use floats for (x, y, z)
        #  - display both indices (dim0, dim1, dim2) and (x, y, z)
        msg = "Cursor out of range"
        if x is not None and y is not None:
            img_idx = self._browser.value()

            if self._perspective == 0:
                dim0, dim1, dim2 = img_idx, int(y), int(x)
            elif self._perspective == 1:
                dim0, dim1, dim2 = int(y), img_idx, int(x)
            elif self._perspective == 2:
                dim0, dim1, dim2 = int(y), int(x), img_idx

            msg = 'Position: (%d, %d, %d)' % (dim0, dim1, dim2)
        if value is not None:
            msg += ', Value: %g' % value
        if self._dataInfo is not None:
            msg = self._dataInfo + ', ' + msg

        self.statusBar().showMessage(msg)

    def setStack(self, stack, *args, **kwargs):
        """Set the displayed stack.

        See :meth:`StackView.setStack` for details.
        """
        if hasattr(stack, 'dtype') and hasattr(stack, 'shape'):
            assert len(stack.shape) == 3
            nframes, height, width = stack.shape
            self._dataInfo = 'Data: %dx%dx%d (%s)' % (nframes, height, width,
                                                      str(stack.dtype))
            self.statusBar().showMessage(self._dataInfo)
        else:
            self._dataInfo = None

        # Set the new stack in StackView widget
        super(StackViewMainWindow, self).setStack(stack, *args, **kwargs)
        self.setStatusBar(None)
