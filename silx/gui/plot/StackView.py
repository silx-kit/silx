# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
__date__ = "10/10/2018"

import numpy
import logging

import silx
from silx.gui import qt
from .. import icons
from . import items, PlotWindow, actions
from ..colors import Colormap
from ..colors import cursorColorForColormap
from .tools import LimitsToolBar
from .Profile import Profile3DToolBar
from ..widgets.FrameBrowser import HorizontalSliderWithBrowser

from silx.gui.plot.actions import control as actions_control
from silx.utils.array_like import DatasetView, ListOfImages
from silx.math import calibration
from silx.utils.deprecation import deprecated_warning

try:
    import h5py
except ImportError:
    def is_dataset(obj):
        return False
    h5py = None
else:
    from silx.io.utils import is_dataset

_logger = logging.getLogger(__name__)


class StackView(qt.QMainWindow):
    """Stack view widget, to display and browse through stack of
    images.

    The profile tool can be switched to "3D" mode, to compute the profile
    on each image of the stack (not only the active image currently displayed)
    and display the result as a slice.

    :param QWidget parent: the Qt parent, or None
    :param backend: The backend to use for the plot (default: matplotlib).
                    See :class:`.PlotWidget` for the list of supported backend.
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

    sigFrameChanged = qt.Signal(int)
    """Signal emitter when the frame number has changed.

    This signal provides the current frame number.
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

        self._first_stack_dimension = 0
        """Used for dimension labels and combobox"""

        self._titleCallback = self._defaultTitleCallback
        """Function returning the plot title based on the frame index.
        It can be set to a custom function using :meth:`setTitleCallback`"""

        self.calibrations3D = (calibration.NoCalibration(),
                               calibration.NoCalibration(),
                               calibration.NoCalibration())

        central_widget = qt.QWidget(self)

        self._plot = PlotWindow(parent=central_widget, backend=backend,
                                resetzoom=resetzoom, autoScale=autoScale,
                                logScale=logScale, grid=grid,
                                curveStyle=False, colormap=colormap,
                                aspectRatio=aspectRatio, yInverted=yinverted,
                                copy=copy, save=save, print_=print_,
                                control=control, position=position,
                                roi=False, mask=mask)
        self._plot.getIntensityHistogramAction().setVisible(True)
        self.sigInteractiveModeChanged = self._plot.sigInteractiveModeChanged
        self.sigActiveImageChanged = self._plot.sigActiveImageChanged
        self.sigPlotSignal = self._plot.sigPlotSignal

        if silx.config.DEFAULT_PLOT_IMAGE_Y_AXIS_ORIENTATION == 'downward':
            self._plot.getYAxis().setInverted(True)

        self._addColorBarAction()

        self._plot.profile = Profile3DToolBar(parent=self._plot,
                                              stackview=self)
        self._plot.addToolBar(self._plot.profile)
        self._plot.getXAxis().setLabel('Columns')
        self._plot.getYAxis().setLabel('Rows')
        self._plot.sigPlotSignal.connect(self._plotCallback)

        self.__planeSelection = PlanesWidget(self._plot)
        self.__planeSelection.sigPlaneSelectionChanged.connect(self.setPerspective)

        self._browser_label = qt.QLabel("Image index (Dim0):")

        self._browser = HorizontalSliderWithBrowser(central_widget)
        self._browser.setRange(0, 0)
        self._browser.valueChanged[int].connect(self.__updateFrameNumber)
        self._browser.setEnabled(False)

        layout = qt.QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot, 0, 0, 1, 3)
        layout.addWidget(self.__planeSelection, 1, 0)
        layout.addWidget(self._browser_label, 1, 1)
        layout.addWidget(self._browser, 1, 2)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # clear profile lines when the perspective changes (plane browsed changed)
        self.__planeSelection.sigPlaneSelectionChanged.connect(
            self._plot.profile.getProfilePlot().clear)
        self.__planeSelection.sigPlaneSelectionChanged.connect(
            self._plot.profile.clearProfile)

    def _addColorBarAction(self):
        self._plot.getColorBarWidget().setVisible(True)
        actions = self._plot.toolBar().actions()
        for index, action in enumerate(actions):
            if action is self._plot.getColormapAction():
                break
        self._colorbarAction = actions_control.ColorBarAction(self._plot, self._plot)
        self._plot.toolBar().insertAction(actions[index + 1], self._colorbarAction)

    def _plotCallback(self, eventDict):
        """Callback for plot events.

        Emit :attr:`valueChanged` signal, with (x, y, value) tuple of the
        cursor location in the plot."""
        if eventDict['event'] == 'mouseMoved':
            activeImage = self._plot.getActiveImage()
            if activeImage is not None:
                data = activeImage.getData()
                height, width = data.shape

                # Get corresponding coordinate in image
                origin = activeImage.getOrigin()
                scale = activeImage.getScale()
                x = int((eventDict['x'] - origin[0]) / scale[0])
                y = int((eventDict['y'] - origin[1]) / scale[1])

                if 0 <= x < width and 0 <= y < height:
                    self.valueChanged.emit(float(x), float(y),
                                           data[y][x])
                else:
                    self.valueChanged.emit(float(x), float(y),
                                           None)

    def getPerspective(self):
        """Returns the index of the dimension the stack is browsed with

        Possible values are: 0, 1, or 2.

        :rtype: int
        """
        return self._perspective

    def setPerspective(self, perspective):
        """Set the index of the dimension the stack is browsed with:

        - slice plane Dim1-Dim2: perspective 0
        - slice plane Dim0-Dim2: perspective 1
        - slice plane Dim0-Dim1: perspective 2

        :param int perspective: Orthogonal dimension number (0, 1, or 2)
        """
        if perspective == self._perspective:
            return
        else:
            if perspective > 2 or perspective < 0:
                raise ValueError(
                    "Perspective must be 0, 1 or 2, not %s" % perspective)

            self._perspective = int(perspective)
            self.__createTransposedView()
            self.__updateFrameNumber(self._browser.value())
            self._plot.resetZoom()
            self.__updatePlotLabels()
            self._updateTitle()
            self._browser_label.setText("Image index (Dim%d):" %
                                        (self._first_stack_dimension + perspective))

            self.sigPlaneSelectionChanged.emit(perspective)
            self.sigStackChanged.emit(self._stack.size if
                                      self._stack is not None else 0)
            self.__planeSelection.sigPlaneSelectionChanged.disconnect(self.setPerspective)
            self.__planeSelection.setPerspective(self._perspective)
            self.__planeSelection.sigPlaneSelectionChanged.connect(self.setPerspective)

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

        # ensure we have the stack encapsulated in an array-like object
        # having a transpose() method
        if isinstance(self._stack, numpy.ndarray):
            self.__transposed_view = self._stack

        elif is_dataset(self._stack) or isinstance(self._stack, DatasetView):
            self.__transposed_view = DatasetView(self._stack)

        elif isinstance(self._stack, ListOfImages):
            self.__transposed_view = ListOfImages(self._stack)

        # transpose the array-like object if necessary
        if self._perspective == 1:
            self.__transposed_view = self.__transposed_view.transpose((1, 0, 2))
        elif self._perspective == 2:
            self.__transposed_view = self.__transposed_view.transpose((2, 0, 1))

        self._browser.setRange(0, self.__transposed_view.shape[0] - 1)
        self._browser.setValue(0)

    def __updateFrameNumber(self, index):
        """Update the current image.

        :param index: index of the frame to be displayed
        """
        if self.__transposed_view is None:
            # no data set
            return
        self._plot.addImage(self.__transposed_view[index, :, :],
                            origin=self._getImageOrigin(),
                            scale=self._getImageScale(),
                            legend=self.__imageLegend,
                            resetzoom=False)
        self._updateTitle()
        self.sigFrameChanged.emit(index)

    def _set3DScaleAndOrigin(self, calibrations):
        """Set scale and origin for all 3 axes, to be used when plotting
        an image.

        See setStack for parameter documentation
        """
        if calibrations is None:
            self.calibrations3D = (calibration.NoCalibration(),
                                   calibration.NoCalibration(),
                                   calibration.NoCalibration())
        else:
            self.calibrations3D = []
            for i, calib in enumerate(calibrations):
                if hasattr(calib, "__len__") and len(calib) == 2:
                    calib = calibration.LinearCalibration(calib[0], calib[1])
                elif calib is None:
                    calib = calibration.NoCalibration()
                elif not isinstance(calib, calibration.AbstractCalibration):
                    raise TypeError("calibration must be a 2-tuple, None or" +
                                    " an instance of an AbstractCalibration " +
                                    "subclass")
                elif not calib.is_affine():
                    _logger.warning(
                            "Calibration for dimension %d is not linear, "
                            "it will be ignored for scaling the graph axes.",
                            i)
                self.calibrations3D.append(calib)

    def getCalibrations(self, order='array'):
        """Returns currently used calibrations for each axis

        Returned calibrations might differ from the ones that were set as
        non-linear calibrations used for image axes are temporarily ignored.

        :param str order:
            'array' to sort calibrations as data array (dim0, dim1, dim2),
            'axes' to sort calibrations as currently selected x, y and z axes.
        :return: Calibrations ordered depending on order
        :rtype: List[~silx.math.calibration.AbstractCalibration]
        """
        assert order in ('array', 'axes')
        calibs = []

        # filter out non-linear calibration for graph axes
        for index, calib in enumerate(self.calibrations3D):
            if index != self._perspective and not calib.is_affine():
                calib = calibration.NoCalibration()
            calibs.append(calib)

        if order == 'axes':  # Move 'z' axis to the end
            xy_dims = [d for d in (0, 1, 2) if d != self._perspective]
            calibs = [calibs[max(xy_dims)],
                      calibs[min(xy_dims)],
                      calibs[self._perspective]]

        return tuple(calibs)

    def _getImageScale(self):
        """
        :return: 2-tuple (XScale, YScale) for current image view
        """
        xcalib, ycalib, _zcalib = self.getCalibrations(order='axes')
        return xcalib.get_slope(), ycalib.get_slope()

    def _getImageOrigin(self):
        """
        :return: 2-tuple (XOrigin, YOrigin) for current image view
        """
        xcalib, ycalib, _zcalib = self.getCalibrations(order='axes')
        return xcalib(0), ycalib(0)

    def _getImageZ(self, index):
        """
        :param idx: 0-based image index in the stack
        :return: calibrated Z value corresponding to the image idx
        """
        _xcalib, _ycalib, zcalib = self.getCalibrations(order='axes')
        return zcalib(index)

    def _updateTitle(self):
        frame_idx = self._browser.value()
        self._plot.setGraphTitle(self._titleCallback(frame_idx))

    def _defaultTitleCallback(self, index):
        return "Image z=%g" % self._getImageZ(index)

    # public API, stack specific methods
    def setStack(self, stack, perspective=None, reset=True, calibrations=None):
        """Set the 3D stack.

        The perspective parameter is used to define which dimension of the 3D
        array is to be used as frame index. The lowest remaining dimension
        number is the row index of the displayed image (Y axis), and the highest
        remaining dimension is the column index (X axis).

        :param stack: 3D stack, or `None` to clear plot.
        :type stack: 3D numpy.ndarray, or 3D h5py.Dataset, or list/tuple of 2D
            numpy arrays, or None.
        :param int perspective: Dimension for the frame index: 0, 1 or 2.
            Use ``None`` to keep the current perspective (default).
        :param bool reset: Whether to reset zoom or not.
        :param calibrations: Sequence of 3 calibration objects for each axis.
            These objects can be a subclass of :class:`AbstractCalibration`,
            or 2-tuples *(a, b)* where *a* is the y-intercept and *b* is the
            slope of a linear calibration (:math:`x \mapsto a + b x`)
        """
        if stack is None:
            self.clear()
            self.sigStackChanged.emit(0)
            return

        self._set3DScaleAndOrigin(calibrations)

        # stack as list of 2D arrays: must be converted into an array_like
        if not isinstance(stack, numpy.ndarray):
            if not is_dataset(stack):
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

        perspective_changed = False
        if perspective not in [None, self._perspective]:
            perspective_changed = True
            self.setPerspective(perspective)

        # This call to setColormap redefines the meaning of autoscale
        # for 3D volume: take global min/max rather than frame min/max
        if self.__autoscaleCmap:
            self.setColormap(autoscale=True)

        # init plot
        self._plot.addImage(self.__transposed_view[0, :, :],
                            legend=self.__imageLegend,
                            colormap=self.getColormap(),
                            origin=self._getImageOrigin(),
                            scale=self._getImageScale(),
                            replace=True,
                            resetzoom=False)
        self._plot.setActiveImage(self.__imageLegend)
        self.__updatePlotLabels()
        self._updateTitle()

        if reset:
            self._plot.resetZoom()

        # enable and init browser
        self._browser.setEnabled(True)

        if not perspective_changed:    # avoid double signal (see self.setPerspective)
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
        image = self._plot.getActiveImage()
        if image is None:
            return None

        if isinstance(image, items.ColormapMixIn):
            colormap = image.getColormap()
        else:
            colormap = None

        params = {
            'info': image.getInfo(),
            'origin': image.getOrigin(),
            'scale': image.getScale(),
            'z': image.getZValue(),
            'selectable': image.isSelectable(),
            'draggable': image.isDraggable(),
            'colormap': colormap,
            'xlabel': image.getXLabel(),
            'ylabel': image.getYLabel(),
        }
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
        image = self._plot.getActiveImage()
        if image is None:
            return None

        if isinstance(image, items.ColormapMixIn):
            colormap = image.getColormap()
        else:
            colormap = None

        params = {
            'info': image.getInfo(),
            'origin': image.getOrigin(),
            'scale': image.getScale(),
            'z': image.getZValue(),
            'selectable': image.isSelectable(),
            'draggable': image.isDraggable(),
            'colormap': colormap,
            'xlabel': image.getXLabel(),
            'ylabel': image.getYLabel(),
        }
        if returnNumpyArray or copy:
            return numpy.array(self.__transposed_view, copy=copy), params
        return self.__transposed_view, params

    def setFrameNumber(self, number):
        """Set the frame selection to a specific value

        :param int number: Number of the frame
        """
        self._browser.setValue(number)

    def getFrameNumber(self):
        """Set the frame selection to a specific value

        :return: Index of currently displayed frame
        :rtype: int
        """
        return self._browser.value()

    def setFirstStackDimension(self, first_stack_dimension):
        """When viewing the last 3 dimensions of an n-D array (n>3), you can
        use this method to change the text in the combobox.

        For instance, for a 7-D array, first stack dim is 4, so the default
        "Dim1-Dim2" text should be replaced with "Dim5-Dim6" (dimensions
        numbers are 0-based).

        :param int first_stack_dim: First stack dimension (n-3) when viewing the
            last 3 dimensions of an n-D array.
        """
        old_state = self.__planeSelection.blockSignals(True)
        self.__planeSelection.setFirstStackDimension(first_stack_dimension)
        self.__planeSelection.blockSignals(old_state)
        self._first_stack_dimension = first_stack_dimension
        self._browser_label.setText("Image index (Dim%d):" % first_stack_dimension)

    def setTitleCallback(self, callback):
        """Set a user defined function to generate the plot title based on the
        image/frame index.

        The callback function must accept an integer as a its first positional
        parameter and must not require any other mandatory parameter.
        It must return a string.

        To switch back the default behavior, you can pass ``None``::

            mystackview.setTitleCallback(None)

        To have no title, pass a function that returns an empty string::

            mystackview.setTitleCallback(lambda idx: "")

        :param callback: Callback function generating the stack title based
            on the frame number.
        """

        if callback is None:
            self._titleCallback = self._defaultTitleCallback
        elif callable(callback):
            self._titleCallback = callback
        else:
            raise TypeError("Provided callback is not callable")
        self._updateTitle()

    def clear(self):
        """Clear the widget:

         - clear the plot
         - clear the loaded data volume
        """
        self._stack = None
        self.__transposed_view = None
        self._perspective = 0
        self._browser.setEnabled(False)
        # reset browser range
        self._browser.setRange(0, 0)
        self._plot.clear()

    def setLabels(self, labels=None):
        """Set the labels to be displayed on the plot axes.

        You must provide a sequence of 3 strings, corresponding to the 3
        dimensions of the original data volume.
        The proper label will automatically be selected for each plot axis
        when the volume is rotated (when different axes are selected as the
        X and Y axes).

        :param List[str] labels: 3 labels corresponding to the 3 dimensions
             of the data volumes.
        """

        default_labels = ["Dimension %d" % self._first_stack_dimension,
                          "Dimension %d" % (self._first_stack_dimension + 1),
                          "Dimension %d" % (self._first_stack_dimension + 2)]
        if labels is None:
            new_labels = default_labels
        else:
            # filter-out None
            new_labels = []
            for i, label in enumerate(labels):
                new_labels.append(label or default_labels[i])

        self.__dimensionsLabels = new_labels
        self.__updatePlotLabels()

    def getLabels(self):
        """Return dimension labels displayed on the plot axes

        :return: List of three strings corresponding to the 3 dimensions
            of the stack: (name_dim0, name_dim1, name_dim2)
        """
        return self.__dimensionsLabels

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
            Or a :class`.Colormap` object.
        :type colormap: dict or str.
        :param str normalization: Colormap mapping: 'linear' or 'log'.
        :param bool autoscale: Whether to use autoscale or [vmin, vmax] range.
            Default value of autoscale is False. This option is not compatible
            with h5py datasets.
        :param float vmin: The minimum value of the range to use if
                           'autoscale' is False.
        :param float vmax: The maximum value of the range to use if
                           'autoscale' is False.
        :param numpy.ndarray colors: Only used if name is None.
            Custom colormap colors as Nx3 or Nx4 RGB or RGBA arrays
        """
        # if is a colormap object or a dictionary
        if isinstance(colormap, Colormap) or isinstance(colormap, dict):
            # Support colormap parameter as a dict
            errmsg = "If colormap is provided as a Colormap object, all other parameters"
            errmsg += " must not be specified when calling setColormap"
            assert normalization is None, errmsg
            assert autoscale is None, errmsg
            assert vmin is None, errmsg
            assert vmax is None, errmsg
            assert colors is None, errmsg

            if isinstance(colormap, dict):
                reason = 'colormap parameter should now be an object'
                replacement = 'Colormap()'
                since_version = '0.6'
                deprecated_warning(type_='function',
                                   name='setColormap',
                                   reason=reason,
                                   replacement=replacement,
                                   since_version=since_version)
                _colormap = Colormap._fromDict(colormap)
            else:
                _colormap = colormap
        else:
            norm = normalization if normalization is not None else 'linear'
            name = colormap if colormap is not None else 'gray'
            _colormap = Colormap(name=name,
                                 normalization=norm,
                                 vmin=vmin,
                                 vmax=vmax,
                                 colors=colors)

            # Patch: since we don't apply this colormap to a single 2D data but
            # a 2D stack we have to deal manually with vmin, vmax
            if autoscale is None:
                # set default
                autoscale = False
            elif autoscale and is_dataset(self._stack):
                # h5py dataset has no min()/max() methods
                raise RuntimeError(
                    "Cannot auto-scale colormap for a h5py dataset")
            else:
                autoscale = autoscale
            self.__autoscaleCmap = autoscale

            if autoscale and (self._stack is not None):
                _vmin, _vmax = _colormap.getColormapRange(data=self._stack)
                _colormap.setVRange(vmin=_vmin, vmax=_vmax)
            else:
                if vmin is None and self._stack is not None:
                    _colormap.setVMin(self._stack.min())
                else:
                    _colormap.setVMin(vmin)
                if vmax is None and self._stack is not None:
                    _colormap.setVMax(self._stack.max())
                else:
                    _colormap.setVMax(vmax)

        cursorColor = cursorColorForColormap(_colormap.getName())
        self._plot.setInteractiveMode('zoom', color=cursorColor)

        self._plot.setDefaultColormap(_colormap)

        # Update active image colormap
        activeImage = self._plot.getActiveImage()
        if isinstance(activeImage, items.ColormapMixIn):
            activeImage.setColormap(self.getColormap())

    def getPlot(self):
        """Return the :class:`PlotWidget`.

        This gives access to advanced plot configuration options.
        Be warned that modifying the plot can cause issues, and some changes
        you make to the plot could be overwritten by the :class:`StackView`
        widget's internal methods and callbacks.

        :return: instance of :class:`PlotWidget` used in widget
        """
        return self._plot

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

    def setOptionVisible(self, isVisible):
        """
        Set the visibility of the browsing options.

        :param bool isVisible: True to have the options visible, else False
        """
        self._browser.setVisible(isVisible)
        self.__planeSelection.setVisible(isVisible)

    # proxies to PlotWidget or PlotWindow methods
    def getProfileToolbar(self):
        """Profile tools attached to this plot

        See :class:`silx.gui.plot.Profile.Profile3DToolBar`
        """
        return self._plot.profile

    def getGraphTitle(self):
        """Return the plot main title as a str.
        """
        return self._plot.getGraphTitle()

    def setGraphTitle(self, title=""):
        """Set the plot main title.

        :param str title: Main title of the plot (default: '')
        """
        return self._plot.setGraphTitle(title)

    def getGraphXLabel(self):
        """Return the current horizontal axis label as a str.
        """
        return self._plot.getXAxis().getLabel()

    def setGraphXLabel(self, label=None):
        """Set the plot horizontal axis label.

        :param str label: The horizontal axis label
        """
        if label is None:
            label = self.__dimensionsLabels[1 if self._perspective == 2 else 2]
        self._plot.getXAxis().setLabel(label)

    def getGraphYLabel(self, axis='left'):
        """Return the current vertical axis label as a str.

        :param str axis: The Y axis for which to get the label (left or right)
        """
        return self._plot.getYAxis().getLabel(axis)

    def setGraphYLabel(self, label=None, axis='left'):
        """Set the vertical axis label on the plot.

        :param str label: The Y axis label
        :param str axis: The Y axis for which to set the label (left or right)
        """
        if label is None:
            label = self.__dimensionsLabels[1 if self._perspective == 0 else 0]
        self._plot.getYAxis(axis=axis).setLabel(label)

    def resetZoom(self):
        """Reset the plot limits to the bounds of the data and redraw the plot.

        This method is a simple proxy to the legacy :class:`PlotWidget` method
        of the same name. Using the object oriented approach is now
        preferred::

            stackview.getPlot().resetZoom()
        """
        self._plot.resetZoom()

    def setYAxisInverted(self, flag=True):
        """Set the Y axis orientation.

        This method is a simple proxy to the legacy :class:`PlotWidget` method
        of the same name. Using the object oriented approach is now
        preferred::

            stackview.getPlot().setYAxisInverted(flag)

        :param bool flag: True for Y axis going from top to bottom,
                          False for Y axis going from bottom to top
        """
        self._plot.setYAxisInverted(flag)

    def isYAxisInverted(self):
        """Return True if Y axis goes from top to bottom, False otherwise.

        This method is a simple proxy to the legacy :class:`PlotWidget` method
        of the same name. Using the object oriented approach is now
        preferred::

            stackview.getPlot().isYAxisInverted()"""
        return self._plot.isYAxisInverted()

    def getSupportedColormaps(self):
        """Get the supported colormap names as a tuple of str.

        The list should at least contain and start by:
        ('gray', 'reversed gray', 'temperature', 'red', 'green', 'blue')

        This method is a simple proxy to the legacy :class:`PlotWidget` method
        of the same name. Using the object oriented approach is now
        preferred::

            stackview.getPlot().getSupportedColormaps()
        """
        return self._plot.getSupportedColormaps()

    def isKeepDataAspectRatio(self):
        """Returns whether the plot is keeping data aspect ratio or not.

        This method is a simple proxy to the legacy :class:`PlotWidget` method
        of the same name. Using the object oriented approach is now
        preferred::

            stackview.getPlot().isKeepDataAspectRatio()"""
        return self._plot.isKeepDataAspectRatio()

    def setKeepDataAspectRatio(self, flag=True):
        """Set whether the plot keeps data aspect ratio or not.

        This method is a simple proxy to the legacy :class:`PlotWidget` method
        of the same name. Using the object oriented approach is now
        preferred::

            stackview.getPlot().setKeepDataAspectRatio(flag)

        :param bool flag: True to respect data aspect ratio
        """
        self._plot.setKeepDataAspectRatio(flag)

    # kind of private methods, but needed by Profile
    def getActiveImage(self, just_legend=False):
        """Returns the currently active image object.

        It returns None in case of not having an active image.

        This method is a simple proxy to the legacy :class:`PlotWidget` method
        of the same name. Using the object oriented approach is now
        preferred::

            stackview.getPlot().getActiveImage()

        :param bool just_legend: True to get the legend of the image,
            False (the default) to get the image data and info.
            Note: :class:`StackView` uses the same legend for all frames.
        :return: legend or image object
        :rtype: str or list or None
        """
        return self._plot.getActiveImage(just_legend=just_legend)

    def getColorBarAction(self):
        """Returns the action managing the visibility of the colorbar.

        .. warning:: to show/hide the plot colorbar call directly the ColorBar
            widget using getColorBarWidget()

        :rtype: QAction
        """
        return self._colorbarAction

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
        self._setCBChoices(first_stack_dimension=0)
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

    def _setCBChoices(self, first_stack_dimension):
        self.qcbAxisSelection.clear()

        dim1dim2 = 'Dim%d-Dim%d' % (first_stack_dimension + 1,
                                    first_stack_dimension + 2)
        dim0dim2 = 'Dim%d-Dim%d' % (first_stack_dimension,
                                    first_stack_dimension + 2)
        dim0dim1 = 'Dim%d-Dim%d' % (first_stack_dimension,
                                    first_stack_dimension + 1)

        self.qcbAxisSelection.addItem(icons.getQIcon("cube-front"), dim1dim2)
        self.qcbAxisSelection.addItem(icons.getQIcon("cube-bottom"), dim0dim2)
        self.qcbAxisSelection.addItem(icons.getQIcon("cube-left"), dim0dim1)

    def setFirstStackDimension(self, first_stack_dim):
        """When viewing the last 3 dimensions of an n-D array (n>3), you can
        use this method to change the text in the combobox.

        For instance, for a 7-D array, first stack dim is 4, so the default
        "Dim1-Dim2" text should be replaced with "Dim5-Dim6" (dimensions
        numbers are 0-based).

        :param int first_stack_dim: First stack dimension (n-3) when viewing the
            last 3 dimensions of an n-D array.
        """
        self._setCBChoices(first_stack_dim)

    def setPerspective(self, perspective):
        """Update the combobox selection.

          - slice plane Dim1-Dim2: perspective 0
          - slice plane Dim0-Dim2: perspective 1
          - slice plane Dim0-Dim1: perspective 2

        :param perspective: Orthogonal dimension number (0, 1, or 2)
        """
        self.qcbAxisSelection.setCurrentIndex(perspective)


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
        menu.addAction(self._plot.getOutputToolBar().getSaveAction())
        menu.addAction(self._plot.getOutputToolBar().getPrintAction())
        menu.addSeparator()
        action = menu.addAction('Quit')
        action.triggered[bool].connect(qt.QApplication.instance().quit)

        menu = self.menuBar().addMenu('Edit')
        menu.addAction(self._plot.getOutputToolBar().getCopyAction())
        menu.addSeparator()
        menu.addAction(self._plot.getResetZoomAction())
        menu.addAction(self._plot.getColormapAction())
        menu.addAction(self.getColorBarAction())

        menu.addAction(actions.control.KeepAspectRatioAction(self._plot, self))
        menu.addAction(actions.control.YAxisInvertedAction(self._plot, self))

        menu = self.menuBar().addMenu('Profile')
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
