import logging

from silx.gui import qt
from silx.math.calibration import ArrayCalibration, LinearCalibration, NoCalibration
from .NumpyAxesSelector import NumpyAxesSelector


_logger = logging.getLogger(__name__)


class ArrayVolumePlot(qt.QWidget):
    """
    Widget for plotting a n-D array (n >= 3) as a 3D scalar field.
    Three axis arrays can be provided to calibrate the axes.

    The signal array can have an arbitrary number of dimensions, the only
    limitation being that the last 3 dimensions must have the same length as
    the axes arrays.

    Sliders are provided to select indices on the first (n - 3) dimensions of
    the signal array, and the plot is updated to load the stack corresponding
    to the selection.
    """

    def __init__(self, parent=None):
        """

        :param parent: Parent QWidget
        """
        super().__init__(parent)

        self.__signal = None
        self.__signal_name = None
        # the Z, Y, X axes apply to the last three dimensions of the signal
        # (in that order)
        self.__z_axis = None
        self.__z_axis_name = None
        self.__y_axis = None
        self.__y_axis_name = None
        self.__x_axis = None
        self.__x_axis_name = None

        from ._VolumeWindow import VolumeWindow

        self._view = VolumeWindow(self)

        self._hline = qt.QFrame(self)
        self._hline.setFrameStyle(qt.QFrame.HLine)
        self._hline.setFrameShadow(qt.QFrame.Sunken)
        self._legend = qt.QLabel(self)
        self._axesSelector = NumpyAxesSelector(self)
        self._axesSelector.setNamedAxesSelectorVisibility(False)
        self.__axes_selector_is_connected = False

        layout = qt.QVBoxLayout()
        layout.addWidget(self._view)
        layout.addWidget(self._hline)
        layout.addWidget(self._legend)
        layout.addWidget(self._axesSelector)

        self.setLayout(layout)

    def getVolumeView(self):
        """Returns the plot used for the display

        :rtype: SceneWindow
        """
        return self._view

    def setData(
        self,
        signal,
        x_axis=None,
        y_axis=None,
        z_axis=None,
        signal_name=None,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        title=None,
    ):
        """

        :param signal: n-D dataset, whose last 3 dimensions are used as the
            3D stack values.
        :param x_axis: 1-D dataset used as the image's x coordinates. If
            provided, its lengths must be equal to the length of the last
            dimension of ``signal``.
        :param y_axis: 1-D dataset used as the image's y. If provided,
            its lengths must be equal to the length of the 2nd to last
            dimension of ``signal``.
        :param z_axis: 1-D dataset used as the image's z. If provided,
            its lengths must be equal to the length of the 3rd to last
            dimension of ``signal``.
        :param signal_name: Label used in the legend
        :param xlabel: Label for X axis
        :param ylabel: Label for Y axis
        :param zlabel: Label for Z axis
        :param title: Graph title
        """
        if self.__axes_selector_is_connected:
            self._axesSelector.selectionChanged.disconnect(self._updateVolume)
            self.__axes_selector_is_connected = False

        self.__signal = signal
        self.__signal_name = signal_name or ""
        self.__x_axis = x_axis
        self.__x_axis_name = xlabel
        self.__y_axis = y_axis
        self.__y_axis_name = ylabel
        self.__z_axis = z_axis
        self.__z_axis_name = zlabel

        self._axesSelector.setData(signal)
        self._axesSelector.setAxisNames(["Y", "X", "Z"])

        self._updateVolume()

        # the legend label shows the selection slice producing the volume
        # (only interesting for ndim > 3)
        if signal.ndim > 3:
            self._axesSelector.setVisible(True)
            self._legend.setVisible(True)
            self._hline.setVisible(True)
        else:
            self._axesSelector.setVisible(False)
            self._legend.setVisible(False)
            self._hline.setVisible(False)

        if not self.__axes_selector_is_connected:
            self._axesSelector.selectionChanged.connect(self._updateVolume)
            self.__axes_selector_is_connected = True

    def _updateVolume(self):
        """Update displayed stack according to the current axes selector
        data."""
        x_axis = self.__x_axis
        y_axis = self.__y_axis
        z_axis = self.__z_axis

        offset = []
        scale = []
        for axis in [x_axis, y_axis, z_axis]:
            if axis is None:
                calibration = NoCalibration()
            elif len(axis) == 2:
                calibration = LinearCalibration(y_intercept=axis[0], slope=axis[1])
            else:
                calibration = ArrayCalibration(axis)
            if not calibration.is_affine():
                _logger.warning("Axis has not linear values, ignored")
                offset.append(0.0)
                scale.append(1.0)
            else:
                offset.append(calibration(0))
                scale.append(calibration.get_slope())

        legend = self.__signal_name + "["
        for sl in self._axesSelector.selection():
            if sl == slice(None):
                legend += ":, "
            else:
                legend += str(sl) + ", "
        legend = legend[:-2] + "]"
        self._legend.setText("Displayed data: " + legend)

        # Update SceneWidget
        data = self._axesSelector.selectedData()

        volumeView = self.getVolumeView()
        volumeView.setData(data, offset=offset, scale=scale)
        volumeView.setAxesLabels(
            self.__x_axis_name, self.__y_axis_name, self.__z_axis_name
        )

    def clear(self):
        old = self._axesSelector.blockSignals(True)
        self._axesSelector.clear()
        self._axesSelector.blockSignals(old)
        self.getVolumeView().clear()
