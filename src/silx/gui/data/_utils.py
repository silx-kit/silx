import numbers

import h5py
import numpy

from ...io.commonh5 import Dataset
from ...io.nxdata.parse import NXdata
from ...math.calibration import (
    AbstractCalibration,
    ArrayCalibration,
    LinearCalibration,
    NoCalibration,
)
from .. import qt
from ..hdf5 import H5Node
from ..plot import PlotWidget
from ..plot.actions import PlotAction
from ..plot.items import ImageBase

ImageAxis = h5py.Dataset | Dataset | None


class _ToggleScaleToCursorAction(PlotAction):
    """Toggle the colormap dialog "Cursor" mode with the W key.

    The shortcut is active when the plot has the keyboard focus.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`.
        Set it (e.g., to the plot) to keep the action alive,
        as :meth:`QWidget.addAction` does not take ownership.
    """

    def __init__(self, plot: PlotWidget, parent: qt.QObject | None = None):
        super().__init__(
            plot,
            icon=qt.QIcon(),
            text="Scale to cursor",
            tooltip="Toggle the colormap range following the mouse cursor",
            triggered=self._actionTriggered,
            parent=parent,
        )
        self.setShortcut(qt.QKeySequence(qt.Qt.Key_W))
        self.setShortcutContext(qt.Qt.WidgetWithChildrenShortcut)

    def _actionTriggered(self, checked: bool = False):
        plot = self.plot
        if plot is None:
            return
        colormapAction = plot.getColormapAction()
        action = colormapAction.getColormapDialog().getScaleToCursorAction()
        if not action.isChecked():
            # The dialog can be shared with other plots: target this one
            colormapAction._updateColormap()
        action.trigger()


def normalizeData(data):
    """Returns a normalized data.

    If the data embed a numpy data or a dataset it is returned.
    Else returns the input data."""
    if isinstance(data, H5Node):
        if data.is_broken:
            return None
        return data.h5py_object
    return data


def normalizeComplex(data):
    """Returns a normalized complex data.

    If the data is a numpy data with complex, returns the
    absolute value.
    Else returns the input data."""
    if hasattr(data, "dtype"):
        isComplex = numpy.issubdtype(data.dtype, numpy.complexfloating)
    else:
        isComplex = isinstance(data, numbers.Complex)
    if isComplex:
        data = numpy.absolute(data)
    return data


def isRgba(nxd: NXdata) -> bool:
    return (
        nxd.interpretation in ("rgb-image", "rgba-image")
        and nxd.signal.ndim >= 3
        and nxd.signal.shape[-1] in (3, 4)
    )


def isScatter(nxd: NXdata, naxes: int) -> bool:
    if nxd.signal.ndim != 1:
        return False

    # Check that all axes match the signal number of values
    if any(axis is None or axis.shape != nxd.signal.shape for axis in nxd.axes):
        return False
    return len(nxd.axes) == naxes


def _getAxisCalib(axis: ImageAxis, axis_length: int) -> AbstractCalibration:
    if axis is None:
        return ArrayCalibration(numpy.arange(axis_length))
    if numpy.isscalar(axis) or len(axis) == 1:
        # constant axis
        return ArrayCalibration(axis * numpy.ones((axis_length,)))
    if len(axis) == 2:
        # linear calibration
        return LinearCalibration(slope=axis[0], y_intercept=axis[1])
    # array
    return ArrayCalibration(axis)


def getAxesCalib(
    image_shape: tuple[int, int], x_axis: ImageAxis, y_axis: ImageAxis
) -> tuple[AbstractCalibration, AbstractCalibration]:
    if x_axis is None and y_axis is None:
        return NoCalibration(), NoCalibration()

    return _getAxisCalib(x_axis, image_shape[1]), _getAxisCalib(y_axis, image_shape[0])


def setImageCoords(
    item: ImageBase, xcalib: AbstractCalibration, ycalib: AbstractCalibration
):
    item.setOrigin((xcalib(0), ycalib(0)))
    item.setScale((xcalib.get_slope(), ycalib.get_slope()))
