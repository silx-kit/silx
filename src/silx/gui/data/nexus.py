import logging

import numpy

import silx.io
from silx.gui import icons, qt
from silx.io import nxdata
from silx.io.nxdata import get_attr_as_unicode

from ._DataInfo import DataInfo
from .composite import CompositeDataView, SelectManyDataView
from .modes import (
    NXDATA_CURVE_MODE,
    NXDATA_IMAGE_MODE,
    NXDATA_INVALID_MODE,
    NXDATA_MODE,
    NXDATA_SCALAR_MODE,
    NXDATA_VOLUME_AS_STACK_MODE,
    NXDATA_VOLUME_MODE,
    NXDATA_XYVSCATTER_MODE,
)
from .NXdataWidgets import ArrayImagePlot
from .views import DataView
from ._utils import normalizeComplex

_logger = logging.getLogger(__name__)


class _InvalidNXdataView(DataView):
    """DataView showing a simple label with an error message
    to inform that a group with @NX_class=NXdata cannot be
    interpreted by any NXDataview."""

    def __init__(self, parent):
        DataView.__init__(self, parent, modeId=NXDATA_INVALID_MODE)
        self._msg = ""

    def createWidget(self, parent):
        widget = qt.QLabel(parent)
        widget.setWordWrap(True)
        widget.setStyleSheet("QLabel { color : red; }")
        return widget

    def axesNames(self, data, info):
        return []

    def clear(self):
        self.getWidget().setText("")

    def setData(self, data):
        self.getWidget().setText(self._msg)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)

        if not info.isInvalidNXdata:
            return DataView.UNSUPPORTED

        if info.hasNXdata:
            self._msg = "NXdata seems valid, but cannot be displayed "
            self._msg += "by any existing plot widget."
        else:
            nx_class = get_attr_as_unicode(data, "NX_class")
            if nx_class == "NXdata":
                # invalid: could not even be parsed by NXdata
                self._msg = "Group has @NX_class = NXdata, but could not be interpreted"
                self._msg += " as valid NXdata."
            elif nx_class == "NXroot" or silx.io.is_file(data):
                default_entry = data[data.attrs["default"]]
                default_nxdata_name = default_entry.attrs["default"]
                self._msg = "NXroot group provides a @default attribute "
                self._msg += "pointing to a NXentry which defines its own "
                self._msg += "@default attribute, "
                if default_nxdata_name not in default_entry:
                    self._msg += " but no corresponding NXdata group exists."
                elif (
                    get_attr_as_unicode(default_entry[default_nxdata_name], "NX_class")
                    != "NXdata"
                ):
                    self._msg += " but the corresponding item is not a "
                    self._msg += "NXdata group."
                else:
                    self._msg += " but the corresponding NXdata seems to be"
                    self._msg += " malformed."
            else:
                self._msg = "Group provides a @default attribute,"
                default_nxdata_name = data.attrs["default"]
                if default_nxdata_name not in data:
                    self._msg += " but no corresponding NXdata group exists."
                elif (
                    get_attr_as_unicode(data[default_nxdata_name], "NX_class")
                    != "NXdata"
                ):
                    self._msg += " but the corresponding item is not a "
                    self._msg += "NXdata group."
                else:
                    self._msg += " but the corresponding NXdata seems to be"
                    self._msg += " malformed."
        return 100


class _NXdataBaseDataView(DataView):
    """Base class for NXdata DataView"""

    def __init__(self, *args, **kwargs):
        DataView.__init__(self, *args, **kwargs)

    def _updateColormap(self, nxdata):
        """Update used colormap according to nxdata's SILX_style"""
        cmap_norm = nxdata.plot_style.signal_scale_type
        if cmap_norm is not None:
            self.defaultColormap().setNormalization(
                "log" if cmap_norm == "log" else "linear"
            )


class _NXdataScalarView(_NXdataBaseDataView):
    """DataView using a table view for displaying NXdata scalars:
    0-D signal or n-D signal with *@interpretation=scalar*"""

    def __init__(self, parent):
        _NXdataBaseDataView.__init__(self, parent, modeId=NXDATA_SCALAR_MODE)

    def createWidget(self, parent):
        from silx.gui.data.ArrayTableWidget import ArrayTableWidget

        widget = ArrayTableWidget(parent)
        # widget.displayAxesSelector(False)
        return widget

    def axesNames(self, data, info):
        return ["col", "row"]

    def clear(self):
        self.getWidget().setArrayData(numpy.array([[]]), labels=True)

    def setData(self, data):
        data = self.normalizeData(data)
        # data could be a NXdata or an NXentry
        nxd = nxdata.get_default(data, validate=False)
        signal = nxd.signal
        self.getWidget().setArrayData(signal, labels=True)

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)

        if info.hasNXdata and not info.isInvalidNXdata:
            nxd = nxdata.get_default(data, validate=False)
            if nxd.signal_is_0d or nxd.interpretation in ["scalar", "scaler"]:
                return 100
        return DataView.UNSUPPORTED


class _NXdataCurveView(_NXdataBaseDataView):
    """DataView using a Plot1D for displaying NXdata curves:
    1-D signal or n-D signal with *@interpretation=spectrum*.

    It also handles basic scatter plots:
    a 1-D signal with one axis whose values are not monotonically increasing.
    """

    def __init__(self, parent):
        _NXdataBaseDataView.__init__(self, parent, modeId=NXDATA_CURVE_MODE)

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import ArrayCurvePlot

        widget = ArrayCurvePlot(parent)
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return None

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = nxdata.get_default(data, validate=False)
        signals_names = [nxd.signal_name] + nxd.auxiliary_signals_names
        if nxd.axes_dataset_names[-1] is not None:
            x_errors = nxd.get_axis_errors(nxd.axes_dataset_names[-1])
        else:
            x_errors = None

        self.getWidget().setCurvesData(
            [nxd.signal] + nxd.auxiliary_signals,
            nxd.axes[-1],
            yerror=nxd.errors,
            xerror=x_errors,
            ylabels=signals_names,
            xlabel=nxd.axes_names[-1],
            title=nxd.title or signals_names[0],
            xscale=nxd.plot_style.axes_scale_types[-1],
            yscale=nxd.plot_style.signal_scale_type,
        )

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.hasNXdata and not info.isInvalidNXdata:
            if nxdata.get_default(data, validate=False).is_curve:
                return 100
        return DataView.UNSUPPORTED


class _NXdataXYVScatterView(_NXdataBaseDataView):
    """DataView using a Plot1D for displaying NXdata 3D scatters as
    a scatter of coloured points (1-D signal with 2 axes)"""

    def __init__(self, parent):
        _NXdataBaseDataView.__init__(self, parent, modeId=NXDATA_XYVSCATTER_MODE)

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import XYVScatterPlot

        widget = XYVScatterPlot(parent)
        widget.getScatterView().setColormap(self.defaultColormap())
        widget.getScatterView().getScatterToolBar().getColormapAction().setColormapDialog(
            self.defaultColorDialog()
        )
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return None

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = nxdata.get_default(data, validate=False)

        x_axis, y_axis = nxd.axes[-2:]
        if x_axis is None:
            x_axis = numpy.arange(nxd.signal.size)
        if y_axis is None:
            y_axis = numpy.arange(nxd.signal.size)

        x_label, y_label = nxd.axes_names[-2:]
        if x_label is not None:
            x_errors = nxd.get_axis_errors(x_label)
        else:
            x_errors = None

        if y_label is not None:
            y_errors = nxd.get_axis_errors(y_label)
        else:
            y_errors = None

        self._updateColormap(nxd)

        self.getWidget().setScattersData(
            y_axis,
            x_axis,
            values=[nxd.signal] + nxd.auxiliary_signals,
            yerror=y_errors,
            xerror=x_errors,
            ylabel=y_label,
            xlabel=x_label,
            title=nxd.title,
            scatter_titles=[nxd.signal_name] + nxd.auxiliary_signals_names,
            xscale=nxd.plot_style.axes_scale_types[-2],
            yscale=nxd.plot_style.axes_scale_types[-1],
        )

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.hasNXdata and not info.isInvalidNXdata:
            if nxdata.get_default(data, validate=False).is_x_y_value_scatter:
                # It have to be a little more than a NX curve priority
                return 110

        return DataView.UNSUPPORTED


class _NXdataImageView(_NXdataBaseDataView):
    """DataView using a Plot2D for displaying NXdata images:
    2-D signal or n-D signals with *@interpretation=image*."""

    def __init__(self, parent):
        _NXdataBaseDataView.__init__(self, parent, modeId=NXDATA_IMAGE_MODE)

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import ArrayImagePlot

        widget = ArrayImagePlot(parent)
        widget.getPlot().setDefaultColormap(self.defaultColormap())
        widget.getPlot().getColormapAction().setColormapDialog(
            self.defaultColorDialog()
        )
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return None

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = nxdata.get_default(data, validate=False)
        if nxd is None:
            return
        isRgba = nxd.interpretation == "rgba-image"

        self._updateColormap(nxd)

        widget: ArrayImagePlot = self.getWidget()
        widget.setImageData(
            [nxd.signal] + nxd.auxiliary_signals,
            axes=nxd.axes,
            signals_names=[nxd.signal_name] + nxd.auxiliary_signals_names,
            axes_names=nxd.axes_names,
            axes_scales=nxd.plot_style.axes_scale_types,
            title=nxd.title,
            isRgba=isRgba,
        )

    def getDataPriority(self, data, info: DataInfo):
        data = self.normalizeData(data)

        if info.hasNXdata and not info.isInvalidNXdata:
            default = nxdata.get_default(data, validate=False)
            if default is None:
                return DataView.UNSUPPORTED

            if default.is_image or default.is_stack:
                return 100

        return DataView.UNSUPPORTED


class _NXdataComplexImageView(_NXdataBaseDataView):
    """DataView using a ComplexImageView for displaying NXdata complex images:
    2-D signal or n-D signals with *@interpretation=image*."""

    def __init__(self, parent):
        _NXdataBaseDataView.__init__(self, parent, modeId=NXDATA_IMAGE_MODE)

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import ArrayComplexImagePlot

        widget = ArrayComplexImagePlot(parent, colormap=self.defaultColormap())
        widget.getPlot().getColormapAction().setColormapDialog(
            self.defaultColorDialog()
        )
        return widget

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = nxdata.get_default(data, validate=False)

        self._updateColormap(nxd)

        # last two axes are Y & X
        img_slicing = slice(-2, None)
        y_axis, x_axis = nxd.axes[img_slicing]
        y_label, x_label = nxd.axes_names[img_slicing]
        x_units = get_attr_as_unicode(x_axis, "units") if x_axis else None
        y_units = get_attr_as_unicode(y_axis, "units") if y_axis else None

        self.getWidget().setImageData(
            [nxd.signal] + nxd.auxiliary_signals,
            x_axis=x_axis,
            y_axis=y_axis,
            signals_names=[nxd.signal_name] + nxd.auxiliary_signals_names,
            xlabel=x_label,
            ylabel=y_label,
            title=nxd.title,
            keep_ratio=(x_units == y_units),
        )

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return None

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)

        if info.hasNXdata and not info.isInvalidNXdata:
            nxd = nxdata.get_default(data, validate=False)
            if nxd.is_image and numpy.iscomplexobj(nxd.signal):
                return 100

        return DataView.UNSUPPORTED


class _NXdataVolumeView(_NXdataBaseDataView):
    def __init__(self, parent):
        _NXdataBaseDataView.__init__(
            self,
            parent,
            label="NXdata (3D)",
            icon=icons.getQIcon("view-nexus"),
            modeId=NXDATA_VOLUME_MODE,
        )
        try:
            import silx.gui.plot3d  # noqa
        except ImportError:
            _logger.warning("Plot3dView is not available")
            _logger.debug("Backtrace", exc_info=True)
            raise

    def normalizeData(self, data):
        data = super().normalizeData(data)
        data = normalizeComplex(data)
        return data

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import ArrayVolumePlot

        widget = ArrayVolumePlot(parent)
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return None

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = nxdata.get_default(data, validate=False)
        signal_name = nxd.signal_name
        z_axis, y_axis, x_axis = nxd.axes[-3:]
        z_label, y_label, x_label = nxd.axes_names[-3:]
        title = nxd.title or signal_name

        widget = self.getWidget()
        widget.setData(
            nxd.signal,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            signal_name=signal_name,
            xlabel=x_label,
            ylabel=y_label,
            zlabel=z_label,
            title=title,
        )

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.hasNXdata and not info.isInvalidNXdata:
            if nxdata.get_default(data, validate=False).is_volume:
                return 150

        return DataView.UNSUPPORTED


class _NXdataVolumeAsStackView(_NXdataBaseDataView):
    def __init__(self, parent):
        _NXdataBaseDataView.__init__(
            self,
            parent,
            label="NXdata (2D)",
            icon=icons.getQIcon("view-nexus"),
            modeId=NXDATA_VOLUME_AS_STACK_MODE,
        )

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import ArrayStackPlot

        widget = ArrayStackPlot(parent)
        widget.getStackView().setColormap(self.defaultColormap())
        widget.getStackView().getPlotWidget().getColormapAction().setColormapDialog(
            self.defaultColorDialog()
        )
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return None

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = nxdata.get_default(data, validate=False)
        signal_name = nxd.signal_name
        z_axis, y_axis, x_axis = nxd.axes[-3:]
        z_label, y_label, x_label = nxd.axes_names[-3:]
        title = nxd.title or signal_name

        self._updateColormap(nxd)

        widget = self.getWidget()
        widget.setStackData(
            nxd.signal,
            x_axis=x_axis,
            y_axis=y_axis,
            z_axis=z_axis,
            signal_name=signal_name,
            xlabel=x_label,
            ylabel=y_label,
            zlabel=z_label,
            title=title,
        )
        # Override the colormap, while setStack overwrite it
        widget.getStackView().setColormap(self.defaultColormap())

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if info.isComplex:
            return DataView.UNSUPPORTED
        if info.hasNXdata and not info.isInvalidNXdata:
            if nxdata.get_default(data, validate=False).is_volume:
                return 200

        return DataView.UNSUPPORTED


class _NXdataComplexVolumeAsStackView(_NXdataBaseDataView):
    def __init__(self, parent):
        _NXdataBaseDataView.__init__(
            self,
            parent,
            label="NXdata (2D)",
            icon=icons.getQIcon("view-nexus"),
            modeId=NXDATA_VOLUME_AS_STACK_MODE,
        )
        self._is_complex_data = False

    def createWidget(self, parent):
        from silx.gui.data.NXdataWidgets import ArrayComplexImagePlot

        widget = ArrayComplexImagePlot(parent, colormap=self.defaultColormap())
        widget.getPlot().getColormapAction().setColormapDialog(
            self.defaultColorDialog()
        )
        return widget

    def axesNames(self, data, info):
        # disabled (used by default axis selector widget in Hdf5Viewer)
        return None

    def clear(self):
        self.getWidget().clear()

    def setData(self, data):
        data = self.normalizeData(data)
        nxd = nxdata.get_default(data, validate=False)
        z_axis, y_axis, x_axis = nxd.axes[-3:]
        z_label, y_label, x_label = nxd.axes_names[-3:]

        self._updateColormap(nxd)

        self.getWidget().setImageData(
            [nxd.signal] + nxd.auxiliary_signals,
            x_axis=x_axis,
            y_axis=y_axis,
            signals_names=[nxd.signal_name] + nxd.auxiliary_signals_names,
            xlabel=x_label,
            ylabel=y_label,
            title=nxd.title,
        )

    def getDataPriority(self, data, info):
        data = self.normalizeData(data)
        if not info.isComplex:
            return DataView.UNSUPPORTED
        if info.hasNXdata and not info.isInvalidNXdata:
            if nxdata.get_default(data, validate=False).is_volume:
                return 200

        return DataView.UNSUPPORTED


class _NXdataView(CompositeDataView):
    """Composite view displaying NXdata groups using the most adequate
    widget depending on the dimensionality."""

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            label="NXdata",
            modeId=NXDATA_MODE,
            icon=icons.getQIcon("view-nexus"),
        )

        self.addView(_InvalidNXdataView(parent))
        self.addView(_NXdataScalarView(parent))
        self.addView(_NXdataCurveView(parent))
        self.addView(_NXdataXYVScatterView(parent))
        self.addView(_NXdataComplexImageView(parent))
        self.addView(_NXdataImageView(parent))

        # The 3D view can be displayed using 2 ways
        nx3dViews = SelectManyDataView(parent)
        nx3dViews.addView(_NXdataVolumeAsStackView(parent))
        nx3dViews.addView(_NXdataComplexVolumeAsStackView(parent))
        try:
            nx3dViews.addView(_NXdataVolumeView(parent))
        except Exception:
            _logger.warning("NXdataVolumeView is not available")
            _logger.debug("Backtrace", exc_info=True)
        self.addView(nx3dViews)
