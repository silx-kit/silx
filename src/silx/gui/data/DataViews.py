"""This module defines a views used by :class:`silx.gui.data.DataViewer`."""

from silx.utils.deprecation import deprecated_warning

from ._DataInfo import DataInfo
from ._DataView import DataView, DataViewHooks
from .composite import (
    CompositeDataView,
    SelectManyDataView,
    SelectOneDataView,
    _RawView,
)
from .modes import (
    COMPLEX_PLOT2D_MODE,
    EMPTY_MODE,
    HDF5_MODE,
    IMAGE_MODE,
    NXDATA_CURVE_MODE,
    NXDATA_IMAGE_MODE,
    NXDATA_INVALID_MODE,
    NXDATA_MODE,
    NXDATA_SCALAR_MODE,
    NXDATA_STACK_MODE,
    NXDATA_VOLUME_AS_STACK_MODE,
    NXDATA_VOLUME_MODE,
    NXDATA_XYVSCATTER_MODE,
    PLOT1D_MODE,
    PLOT2D_MODE,
    PLOT3D_MODE,
    RAW_ARRAY_MODE,
    RAW_HEXA_MODE,
    RAW_MODE,
    RAW_RECORD_MODE,
    RAW_SCALAR_MODE,
    RECORD_PLOT_MODE,
    STACK_MODE,
)
from .nexus import (
    _InvalidNXdataView,
    _NXdataComplexImageView,
    _NXdataComplexVolumeAsStackView,
    _NXdataCurveView,
    _NXdataImageView,
    _NXdataScalarView,
    _NXdataView,
    _NXdataVolumeAsStackView,
    _NXdataXYVScatterView,
)
from .views import (
    _ArrayView,
    _ComplexImageView,
    _EmptyView,
    _Hdf5View,
    _HexaView,
    _Plot1dView,
    _Plot2dRecordView,
    _Plot2dView,
    _Plot3dView,
    _RecordView,
    _ScalarView,
)

deprecated_warning(
    "module",
    name="silx.gui.data.DataViews",
    reason="Broken down in several modules",
    replacement="silx.gui.data.composite, silx.gui.data._DataView, silx.gui.data._DataInfo, silx.gui.data.modes, silx.gui.data.nexus, silx.gui.data.views",
    since_version="3.0.0",
)
