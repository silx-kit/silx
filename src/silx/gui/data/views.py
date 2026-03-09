import numpy
import logging

import silx.io
from silx.gui import icons, qt
from silx.gui.plot.actions.image import AggregationModeAction
from silx.gui.plot.items import ImageDataAggregated

from ._DataView import DataView
from ._utils import normalizeComplex
from .modes import (
    COMPLEX_PLOT2D_MODE,
    EMPTY_MODE,
    HDF5_MODE,
    PLOT1D_MODE,
    PLOT2D_MODE,
    PLOT3D_MODE,
    RAW_ARRAY_MODE,
    RAW_HEXA_MODE,
    RAW_RECORD_MODE,
    RAW_SCALAR_MODE,
    RECORD_PLOT_MODE,
)
from .TextFormatter import TextFormatter

_logger = logging.getLogger(__name__)


class _EmptyView(DataView):
    """Dummy view to display nothing"""

    def __init__(self, parent):
        DataView.__init__(self, parent, modeId=EMPTY_MODE)

    def axesNames(self, data, info):
        return None

    def createWidget(self, parent):
        return qt.QLabel(parent)

    def getDataPriority(self, data, info):
        return DataView.UNSUPPORTED


class _Plot1dView(DataView):
    """View displaying data using a 1d plot"""

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            modeId=PLOT1D_MODE,
            label="Curve",
            icon=icons.getQIcon("view-1d"),
        )
        self.__resetZoomNextTime = True

    def createWidget(self, parent):
        from silx.gui import plot

        widget = plot.Plot1D(parent=parent)
        widget.setGraphGrid(True)
        return widget

    def clear(self):
        self.getWidget().clear()
        self.__resetZoomNextTime = True

    def normalizeData(self, data):
        data = DataView.normalizeData(self, data)
        data = normalizeComplex(data)
        return data

    def setData(self, data):
        data = self.normalizeData(data)
        plotWidget = self.getWidget()
        legend = "data"
        plotWidget.addCurve(
            legend=legend,
            x=range(len(data)),
            y=data,
            resetzoom=self.__resetZoomNextTime,
        )
        plotWidget.setActiveCurve(legend)
        self.__resetZoomNextTime = True

    def setDataSelection(self, selection):
        self.getWidget().setGraphTitle(self.titleForSelection(selection))

    def axesNames(self, data, info):
        return ["y"]

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
        if data is None or not info.isArray or not info.isNumeric:
            return DataView.UNSUPPORTED
        if info.dim < 1:
            return DataView.UNSUPPORTED
        if info.interpretation == "spectrum":
            return 1000
        if info.dim == 2 and info.shape[0] == 1:
            return 210
        if info.dim == 1:
            return 100
        else:
            return 10


class _Plot2dRecordView(DataView):
    def __init__(self, parent):
        super().__init__(
            parent=parent,
            modeId=RECORD_PLOT_MODE,
            label="Curve",
            icon=icons.getQIcon("view-1d"),
        )
        self.__resetZoomNextTime = True
        self._data = None
        self._xAxisDropDown = None
        self._yAxisDropDown = None
        self.__fields = None

    def createWidget(self, parent):
        from ._RecordPlot import RecordPlot

        return RecordPlot(parent=parent)

    def clear(self):
        self.getWidget().clear()
        self.__resetZoomNextTime = True

    def normalizeData(self, data):
        data = DataView.normalizeData(self, data)
        data = normalizeComplex(data)
        return data

    def setData(self, data):
        self._data = self.normalizeData(data)

        all_fields = sorted(self._data.dtype.fields.items(), key=lambda e: e[1][1])
        numeric_fields = [
            f[0] for f in all_fields if numpy.issubdtype(f[1][0], numpy.number)
        ]
        if numeric_fields == self.__fields:  # Reuse previously selected fields
            fieldNameX = self.getWidget().getXAxisFieldName()
            fieldNameY = self.getWidget().getYAxisFieldName()
        else:
            self.__fields = numeric_fields

            self.getWidget().setSelectableXAxisFieldNames(numeric_fields)
            self.getWidget().setSelectableYAxisFieldNames(numeric_fields)
            fieldNameX = None
            fieldNameY = numeric_fields[0]

            # If there is a field called time, use it for the x-axis by default
            if "time" in numeric_fields:
                fieldNameX = "time"
            # Use the first field that is not "time" for the y-axis
            if fieldNameY == "time" and len(numeric_fields) >= 2:
                fieldNameY = numeric_fields[1]

        self._plotData(fieldNameX, fieldNameY)

        if not self._xAxisDropDown:
            self._xAxisDropDown = (
                self.getWidget().getAxesSelectionToolBar().getXAxisDropDown()
            )
            self._yAxisDropDown = (
                self.getWidget().getAxesSelectionToolBar().getYAxisDropDown()
            )
            self._xAxisDropDown.activated.connect(self._onAxesSelectionChaned)
            self._yAxisDropDown.activated.connect(self._onAxesSelectionChaned)

    def setDataSelection(self, selection):
        self.getWidget().setGraphTitle(self.titleForSelection(selection))

    def _onAxesSelectionChaned(self):
        fieldNameX = self._xAxisDropDown.currentData()
        self._plotData(fieldNameX, self._yAxisDropDown.currentText())

    def _plotData(self, fieldNameX, fieldNameY):
        self.clear()
        ydata = self._data[fieldNameY]
        if fieldNameX is None:
            xdata = numpy.arange(len(ydata))
        else:
            xdata = self._data[fieldNameX]
        self.getWidget().addCurve(
            legend="data", x=xdata, y=ydata, resetzoom=self.__resetZoomNextTime
        )
        self.getWidget().setXAxisFieldName(fieldNameX)
        self.getWidget().setYAxisFieldName(fieldNameY)
        self.__resetZoomNextTime = True

    def axesNames(self, data, info):
        return ["data"]

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
        if data is None or not info.isRecord:
            return DataView.UNSUPPORTED
        if info.dim < 1:
            return DataView.UNSUPPORTED
        if info.countNumericColumns < 2:
            return DataView.UNSUPPORTED
        if info.interpretation == "spectrum":
            return 1000
        if info.dim == 2 and info.shape[0] == 1:
            return 210
        if info.dim == 1:
            return 40
        else:
            return 10


class _Plot2dView(DataView):
    """View displaying data using a 2d plot"""

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            modeId=PLOT2D_MODE,
            label="Image",
            icon=icons.getQIcon("view-2d"),
        )
        self.__resetZoomNextTime = True

    def createWidget(self, parent):
        from silx.gui import plot

        widget = plot.Plot2D(parent=parent)
        widget.setDefaultColormap(self.defaultColormap())
        widget.getColormapAction().setColormapDialog(self.defaultColorDialog())
        widget.getIntensityHistogramAction().setVisible(True)

        self.__aggregationModeAction = AggregationModeAction(parent=widget)
        widget.toolBar().addAction(self.__aggregationModeAction)
        self.__aggregationModeAction.sigAggregationModeChanged.connect(
            self._aggregationModeChanged
        )

        self.__imageItem = ImageDataAggregated()
        self.__imageItem.setAggregationMode(
            self.__aggregationModeAction.getAggregationMode()
        )
        self.__imageItem.setName("data")
        self.__imageItem.setColormap(widget.getDefaultColormap())
        widget.addItem(self.__imageItem)
        widget.setActiveImage(self.__imageItem)

        widget.setKeepDataAspectRatio(True)
        widget.getXAxis().setLabel("X")
        widget.getYAxis().setLabel("Y")
        maskToolsWidget = widget.getMaskToolsDockWidget().widget()
        maskToolsWidget.setItemMaskUpdated(True)
        return widget

    def _aggregationModeChanged(self):
        self.__imageItem.setAggregationMode(
            self.__aggregationModeAction.getAggregationMode()
        )

    def clear(self):
        self.__imageItem.setData(numpy.zeros((0, 0), dtype=numpy.float32))
        self.__resetZoomNextTime = True

    def normalizeData(self, data):
        data = DataView.normalizeData(self, data)
        data = normalizeComplex(data)
        return data

    def setData(self, data):
        data = self.normalizeData(data)
        plot = self.getWidget()

        self.__imageItem.setData(data=data)
        if self.__resetZoomNextTime:
            plot.resetZoom()
        self.__resetZoomNextTime = False

    def setDataSelection(self, selection):
        self.getWidget().setGraphTitle(self.titleForSelection(selection))

    def axesNames(self, data, info):
        return ["y", "x"]

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
        if data is None or not info.isArray or not (info.isNumeric or info.isBoolean):
            return DataView.UNSUPPORTED
        if info.isComplex:
            return DataView.UNSUPPORTED
        if info.dim < 2:
            return DataView.UNSUPPORTED
        if info.interpretation == "image":
            return 1000
        if info.dim == 2:
            return 200
        else:
            return 190


class _Plot3dView(DataView):
    """View displaying data using a 3d plot"""

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            modeId=PLOT3D_MODE,
            label="Cube",
            icon=icons.getQIcon("view-3d"),
        )
        try:
            from ._VolumeWindow import VolumeWindow  # noqa
        except ImportError:
            _logger.warning("3D visualization is not available")
            _logger.debug("Backtrace", exc_info=True)
            raise
        self.__resetZoomNextTime = True

    def createWidget(self, parent):
        from ._VolumeWindow import VolumeWindow

        plot = VolumeWindow(parent)
        plot.setAxesLabels(*reversed(self.axesNames(None, None)))
        return plot

    def clear(self):
        self.getWidget().clear()
        self.__resetZoomNextTime = True

    def setData(self, data):
        data = self.normalizeData(data)
        self.getWidget().setData(data)
        self.__resetZoomNextTime = False

    def axesNames(self, data, info):
        return ["z", "y", "x"]

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
        if data is None or not info.isArray or not info.isNumeric:
            return DataView.UNSUPPORTED
        if info.dim < 3:
            return DataView.UNSUPPORTED
        if min(data.shape) < 2:
            return DataView.UNSUPPORTED
        if info.dim == 3:
            return 100
        else:
            return 10


class _ComplexImageView(DataView):
    """View displaying data using a ComplexImageView"""

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            modeId=COMPLEX_PLOT2D_MODE,
            label="Image",
            icon=icons.getQIcon("view-2d"),
        )

    def createWidget(self, parent):
        from silx.gui.plot.ComplexImageView import ComplexImageView

        widget = ComplexImageView(parent=parent)
        widget.setColormap(
            self.defaultColormap(), mode=ComplexImageView.ComplexMode.ABSOLUTE
        )
        widget.setColormap(
            self.defaultColormap(), mode=ComplexImageView.ComplexMode.SQUARE_AMPLITUDE
        )
        widget.setColormap(
            self.defaultColormap(), mode=ComplexImageView.ComplexMode.REAL
        )
        widget.setColormap(
            self.defaultColormap(), mode=ComplexImageView.ComplexMode.IMAGINARY
        )
        widget.getPlot().getColormapAction().setColormapDialog(
            self.defaultColorDialog()
        )
        widget.getPlot().getIntensityHistogramAction().setVisible(True)
        widget.getPlot().setKeepDataAspectRatio(True)
        widget.getXAxis().setLabel("X")
        widget.getYAxis().setLabel("Y")
        maskToolsWidget = widget.getPlot().getMaskToolsDockWidget().widget()
        maskToolsWidget.setItemMaskUpdated(True)
        return widget

    def clear(self):
        self.getWidget().setData(None)

    def normalizeData(self, data):
        data = DataView.normalizeData(self, data)
        return data

    def setData(self, data):
        data = self.normalizeData(data)
        self.getWidget().setData(data)

    def setDataSelection(self, selection):
        self.getWidget().getPlot().setGraphTitle(self.titleForSelection(selection))

    def axesNames(self, data, info):
        return ["y", "x"]

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
        if data is None or not info.isArray or not info.isComplex:
            return DataView.UNSUPPORTED
        if info.dim < 2:
            return DataView.UNSUPPORTED
        if info.interpretation == "image":
            return 1000
        if info.dim == 2:
            return 200
        else:
            return 190


class _ArrayView(DataView):
    """View displaying data using a 2d table"""

    def __init__(self, parent):
        DataView.__init__(self, parent, modeId=RAW_ARRAY_MODE)

    def createWidget(self, parent):
        from silx.gui.data.ArrayTableWidget import ArrayTableWidget

        widget = ArrayTableWidget(parent)
        widget.displayAxesSelector(False)
        return widget

    def clear(self):
        self.getWidget().setArrayData(numpy.array([[]]))

    def setData(self, data):
        data = self.normalizeData(data)
        self.getWidget().setArrayData(data)

    def axesNames(self, data, info):
        return ["col", "row"]

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
        if data is None or not info.isArray or info.isRecord:
            return DataView.UNSUPPORTED
        if info.dim < 2:
            return DataView.UNSUPPORTED
        if info.interpretation in ["scalar", "scaler"]:
            return 1000
        return 500


class _ScalarView(DataView):
    """View displaying data using text"""

    def __init__(self, parent):
        DataView.__init__(self, parent, modeId=RAW_SCALAR_MODE)

    def createWidget(self, parent):
        widget = qt.QTextEdit(parent)
        widget.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        widget.setAlignment(qt.Qt.AlignLeft | qt.Qt.AlignTop)
        self.__formatter = TextFormatter(parent)
        return widget

    def clear(self):
        self.getWidget().setText("")

    def setData(self, data):
        d = self.normalizeData(data)
        if silx.io.is_dataset(d):
            d = d[()]
        dtype = None
        if data is not None:
            if hasattr(data, "dtype"):
                dtype = data.dtype
        text = self.__formatter.toString(d, dtype)
        self.getWidget().setText(text)

    def axesNames(self, data, info):
        return []

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
        data = self.normalizeData(data)
        if info.shape is None:
            return DataView.UNSUPPORTED
        if data is None:
            return DataView.UNSUPPORTED
        if silx.io.is_group(data):
            return DataView.UNSUPPORTED
        return 2


class _RecordView(DataView):
    """View displaying data using text"""

    def __init__(self, parent):
        DataView.__init__(self, parent, modeId=RAW_RECORD_MODE)

    def createWidget(self, parent):
        from .RecordTableView import RecordTableView

        widget = RecordTableView(parent)
        widget.setWordWrap(False)
        return widget

    def clear(self):
        self.getWidget().setArrayData(None)

    def setData(self, data):
        data = self.normalizeData(data)
        widget = self.getWidget()
        widget.setArrayData(data)
        if len(data) < 100:
            widget.resizeRowsToContents()
            widget.resizeColumnsToContents()

    def axesNames(self, data, info):
        return ["data"]

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
        if info.isRecord:
            return 40
        if data is None or not info.isArray:
            return DataView.UNSUPPORTED
        if info.dim == 1:
            if info.interpretation in ["scalar", "scaler"]:
                return 1000
            if info.shape[0] == 1:
                return 510
            return 500
        elif info.isRecord:
            return 40
        return DataView.UNSUPPORTED


class _HexaView(DataView):
    """View displaying data using text"""

    def __init__(self, parent):
        DataView.__init__(self, parent, modeId=RAW_HEXA_MODE)

    def createWidget(self, parent):
        from .HexaTableView import HexaTableView

        widget = HexaTableView(parent)
        return widget

    def clear(self):
        self.getWidget().setArrayData(None)

    def setData(self, data):
        data = self.normalizeData(data)
        widget = self.getWidget()
        widget.setArrayData(data)

    def axesNames(self, data, info):
        return []

    def getDataPriority(self, data, info):
        if info.size <= 0:
            return DataView.UNSUPPORTED
        if info.isVoid:
            return 2000
        return DataView.UNSUPPORTED


class _Hdf5View(DataView):
    """View displaying data using text"""

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            modeId=HDF5_MODE,
            label="HDF5",
            icon=icons.getQIcon("view-hdf5"),
        )

    def createWidget(self, parent):
        from .Hdf5TableView import Hdf5TableView

        widget = Hdf5TableView(parent)
        return widget

    def clear(self):
        widget = self.getWidget()
        widget.setData(None)

    def setData(self, data):
        widget = self.getWidget()
        widget.setData(data)

    def axesNames(self, data, info):
        return None

    def getDataPriority(self, data, info):
        widget = self.getWidget()
        if widget.isSupportedData(data):
            return 1
        else:
            return DataView.UNSUPPORTED
