from silx.gui.plot.PlotWindow import PlotWindow
from silx.gui.plot.PlotWidget import PlotWidget
from .. import qt


class RecordPlot(PlotWindow):
    def __init__(self, parent=None, backend=None):
        super(RecordPlot, self).__init__(parent=parent, backend=backend,
                                         resetzoom=True, autoScale=True,
                                         logScale=True, grid=True,
                                         curveStyle=True, colormap=False,
                                         aspectRatio=False, yInverted=False,
                                         copy=True, save=True, print_=True,
                                         control=True, position=True,
                                         roi=True, mask=False, fit=True)
        if parent is None:
            self.setWindowTitle('RecordPlot')
        self._axesSelectionToolBar = AxesSelectionToolBar(parent=self, plot=self)
        self.addToolBar(qt.Qt.BottomToolBarArea, self._axesSelectionToolBar)

    def setXAxisFieldName(self, value):
        """Set the current selected field for the X axis.

        :param Union[str,None] value:
        """
        label = '' if value is None else value
        index = self._axesSelectionToolBar.getXAxisDropDown().findData(value)

        if index >= 0:
            self.getXAxis().setLabel(label)
            self._axesSelectionToolBar.getXAxisDropDown().setCurrentIndex(index)

    def getXAxisFieldName(self):
        """Returns currently selected field for the X axis or None.

        rtype: Union[str,None]
        """
        return self._axesSelectionToolBar.getXAxisDropDown().currentData()

    def setYAxisFieldName(self, value):
        self.getYAxis().setLabel(value)
        index = self._axesSelectionToolBar.getYAxisDropDown().findText(value)
        if index >= 0:
            self._axesSelectionToolBar.getYAxisDropDown().setCurrentIndex(index)

    def getYAxisFieldName(self):
        return self._axesSelectionToolBar.getYAxisDropDown().currentText()

    def setSelectableXAxisFieldNames(self, fieldNames):
        """Add list of field names to X axis

        :param List[str] fieldNames:
        """
        comboBox = self._axesSelectionToolBar.getXAxisDropDown()
        comboBox.clear()
        comboBox.addItem('-', None)
        comboBox.insertSeparator(1)
        for name in fieldNames:
            comboBox.addItem(name, name)

    def setSelectableYAxisFieldNames(self, fieldNames):
        self._axesSelectionToolBar.getYAxisDropDown().clear()
        self._axesSelectionToolBar.getYAxisDropDown().addItems(fieldNames)

    def getAxesSelectionToolBar(self):
        return self._axesSelectionToolBar

class AxesSelectionToolBar(qt.QToolBar):
    def __init__(self, parent=None, plot=None, title='Plot Axes Selection'):
        super(AxesSelectionToolBar, self).__init__(title, parent)

        assert isinstance(plot, PlotWidget)

        self.addWidget(qt.QLabel("Field selection: "))

        self._labelXAxis = qt.QLabel(" X: ")
        self.addWidget(self._labelXAxis)

        self._selectXAxisDropDown = qt.QComboBox()
        self.addWidget(self._selectXAxisDropDown)

        self._labelYAxis = qt.QLabel(" Y: ")
        self.addWidget(self._labelYAxis)

        self._selectYAxisDropDown = qt.QComboBox()
        self.addWidget(self._selectYAxisDropDown)

    def getXAxisDropDown(self):
        return self._selectXAxisDropDown

    def getYAxisDropDown(self):
        return self._selectYAxisDropDown