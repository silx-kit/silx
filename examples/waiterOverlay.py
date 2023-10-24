import numpy.random
from silx.gui import qt
from silx.gui.widgets.WaitingOverlay import WaitingOverlay
from silx.gui.plot import Plot2D


class MyMainWindow(qt.QMainWindow):
    """
    Dummy demonstration window that create an image in a thread and update the plot
    """

    WAITING_TIME = 2000  # ms

    def __init__(self, parent=None):
        super().__init__(parent)

        # central plot
        self._plot = Plot2D()
        self._waitingOverlay = WaitingOverlay(self._plot)
        self.setCentralWidget(self._plot)

        # button to trigger image generation
        self._rightPanel = qt.QWidget(self)
        self._rightPanel.setLayout(qt.QVBoxLayout())
        self._button = qt.QPushButton("generate image", self)
        self._rightPanel.layout().addWidget(self._button)

        self._dockWidget = qt.QDockWidget()
        self._dockWidget.setWidget(self._rightPanel)
        self.addDockWidget(qt.Qt.RightDockWidgetArea, self._dockWidget)

        # set up
        self._waitingOverlay.hide()
        self._waitingOverlay.setIconSize(qt.QSize(60, 60))
        # connect signal / slot
        self._button.released.connect(self._triggerImageCalculation)

    def _generateRandomData(self):
        self.setData(numpy.random.random(1000 * 500).reshape((1000, 500)))
        self._button.setEnabled(True)

    def setData(self, data):
        self._plot.addImage(data)
        self._waitingOverlay.hide()

    def _triggerImageCalculation(self):
        self._plot.clear()
        self._button.setEnabled(False)
        self._waitingOverlay.show()
        qt.QTimer.singleShot(self.WAITING_TIME, self._generateRandomData)


qapp = qt.QApplication([])
window = MyMainWindow()
window.show()
qapp.exec_()
