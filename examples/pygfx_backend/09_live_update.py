"""Live data update with pygfx backend.

Demonstrates: real-time curve and image updates from a timer.
"""

import numpy
from silx.gui import qt
from silx.gui.plot import Plot1D, Plot2D


class LiveCurveWindow(qt.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pygfx - Live Curve Update")

        self._plot = Plot1D(backend="pygfx")
        self._plot.setGraphTitle("Live Sine Wave")
        self._plot.setGraphXLabel("X")
        self._plot.setGraphYLabel("Y")
        self._plot.setGraphYLimits(-1.5, 1.5)
        self.setCentralWidget(self._plot)

        self._phase = 0.0
        self._x = numpy.linspace(0, 4 * numpy.pi, 500)

        self._timer = qt.QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(30)  # ~33 fps

    def _update(self):
        self._phase += 0.05
        y = numpy.sin(self._x + self._phase)
        self._plot.addCurve(
            self._x, y, legend="live", color="blue", linewidth=2, resetzoom=False
        )


class LiveImageWindow(qt.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pygfx - Live Image Update")

        self._plot = Plot2D(backend="pygfx")
        self._plot.setGraphTitle("Live 2D Gaussian")
        self._plot.getDefaultColormap().setName("inferno")
        self.setCentralWidget(self._plot)

        self._size = 128
        self._x0 = 0.0
        self._y0 = 0.0
        x = numpy.linspace(-3, 3, self._size)
        self._xx, self._yy = numpy.meshgrid(x, x)

        self._timer = qt.QTimer(self)
        self._timer.timeout.connect(self._update)
        self._timer.start(50)  # ~20 fps

    def _update(self):
        self._x0 += 0.05 * (numpy.random.random() - 0.5)
        self._y0 += 0.05 * (numpy.random.random() - 0.5)
        image = numpy.exp(-((self._xx - self._x0) ** 2 + (self._yy - self._y0) ** 2))
        image += 0.1 * numpy.random.random(image.shape)
        self._plot.addImage(image, resetzoom=False)


def main():
    app = qt.QApplication([])

    w1 = LiveCurveWindow()
    w1.resize(700, 400)
    w1.show()

    w2 = LiveImageWindow()
    w2.resize(600, 500)
    w2.move(720, 0)
    w2.show()

    app.exec()


if __name__ == "__main__":
    main()
