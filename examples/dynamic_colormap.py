import numpy
from silx.gui import qt
from silx.gui.plot import Plot2D


def main():
    app = qt.QApplication([])

    # Create the ad hoc plot widget and change its default colormap
    x = numpy.zeros((100, 100),dtype=numpy.float32)
    x[:50,:50] = numpy.random.randn(50,50)
    x[:50,50:] = 10 * numpy.random.randn(50,50)
    x[50:,:50] = 100 * numpy.random.randn(50,50)
    x[50:,50:] = 5 * numpy.random.randn(50,50)

    example = Plot2D()
    example.addImage(x)
    example.show()

    app.exec()


if __name__ == "__main__":
    main()
