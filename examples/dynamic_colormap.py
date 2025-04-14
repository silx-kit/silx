import numpy
from silx.gui import qt
from silx.gui.plot import Plot2D


def main():
    app = qt.QApplication([])

    # Create the ad hoc plot widget and change its default colormap
    example = Plot2D()
    example.addImage(
        numpy.linspace(0, 100 * 100, 100 * 100, endpoint=False).reshape(100, 100)
    )
    example.show()

    app.exec()


if __name__ == "__main__":
    main()
