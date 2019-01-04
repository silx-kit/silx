#!/usr/bin/env python3
# pylint: skip-file

""" Small demo for demonstrating the silx bg color.

"""

import enum
import logging
import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtGui import QColor, QPalette
from silx.gui.plot import Plot1D, Plot2D, TickMode
from silx.gui.plot.Colormap import Colormap


logger = logging.getLogger("main")

if 0:
    BACKEND = 'mpl'
else:
    BACKEND = 'gl'


@enum.unique
class PaletteRoleEnum(enum.Enum):
    """ Enumeration for the role in a QPalette.

        # See http://doc.qt.io/archives/qt-5.6/qpalette.html for the full descriptions.
    """
    # Central roles
    Window        = QPalette.Window         # A general background color.
    WindowText    = QPalette.WindowText     # A general foreground color.
    Base          = QPalette.Base           # Typically the background color for text entry widgets.
    AlternateBase = QPalette.AlternateBase  # Alternate background color when alternating rows
    ToolTipBase   = QPalette.ToolTipBase	# Used as the background color for QToolTip
    ToolTipText   = QPalette.ToolTipText    # Used as the foreground color for QToolTip
    Text          = QPalette.Text           # The foreground color used with Base.
    Button        = QPalette.Button	        # The general button background color.
    ButtonText    = QPalette.ButtonText     # A foreground color used with the Button color.
    BrightText    = QPalette.BrightText     # A text color that is very different from WindowText

    Light         = QPalette.Light          # Lighter than Button color.
    Midlight      = QPalette.Midlight       # Between Button and Light.
    Dark          = QPalette.Dark	        # Darker than Button.
    Mid           = QPalette.Mid            # Between Button and Dark.
    Shadow        = QPalette.Shadow         # A very dark color. By default it's Qt::black.

    Highlight       = QPalette.Highlight        # Indicates a selected item or the current item.
    HighlightedText = QPalette.HighlightedText	# A text color that contrasts with Highlight.
    Link          = QPalette.Link           # A text color used for unvisited hyperlinks.
    LinkVisited   = QPalette.LinkVisited    # A text color used for already visited hyperlinks.


def setWidgetColor(widget, color, role, group=None):
    """ Sets the background color of a widget.

        Note: if the widget remains transparent use: widget.setAutoFillBackground(True)

        :param QWidget widget: the Qt Widget
        :param QtGui.QColor color: the color to set
        :param PaletteGroupEnum group: the Palette group that is used.(None = current):
        :param PaletteRoleEnum role: the role (foreground color, back ground color,etc)
    """
    pal = widget.palette()
    if group is None:
        pal.setColor(role.value, color)
    else:
        pal.setColor(group.value, role.value, color)
    widget.setPalette(pal)


def makePlot1D():

    CREATE_DATES = False

    if not CREATE_DATES:
        y = np.random.normal(size=int(1e4))
        y[1000] = 23

        x = np.arange(len(y)) + 900
        x = list(x)

    plot1D = Plot1D(backend=BACKEND)
    plot1D.setGraphGrid("both")
    xAxis = plot1D.getXAxis()
    xAxis.setLimits(-11, 35)

    foregroundColor = QColor(0, 0, 0)
    #gridColor = QColor(40, 40, 40)
    gridColor = QColor(255, 0, 0,)

    backgroundColor = QColor(255, 255, 255)
    dataBackgroundColor = QColor(200, 200, 200)

    plot1D.setGraphTitle("My Title")
    plot1D.setForegroundColor(foregroundColor)
    plot1D.setGridColor(gridColor)
    plot1D.setBackgroundColor(backgroundColor)
    #plot1D.setDataBackgroundColor(dataBackgroundColor)

    #setWidgetColor(plot1D, testColor, PaletteRoleEnum.Window)
    #setWidgetColor(plot1D, foregroundColor, PaletteRoleEnum.WindowText)

    plot1D.addCurve(x=x, y=y, legend='curve')

    return plot1D



def makePlot2D():
    plot = Plot2D(backend=BACKEND)

    colormap = Colormap(name='viridis', normalization='linear', vmin=0.0, vmax=2.0)
    plot.setDefaultColormap(colormap)

    data = np.random.random(512 * 512).reshape(512, -1)
    plot.addImage(data, legend='random', replace=False)

    plot.setBackgroundColor(QColor(80, 80, 80))

    data = np.arange(512 * 512.).reshape(512, -1)
    plot.addImage(data, legend='arange', replace=False, origin=(512, 512))
    
    return plot



class MyWin(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MyWin, self).__init__(parent=parent)

        # Layout with legend
        self.contentsLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.contentsLayout)

        self.plot = makePlot1D()
        self.contentsLayout.addWidget(self.plot)

        self.myTestAction = QtWidgets.QAction("My test", self, triggered=self.myTest)
        self.myTestAction.setShortcut("Ctrl+T")
        self.addAction(self.myTestAction)


    def myTest(self):
        logger.info("Called my test")

        y = np.random.normal(size=int(1e5)) + 5
        x = np.arange(len(y)) + 9000
        self.plot.addCurve(x=x, y=y, color='red', legend='test curve')


    

def main():
    fmt = '%(asctime)s %(filename)25s:%(lineno)-4d : %(levelname)-7s: %(message)s'
    logging.basicConfig(level='DEBUG', format=fmt)
    logger.info("START")
    print("START------------------------------------------------")

    app = QtWidgets.QApplication([])
    #win = makePlot2D()
    win = MyWin()
    #win = makeHistogram()
    win.show()
    app.exec_()
    
    
    
if __name__ == "__main__":
    main()

