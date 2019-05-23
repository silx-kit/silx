# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""Image stack view with data prefetch
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "04/03/2019"


from silx.gui import qt
from silx.gui.plot import Plot2D
from silx.gui.widgets.UrlSelectionTable import UrlSelectionTable


class ImageStack(qt.QWidget):
    """
    This widget is made to load on the fly image contained the given urls.
    For avoiding lack impression it will prefetch images close to the one
    displayed.
    """

    N_PRELOAD = 10
    """Num"""

    _BUTTON_ICON = qt.QStyle.SP_ComputerIcon.SP_ToolBarHorizontalExtensionButton  # noqa

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self._n_preload = ImageStack.N_PRELOAD
        """Number of image to load from the active one"""

        self.setLayout(qt.QGridLayout())
        self._plot = Plot2D(parent=self)
        self.layout().addWidget(self._plot, 0, 0, 2, 1)
        self._toggleButton = qt.QPushButton(parent=self)
        self.layout().addWidget(self._toggleButton, 0, 2, 1, 1)
        self._toggleButton.setSizePolicy(qt.QSizePolicy.Fixed,
                                         qt.QSizePolicy.Fixed)

        self._urlTable = UrlSelectionTable(parent=self)
        self.layout().addWidget(self._urlTable, 1, 1, 1, 2)

        # set up
        self._setButtonIcon(show=True)

        # Signal / slot connection
        self._toggleButton.clicked.connect(self.toggleUrlSelectionTable)

    def getPlot(self):
        return self._plot

    def setImageUrls(self, urls):
        pass

    def addImageUrls(self, urls):
        pass

    def clearImageUrls(self):
        pass

    def toggleUrlSelectionTable(self):
        visible = not self.urlSelectionTableIsVisible()
        self._setButtonIcon(show=visible)
        self._urlTable.setVisible(visible)

    def _setButtonIcon(self, show):
        print(show)
        style = qt.QApplication.instance().style()
        # return a QIcon
        icon = style.standardIcon(self._BUTTON_ICON)
        if show is False:
            pixmap = icon.pixmap(32, 32).transformed(qt.QTransform().scale(-1, 1))
            icon = qt.QIcon(pixmap)
        self._toggleButton.setIcon(icon)

    def urlSelectionTableIsVisible(self):
        return self._urlTable.isVisible()


if __name__ == '__main__':
    import numpy
    qapp = qt.QApplication([])
    widget = ImageStack()
    widget._plot.addImage(numpy.random.random((500, 500)))
    widget.show()
    qapp.exec_()
