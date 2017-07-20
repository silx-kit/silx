# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""

- :class:`PrintPreviewAction`
"""
from __future__ import absolute_import

from io import StringIO

from . import PlotAction
from ...widgets.PrintPreview import PrintPreviewDialog
from ... import qt


__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "18/07/2017"


class PrintPreviewAction(PlotAction):
    """QAction to open a :class:`PrintPreviewDialog` (if not already open)
    and add the current plot to its page to be printed.

    :param plot: :class:`.PlotWidget` instance on which to operate
    :param parent: See :class:`QAction`
    """
    def __init__(self, plot, parent=None):
        super(PrintPreviewAction, self).__init__(
            plot, icon='document-print', text='Print preview',
            tooltip='Send plot data to a print preview dialog',
            triggered=self._plotToPrintPreview,
            checkable=False, parent=parent)
        self.printPreviewDialog = None
        self.printConfigurationDialog = None

        self._printConfiguration = {"xOffset": 0.1,
                                    "yOffset": 0.1,
                                    "width": 0.9,
                                    "height": 0.9,
                                    "units": "page",
                                    "keepAspectRatio": True}

    def _plotToPrintPreview(self):
        if self.printPreviewDialog is None:
            self.printPreviewDialog = PrintPreviewDialog(self.parent())
        self.printPreviewDialog.show()

        if qt.HAS_SVG:
            svgRenderer = self._getSvgRenderer()
            self.printPreviewDialog.addSvgItem(svgRenderer)
        else:
            if qt.BINDING in ["PyQt4", "PySide"]:
                pixmap = qt.QPixmap.grabWidget(self.plot.centralWidget())
            else:
                # PyQt5 and hopefully PyQt6+
                pixmap = self.plot.centralWidget().grab()
            self.printPreviewDialog.addPixmap(pixmap)
        self.printPreviewDialog.raise_()

    def _getSvgRenderer(self):
        """Return a SVG renderer displaying the plot.
        The size of the renderer is adjusted to the printer configuration
        and to the geometry configuration (width, height, ratio) specified
        by the user."""
        imgData = StringIO()
        assert self.plot.saveGraph(imgData, fileFormat="svg"), \
            "Unable to save graph"
        imgData.flush()
        imgData.seek(0)
        svgData = imgData.read()

        svgRenderer = qt.QSvgRenderer()

        printer = self.printPreviewDialog.printer

        self._printConfigurationDialog()     # opens a dialog and updates _printConfiguration
        config = self._printConfiguration
        width = config['width']
        height = config['height']
        xOffset = config['xOffset']
        yOffset = config['yOffset']
        units = config['units']
        keepAspectRatio = config['keepAspectRatio']

        dpix = printer.logicalDpiX()
        dpiy = printer.logicalDpiY()

        availableWidth = printer.width()
        availableHeight = printer.height()

        # convert the offsets to dpi
        if units.lower() in ['inch', 'inches']:
            xOffset = xOffset * dpix
            yOffset = yOffset * dpiy
            if width is not None:
                width = width * dpix
            if height is not None:
                height = height * dpiy
        elif units.lower() in ['cm', 'centimeters']:
            xOffset = (xOffset / 2.54) * dpix
            yOffset = (yOffset / 2.54) * dpiy
            if width is not None:
                width = (width / 2.54) * dpix
            if height is not None:
                height = (height / 2.54) * dpiy
        else:
            # page units
            xOffset = availableWidth * xOffset
            yOffset = availableHeight * yOffset
            if width is not None:
                width = availableWidth * width
            if height is not None:
                height = availableHeight * height

        availableWidth -= xOffset
        availableHeight -= yOffset

        if width is not None:
            if (availableWidth + 0.1) < width:
                txt = "Available width  %f is less than requested width %f" % \
                              (availableWidth, width)
                raise ValueError(txt)
        if height is not None:
            if (availableHeight + 0.1) < height:
                txt = "Available height  %f is less than requested height %f" % \
                              (availableHeight, height)
                raise ValueError(txt)

        if keepAspectRatio:
            # get the aspect ratio
            widget = self.plot.centralWidget()
            graphWidth = float(widget.width())
            graphHeight = float(widget.height())
            graphRatio = graphHeight / graphWidth

            bodyWidth = width or availableWidth
            bodyHeight = bodyWidth * graphRatio

            if bodyHeight > availableHeight:
                bodyHeight = availableHeight
                bodyWidth = bodyHeight / graphRatio
        else:
            bodyWidth = width or availableWidth
            bodyHeight = height or availableHeight

        body = qt.QRectF(xOffset,
                         yOffset,
                         bodyWidth,
                         bodyHeight)

        svgRenderer.setViewBox(body)

        xml_stream = qt.QXmlStreamReader(svgData.encode(errors="replace"))

        if not svgRenderer.load(xml_stream):
            raise RuntimeError("Cannot interpret svg data")

        return svgRenderer

    def _printConfigurationDialog(self):
        """Open a dialog to prompt the user to adjust print parameters."""
        if self.printConfigurationDialog is None:
            self.printConfigurationDialog = PrintGeometryDialog(self.parent())

        self.printConfigurationDialog.setPrintGeometry(self._printConfiguration)
        if self.printConfigurationDialog.exec_():
            self._printConfiguration = self.printConfigurationDialog.getPrintGeometry()


class PrintGeometryWidget(qt.QWidget):
    """Widget to specify the size and aspect ratio of an item
    before sending it to the print preview dialog.
    """
    def __init__(self, parent=None):
        super(PrintGeometryWidget, self).__init__(parent)
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)
        hbox = qt.QWidget()
        hboxLayout = qt.QHBoxLayout(hbox)
        hboxLayout.setContentsMargins(0, 0, 0, 0)
        hboxLayout.setSpacing(2)
        label = qt.QLabel(self)
        label.setText("Units")
        label.setAlignment(qt.Qt.AlignCenter)
        self._pageButton = qt.QRadioButton()
        self._pageButton.setText("Page")
        self._inchButton = qt.QRadioButton()
        self._inchButton.setText("Inches")
        self._cmButton = qt.QRadioButton()
        self._cmButton.setText("Centimeters")
        self._buttonGroup = qt.QButtonGroup(self)
        self._buttonGroup.addButton(self._pageButton)
        self._buttonGroup.addButton(self._inchButton)
        self._buttonGroup.addButton(self._cmButton)
        self._buttonGroup.setExclusive(True)

        # units
        self.mainLayout.addWidget(label, 0, 0, 1, 4)
        hboxLayout.addWidget(self._pageButton)
        hboxLayout.addWidget(self._inchButton)
        hboxLayout.addWidget(self._cmButton)
        self.mainLayout.addWidget(hbox, 1, 0, 1, 4)
        self._pageButton.setChecked(True)

        # xOffset
        label = qt.QLabel(self)
        label.setText("X Offset:")
        self.mainLayout.addWidget(label, 2, 0)
        self._xOffset = qt.QLineEdit(self)
        validator = qt.QDoubleValidator(None)
        self._xOffset.setValidator(validator)
        self._xOffset.setText("0.0")
        self.mainLayout.addWidget(self._xOffset, 2, 1)

        # yOffset
        label = qt.QLabel(self)
        label.setText("Y Offset:")
        self.mainLayout.addWidget(label, 2, 2)
        self._yOffset = qt.QLineEdit(self)
        validator = qt.QDoubleValidator(None)
        self._yOffset.setValidator(validator)
        self._yOffset.setText("0.0")
        self.mainLayout.addWidget(self._yOffset, 2, 3)

        # width
        label = qt.QLabel(self)
        label.setText("Width:")
        self.mainLayout.addWidget(label, 3, 0)
        self._width = qt.QLineEdit(self)
        validator = qt.QDoubleValidator(None)
        self._width.setValidator(validator)
        self._width.setText("0.5")
        self.mainLayout.addWidget(self._width, 3, 1)

        # height
        label = qt.QLabel(self)
        label.setText("Height:")
        self.mainLayout.addWidget(label, 3, 2)
        self._height = qt.QLineEdit(self)
        validator = qt.QDoubleValidator(None)
        self._height.setValidator(validator)
        self._height.setText("0.5")
        self.mainLayout.addWidget(self._height, 3, 3)

        # aspect ratio
        self._aspect = qt.QCheckBox(self)
        self._aspect.setText("Keep screen aspect ratio")
        self._aspect.setChecked(True)
        self.mainLayout.addWidget(self._aspect, 4, 1, 1, 2)

    def getPrintGeometry(self):
        ddict = {}
        if self._inchButton.isChecked():
            ddict['units'] = "inches"
        elif self._cmButton.isChecked():
            ddict['units'] = "centimeters"
        else:
            ddict['units'] = "page"

        ddict['xOffset'] = float(self._xOffset.text())
        ddict['yOffset'] = float(self._yOffset.text())
        ddict['width'] = float(self._width.text())
        ddict['height'] = float(self._height.text())

        if self._aspect.isChecked():
            ddict['keepAspectRatio'] = True
        else:
            ddict['keepAspectRatio'] = False
        return ddict

    def setPrintGeometry(self, ddict=None):
        if ddict is None:
            ddict = {}
        oldDict = self.getPrintGeometry()
        for key in ["units", "xOffset", "yOffset",
                    "width", "height", "keepAspectRatio"]:
            ddict[key] = ddict.get(key, oldDict[key])

        if ddict['units'].lower().startswith("inc"):
            self._inchButton.setChecked(True)
        elif ddict['units'].lower().startswith("c"):
            self._cmButton.setChecked(True)
        else:
            self._pageButton.setChecked(True)

        self._xOffset.setText("%s" % float(ddict['xOffset']))
        self._yOffset.setText("%s" % float(ddict['yOffset']))
        self._width.setText("%s" % float(ddict['width']))
        self._height.setText("%s" % float(ddict['height']))
        if ddict['keepAspectRatio']:
            self._aspect.setChecked(True)
        else:
            self._aspect.setChecked(False)


class PrintGeometryDialog(qt.QDialog):
    def __init__(self, parent=None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("Set print size preferences")
        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.configurationWidget = PrintGeometryWidget(self)
        hbox = qt.QWidget(self)
        hboxLayout = qt.QHBoxLayout(hbox)
        self.okButton = qt.QPushButton(hbox)
        self.okButton.setText("Accept")
        self.okButton.setAutoDefault(False)
        self.rejectButton = qt.QPushButton(hbox)
        self.rejectButton.setText("Dismiss")
        self.rejectButton.setAutoDefault(False)
        self.okButton.clicked.connect(self.accept)
        self.rejectButton.clicked.connect(self.reject)
        hboxLayout.setContentsMargins(0, 0, 0, 0)
        hboxLayout.setSpacing(2)
        # hboxLayout.addWidget(qt.HorizontalSpacer(hbox))
        hboxLayout.addWidget(self.okButton)
        hboxLayout.addWidget(self.rejectButton)
        # hboxLayout.addWidget(qt.HorizontalSpacer(hbox))
        layout.addWidget(self.configurationWidget)
        layout.addWidget(hbox)

    def setPrintGeometry(self, geometry):
        self.configurationWidget.setPrintGeometry(geometry)

    def getPrintGeometry(self):
        return self.configurationWidget.getPrintGeometry()


if __name__ == '__main__':
    from silx.gui.plot import PlotWidget
    import numpy
    app = qt.QApplication([])

    pw = PlotWidget()
    toolbar = qt.QToolBar()
    action = PrintPreviewAction(plot=pw)
    pw.addToolBar(toolbar)
    toolbar.addAction(action)
    pw.show()

    x = numpy.arange(1000)
    y = x / numpy.sin(x)
    pw.addCurve(x, y)

    app.exec_()
