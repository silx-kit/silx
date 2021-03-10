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


from silx.gui import qt
from silx.gui.widgets.FloatEdit import FloatEdit


class PrintGeometryWidget(qt.QWidget):
    """Widget to specify the size and aspect ratio of an item
    before sending it to the print preview dialog.

    Use methods :meth:`setPrintGeometry` and :meth:`getPrintGeometry`
    to interact with the widget.
    """
    def __init__(self, parent=None):
        super(PrintGeometryWidget, self).__init__(parent)
        self.mainLayout = qt.QGridLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)
        hbox = qt.QWidget(self)
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
        self._xOffset = FloatEdit(self, 0.1)
        self.mainLayout.addWidget(self._xOffset, 2, 1)

        # yOffset
        label = qt.QLabel(self)
        label.setText("Y Offset:")
        self.mainLayout.addWidget(label, 2, 2)
        self._yOffset = FloatEdit(self, 0.1)
        self.mainLayout.addWidget(self._yOffset, 2, 3)

        # width
        label = qt.QLabel(self)
        label.setText("Width:")
        self.mainLayout.addWidget(label, 3, 0)
        self._width = FloatEdit(self, 0.9)
        self.mainLayout.addWidget(self._width, 3, 1)

        # height
        label = qt.QLabel(self)
        label.setText("Height:")
        self.mainLayout.addWidget(label, 3, 2)
        self._height = FloatEdit(self, 0.9)
        self.mainLayout.addWidget(self._height, 3, 3)

        # aspect ratio
        self._aspect = qt.QCheckBox(self)
        self._aspect.setText("Keep screen aspect ratio")
        self._aspect.setChecked(True)
        self.mainLayout.addWidget(self._aspect, 4, 1, 1, 2)

    def getPrintGeometry(self):
        """Return the print geometry dictionary.

        See :meth:`setPrintGeometry` for documentation about the
        print geometry dictionary."""
        ddict = {}
        if self._inchButton.isChecked():
            ddict['units'] = "inches"
        elif self._cmButton.isChecked():
            ddict['units'] = "centimeters"
        else:
            ddict['units'] = "page"

        ddict['xOffset'] = self._xOffset.value()
        ddict['yOffset'] = self._yOffset.value()
        ddict['width'] = self._width.value()
        ddict['height'] = self._height.value()

        if self._aspect.isChecked():
            ddict['keepAspectRatio'] = True
        else:
            ddict['keepAspectRatio'] = False
        return ddict

    def setPrintGeometry(self, geometry=None):
        """Set the print geometry.

        The geometry parameters must be provided as a dictionary with
        the following keys:

         - *"xOffset"* (float)
         - *"yOffset"* (float)
         - *"width"* (float)
         - *"height"* (float)
         - *"units"*: possible values *"page", "inch", "cm"*
         - *"keepAspectRatio"*: *True* or *False*

        If *units* is *"page"*, the values should be floats in [0, 1.]
        and are interpreted as a fraction of the page width or height.

        :param dict geometry: Geometry parameters, as a dictionary."""
        if geometry is None:
            geometry = {}
        oldDict = self.getPrintGeometry()
        for key in ["units", "xOffset", "yOffset",
                    "width", "height", "keepAspectRatio"]:
            geometry[key] = geometry.get(key, oldDict[key])

        if geometry['units'].lower().startswith("inc"):
            self._inchButton.setChecked(True)
        elif geometry['units'].lower().startswith("c"):
            self._cmButton.setChecked(True)
        else:
            self._pageButton.setChecked(True)

        self._xOffset.setText("%s" % float(geometry['xOffset']))
        self._yOffset.setText("%s" % float(geometry['yOffset']))
        self._width.setText("%s" % float(geometry['width']))
        self._height.setText("%s" % float(geometry['height']))
        if geometry['keepAspectRatio']:
            self._aspect.setChecked(True)
        else:
            self._aspect.setChecked(False)


class PrintGeometryDialog(qt.QDialog):
    """Dialog embedding a :class:`PrintGeometryWidget`.

    Use methods :meth:`setPrintGeometry` and :meth:`getPrintGeometry`
    to interact with the widget.

    Execute method :meth:`exec_` to run the dialog.
    The return value of that method is *True* if the geometry was set
    (*Ok* button clicked) or *False* if the user clicked the *Cancel*
    button.
    """

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
        """Return the print geometry dictionary.

        See :meth:`PrintGeometryWidget.setPrintGeometry` for documentation on
        print geometry dictionary.

        :param dict geometry: Print geometry parameters dictionary.
        """
        self.configurationWidget.setPrintGeometry(geometry)

    def getPrintGeometry(self):
        """Return the print geometry dictionary.

        See :meth:`PrintGeometryWidget.setPrintGeometry` for documentation on
        print geometry dictionary."""
        return self.configurationWidget.getPrintGeometry()
