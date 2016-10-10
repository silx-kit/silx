# coding: utf-8
#/*##########################################################################
# Copyright (C) 2004-2016 V.A. Sole, European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
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
# #########################################################################*/
"""Background configuration widget"""
import sys
import numpy
from silx.gui import qt
from silx.gui.plot import PlotWindow
from silx.math.fit import filters

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "05/10/2016"


class HorizontalSpacer(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setSizePolicy(qt.QSizePolicy(qt.QSizePolicy.Expanding,
                                          qt.QSizePolicy.Fixed))


class BackgroundParamWidget(qt.QWidget):
    """Background configuration widget.

    Strip and snip filters parameters can be adjusted, and
    the computed backgrounds are plotted next to the original data to
    show the result."""
    sigBackgroundParamWidgetSignal = qt.pyqtSignal(object)

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        self.mainLayout = qt.QGridLayout(self)

        # Strip parameters ---------------------------------------------------
        self.stripGroup = qt.QGroupBox("Strip", self)
        self.stripGroup.setCheckable(True)
        self.stripGroup.toggled.connect(self._stripGroupToggled)
        stripLayout = qt.QGridLayout(self.stripGroup)
        self.stripGroup.setLayout(stripLayout)

        self.stripWidthLabel = qt.QLabel(self.stripGroup)
        self.stripWidthLabel.setText("Strip Width")

        self.stripWidthSpin = qt.QSpinBox(self.stripGroup)
        self.stripWidthSpin.setMaximum(100)
        self.stripWidthSpin.setMinimum(1)
        self.stripWidthSpin.valueChanged[int].connect(self._emitSignal)

        # Strip iterations
        self.stripIterLabel = qt.QLabel(self.stripGroup)
        self.stripIterLabel.setText("Strip Iterations")
        self.stripIterValue = qt.QLineEdit(self.stripGroup)
        validator = qt.QIntValidator(self.stripIterValue)
        self.stripIterValue._v = validator
        self.stripIterValue.setText("0")
        self.stripIterValue.editingFinished[()].connect(self._emitSignal)

        stripLayout.setColumnStretch(1, 1)
        stripLayout.addWidget(self.stripWidthLabel, 0, 0)
        stripLayout.addWidget(self.stripWidthSpin, 0, 2)
        stripLayout.addWidget(self.stripIterLabel, 1, 0)
        stripLayout.addWidget(self.stripIterValue, 1, 2)

        # Snip parameters ----------------------------------------------------
        self.snipGroup = qt.QGroupBox("Snip", self)
        self.snipGroup.setCheckable(True)
        self.snipGroup.toggled.connect(self._snipGroupToggled)
        snipLayout = qt.QGridLayout(self.snipGroup)
        self.snipGroup.setLayout(snipLayout)

        self.snipWidthLabel = qt.QLabel(self.snipGroup)
        self.snipWidthLabel.setText("SNIP Width")

        self.snipWidthSpin = qt.QSpinBox(self.snipGroup)
        self.snipWidthSpin.setMaximum(300)
        self.snipWidthSpin.setMinimum(0)
        self.snipWidthSpin.valueChanged[int].connect(self._emitSignal)

        snipLayout.setColumnStretch(1, 1)
        snipLayout.addWidget(self.snipWidthLabel, 0, 0)
        snipLayout.addWidget(self.snipWidthSpin, 0, 2)

        # Smoothing parameters -----------------------------------------------
        self.smoothingGroup = qt.QGroupBox("Smoothing", self)
        self.smoothingGroup.setCheckable(True)
        self.smoothingGroup.toggled.connect(self._smoothingGroupToggled)
        smoothingLayout = qt.QGridLayout(self.smoothingGroup)
        self.smoothingGroup.setLayout(smoothingLayout)

        self.filterLabel = qt.QLabel(self.smoothingGroup)
        self.filterLabel.setText("Smoothing Width (Savitsky-Golay)")

        self.filterSpin = qt.QSpinBox(self.smoothingGroup)
        self.filterSpin.setMinimum(3)
        #self.filterSpin.setMaximum(40)
        self.filterSpin.setSingleStep(2)
        self.filterSpin.valueChanged[int].connect(self._emitSignal)

        smoothingLayout.setColumnStretch(1, 1)
        smoothingLayout.addWidget(self.filterLabel, 0, 0)
        smoothingLayout.addWidget(self.filterSpin, 0, 2)

        # Anchors ------------------------------------------------------------

        # Strip anchors
        self.anchorsGroup = qt.QGroupBox("Anchors", self)
        self.anchorsGroup.setToolTip(
                 "X coordinates of values that must remain constant")
        self.anchorsGroup.setCheckable(True)
        self.anchorsGroup.toggled.connect(self._emitSignal)
        anchorsLayout = qt.QHBoxLayout(self.anchorsGroup)
        anchorsLayout.setSpacing(2)
        self.anchorsGroup.setLayout(anchorsLayout)

        # self.stripAnchorsFlagCheck = qt.QCheckBox(self.anchorsGroup)
        # self.stripAnchorsFlagCheck.setText("Use anchors")
        # self.stripAnchorsFlagCheck.setToolTip(
        #         "X coordinates of values that must not be modified")
        # self.stripAnchorsFlagCheck.stateChanged[int].connect(
        #         self._emitSignal)
        # anchorsLayout.addWidget(self.stripAnchorsFlagCheck)

        maxnchannel = 16384 * 4    # Fixme ?
        self.anchorsList = []
        for i in range(4):
            anchorSpin = qt.QSpinBox(self.anchorsGroup)
            anchorSpin.setMinimum(0)
            anchorSpin.setMaximum(maxnchannel)
            anchorSpin.valueChanged[int].connect(self._emitSignal)
            anchorsLayout.addWidget(anchorSpin)
            self.anchorsList.append(anchorSpin)

        # --------------------------------------------------------------------
        self.mainLayout.addWidget(self.stripGroup)
        self.mainLayout.addWidget(self.snipGroup)
        self.mainLayout.addWidget(self.smoothingGroup)
        self.mainLayout.addWidget(self.anchorsGroup)

        margin = 6
        # margins left, top, right of groups
        for layout in [self.mainLayout, stripLayout, snipLayout,
                       smoothingLayout, anchorsLayout]:
            layout.setContentsMargins(margin, margin, margin, 0)
            layout.setSpacing(margin / 2)
        # margin above widgets inside group
        # for layout in []:
        #     self.layout.setSpacing(6)
        #     layout.setContentsMargins(0, margin, 0, 0)

        self.stripGroup.setChecked(True)
        self.snipGroup.setChecked(False)
        self.algorithm = "strip"

    def _stripGroupToggled(self, is_checked):
        """Slot called when the Strip group is activated

        :param is_checked: If *True*, use Strip algorithm, else use Snip
        """
        # snip and strip groups are mutually exclusive
        self.snipGroup.setChecked(not is_checked)
        self.algorithm = "strip" if is_checked else "snip"
        self._emitSignal()

    def _snipGroupToggled(self, is_checked):
        """Slot called when the Snip group is selectd

        :param is_checked: If *True*, use Strip algorithm, else use Snip
        """
        # snip and strip groups are mutually exclusive
        self.stripGroup.setChecked(not is_checked)
        self.algorithm = "snip" if is_checked else "strip"
        self._emitSignal()

    def _smoothingGroupToggled(self, is_checked):
        self._emitSignal()

    def setParameters(self, ddict):
        # if 'fit' in ddict:
        #     pars = ddict['fit']
        # else:
        #     pars = ddict

        if "algorithm" in ddict:
            if ddict["algorithm"] == "strip":
                self.stripGroup.setChecked(True)
            elif ddict["algorithm"] == "snip":
                self.snipGroup.setChecked(True)
            else:
                raise ValueError(
                        "Unknown background filter algorithm %s" % ddict["algorithm"])

        if "snipwidth" in ddict:
            self.snipWidthSpin.setValue(int(ddict["snipwidth"]))

        if "stripwidth" in ddict:
            self.stripWidthSpin.setValue(int(ddict["stripwidth"]))

        if "stripiterations" in ddict:
            self.stripIterValue.setText("%d" % int(ddict["stripiterations"]))

        if "smoothingflag" in ddict:
            self.smoothingGroup.setChecked(bool(ddict["smoothingflag"]))

        if "smoothingwidth" in ddict:
            self.filterSpin.setValue(int(ddict["smoothingwidth"]))

        if "anchorsflag" in ddict:
            self.anchorsGroup.setChecked(bool(ddict["anchorsflag"]))

        if "anchorslist" in ddict:
            anchorslist = ddict["anchorslist"]
            if anchorslist in [None, 'None']:
                anchorslist = []
            for spin in self.anchorsList:
                spin.setValue(0)

            i = 0
            for value in anchorslist:
                self.anchorsList[i].setValue(int(value))
                i += 1

    def getParameters(self):
        """Return dictionary of parameters defined in the GUI

        The returned dictionary contains following values:

            - *algorithm*: *"strip"* or *"snip"*
            - *stripwidth*: width of strip iterator
            - *stripiterations*: number of iterations
            - *snipwidth*: width of snip algorithm
            - *smoothingflag*: flag to enable/disable smoothing
            - *smoothingwidth*: width of Savitsky-Golay smoothing filter
            - *anchorsflag*: flag to enable/disable anchors
            - *anchorslist*: list of anchors (X coordinates of fixed values)
            """
        stripitertext = self.stripIterValue.text()
        stripiter = int(stripitertext) if len(stripitertext) else 0

        return {"algorithm": self.algorithm,
                "stripthreshold": 1.0,
                "snipwidth": self.snipWidthSpin.value(),
                "stripiterations": stripiter,
                "stripwidth": self.stripWidthSpin.value(),
                "smoothingflag": self.smoothingGroup.isChecked(),
                "smoothingwidth": self.filterSpin.value(),
                "anchorsflag": self.anchorsGroup.isChecked(),
                "anchorslist": [spin.value() for spin in self.anchorsList]}

    def _emitSignal(self, dummy=None):
        self.sigBackgroundParamWidgetSignal.emit(
            {'event': 'ParametersChanged',
             'parameters': self.getParameters()})


class BackgroundWidget(qt.QWidget):
    """Background configuration widget, with a :class:`PlotWindow`.

    Strip and snip filters parameters can be adjusted, and
    the computed backgrounds are plotted next to the original data to
    show the result."""
    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle("Strip and SNIP Configuration Window")
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)
        self.parametersWidget = BackgroundParamWidget(self)
        self.graphWidget = PlotWindow(parent=self,
                                      colormap=False,
                                      roi=False,
                                      mask=False,
                                      fit=False)
        self.mainLayout.addWidget(self.parametersWidget)
        self.mainLayout.addWidget(self.graphWidget)
        self.getParameters = self.parametersWidget.getParameters
        self.setParameters = self.parametersWidget.setParameters
        self._x = None
        self._y = None
        self.parametersWidget.sigBackgroundParamWidgetSignal.connect(self._slot)

    def setData(self, x, y):
        self._x = x
        self._y = y
        self.update(resetzoom=True)

    def _slot(self, ddict):
        self.update()

    def update(self, resetzoom=False):
        if self._y is None:
            return

        pars = self.getParameters()

        # smoothed data
        y = numpy.ravel(numpy.array(self._y)).astype(numpy.float)
        ysmooth = filters.savitsky_golay(y, pars['smoothingwidth'])
        f = [0.25, 0.5, 0.25]
        ysmooth[1:-1] = numpy.convolve(ysmooth, f, mode=0)
        ysmooth[0] = 0.5 * (ysmooth[0] + ysmooth[1])
        ysmooth[-1] = 0.5 * (ysmooth[-1] + ysmooth[-2])

        # loop for anchors
        x = self._x
        niter = pars['stripiterations']
        anchors_indices = []
        if pars['anchorsflag'] and pars['anchorslist'] is not None:
            ravelled = x
            for channel in pars['anchorslist']:
                if channel <= ravelled[0]:
                    continue
                index = numpy.nonzero(ravelled >= channel)[0]
                if len(index):
                    index = min(index)
                    if index > 0:
                        anchors_indices.append(index)

        stripBackground = filters.strip(ysmooth,
                                        w=pars['stripwidth'],
                                        niterations=niter,
                                        factor=pars['stripthreshold'],
                                        anchors=anchors_indices)

        if niter >= 1000:
            # final smoothing
            stripBackground = filters.strip(stripBackground,
                                            w=1,
                                            niterations=50*pars['stripwidth'],
                                            factor=pars['stripthreshold'],
                                            anchors=anchors_indices)

        if len(anchors_indices) == 0:
            anchors_indices = [0, len(ysmooth)-1]
        anchors_indices.sort()
        snipBackground = 0.0 * ysmooth
        lastAnchor = 0
        width = pars['snipwidth']
        for anchor in anchors_indices:
            if (anchor > lastAnchor) and (anchor < len(ysmooth)):
                snipBackground[lastAnchor:anchor] =\
                            filters.snip1d(ysmooth[lastAnchor:anchor], width)
                lastAnchor = anchor
        if lastAnchor < len(ysmooth):
            snipBackground[lastAnchor:] =\
                            filters.snip1d(ysmooth[lastAnchor:], width)

        self.graphWidget.addCurve(x, y,
                                  legend='Input Data',
                                  replace=True,
                                  resetzoom=resetzoom)
        self.graphWidget.addCurve(x, stripBackground,
                                  legend='Strip Background',
                                  resetzoom=False)
        self.graphWidget.addCurve(x, snipBackground,
                                  legend='SNIP Background',
                                  resetzoom=False)


class BackgroundDialog(qt.QDialog):
    def __init__(self, parent=None):
        qt.QDialog.__init__(self, parent)
        self.setWindowTitle("Strip and SNIP Configuration Window")
        self.mainLayout = qt.QVBoxLayout(self)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.setSpacing(2)
        self.parametersWidget = BackgroundWidget(self)
        self.setData = self.parametersWidget.setData
        self.getParameters = self.parametersWidget.getParameters
        self.setParameters = self.parametersWidget.setParameters
        self.mainLayout.addWidget(self.parametersWidget)
        hbox = qt.QWidget(self)
        hboxLayout = qt.QHBoxLayout(hbox)
        hboxLayout.setContentsMargins(0, 0, 0, 0)
        hboxLayout.setSpacing(2)
        self.okButton = qt.QPushButton(hbox)
        self.okButton.setText("OK")
        self.okButton.setAutoDefault(False)
        self.dismissButton = qt.QPushButton(hbox)
        self.dismissButton.setText("Cancel")
        self.dismissButton.setAutoDefault(False)
        hboxLayout.addWidget(HorizontalSpacer(hbox))
        hboxLayout.addWidget(self.okButton)
        hboxLayout.addWidget(self.dismissButton)
        self.mainLayout.addWidget(hbox)
        self.dismissButton.clicked.connect(self.reject)
        self.okButton.clicked.connect(self.accept)

    def sizeHint(self):
        return qt.QSize(int(1.5*qt.QDialog.sizeHint(self).width()),
                        qt.QDialog.sizeHint(self).height())


def main():
    # synthetic data
    from silx.math.fit.functions import sum_gauss

    x = numpy.arange(5000)
    # (height1, center1, fwhm1, ...) 5 peaks
    params1 = (50, 500, 100,
               20, 2000, 200,
               50, 2250, 100,
               40, 3000, 75,
               23, 4000, 150)
    y0 = sum_gauss(x, *params1)

    # random values between [-1;1]
    noise = 2 * numpy.random.random(5000) - 1
    # make it +- 5%
    noise *= 0.05

    # 2 gaussians with very large fwhm, as background signal
    actual_bg = sum_gauss(x, 15, 3500, 3000, 5, 1000, 1500)

    # Add 5% random noise to gaussians and add background
    y = y0 + numpy.average(y0) * noise + actual_bg

    # Open widget
    a = qt.QApplication(sys.argv)
    a.lastWindowClosed.connect(a.quit)

    def mySlot(ddict):
        print(ddict)

    w = BackgroundDialog()
    w.parametersWidget.parametersWidget.sigBackgroundParamWidgetSignal.connect(mySlot)
    w.setData(x, y)
    w.exec_()
    #a.exec_()

if __name__ == "__main__":
    main()
