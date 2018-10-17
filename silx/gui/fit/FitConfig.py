# coding: utf-8
# /*##########################################################################
# Copyright (C) 2004-2018 V.A. Sole, European Synchrotron Radiation Facility
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
# ######################################################################### */
"""This module defines widgets used to build a fit configuration dialog.
The resulting dialog widget outputs a dictionary of configuration parameters.
"""
from silx.gui import qt

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "30/11/2016"


class TabsDialog(qt.QDialog):
    """Dialog widget containing a QTabWidget :attr:`tabWidget`
    and a buttons:

        # - buttonHelp
        - buttonDefaults
        - buttonOk
        - buttonCancel

    This dialog defines a __len__ returning the number of tabs,
    and an __iter__ method yielding the tab widgets.
    """
    def __init__(self, parent=None):
        qt.QDialog.__init__(self, parent)
        self.tabWidget = qt.QTabWidget(self)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.tabWidget)

        layout2 = qt.QHBoxLayout(None)

        # self.buttonHelp = qt.QPushButton(self)
        # self.buttonHelp.setText("Help")
        # layout2.addWidget(self.buttonHelp)

        self.buttonDefault = qt.QPushButton(self)
        self.buttonDefault.setText("Undo changes")
        layout2.addWidget(self.buttonDefault)

        spacer = qt.QSpacerItem(20, 20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        layout2.addItem(spacer)

        self.buttonOk = qt.QPushButton(self)
        self.buttonOk.setText("OK")
        layout2.addWidget(self.buttonOk)

        self.buttonCancel = qt.QPushButton(self)
        self.buttonCancel.setText("Cancel")
        layout2.addWidget(self.buttonCancel)

        layout.addLayout(layout2)

        self.buttonOk.clicked.connect(self.accept)
        self.buttonCancel.clicked.connect(self.reject)

    def __len__(self):
        """Return number of tabs"""
        return self.tabWidget.count()

    def __iter__(self):
        """Return the next tab widget in  :attr:`tabWidget` every
        time this method is called.

        :return: Tab widget
        :rtype: QWidget
        """
        for widget_index in range(len(self)):
            yield self.tabWidget.widget(widget_index)

    def addTab(self, page, label):
        """Add a new tab

        :param page: Content of new page. Must be a widget with
            a get() method returning a dictionary.
        :param str label: Tab label
        """
        self.tabWidget.addTab(page, label)

    def getTabLabels(self):
        """
        Return a list of all tab labels in :attr:`tabWidget`
        """
        return [self.tabWidget.tabText(i) for i in range(len(self))]


class TabsDialogData(TabsDialog):
    """This dialog adds a data attribute to :class:`TabsDialog`.

    Data input in widgets, such as text entries or checkboxes, is stored in an
    attribute :attr:`output` when the user clicks the OK button.

    A default dictionary can be supplied when this dialog is initialized, to
    be used as default data for :attr:`output`.
    """
    def __init__(self, parent=None, modal=True, default=None):
        """

        :param parent: Parent :class:`QWidget`
        :param modal: If `True`, dialog is modal, meaning this dialog remains
            in front of it's parent window and disables it until the user is
            done interacting with the dialog
        :param default: Default dictionary, used to initialize and reset
            :attr:`output`.
        """
        TabsDialog.__init__(self, parent)
        self.setModal(modal)
        self.setWindowTitle("Fit configuration")

        self.output = {}

        self.default = {} if default is None else default

        self.buttonDefault.clicked.connect(self._resetDefault)
        # self.keyPressEvent(qt.Qt.Key_Enter).

    def keyPressEvent(self, event):
        """Redefining this method to ignore Enter key
        (for some reason it activates buttonDefault callback which
        resets all widgets)
        """
        if event.key() in [qt.Qt.Key_Enter, qt.Qt.Key_Return]:
            return
        TabsDialog.keyPressEvent(self, event)

    def accept(self):
        """When *OK* is clicked, update :attr:`output` with data from
        various widgets
        """
        self.output.update(self.default)

        # loop over all tab widgets (uses TabsDialog.__iter__)
        for tabWidget in self:
            self.output.update(tabWidget.get())

        # avoid pathological None cases
        for key in self.output.keys():
            if self.output[key] is None:
                if key in self.default:
                    self.output[key] = self.default[key]
        super(TabsDialogData, self).accept()

    def reject(self):
        """When the *Cancel* button is clicked, reinitialize :attr:`output`
        and quit
        """
        self.setDefault()
        super(TabsDialogData, self).reject()

    def _resetDefault(self, checked):
        self.setDefault()

    def setDefault(self, newdefault=None):
        """Reinitialize :attr:`output` with :attr:`default` or with
        new dictionary ``newdefault`` if provided.
        Call :meth:`setDefault` for each tab widget, if available.
        """
        self.output = {}
        if newdefault is None:
            newdefault = self.default
        else:
            self.default = newdefault
        self.output.update(newdefault)

        for tabWidget in self:
            if hasattr(tabWidget, "setDefault"):
                tabWidget.setDefault(self.output)


class ConstraintsPage(qt.QGroupBox):
    """Checkable QGroupBox widget filled with QCheckBox widgets,
    to configure the fit estimation for standard fit theories.
    """
    def __init__(self, parent=None, title="Set constraints"):
        super(ConstraintsPage, self).__init__(parent)
        self.setTitle(title)
        self.setToolTip("Disable 'Set constraints' to remove all " +
                        "constraints on all fit parameters")
        self.setCheckable(True)

        layout = qt.QVBoxLayout(self)
        self.setLayout(layout)

        self.positiveHeightCB = qt.QCheckBox("Force positive height/area", self)
        self.positiveHeightCB.setToolTip("Fit must find positive peaks")
        layout.addWidget(self.positiveHeightCB)

        self.positionInIntervalCB = qt.QCheckBox("Force position in interval", self)
        self.positionInIntervalCB.setToolTip(
                "Fit must position peak within X limits")
        layout.addWidget(self.positionInIntervalCB)

        self.positiveFwhmCB = qt.QCheckBox("Force positive FWHM", self)
        self.positiveFwhmCB.setToolTip("Fit must find a positive FWHM")
        layout.addWidget(self.positiveFwhmCB)

        self.sameFwhmCB = qt.QCheckBox("Force same FWHM for all peaks", self)
        self.sameFwhmCB.setToolTip("Fit must find same FWHM for all peaks")
        layout.addWidget(self.sameFwhmCB)

        self.quotedEtaCB = qt.QCheckBox("Force Eta between 0 and 1", self)
        self.quotedEtaCB.setToolTip(
                "Fit must find Eta between 0 and 1 for pseudo-Voigt function")
        layout.addWidget(self.quotedEtaCB)

        layout.addStretch()

        self.setDefault()

    def setDefault(self, default_dict=None):
        """Set default state for all widgets.

        :param default_dict: If a default config dictionary is provided as
            a parameter, its values are used as default state."""
        if default_dict is None:
            default_dict = {}
        # this one uses reverse logic: if checked, NoConstraintsFlag must be False
        self.setChecked(
                not default_dict.get('NoConstraintsFlag', False))
        self.positiveHeightCB.setChecked(
                default_dict.get('PositiveHeightAreaFlag', True))
        self.positionInIntervalCB.setChecked(
                default_dict.get('QuotedPositionFlag', False))
        self.positiveFwhmCB.setChecked(
                default_dict.get('PositiveFwhmFlag', True))
        self.sameFwhmCB.setChecked(
                default_dict.get('SameFwhmFlag', False))
        self.quotedEtaCB.setChecked(
                default_dict.get('QuotedEtaFlag', False))

    def get(self):
        """Return a dictionary of constraint flags, to be processed by the
        :meth:`configure` method of the selected fit theory."""
        ddict = {
            'NoConstraintsFlag': not self.isChecked(),
            'PositiveHeightAreaFlag': self.positiveHeightCB.isChecked(),
            'QuotedPositionFlag': self.positionInIntervalCB.isChecked(),
            'PositiveFwhmFlag': self.positiveFwhmCB.isChecked(),
            'SameFwhmFlag': self.sameFwhmCB.isChecked(),
            'QuotedEtaFlag': self.quotedEtaCB.isChecked(),
        }
        return ddict


class SearchPage(qt.QWidget):
    def __init__(self, parent=None):
        super(SearchPage, self).__init__(parent)
        layout = qt.QVBoxLayout(self)

        self.manualFwhmGB = qt.QGroupBox("Define FWHM manually", self)
        self.manualFwhmGB.setCheckable(True)
        self.manualFwhmGB.setToolTip(
            "If disabled, the FWHM parameter used for peak search is " +
            "estimated based on the highest peak in the data")
        layout.addWidget(self.manualFwhmGB)
        # ------------ GroupBox fwhm--------------------------
        layout2 = qt.QHBoxLayout(self.manualFwhmGB)
        self.manualFwhmGB.setLayout(layout2)

        label = qt.QLabel("Fwhm Points", self.manualFwhmGB)
        layout2.addWidget(label)

        self.fwhmPointsSpin = qt.QSpinBox(self.manualFwhmGB)
        self.fwhmPointsSpin.setRange(0, 999999)
        self.fwhmPointsSpin.setToolTip("Typical peak fwhm (number of data points)")
        layout2.addWidget(self.fwhmPointsSpin)
        # ----------------------------------------------------

        self.manualScalingGB = qt.QGroupBox("Define scaling manually", self)
        self.manualScalingGB.setCheckable(True)
        self.manualScalingGB.setToolTip(
            "If disabled, the Y scaling used for peak search is " +
            "estimated automatically")
        layout.addWidget(self.manualScalingGB)
        # ------------ GroupBox scaling-----------------------
        layout3 = qt.QHBoxLayout(self.manualScalingGB)
        self.manualScalingGB.setLayout(layout3)

        label = qt.QLabel("Y Scaling", self.manualScalingGB)
        layout3.addWidget(label)

        self.yScalingEntry = qt.QLineEdit(self.manualScalingGB)
        self.yScalingEntry.setToolTip(
                "Data values will be multiplied by this value prior to peak" +
                " search")
        self.yScalingEntry.setValidator(qt.QDoubleValidator(self))
        layout3.addWidget(self.yScalingEntry)
        # ----------------------------------------------------

        # ------------------- grid layout --------------------
        containerWidget = qt.QWidget(self)
        layout4 = qt.QHBoxLayout(containerWidget)
        containerWidget.setLayout(layout4)

        label = qt.QLabel("Sensitivity", containerWidget)
        layout4.addWidget(label)

        self.sensitivityEntry = qt.QLineEdit(containerWidget)
        self.sensitivityEntry.setToolTip(
            "Peak search sensitivity threshold, expressed as a multiple " +
            "of the standard deviation of the noise.\nMinimum value is 1 " +
            "(to be detected, peak must be higher than the estimated noise)")
        sensivalidator = qt.QDoubleValidator(self)
        sensivalidator.setBottom(1.0)
        self.sensitivityEntry.setValidator(sensivalidator)
        layout4.addWidget(self.sensitivityEntry)
        # ----------------------------------------------------
        layout.addWidget(containerWidget)

        self.forcePeakPresenceCB = qt.QCheckBox("Force peak presence", self)
        self.forcePeakPresenceCB.setToolTip(
                "If peak search algorithm is unsuccessful, place one peak " +
                "at the maximum of the curve")
        layout.addWidget(self.forcePeakPresenceCB)

        layout.addStretch()

        self.setDefault()

    def setDefault(self, default_dict=None):
        """Set default values for all widgets.

        :param default_dict: If a default config dictionary is provided as
            a parameter, its values are used as default values."""
        if default_dict is None:
            default_dict = {}
        self.manualFwhmGB.setChecked(
                not default_dict.get('AutoFwhm', True))
        self.fwhmPointsSpin.setValue(
               default_dict.get('FwhmPoints', 8))
        self.sensitivityEntry.setText(
                str(default_dict.get('Sensitivity', 1.0)))
        self.manualScalingGB.setChecked(
                not default_dict.get('AutoScaling', False))
        self.yScalingEntry.setText(
                str(default_dict.get('Yscaling', 1.0)))
        self.forcePeakPresenceCB.setChecked(
                default_dict.get('ForcePeakPresence', False))

    def get(self):
        """Return a dictionary of peak search parameters, to be processed by
        the :meth:`configure` method of the selected fit theory."""
        ddict = {
            'AutoFwhm': not self.manualFwhmGB.isChecked(),
            'FwhmPoints': self.fwhmPointsSpin.value(),
            'Sensitivity': safe_float(self.sensitivityEntry.text()),
            'AutoScaling': not self.manualScalingGB.isChecked(),
            'Yscaling': safe_float(self.yScalingEntry.text()),
            'ForcePeakPresence': self.forcePeakPresenceCB.isChecked()
        }
        return ddict


class BackgroundPage(qt.QGroupBox):
    """Background subtraction configuration, specific to fittheories
    estimation functions."""
    def __init__(self, parent=None,
                 title="Subtract strip background prior to estimation"):
        super(BackgroundPage, self).__init__(parent)
        self.setTitle(title)
        self.setCheckable(True)
        self.setToolTip(
            "The strip algorithm strips away peaks to compute the " +
            "background signal.\nAt each iteration, a sample is compared " +
            "to the average of the two samples at a given distance in both" +
            " directions,\n and if its value is higher than the average,"
            "it is replaced by the average.")

        layout = qt.QGridLayout(self)
        self.setLayout(layout)

        for i, label_text in enumerate(
                ["Strip width (in samples)",
                 "Number of iterations",
                 "Strip threshold factor"]):
            label = qt.QLabel(label_text)
            layout.addWidget(label, i, 0)

        self.stripWidthSpin = qt.QSpinBox(self)
        self.stripWidthSpin.setToolTip(
            "Width, in number of samples, of the strip operator")
        self.stripWidthSpin.setRange(1, 999999)

        layout.addWidget(self.stripWidthSpin, 0, 1)

        self.numIterationsSpin = qt.QSpinBox(self)
        self.numIterationsSpin.setToolTip(
            "Number of iterations of the strip algorithm")
        self.numIterationsSpin.setRange(1, 999999)
        layout.addWidget(self.numIterationsSpin, 1, 1)

        self.thresholdFactorEntry = qt.QLineEdit(self)
        self.thresholdFactorEntry.setToolTip(
            "Factor used by the strip algorithm to decide whether a sample" +
            "value should be stripped.\nThe value must be higher than the " +
            "average of the 2 samples at +- w times this factor.\n")
        self.thresholdFactorEntry.setValidator(qt.QDoubleValidator(self))
        layout.addWidget(self.thresholdFactorEntry, 2, 1)

        self.smoothStripGB = qt.QGroupBox("Apply smoothing prior to strip", self)
        self.smoothStripGB.setCheckable(True)
        self.smoothStripGB.setToolTip(
                "Apply a smoothing before subtracting strip background" +
                " in fit and estimate processes")
        smoothlayout = qt.QHBoxLayout(self.smoothStripGB)
        label = qt.QLabel("Smoothing width (Savitsky-Golay)")
        smoothlayout.addWidget(label)
        self.smoothingWidthSpin = qt.QSpinBox(self)
        self.smoothingWidthSpin.setToolTip(
            "Width parameter for Savitsky-Golay smoothing (number of samples, must be odd)")
        self.smoothingWidthSpin.setRange(3, 101)
        self.smoothingWidthSpin.setSingleStep(2)
        smoothlayout.addWidget(self.smoothingWidthSpin)

        layout.addWidget(self.smoothStripGB, 3, 0, 1, 2)

        layout.setRowStretch(4, 1)

        self.setDefault()

    def setDefault(self, default_dict=None):
        """Set default values for all widgets.

        :param default_dict: If a default config dictionary is provided as
            a parameter, its values are used as default values."""
        if default_dict is None:
            default_dict = {}

        self.setChecked(
                default_dict.get('StripBackgroundFlag', True))

        self.stripWidthSpin.setValue(
                default_dict.get('StripWidth', 2))
        self.numIterationsSpin.setValue(
                default_dict.get('StripIterations', 5000))
        self.thresholdFactorEntry.setText(
                str(default_dict.get('StripThreshold', 1.0)))
        self.smoothStripGB.setChecked(
                default_dict.get('SmoothingFlag', False))
        self.smoothingWidthSpin.setValue(
                default_dict.get('SmoothingWidth', 3))

    def get(self):
        """Return a dictionary of background subtraction parameters, to be
        processed by the :meth:`configure` method of the selected fit theory.
        """
        ddict = {
            'StripBackgroundFlag': self.isChecked(),
            'StripWidth': self.stripWidthSpin.value(),
            'StripIterations': self.numIterationsSpin.value(),
            'StripThreshold': safe_float(self.thresholdFactorEntry.text()),
            'SmoothingFlag': self.smoothStripGB.isChecked(),
            'SmoothingWidth': self.smoothingWidthSpin.value()
        }
        return ddict


def safe_float(string_, default=1.0):
    """Convert a string into a float.
    If the conversion fails, return the default value.
    """
    try:
        ret = float(string_)
    except ValueError:
        return default
    else:
        return ret


def safe_int(string_, default=1):
    """Convert a string into a integer.
    If the conversion fails, return the default value.
    """
    try:
        ret = int(float(string_))
    except ValueError:
        return default
    else:
        return ret


def getFitConfigDialog(parent=None, default=None, modal=True):
    """Instantiate and return a fit configuration dialog, adapted
    for configuring standard fit theories from
    :mod:`silx.math.fit.fittheories`.

    :return: Instance of :class:`TabsDialogData` with 3 tabs:
        :class:`ConstraintsPage`, :class:`SearchPage` and
        :class:`BackgroundPage`
    """
    tdd = TabsDialogData(parent=parent, default=default)
    tdd.addTab(ConstraintsPage(), label="Constraints")
    tdd.addTab(SearchPage(), label="Peak search")
    tdd.addTab(BackgroundPage(), label="Background")
    # apply default to newly added pages
    tdd.setDefault()

    return tdd


def main():
    a = qt.QApplication([])

    mw = qt.QMainWindow()
    mw.show()

    tdd = getFitConfigDialog(mw, default={"a": 1})
    tdd.show()
    tdd.exec_()
    print("TabsDialogData result: ", tdd.result())
    print("TabsDialogData output: ", tdd.output)

    a.exec_()

if __name__ == "__main__":
    main()
