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
"""This module provides a widget designed to configure and run a fitting
process with constraints on parameters.

The main class is :class:`FitWidget`. It relies on
:mod:`silx.math.fit.fitmanager`.

The user can choose between functions before running the fit. These function can
be user defined, or by default are loaded from
:mod:`silx.math.fit.fittheories`.
"""
import logging
import sys
import traceback

from silx.math.fit import fittheories
from silx.math.fit import fitmanager, functions
from silx.gui import qt
from .FitWidgets import (FitActionsButtons, FitStatusLines,
                         FitConfigWidget, ParametersTab)
from .QScriptOption import QScriptOption

QTVERSION = qt.qVersion()

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "19/07/2016"

DEBUG = 0
_logger = logging.getLogger(__name__)


class FitWidget(qt.QWidget):
    """Widget to configure, run and display results of a fitting.
    It works hand in hand with a :class:`silx.math.fit.fitmanager.FitManager`
    object that handles the fit functions and calls the iterative least-square
    fitting algorithm.
    """
    sigFitWidgetSignal = qt.Signal(object)

    def __init__(self, parent=None, name=None, fitinstance=None,
                 enableconfig=True, enablestatus=True, enablebuttons=True):
        """

        :param parent: Parent widget
        :param name: Window title
        :param fitinstance: User defined instance of
            :class:`silx.math.fit.fitmanager.FitManager`, or ``None``
        :param enableconfig: If ``True``, activate widgets to modify the fit
            configuration (select between several fit functions or background
            functions, apply global constraints, peak search parameters…)
        :param enablestatus: If ``True``, add a fit status widget, to display
            a message when fit estimation is available and when fit results
            are available, as well as a measure of the fit error.
        :param enablebuttons: If ``True``, add buttons to run estimation and
            fitting.
        """
        if name is None:
            name = "FitWidget"
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle(name)
        layout = qt.QVBoxLayout(self)

        self.fitmanager = self._set_fitmanager(fitinstance)
        """Instance of :class:`FitManager`. If no theories are defined,
        we import the default ones from :mod:`silx.math.fit.fittheories`."""

        # copy fitmanager.configure method for direct access
        self.configure = self.fitmanager.configure
        self.fitconfig = self.fitmanager.fitconfig

        self.guiconfig = None
        """Configuration widget at the top of FitWidget, to select
        fit function, background function, and open an advanced
        configuration dialog."""

        self.guiparameters = ParametersTab(self)
        """Table widget for display of fit parameters and constraints"""
        # self.guiparameters.sigMultiParametersSignal.connect(self.__forward)  # mca related

        if enableconfig:
            self.guiconfig = FitConfigWidget(self)

            # self.guiconfig.MCACheckBox.stateChanged[int].connect(self.mcaevent)
            # self.guiconfig.WeightCheckBox.stateChanged[
            #     int].connect(self.weightevent)
            self.guiconfig.AutoFWHMCheckBox.stateChanged[
                int].connect(self.autofwhmevent)
            self.guiconfig.AutoScalingCheckBox.stateChanged[
                int].connect(self.autoscaleevent)
            self.guiconfig.ConfigureButton.clicked.connect(
                self.__configureGuiSlot)
            self.guiconfig.BkgComBox.activated[str].connect(self.bkgevent)
            self.guiconfig.FunComBox.activated[str].connect(self.funevent)
            layout.addWidget(self.guiconfig)

            for theory_name in self.fitmanager.bkgdict:
                self.guiconfig.BkgComBox.addItem(theory_name)
                self.guiconfig.BkgComBox.setItemData(
                    self.guiconfig.BkgComBox.findText(theory_name),
                    self.fitmanager.bkgdict[theory_name]["description"],
                    qt.Qt.ToolTipRole)

            for theory_name in self.fitmanager.theorydict:
                self.guiconfig.FunComBox.addItem(theory_name)
                self.guiconfig.FunComBox.setItemData(
                    self.guiconfig.FunComBox.findText(theory_name),
                    self.fitmanager.theorydict[theory_name]["description"],
                    qt.Qt.ToolTipRole)

            if fitinstance is not None:
                # customized FitManager provided in __init__:
                #    - activate selected fit theory (if any)
                #    - activate selected bg theory (if any)
                configuration = fitinstance.configure()
                if configuration['fittheory'] is None:
                    # take the first one by default
                    self.guiconfig.FunComBox.setCurrentIndex(1)
                    self.funevent(self.fitmanager.theorydict.keys[0])
                else:
                    self.funevent(configuration['fittheory'])
                if configuration['fitbkg'] is None:
                    self.guiconfig.BkgComBox.setCurrentIndex(1)
                    self.bkgevent(list(self.fitmanager.bkgdict.keys())[0])
                else:
                    self.bkgevent(configuration['fitbkg'])
            else:
                # Default FitManager and fittheories used:
                #    - activate first fit theory (gauss)
                #    - activate first bg theory (no bg)
                configuration = {}
                self.guiconfig.BkgComBox.setCurrentIndex(1)
                self.guiconfig.FunComBox.setCurrentIndex(1)
                self.funevent(list(self.fitmanager.theorydict.keys())[0])
                self.bkgevent(list(self.fitmanager.bkgdict.keys())[0])
            configuration.update(self.configure())

            # if configuration['McaMode']:
            #     self.guiconfig.MCACheckBox.setChecked(1)
            # else:
            #     self.guiconfig.MCACheckBox.setChecked(0)

            # if configuration['WeightFlag']:
            #     self.guiconfig.WeightCheckBox.setChecked(1)
            # else:
            #     self.guiconfig.WeightCheckBox.setChecked(0)

            if configuration['AutoFwhm']:
                self.guiconfig.AutoFWHMCheckBox.setChecked(1)
            else:
                self.guiconfig.AutoFWHMCheckBox.setChecked(0)

            if configuration['AutoScaling']:
                self.guiconfig.AutoScalingCheckBox.setChecked(1)
            else:
                self.guiconfig.AutoScalingCheckBox.setChecked(0)

        layout.addWidget(self.guiparameters)

        if enablestatus:
            self.guistatus = FitStatusLines(self)
            """Status bar"""
            layout.addWidget(self.guistatus)

        if enablebuttons:
            self.guibuttons = FitActionsButtons(self)
            """Widget with estimate, start fit and dismiss buttons"""
            self.guibuttons.EstimateButton.clicked.connect(self.estimate)
            self.guibuttons.StartfitButton.clicked.connect(self.startfit)
            self.guibuttons.DismissButton.clicked.connect(self.dismiss)
            layout.addWidget(self.guibuttons)

    # def updateGui(self, configuration=None):
    #     self.__configureGui(configuration)

    def _set_fitmanager(self, fitinstance):
        """Initialize a :class:`FitManager` instance, to be assigned to
        :attr:`fitmanager`"""
        if isinstance(fitinstance, fitmanager.FitManager):
            fitmngr = fitinstance
        else:
            fitmngr = fitmanager.FitManager()

        # initialize the default fitting functions in case
        # none is present
        if not len(fitmngr.theorydict):
            fitmngr.loadtheories(fittheories)
        return fitmngr

    def setdata(self, x, y, sigmay=None, xmin=None, xmax=None):
        """Set data to be fitted.

        :param x: Abscissa data. If ``None``, :attr:`xdata`` is set to
            ``numpy.array([0.0, 1.0, 2.0, ..., len(y)-1])``
        :type x: Sequence or numpy array or None
        :param y: The dependant data ``y = f(x)``. ``y`` must have the same
            shape as ``x`` if ``x`` is not ``None``.
        :type y: Sequence or numpy array or None
        :param sigmay: The uncertainties in the ``ydata`` array. These are
            used as weights in the least-squares problem.
            If ``None``, the uncertainties are assumed to be 1.
        :type sigmay: Sequence or numpy array or None
        :param xmin: Lower value of x values to use for fitting
        :param xmax: Upper value of x values to use for fitting
        """
        self.fitmanager.setdata(x=x, y=y, sigmay=sigmay,
                                xmin=xmin, xmax=xmax)

    def _emitSignal(self, ddict):
        """Emit pyqtSignal after estimation completed
        (``ddict = {'event': 'EstimateFinished', 'data': fit_results}``)
        and after fit completed
        (``ddict = {'event': 'FitFinished', 'data': fit_results}``)"""
        self.sigFitWidgetSignal.emit(ddict)

    def __configureGuiSlot(self):
        """Open an advanced configuration dialog widget"""
        self.__configureGui()

    def __configureGui(self, newconfiguration=None):
        """Open an advanced configuration dialog widget to get a configuration
        dictionary, or use a supplied configuration dictionary. Call
        :meth:`configure` with this dictionary as a parameter. Update the gui
        accordingly. Reinitialize the fit results in the table and in
        :attr:`fitmanager`.

        :param newconfiguration: User supplied configuration dictionary. If ``None``,
            open a dialog widget that returns a dictionary."""
        configuration = self.configure()
        # get new dictionary
        if newconfiguration is None:
            newconfiguration = self.configureGui(configuration)
        # update configuration
        configuration.update(self.configure(**newconfiguration))
        # set fit function theory
        try:
            i = 1 + \
                list(self.fitmanager.theorydict.keys()).index(
                    self.fitmanager.fitconfig['fittheory'])
            self.guiconfig.FunComBox.setCurrentIndex(i)
            self.funevent(self.fitmanager.fitconfig['fittheory'])
        except ValueError:
            _logger.error("Function not in list %s",
                          self.fitmanager.fitconfig['fittheory'])
            self.funevent(list(self.fitmanager.theorydict.keys())[0])
        # current background
        try:
            i = 1 + list(self.fitmanager.bkgdict.keys()
                         ).index(self.fitmanager.fitconfig['fitbkg'])
            self.guiconfig.BkgComBox.setCurrentIndex(i)
        except ValueError:
            _logger.error("Background not in list %s",
                          self.fitmanager.fitconfig['fitbkg'])
            self.bkgevent(list(self.fitmanager.bkgdict.keys())[0])

        # and all the rest
        # if configuration['McaMode']:
        #     self.guiconfig.MCACheckBox.setChecked(1)
        # else:
        #     self.guiconfig.MCACheckBox.setChecked(0)

        # if configuration['WeightFlag']:
        #     self.guiconfig.WeightCheckBox.setChecked(1)
        # else:
        #     self.guiconfig.WeightCheckBox.setChecked(0)

        if configuration['AutoFwhm']:
            self.guiconfig.AutoFWHMCheckBox.setChecked(1)
        else:
            self.guiconfig.AutoFWHMCheckBox.setChecked(0)

        if configuration['AutoScaling']:
            self.guiconfig.AutoScalingCheckBox.setChecked(1)
        else:
            self.guiconfig.AutoScalingCheckBox.setChecked(0)
        # update the Gui
        self.__initialparameters()

    def configureGui(self, oldconfiguration):
        """Display a dialog, allowing the user to define fit configuration
        parameters:

            - ``PositiveHeightAreaFlag``
            - ``QuotedPositionFlag``
            - ``PositiveFwhmFlag``
            - ``SameFwhmFlag``
            - ``QuotedEtaFlag``
            - ``NoConstraintsFlag``
            - ``FwhmPoints``
            - ``Sensitivity``
            - ``Yscaling``
            - ``ForcePeakPresence``

        :return: User defined parameters in a dictionary"""
        # this method can be overwritten for custom
        # it should give back a new dictionary
        newconfiguration = {}
        newconfiguration.update(oldconfiguration)

        # example script options like
        sheet1 = {'notetitle': 'Restrains',
                  'fields': (["CheckField", 'PositiveHeightAreaFlag',
                              'Force positive Height/Area'],
                             ["CheckField", 'QuotedPositionFlag',
                              'Force position in interval'],
                             ["CheckField", 'PositiveFwhmFlag',
                                 'Force positive FWHM'],
                             ["CheckField", 'SameFwhmFlag', 'Force same FWHM'],
                             ["CheckField", 'QuotedEtaFlag',
                                 'Force Eta between 0 and 1'],
                             ["CheckField", 'NoConstraintsFlag', 'Ignore Restrains'])}

        sheet2 = {'notetitle': 'Search',
                  'fields': (["EntryField", 'FwhmPoints', 'Fwhm Points: '],
                             ["EntryField", 'Sensitivity', 'Sensitivity: '],
                             ["EntryField", 'Yscaling',   'Y Factor   : '],
                             ["CheckField", 'ForcePeakPresence',   'Force peak presence '])}
        w = QScriptOption(self, name='Fit Configuration',
                          sheets=(sheet1, sheet2),
                          default=oldconfiguration)

        w.show()
        w.exec_()
        if w.result():
            newconfiguration.update(w.output)
        # we do not need the dialog any longer
        del w

        newconfiguration['FwhmPoints'] = int(
            float(newconfiguration['FwhmPoints']))
        newconfiguration['Sensitivity'] = float(
            newconfiguration['Sensitivity'])
        newconfiguration['Yscaling'] = float(newconfiguration['Yscaling'])

        return newconfiguration

    def estimate(self):
        """Run parameter estimation function then emit
        :attr:`sigFitWidgetSignal` with a dictionary containing a status
        message *'EstimateFinished'* and a list of fit parameters estimations
        in the format defined in
        :attr:`silx.math.fit.fitmanager.FitManager.fit_results`
        """
        try:
            theory_name = self.fitmanager.fitconfig['fittheory']
            estimation_function = self.fitmanager.theorydict[theory_name]["estimate"]
            if estimation_function is not None:
                self.fitmanager.estimate(callback=self.fitstatus)
            else:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Information)
                text = "Function does not define a way to estimate\n"
                text += "the initial parameters. Please, fill them\n"
                text += "yourself in the table and press Start Fit\n"
                msg.setText(text)
                msg.setWindowTitle('FitWidget Message')
                msg.exec_()
                return
        except:    # noqa (we want to catch and report all errors)
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error on estimate: %s" % traceback.format_exc())
            msg.exec_()
            return

        self.guiparameters.fillfromfit(
            self.fitmanager.fit_results, view='Fit')
        self.guiparameters.removeallviews(keep='Fit')
        ddict = {}
        ddict['event'] = 'EstimateFinished'
        ddict['data'] = self.fitmanager.fit_results
        self._emitSignal(ddict)

    # related to MCA
    # def __forward(self, ddict):
    #     self._emitSignal(ddict)

    def startfit(self):
        """Run fit, then emit :attr:`sigFitWidgetSignal` with a dictionary
        containing a status
        message *'FitFinished'* and a list of fit parameters results
        in the format defined in
        :attr:`silx.math.fit.fitmanager.FitManager.fit_results`
        """
        self.fitmanager.fit_results = self.guiparameters.getfitresults()
        try:
            self.fitmanager.startfit(callback=self.fitstatus)
        except:  # noqa (we want to catch and report all errors)
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error on Fit: %s" % traceback.format_exc())
            msg.exec_()
            return

        self.guiparameters.fillfromfit(
            self.fitmanager.fit_results, view='Fit')
        self.guiparameters.removeallviews(keep='Fit')
        ddict = {}
        ddict['event'] = 'FitFinished'
        ddict['data'] = self.fitmanager.fit_results
        self._emitSignal(ddict)
        return

    def autofwhmevent(self, item):
        """Set :attr:`fitmanager"fitconfig['AutoFwhm']`"""
        if int(item):
            self.configure(AutoFwhm=True)
        else:
            self.configure(AutoFwhm=False)

    def autoscaleevent(self, item):
        """Set :attr:`fitmanager"fitconfig['AutoScaling']`"""
        if int(item):
            self.configure(AutoScaling=True)
        else:
            self.configure(AutoScaling=False)

    def bkgevent(self, bgtheory):
        """Select background theory, then reinitialize parameters"""
        bgtheory = str(bgtheory)
        if bgtheory in self.fitmanager.bkgdict:
            self.fitmanager.setbackground(bgtheory)
        else:
            qt.QMessageBox.information(
                self, "Info",
                "%s is not a known background theory. Known theories are: " +
                ", ".join(self.fitmanager.bkgdict)
            )
            return
        self.__initialparameters()

    def funevent(self, theoryname):
        """Select a fit theory to be used for fitting. If this theory exists
        in :attr:`fitmanager`, use it. Then, reinitialize table.

        :param theoryname: Name of the fit theory to use for fitting. If this theory
            exists in :attr:`fitmanager`, use it. Else, open a file dialog to open
            a custom fit function definition file with
            :meth:`fitmanager.loadtheories`.
        """
        theoryname = str(theoryname)
        if theoryname in self.fitmanager.theorydict:
            self.fitmanager.settheory(theoryname)
        else:
            # open a load file dialog
            functionsfile = qt.QFileDialog.getOpenFileName(
                self, "Select python module with your function(s)", "",
                "Python Files (*.py);;All Files (*)")

            if len(functionsfile):
                try:
                    self.fitmanager.loadtheories(functionsfile)
                except ImportError:
                    qt.QMessageBox.critical(self, "ERROR",
                                            "Function not imported")
                    return
                else:
                    # empty the ComboBox
                    while(self.guiconfig.FunComBox.count() > 1):
                        self.guiconfig.FunComBox.removeItem(1)
                    # and fill it again
                    for key in self.fitmanager.theorydict:
                        self.guiconfig.FunComBox.addItem(str(key))

            i = 1 + \
                list(self.fitmanager.theorydict.keys()).index(
                    self.fitmanager.fitconfig['fittheory'])
            self.guiconfig.FunComBox.setCurrentIndex(i)
        self.__initialparameters()

    def __initialparameters(self):
        """Fill the fit parameters names with names of the parameters of
        the selected background theory and the selected fit theory.
        Initialize :attr:`fitmanager.fit_results` with these names, and
        initialize the table with them. This creates a view called "Fit"
        in :attr:`guiparameters`"""
        self.fitmanager.parameter_names = []
        self.fitmanager.fit_results = []
        for pname in self.fitmanager.bkgdict[self.fitmanager.fitconfig['fitbkg']]["parameters"]:
            self.fitmanager.parameter_names.append(pname)
            self.fitmanager.fit_results.append({'name': pname,
                                           'estimation': 0,
                                           'group': 0,
                                           'code': 'FREE',
                                           'cons1': 0,
                                           'cons2': 0,
                                           'fitresult': 0.0,
                                           'sigma': 0.0,
                                           'xmin': None,
                                           'xmax': None})
        if self.fitmanager.fitconfig['fittheory'] is not None:
            theory = self.fitmanager.fitconfig['fittheory']
            for pname in self.fitmanager.theorydict[theory]["parameters"]:
                self.fitmanager.parameter_names.append(pname + "1")
                self.fitmanager.fit_results.append({'name': pname + "1",
                                               'estimation': 0,
                                               'group': 1,
                                               'code': 'FREE',
                                               'cons1': 0,
                                               'cons2': 0,
                                               'fitresult': 0.0,
                                               'sigma': 0.0,
                                               'xmin': None,
                                               'xmax': None})
        # if self.fitmanager.fitconfig['McaMode']:
        #     self.guiparameters.fillfromfit(
        #         self.fitmanager.fit_results, view='Region 1')
        #     self.guiparameters.removeallviews(keep='Region 1')
        # else:
        self.guiparameters.fillfromfit(
            self.fitmanager.fit_results, view='Fit')

    def fitstatus(self, data):
        """Set *status* and *chisq* in status bar"""
        if 'chisq' in data:
            if data['chisq'] is None:
                self.guistatus.ChisqLine.setText(" ")
            else:
                chisq = data['chisq']
                self.guistatus.ChisqLine.setText("%6.2f" % chisq)

        if 'status' in data:
            status = data['status']
            self.guistatus.StatusLine.setText(str(status))

    def dismiss(self):
        """Close FitWidget"""
        self.close()


if __name__ == "__main__":
    import numpy

    x = numpy.arange(1500).astype(numpy.float)
    constant_bg = 3.14

    p = [1000, 100., 30.0,
         500, 300., 25.,
         1700, 500., 35.,
         750, 700., 30.0,
         1234, 900., 29.5,
         302, 1100., 30.5,
         75, 1300., 210.]
    y = functions.sum_gauss(x, *p) + constant_bg

    a = qt.QApplication(sys.argv)
    w = FitWidget()
    w.setdata(x=x, y=y)
    w.show()
    a.exec_()
