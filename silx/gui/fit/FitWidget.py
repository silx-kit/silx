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
import warnings

from silx.math.fit import fittheories
from silx.math.fit import fitmanager, functions
from silx.gui import qt
from .FitWidgets import (FitActionsButtons, FitStatusLines,
                         FitConfigWidget, ParametersTab)
from .QScriptOption import QScriptOption

QTVERSION = qt.qVersion()

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "11/08/2016"

DEBUG = 0
_logger = logging.getLogger(__name__)


class FitWidget(qt.QWidget):
    """Widget to configure, run and display results of a fitting.
    It works hand in hand with a :class:`silx.math.fit.fitmanager.FitManager`
    object that handles the fit functions and calls the iterative least-square
    fitting algorithm.
    """
    sigFitWidgetSignal = qt.Signal(object)
    """This signal is emitted when:

        - estimation is complete
        - fit is complete

    It carries a dictionary with two items:

        - *event*: *EstimateFinished* or *FitFinished*
        - *data*: fit result (see documentation for
          :attr:`silx.math.fit.fitmanager.FitManager.fit_results`)
    """

    def __init__(self, parent=None, title=None, fitmngr=None,
                 enableconfig=True, enablestatus=True, enablebuttons=True):
        """

        :param parent: Parent widget
        :param title: Window title
        :param fitmngr: User defined instance of
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
        if title is None:
            title = "FitWidget"
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle(title)
        layout = qt.QVBoxLayout(self)

        self.fitmanager = self._setFitManager(fitmngr)
        """Instance of :class:`FitManager`. If no theories are defined,
        we import the default ones from :mod:`silx.math.fit.fittheories`."""

        # reference fitmanager.configure method for direct access
        self.configure = self.fitmanager.configure
        self.fitConfig = self.fitmanager.fitconfig

        self.guiConfig = None
        """Configuration widget at the top of FitWidget, to select
        fit function, background function, and open an advanced
        configuration dialog."""

        self.guiParameters = ParametersTab(self)
        """Table widget for display of fit parameters and constraints"""
        # self.guiParameters.sigMultiParametersSignal.connect(self.__forward)  # mca related

        if enableconfig:
            # todo:
            #     - separate theory selection from config
            #     - replace config with FitTheory.configWidget (if provided)

            self.guiConfig = FitConfigWidget(self)

            # self.guiConfig.MCACheckBox.stateChanged[int].connect(self.mcaevent)
            # self.guiConfig.WeightCheckBox.stateChanged[
            #     int].connect(self.weightevent)
            self.guiConfig.AutoFWHMCheckBox.stateChanged[
                int].connect(self.autoFwhmEvent)
            # self.guiConfig.AutoScalingCheckBox.stateChanged[
            #     int].connect(self.autoscaleevent)
            self.guiConfig.ConfigureButton.clicked.connect(
                self.__configureGuiSlot)
            self.guiConfig.BkgComBox.activated[str].connect(self.bkgEvent)
            self.guiConfig.FunComBox.activated[str].connect(self.funEvent)
            layout.addWidget(self.guiConfig)

            for theory_name in self.fitmanager.bgtheories:
                self.guiConfig.BkgComBox.addItem(theory_name)
                self.guiConfig.BkgComBox.setItemData(
                    self.guiConfig.BkgComBox.findText(theory_name),
                    self.fitmanager.bgtheories[theory_name].description,
                    qt.Qt.ToolTipRole)

            for theory_name in self.fitmanager.theories:
                self.guiConfig.FunComBox.addItem(theory_name)
                self.guiConfig.FunComBox.setItemData(
                    self.guiConfig.FunComBox.findText(theory_name),
                    self.fitmanager.theories[theory_name].description,
                    qt.Qt.ToolTipRole)

            if fitmngr is not None:
                # customized FitManager provided in __init__:
                #    - activate selected fit theory (if any)
                #    - activate selected bg theory (if any)
                configuration = fitmngr.configure()
                if fitmngr.selectedtheory is None:
                    # take the first one by default
                    self.guiConfig.FunComBox.setCurrentIndex(1)
                    self.funEvent(list(self.fitmanager.theories.keys())[0])
                else:
                    self.funEvent(fitmngr.selectedtheory)
                if fitmngr.selectedbg is None:
                    self.guiConfig.BkgComBox.setCurrentIndex(0)
                    self.bkgEvent(list(self.fitmanager.bgtheories.keys())[0])
                else:
                    self.bkgEvent(fitmngr.selectedbg)
            else:
                # Default FitManager and fittheories used:
                #    - activate first fit theory (gauss)
                #    - activate first bg theory (no bg)
                configuration = {}
                self.guiConfig.BkgComBox.setCurrentIndex(0)
                self.guiConfig.FunComBox.setCurrentIndex(1)  # Index 0 is "Add function"
                self.funEvent(list(self.fitmanager.theories.keys())[0])
                self.bkgEvent(list(self.fitmanager.bgtheories.keys())[0])
            configuration.update(self.configure())

            # if configuration['WeightFlag']:
            #     self.guiConfig.WeightCheckBox.setChecked(1)
            # else:
            #     self.guiConfig.WeightCheckBox.setChecked(0)

            if configuration['AutoFwhm']:
                self.guiConfig.AutoFWHMCheckBox.setChecked(1)
            else:
                self.guiConfig.AutoFWHMCheckBox.setChecked(0)

            # if configuration['AutoScaling']:
            #     self.guiConfig.AutoScalingCheckBox.setChecked(1)
            # else:
            #     self.guiConfig.AutoScalingCheckBox.setChecked(0)

        layout.addWidget(self.guiParameters)

        if enablestatus:
            self.guistatus = FitStatusLines(self)
            """Status bar"""
            layout.addWidget(self.guistatus)

        if enablebuttons:
            self.guibuttons = FitActionsButtons(self)
            """Widget with estimate, start fit and dismiss buttons"""
            self.guibuttons.EstimateButton.clicked.connect(self.estimate)
            self.guibuttons.StartFitButton.clicked.connect(self.startFit)
            self.guibuttons.DismissButton.clicked.connect(self.dismiss)
            layout.addWidget(self.guibuttons)

    # def updateGui(self, configuration=None):
    #     self.__configureGui(configuration)

    def _setFitManager(self, fitinstance):
        """Initialize a :class:`FitManager` instance, to be assigned to
        :attr:`fitmanager`"""
        if isinstance(fitinstance, fitmanager.FitManager):
            fitmngr = fitinstance
        else:
            fitmngr = fitmanager.FitManager()

        # initialize the default fitting functions in case
        # none is present
        if not len(fitmngr.theories):
            fitmngr.loadtheories(fittheories)
        return fitmngr

    def setdata(self, x, y, sigmay=None, xmin=None, xmax=None):
        warnings.warn("Method renamed to setData",
                      DeprecationWarning)
        self.setData(x, y, sigmay, xmin, xmax)

    def setData(self, x, y, sigmay=None, xmin=None, xmax=None):
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
            newconfiguration = self.configureDialog(configuration)
        # update configuration
        configuration.update(self.configure(**newconfiguration))
        # set fit function theory
        try:
            i = 1 + \
                list(self.fitmanager.theories.keys()).index(
                    self.fitmanager.selectedtheory)
            self.guiConfig.FunComBox.setCurrentIndex(i)
            self.funEvent(self.fitmanager.selectedtheory)
        except ValueError:
            _logger.error("Function not in list %s",
                          self.fitmanager.selectedtheory)
            self.funEvent(list(self.fitmanager.theories.keys())[0])
        # current background
        try:
            i = list(self.fitmanager.bgtheories.keys()
                     ).index(self.fitmanager.selectedbg)
            self.guiConfig.BkgComBox.setCurrentIndex(i)
        except ValueError:
            _logger.error("Background not in list %s",
                          self.fitmanager.selectedbg)
            self.bkgEvent(list(self.fitmanager.bgtheories.keys())[0])

        # and all the rest
        # if configuration['McaMode']:
        #     self.guiConfig.MCACheckBox.setChecked(1)
        # else:
        #     self.guiConfig.MCACheckBox.setChecked(0)

        # if configuration['WeightFlag']:
        #     self.guiConfig.WeightCheckBox.setChecked(1)
        # else:
        #     self.guiConfig.WeightCheckBox.setChecked(0)

        if configuration['AutoFwhm']:
            self.guiConfig.AutoFWHMCheckBox.setChecked(1)
        else:
            self.guiConfig.AutoFWHMCheckBox.setChecked(0)

        # if configuration['AutoScaling']:
        #     self.guiConfig.AutoScalingCheckBox.setChecked(1)
        # else:
        #     self.guiConfig.AutoScalingCheckBox.setChecked(0)
        # update the Gui
        self.__initialParameters()

    def configureDialog(self, oldconfiguration):
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
            - ``StripBackgroundFlag``
            - ``StripWidth``
            - ``StripNIterations``
            - ``StripThresholdFactor``

        :return: User defined parameters in a dictionary"""
        # this method can be overwritten for custom
        # it should give back a new dictionary
        newconfiguration = {}
        newconfiguration.update(oldconfiguration)

        theory = self.fitmanager.selectedtheory
        custom_config_widget = self.fitmanager.theories[theory].config_widget

        if custom_config_widget is not None:
            dialog_widget = custom_config_widget()
            for mandatory_attr in ["show", "exec_", "result", "output"]:
                if not hasattr(dialog_widget, mandatory_attr):
                    raise AttributeError(
                            "Custom configuration widget must define " +
                            "attribute or method " + mandatory_attr)

        else:
            # default config widget, adapted for default fit theories
            sheet1 = {'notetitle': 'Constraints',
                      'fields': (
                          {'name': 'PositiveHeightAreaFlag',
                           'widget type': 'CheckField',
                           'text': 'Force positive Height/Area'},
                          {'name': 'QuotedPositionFlag',
                           'widget type': 'CheckField',
                           'text': 'Force position in interval'},
                          {'name': 'PositiveFwhmFlag',
                           'widget type': 'CheckField',
                           'text': 'Force positive FWHM'},
                          {'name': 'SameFwhmFlag',
                           'widget type': 'CheckField',
                           'text': 'Force same FWHM'},
                          {'name': 'QuotedEtaFlag',
                           'widget type': 'CheckField',
                           'text': 'Force Eta between 0 and 1'},
                          {'name': 'NoConstraintsFlag',
                           'widget type': 'CheckField',
                           'text': 'Ignore constraints'})}

            sheet2 = {'notetitle': 'Search',
                      'fields': (
                          {'widget type': "EntryField",
                           'name': 'FwhmPoints',
                           'text': 'Fwhm Points: ',
                           'data type': "int",
                           'min value': 3,
                           'tooltip': "Number of data points for typical peak fwhm, used by " +
                                      " peak detection algorithm to filter out noise. "
                                      "Lower limit: 3"},
                          {'widget type': "EntryField",
                           'name': 'Sensitivity',
                           'text': 'Sensitivity: ',
                           'data type': "float",
                           'min value': 1.,
                           'tooltip': "Sensitivity parameter for the peak detection algorithm." +
                                      " Peaks must have an amplitude larger than sensitivity " +
                                      " times the standard deviation f the noise. " +
                                      "Lower limit is 1."},
                          {'widget type': "EntryField",
                           'name': 'Yscaling',
                           'data type': "float",
                           'text': 'Y Factor: '},
                          {'widget type': "CheckField",
                           'name': 'ForcePeakPresence',
                           'text': 'Force peak presence',
                           'tooltip': "In case no peak is detected by the peak-search" +
                                      " algorithm, put one peak at the max data location."})}
            sheet3 = {'notetitle': 'Background',
                      'fields': (
                          {'widget type': "CheckField",
                           'name': 'StripBackgroundFlag',
                           'text': 'Subtract strip background for estimation',
                           'tooltip': "Background filter useful when fitting narrow peaks"},
                          {'widget type': "EntryField",
                           'name': 'StripWidth',
                           'text': 'Strip width (samples): ',
                           'data type': "int",
                           'tooltip': "Width of strip operator, in number of samples. A sample will be " +
                                      "compared to the average of the 2 samples at a distance of " +
                                      " + or - width samples."},
                          {'widget type': "EntryField",
                           'name': 'StripNIterations',
                           'text': 'Number of iterations: ',
                           'data type': "int",
                           'tooltip': "Number of iterations for strip algorithm"},
                          {'widget type': "EntryField",
                           'name': 'StripThresholdFactor',
                           'text': 'Strip threshold factor: ',
                           'data type': "float",
                           'tooltip': "If a sample is higher than the average of the two samples " +
                                      "multiplied by this factor, it will be replaced by the average."})}

            dialog_widget = QScriptOption(
                                self, name='Fit Configuration',
                                sheets=(sheet1, sheet2, sheet3),
                                default=oldconfiguration)

        dialog_widget.show()
        dialog_widget.exec_()
        if dialog_widget.result():
            newconfiguration.update(dialog_widget.output)
        # we do not need the dialog any longer
        del dialog_widget


        return newconfiguration

    def estimate(self):
        """Run parameter estimation function then emit
        :attr:`sigFitWidgetSignal` with a dictionary containing a status
        message *'EstimateFinished'* and a list of fit parameters estimations
        in the format defined in
        :attr:`silx.math.fit.fitmanager.FitManager.fit_results`
        """
        try:
            theory_name = self.fitmanager.selectedtheory
            estimation_function = self.fitmanager.theories[theory_name].estimate
            if estimation_function is not None:
                self.fitmanager.estimate(callback=self.fitStatus)
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

        self.guiParameters.fillFromFit(
            self.fitmanager.fit_results, view='Fit')
        self.guiParameters.removeAllViews(keep='Fit')
        ddict = {
            'event': 'EstimateFinished',
            'data': self.fitmanager.fit_results}
        self._emitSignal(ddict)


    def startfit(self):
        warnings.warn("Method renamed to startFit",
                      DeprecationWarning)
        self.startFit()

    def startFit(self):
        """Run fit, then emit :attr:`sigFitWidgetSignal` with a dictionary
        containing a status
        message *'FitFinished'* and a list of fit parameters results
        in the format defined in
        :attr:`silx.math.fit.fitmanager.FitManager.fit_results`
        """
        self.fitmanager.fit_results = self.guiParameters.getFitResults()
        try:
            self.fitmanager.runfit(callback=self.fitStatus)
        except:  # noqa (we want to catch and report all errors)
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error on Fit: %s" % traceback.format_exc())
            msg.exec_()
            return

        self.guiParameters.fillFromFit(
            self.fitmanager.fit_results, view='Fit')
        self.guiParameters.removeAllViews(keep='Fit')
        ddict = {
            'event': 'FitFinished',
            'data': self.fitmanager.fit_results
        }
        self._emitSignal(ddict)
        return

    def autoFwhmEvent(self, item):
        """Set :attr:`fitmanager.fitconfig['AutoFwhm']`"""
        if int(item):
            self.configure(AutoFwhm=True)
        else:
            self.configure(AutoFwhm=False)

    # def autoscaleevent(self, item):
    #     """Set :attr:`fitmanager"fitConfig['AutoScaling']`"""
    #     if int(item):
    #         self.configure(AutoScaling=True)
    #     else:
    #         self.configure(AutoScaling=False)

    def bkgEvent(self, bgtheory):
        """Select background theory, then reinitialize parameters"""
        bgtheory = str(bgtheory)
        if bgtheory in self.fitmanager.bgtheories:
            self.fitmanager.setbackground(bgtheory)
        else:
            qt.QMessageBox.information(
                self, "Info",
                "%s is not a known background theory. Known " % bgtheory +
                "theories are: " + ", ".join(self.fitmanager.bgtheories)
            )
            return
        self.__initialParameters()

    def funEvent(self, theoryname):
        """Select a fit theory to be used for fitting. If this theory exists
        in :attr:`fitmanager`, use it. Then, reinitialize table.

        :param theoryname: Name of the fit theory to use for fitting. If this theory
            exists in :attr:`fitmanager`, use it. Else, open a file dialog to open
            a custom fit function definition file with
            :meth:`fitmanager.loadtheories`.
        """
        theoryname = str(theoryname)
        if theoryname in self.fitmanager.theories:
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
                    while self.guiConfig.FunComBox.count() > 1:
                        self.guiConfig.FunComBox.removeItem(1)
                    # and fill it again
                    for key in self.fitmanager.theories:
                        self.guiConfig.FunComBox.addItem(str(key))

            i = 1 + \
                list(self.fitmanager.theories.keys()).index(
                    self.fitmanager.selectedtheory)
            self.guiConfig.FunComBox.setCurrentIndex(i)
        self.__initialParameters()

    def __initialParameters(self):
        """Fill the fit parameters names with names of the parameters of
        the selected background theory and the selected fit theory.
        Initialize :attr:`fitmanager.fit_results` with these names, and
        initialize the table with them. This creates a view called "Fit"
        in :attr:`guiParameters`"""
        self.fitmanager.parameter_names = []
        self.fitmanager.fit_results = []
        for pname in self.fitmanager.bgtheories[self.fitmanager.selectedbg].parameters:
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
        if self.fitmanager.selectedtheory is not None:
            theory = self.fitmanager.selectedtheory
            for pname in self.fitmanager.theories[theory].parameters:
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

        self.guiParameters.fillFromFit(
            self.fitmanager.fit_results, view='Fit')

    def fitStatus(self, data):
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
         75, 1300., 21.]
    y = functions.sum_gauss(x, *p) + constant_bg

    a = qt.QApplication(sys.argv)
    w = FitWidget()
    w.setData(x=x, y=y)
    w.show()
    a.exec_()
