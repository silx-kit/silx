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
import logging
import sys
import traceback

from silx.math.fit import fitestimatefunctions
from silx.math.fit import specfit, functions
from silx.gui import qt
from .specfitwidgets import (FitActionsButtons, FitStatusLines,
                             FitConfigWidget, ParametersTab)

QTVERSION = qt.qVersion()
from .qscriptoption import QScriptOption

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "03/06/2016"

DEBUG = 0
_logger = logging.getLogger(__name__)


class SpecfitGui(qt.QWidget):
    sigSpecfitGuiSignal = qt.pyqtSignal(object)

    def __init__(self, parent=None, name=None, fl=0,
                 specfit_instance=None,
                 config=0, status=0, buttons=0):
        if name is None:
            name = "SpecfitGui"
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle(name)
        layout = qt.QVBoxLayout(self)

        if specfit_instance is None:
            self.specfit = specfit.Specfit()
        else:
            self.specfit = specfit_instance

        # initialize the default fitting functions in case
        # none is present
        if not len(self.specfit.theorydict):
            self.specfit.importfun(fitestimatefunctions.__file__)

        # copy specfit.configure method for direct access
        self.configure = self.specfit.configure
        self.fitconfig = self.specfit.fitconfig

        self.setdata = self.specfit.setdata
        self.guiconfig = None
        if config:
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
            self.guiconfig.PrintPushButton.clicked.connect(self.printps)
            self.guiconfig.BkgComBox.activated[str].connect(self.bkgevent)
            self.guiconfig.FunComBox.activated[str].connect(self.funevent)
            layout.addWidget(self.guiconfig)

        self.guiparameters = ParametersTab(self)
        layout.addWidget(self.guiparameters)
        self.guiparameters.sigMultiParametersSignal.connect(self.__forward)
        if config:
            for key in self.specfit.bkgdict.keys():
                self.guiconfig.BkgComBox.addItem(str(key))
            for key in self.specfit.theorydict:
                self.guiconfig.FunComBox.addItem(str(key))
            configuration = {}
            if specfit_instance is not None:
                configuration = specfit_instance.configure()
                if configuration['fittheory'] is None:
                    self.guiconfig.FunComBox.setCurrentIndex(1)
                    self.funevent(self.specfit.theorydict.keys[0])
                else:
                    self.funevent(configuration['fittheory'])
                if configuration['fitbkg'] is None:
                    self.guiconfig.BkgComBox.setCurrentIndex(1)
                    self.bkgevent(list(self.specfit.bkgdict.keys())[0])
                else:
                    self.bkgevent(configuration['fitbkg'])
            else:
                self.guiconfig.BkgComBox.setCurrentIndex(1)
                self.guiconfig.FunComBox.setCurrentIndex(1)
                self.funevent(list(self.specfit.theorydict.keys())[0])
                self.bkgevent(list(self.specfit.bkgdict.keys())[0])
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

        if status:
            self.guistatus = FitStatusLines(self)
            layout.addWidget(self.guistatus)
        if buttons:
            self.guibuttons = FitActionsButtons(self)
            self.guibuttons.EstimateButton.clicked.connect(self.estimate)
            self.guibuttons.StartfitButton.clicked.connect(self.startfit)
            self.guibuttons.DismissButton.clicked.connect(self.dismiss)
            layout.addWidget(self.guibuttons)

    def updateGui(self, configuration=None):
        self.__configureGui(configuration)

    def _emitSignal(self, ddict):
        self.sigSpecfitGuiSignal.emit(ddict)

    def __configureGuiSlot(self):
        self.__configureGui()

    def __configureGui(self, newconfiguration=None):
        if self.guiconfig is not None:
            # get current dictionary
            # print "before ",self.specfit.fitconfig['fitbkg']
            configuration = self.configure()
            # get new dictionary
            if newconfiguration is None:
                newconfiguration = self.configureGui(configuration)
            # update configuration
            configuration.update(self.configure(**newconfiguration))
            # print "after =",self.specfit.fitconfig['fitbkg']
            # update Gui
            # current function
            # self.funevent(list(self.specfit.theorydict.keys())[0])
            try:
                i = 1 + \
                    list(self.specfit.theorydict.keys()).index(
                        self.specfit.fitconfig['fittheory'])
                self.guiconfig.FunComBox.setCurrentIndex(i)
                self.funevent(self.specfit.fitconfig['fittheory'])
            except:
                print("Function not in list %s" %
                      self.specfit.fitconfig['fittheory'])
                self.funevent(list(self.specfit.theorydict.keys())[0])
            # current background
            try:
                # the list conversion is needed in python 3.
                i = 1 + list(self.specfit.bkgdict.keys()
                             ).index(self.specfit.fitconfig['fitbkg'])
                self.guiconfig.BkgComBox.setCurrentIndex(i)
            except:
                print("Background not in list %s" %
                      self.specfit.fitconfig['fitbkg'])
                self.bkgevent(list(self.specfit.bkgdict.keys())[0])
            # and all the rest
            # if configuration['McaMode']:
            #     self.guiconfig.MCACheckBox.setChecked(1)
            # else:
            # #     self.guiconfig.MCACheckBox.setChecked(0)
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
        """Display a :class:`silx.gui.fit.qscriptoption.QScriptOption`
        dialog, allowing the user to define fit configuration parameters:

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
        :attr:`sigSpecfitGuiSignal` with a dictionary containing a status
        message *'EstimateFinished'* and a list of fit parameters estimations
        in the format defined in
        :attr:`silx.math.fit.specfit.Specfit.fit_results`
        """
        try:
            theory_name = self.specfit.fitconfig['fittheory']
            estimation_function = self.specfit.theorydict[theory_name][2]
            if estimation_function is not None:
                self.specfit.estimate(callback=self.fitstatus)
            else:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Information)
                text = "Function does not define a way to estimate\n"
                text += "the initial parameters. Please, fill them\n"
                text += "yourself in the table and press Start Fit\n"
                msg.setText(text)
                msg.setWindowTitle('SpecfitGui Message')
                msg.exec_()
                return
        except:
            if DEBUG:
                raise
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error on estimate: %s" % traceback.format_exc())
            msg.exec_()
            return
        self.guiparameters.fillfromfit(
            self.specfit.fit_results, current='Fit')
        self.guiparameters.removeallviews(keep='Fit')
        ddict = {}
        ddict['event'] = 'EstimateFinished'
        ddict['data'] = self.specfit.fit_results
        self._emitSignal(ddict)

    def __forward(self, ddict):
        self._emitSignal(ddict)

    def startfit(self):
        """Run fit, then emit :attr:`sigSpecfitGuiSignal` with a dictionary
        containing a status
        message *'FitFinished'* and a list of fit parameters results
        in the format defined in
        :attr:`silx.math.fit.specfit.Specfit.fit_results`
        """
        self.specfit.fit_results = self.guiparameters.fillfitfromtable()
        try:
            self.specfit.startfit(callback=self.fitstatus)
        except:
            msg = qt.QMessageBox(self)
            msg.setIcon(qt.QMessageBox.Critical)
            msg.setText("Error on Fit")
            msg.exec_()
            if DEBUG:
                raise
            return
        self.guiparameters.fillfromfit(
            self.specfit.fit_results, current='Fit')
        self.guiparameters.removeallviews(keep='Fit')
        ddict = {}
        ddict['event'] = 'FitFinished'
        ddict['data'] = self.specfit.fit_results
        self._emitSignal(ddict)
        return

    def printps(self, **kw):
        text = self.guiparameters.gettext(**kw)
        if __name__ == "__main__":
            self.__printps(text)
        else:
            ddict = {}
            ddict['event'] = 'print'
            ddict['text'] = text
            self._emitSignal(ddict)
        return

    def __printps(self, text):
        msg = qt.QMessageBox(self)
        msg.setIcon(qt.QMessageBox.Critical)
        msg.setText("Sorry, Qt4 printing not implemented yet")
        msg.exec_()

    def autofwhmevent(self, item):
        if int(item):
            self.configure(AutoFwhm=1)
        else:
            self.configure(AutoFwhm=0)
        return

    def autoscaleevent(self, item):
        if int(item):
            self.configure(AutoScaling=1)
        else:
            self.configure(AutoScaling=0)
        return

    def bkgevent(self, item):
        item = str(item)
        if item in self.specfit.bkgdict.keys():
            self.specfit.setbackground(item)
        else:
            qt.QMessageBox.information(
                self, "Info", "Function not implemented")
            return
            i = 1 + \
                self.specfit.bkgdict.keys().index(
                    self.specfit.fitconfig['fitbkg'])
            self.guiconfig.BkgComBox.setCurrentIndex(i)
        self.__initialparameters()
        return

    def funevent(self, item):
        item = str(item)
        if item in self.specfit.theorydict:
            self.specfit.settheory(item)
        else:
            functionsfile = qt.QFileDialog.getOpenFileName(
                self, "Select python module with your function(s)", "",
                "Python Files (*.py);;All Files (*)")

            if len(functionsfile):
                try:
                    if self.specfit.importfun(functionsfile):
                        qt.QMessageBox.critical(self, "ERROR",
                                                "Function not imported")
                        return
                    else:
                        # empty the ComboBox
                        n = self.guiconfig.FunComBox.count()
                        while(self.guiconfig.FunComBox.count() > 1):
                            self.guiconfig.FunComBox.removeItem(1)
                        # and fill it again
                        for key in self.specfit.theorydict:
                            if QTVERSION < '4.0.0':
                                self.guiconfig.FunComBox.insertItem(str(key))
                            else:
                                self.guiconfig.FunComBox.addItem(str(key))
                except:
                    qt.QMessageBox.critical(self, "ERROR",
                                            "Function not imported")
            i = 1 + \
                list(self.specfit.theorydict.keys()).index(
                    self.specfit.fitconfig['fittheory'])
            if QTVERSION < '4.0.0':
                self.guiconfig.FunComBox.setCurrentItem(i)
            else:
                self.guiconfig.FunComBox.setCurrentIndex(i)
        self.__initialparameters()
        return

    def __initialparameters(self):
        self.specfit.parameter_names = []
        self.specfit.fit_results = []
        for pname in self.specfit.bkgdict[self.specfit.fitconfig['fitbkg']][1]:
            self.specfit.parameter_names.append(pname)
            self.specfit.fit_results.append({'name': pname,
                                           'estimation': 0,
                                           'group': 0,
                                           'code': 'FREE',
                                           'cons1': 0,
                                           'cons2': 0,
                                           'fitresult': 0.0,
                                           'sigma': 0.0,
                                           'xmin': None,
                                           'xmax': None})
        if self.specfit.fitconfig['fittheory'] is not None:
            for pname in self.specfit.theorydict[self.specfit.fitconfig['fittheory']][1]:
                self.specfit.parameter_names.append(pname + "1")
                self.specfit.fit_results.append({'name': pname + "1",
                                               'estimation': 0,
                                               'group': 1,
                                               'code': 'FREE',
                                               'cons1': 0,
                                               'cons2': 0,
                                               'fitresult': 0.0,
                                               'sigma': 0.0,
                                               'xmin': None,
                                               'xmax': None})
        # if self.specfit.fitconfig['McaMode']:
        #     self.guiparameters.fillfromfit(
        #         self.specfit.fit_results, current='Region 1')
        #     self.guiparameters.removeallviews(keep='Region 1')
        # else:
        self.guiparameters.fillfromfit(
            self.specfit.fit_results, current='Fit')
        self.guiparameters.removeallviews(keep='Fit')

    def fitstatus(self, data):
        if 'chisq' in data:
            if data['chisq'] is None:
                self.guistatus.ChisqLine.setText(" ")
            else:
                chisq = data['chisq']
                self.guistatus.ChisqLine.setText("%6.2f" % chisq)

        if 'status' in data:
            status = data['status']
            self.guistatus.StatusLine.setText(str(status))
        return

    def dismiss(self):
        self.close()
        return


if __name__ == "__main__":
    import numpy

    x = numpy.arange(2000).astype(numpy.float)

    p = numpy.array([1500, 100., 30.0,
                     1500, 300., 30.0,
                     1500, 500., 30.0,
                     1500, 700., 30.0,
                     1500, 900., 30.0,
                     1500, 1100., 30.0,
                     1500, 1300., 30.0,
                     1500, 1500., 30.0,
                     1500, 1700., 30.0,
                     1500, 1900., 30.0])
    y = functions.sum_gauss(x, *p) + 1

    y /= 1000.0

    a = qt.QApplication(sys.argv)
    a.lastWindowClosed.connect(a.quit)
    w = SpecfitGui(config=1, status=1, buttons=1)
    w.setdata(x=x, y=y)
    w.show()
    a.exec_()
