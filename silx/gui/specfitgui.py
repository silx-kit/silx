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
import sys
import traceback
import logging

import PyMca5
from PyMca5.PyMcaCore import EventHandler
from ..math import specfit
from . import qt

QTVERSION = qt.qVersion()
from PyMca5.PyMcaGui.math.fitting import QScriptOption

__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "03/06/2016"

DEBUG = 0
_logger = logging.getLogger(__name__)


class SpecfitGui(qt.QWidget):
    sigSpecfitGuiSignal = qt.pyqtSignal(object)

    def __init__(self, parent=None, name=None, fl=0,
                 specfit_instance=None,
                 config=0, status=0, buttons=0,
                 event_handler=None):
        if name is None:
            name = "SpecfitGui"
        qt.QWidget.__init__(self, parent)
        self.setWindowTitle(name)
        layout = qt.QVBoxLayout(self)
        # layout.setAutoAdd(1)
        if event_handler is None:
            self.eh = EventHandler.EventHandler()
        else:
            self.eh = event_handler
        if specfit_instance is None:
            self.specfit = specfit.Specfit(event_handler=self.eh)
        else:
            self.specfit = specfit_instance

        # initialize the default fitting functions in case
        #none is present
        # FIXME
        if not len(self.specfit.theorydict):
            self.specfit.importfun(
                PyMca5.PyMcaMath.fitting.SpecfitFunctions.__file__)

        # copy specfit configure method for direct access
        self.configure = self.specfit.configure
        self.fitconfig = self.specfit.fitconfig

        self.setdata = self.specfit.setdata
        self.guiconfig = None
        if config:
            self.guiconfig = FitConfigWidget(self)
            self.guiconfig.MCACheckBox.stateChanged[int].connect(self.mcaevent)
            self.guiconfig.WeightCheckBox.stateChanged[
                int].connect(self.weightevent)
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
            if configuration['McaMode']:
                self.guiconfig.MCACheckBox.setChecked(1)
            else:
                self.guiconfig.MCACheckBox.setChecked(0)
            if configuration['WeightFlag']:
                self.guiconfig.WeightCheckBox.setChecked(1)
            else:
                self.guiconfig.WeightCheckBox.setChecked(0)
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
            self.eh.register('FitStatusChanged', self.fitstatus)
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
            if configuration['McaMode']:
                self.guiconfig.MCACheckBox.setChecked(1)
            else:
                self.guiconfig.MCACheckBox.setChecked(0)
            if configuration['WeightFlag']:
                self.guiconfig.WeightCheckBox.setChecked(1)
            else:
                self.guiconfig.WeightCheckBox.setChecked(0)
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
        # this method can be overwritten for custom
        # it should give back a new dictionary
        newconfiguration = {}
        newconfiguration.update(oldconfiguration)
        if (0):
            # example to force a given default configuration
            newconfiguration['FitTheory'] = "Pseudo-Voigt Line"
            newconfiguration['AutoFwhm'] = 1
            newconfiguration['AutoScaling'] = 1

        # example script options like
        if (1):
            sheet1 = {'notetitle': 'Restrains',
                      'fields': (["CheckField", 'HeightAreaFlag', 'Force positive Height/Area'],
                                 ["CheckField", 'PositionFlag',
                                  'Force position in interval'],
                                 ["CheckField", 'PosFwhmFlag',
                                     'Force positive FWHM'],
                                 ["CheckField", 'SameFwhmFlag', 'Force same FWHM'],
                                 ["CheckField", 'EtaFlag',
                                     'Force Eta between 0 and 1'],
                                 ["CheckField", 'NoConstrainsFlag', 'Ignore Restrains'])}

            sheet2 = {'notetitle': 'Search',
                      'fields': (["EntryField", 'FwhmPoints', 'Fwhm Points: '],
                                 ["EntryField", 'Sensitivity', 'Sensitivity: '],
                                 ["EntryField", 'Yscaling',   'Y Factor   : '],
                                 ["CheckField", 'ForcePeakPresence',   'Force peak presence '])}
            w = QScriptOption.QScriptOption(self, name='Fit Configuration',
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
        if self.specfit.fitconfig['McaMode']:
            try:
                mcaresult = self.specfit.mcafit()
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setWindowTitle("Error on mcafit")
                msg.setInformativeText(str(sys.exc_info()[1]))
                msg.setDetailedText(traceback.format_exc())
                msg.exec_()
                ddict = {}
                ddict['event'] = 'FitError'
                self._emitSignal(ddict)
                if DEBUG:
                    raise
                return
            self.guiparameters.fillfrommca(mcaresult)
            ddict = {}
            ddict['event'] = 'McaFitFinished'
            ddict['data'] = mcaresult
            self._emitSignal(ddict)
            #self.guiparameters.removeallviews(keep='Region 1')
        else:
            try:
                if self.specfit.theorydict[self.specfit.fitconfig['fittheory']][2] is not None:
                    self.specfit.estimate()
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
                msg.setText("Error on estimate: %s" % sys.exc_info()[1])
                msg.exec_()
                return
            self.guiparameters.fillfromfit(
                self.specfit.paramlist, current='Fit')
            self.guiparameters.removeallviews(keep='Fit')
            ddict = {}
            ddict['event'] = 'EstimateFinished'
            ddict['data'] = self.specfit.paramlist
            self._emitSignal(ddict)

        return

    def __forward(self, ddict):
        self._emitSignal(ddict)

    def startfit(self):
        if self.specfit.fitconfig['McaMode']:
            try:
                mcaresult = self.specfit.mcafit()
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error on mcafit: %s" % sys.exc_info()[1])
                msg.exec_()
                if DEBUG:
                    raise
                return
            self.guiparameters.fillfrommca(mcaresult)
            ddict = {}
            ddict['event'] = 'McaFitFinished'
            ddict['data'] = mcaresult
            self._emitSignal(ddict)
            # self.guiparameters.removeview(view='Fit')
        else:
            # for param in self.specfit.paramlist:
            #    print param['name'],param['group'],param['estimation']
            self.specfit.paramlist = self.guiparameters.fillfitfromtable()
            if DEBUG:
                for param in self.specfit.paramlist:
                    print(param['name'], param['group'], param['estimation'])
                print("TESTING")
                self.specfit.startfit()
            try:
                self.specfit.startfit()
            except:
                msg = qt.QMessageBox(self)
                msg.setIcon(qt.QMessageBox.Critical)
                msg.setText("Error on Fit")
                msg.exec_()
                if DEBUG:
                    raise
                return
            self.guiparameters.fillfromfit(
                self.specfit.paramlist, current='Fit')
            self.guiparameters.removeallviews(keep='Fit')
            ddict = {}
            ddict['event'] = 'FitFinished'
            ddict['data'] = self.specfit.paramlist
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

    def mcaevent(self, item):
        if int(item):
            self.configure(McaMode=1)
            mode = 1
        else:
            self.configure(McaMode=0)
            mode = 0
        self.__initialparameters()
        ddict = {}
        ddict['event'] = 'McaModeChanged'
        ddict['data'] = mode
        self._emitSignal(ddict)
        return

    def weightevent(self, item):
        if int(item):
            self.configure(WeightFlag=1)
        else:
            self.configure(WeightFlag=0)
        return

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
        self.specfit.final_theory = []
        self.specfit.paramlist = []
        for pname in self.specfit.bkgdict[self.specfit.fitconfig['fitbkg']][1]:
            self.specfit.final_theory.append(pname)
            self.specfit.paramlist.append({'name': pname,
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
                self.specfit.final_theory.append(pname + "1")
                self.specfit.paramlist.append({'name': pname + "1",
                                               'estimation': 0,
                                               'group': 1,
                                               'code': 'FREE',
                                               'cons1': 0,
                                               'cons2': 0,
                                               'fitresult': 0.0,
                                               'sigma': 0.0,
                                               'xmin': None,
                                               'xmax': None})
        if self.specfit.fitconfig['McaMode']:
            self.guiparameters.fillfromfit(
                self.specfit.paramlist, current='Region 1')
            self.guiparameters.removeallviews(keep='Region 1')
        else:
            self.guiparameters.fillfromfit(
                self.specfit.paramlist, current='Fit')
            self.guiparameters.removeallviews(keep='Fit')
        return

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


class FitActionsButtons(qt.QWidget):
    """Widget with 3 ``QPushButton``:

    The buttons can be accessed as public attributes::

        - ``EstimateButton``
        - ``StartfitButton``
        - ``DismissButton``

    You will typically need to access these attributes to connect the buttons
    to actions. For instance, if you have 3 functions ``estimate``,
    ``startfit`` and  ``dismiss``, you can connect them like this::

        >>> fit_actions_buttons = FitActionsButtons()
        >>> fit_actions_buttons.EstimateButton.clicked.connect(estimate)
        >>> fit_actions_buttons.StartfitButton.clicked.connect(startfit)
        >>> fit_actions_buttons.DismissButton.clicked.connect(dismiss)

    """

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        self.resize(234, 53)

        FitActionsGUILayout = qt.QGridLayout(self)
        FitActionsGUILayout.setContentsMargins(11, 11, 11, 11)
        FitActionsGUILayout.setSpacing(6)
        layout = qt.QHBoxLayout(None)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.EstimateButton = qt.QPushButton(self)
        self.EstimateButton.setText("Estimate")
        layout.addWidget(self.EstimateButton)
        spacer = qt.QSpacerItem(20, 20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        layout.addItem(spacer)

        self.StartfitButton = qt.QPushButton(self)
        self.StartfitButton.setText("Start Fit")
        layout.addWidget(self.StartfitButton)
        spacer_2 = qt.QSpacerItem(20, 20,
                                  qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum)
        layout.addItem(spacer_2)

        self.DismissButton = qt.QPushButton(self)
        self.DismissButton.setText("Dismiss")
        layout.addWidget(self.DismissButton)

        FitActionsGUILayout.addLayout(layout, 0, 0)


class FitStatusLines(qt.QWidget):
    """Widget with 2 greyed out write-only ``QLineEdit``.

    These text widgets can be accessed as public attributes::

        - ``StatusLine``
        - ``ChisqLine``

    You will typically need to access these widgets to update the displayed
    text::

        >>> fit_status_lines = FitStatusLines()
        >>> fit_status_lines.StatusLine.setText("Ready")
        >>> fit_status_lines.ChisqLine.setText("%6.2f" % 0.01)

    """

    def __init__(self, parent=None):
        qt.QWidget.__init__(self,  parent)

        self.resize(535, 47)

        layout = qt.QHBoxLayout(self)
        layout.setContentsMargins(11, 11, 11, 11)
        layout.setSpacing(6)

        self.StatusLabel = qt.QLabel(self)
        self.StatusLabel.setText("Status:")
        layout.addWidget(self.StatusLabel)

        self.StatusLine = qt.QLineEdit(self)
        self.StatusLine.setText("Ready")
        self.StatusLine.setReadOnly(1)
        layout.addWidget(self.StatusLine)

        self.ChisqLabel = qt.QLabel(self)
        self.ChisqLabel.setText("Chisq:")
        layout.addWidget(self.ChisqLabel)

        self.ChisqLine = qt.QLineEdit(self)
        self.ChisqLine.setMaximumSize(qt.QSize(16000, 32767))
        self.ChisqLine.setText("")
        self.ChisqLine.setReadOnly(1)
        layout.addWidget(self.ChisqLine)


class FitConfigWidget(qt.QWidget):
    """Widget with 2 ``QComboBox``, 4 ``QCheckBox`` and 2 ``QPushButtons``.

    These text widgets can be accessed as public attributes::

        - ``BkgComBox``
        - ``FunComBox``
        - ``WeightCheckBox``
        - ``MCACheckBox``
        - ``AutoFWHMCheckBox``
        - ``AutoScalingCheckBox``
        - ``PrintPushButton``
        - ``ConfigureButton``
    """

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)

        self.setWindowTitle("FitConfigGUI")

        FitConfigGUILayout = qt.QHBoxLayout(self)
        FitConfigGUILayout.setContentsMargins(11, 11, 11, 11)
        FitConfigGUILayout.setSpacing(6)

        layout9 = qt.QHBoxLayout(None)
        layout9.setContentsMargins(0, 0, 0, 0)
        layout9.setSpacing(6)

        layout2 = qt.QGridLayout(None)
        layout2.setContentsMargins(0, 0, 0, 0)
        layout2.setSpacing(6)

        self.BkgComBox = qt.QComboBox(self)
        self.BkgComBox.addItem("Add Background")

        layout2.addWidget(self.BkgComBox, 1, 1)

        self.BkgLabel = qt.QLabel(self)
        self.BkgLabel.setText("Background")

        layout2.addWidget(self.BkgLabel, 1, 0)

        self.FunComBox = qt.QComboBox(self)
        self.FunComBox.addItem("Add Function(s)")

        layout2.addWidget(self.FunComBox, 0, 1)

        self.FunLabel = qt.QLabel(self)
        self.FunLabel.setText("Function")

        layout2.addWidget(self.FunLabel, 0, 0)
        layout9.addLayout(layout2)
        spacer = qt.QSpacerItem(20, 20,
                                qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        layout9.addItem(spacer)

        layout6 = qt.QGridLayout(None)
        layout6.setContentsMargins(0, 0, 0, 0)
        layout6.setSpacing(6)

        self.WeightCheckBox = qt.QCheckBox(self)
        self.WeightCheckBox.setText("Weight")

        layout6.addWidget(self.WeightCheckBox, 0, 0)

        self.MCACheckBox = qt.QCheckBox(self)
        self.MCACheckBox.setText("MCA Mode")

        layout6.addWidget(self.MCACheckBox, 1, 0)
        layout9.addLayout(layout6)

        layout6_2 = qt.QGridLayout(None)
        layout6_2.setContentsMargins(0, 0, 0, 0)
        layout6_2.setSpacing(6)

        self.AutoFWHMCheckBox = qt.QCheckBox(self)
        self.AutoFWHMCheckBox.setText("Auto FWHM")

        layout6_2.addWidget(self.AutoFWHMCheckBox, 0, 0)

        self.AutoScalingCheckBox = qt.QCheckBox(self)
        self.AutoScalingCheckBox.setText("Auto Scaling")

        layout6_2.addWidget(self.AutoScalingCheckBox, 1, 0)
        layout9.addLayout(layout6_2)
        spacer_2 = qt.QSpacerItem(20, 20, qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum)
        layout9.addItem(spacer_2)

        layout5 = qt.QGridLayout(None)
        layout5.setContentsMargins(0, 0, 0, 0)
        layout5.setSpacing(6)

        self.PrintPushButton = qt.QPushButton(self)
        self.PrintPushButton.setText("Print")

        layout5.addWidget(self.PrintPushButton, 1, 0)

        self.ConfigureButton = qt.QPushButton(self)
        self.ConfigureButton.setText("Configure")

        layout5.addWidget(self.ConfigureButton, 0, 0)
        layout9.addLayout(layout5)
        FitConfigGUILayout.addLayout(layout9)


class McaTable(qt.QTableWidget):
    sigMcaTableSignal = qt.pyqtSignal(object)

    def __init__(self, labels=None, *args, **kw):
        qt.QTableWidget.__init__(self, *args)
        self.setRowCount(1)
        self.setColumnCount(1)

        self.code_options = ["FREE", "POSITIVE", "QUOTED",
                             "FIXED", "FACTOR", "DELTA", "SUM", "IGNORE", "ADD", "SHOW"]

        if labels is not None:
            self.labels = labels
        else:
            self.labels = ['Position', 'Fit Area', 'MCA Area', 'Sigma',
                           'Fwhm', 'Chisq', 'Region', 'XBegin', 'XEnd']

        self.setColumnCount(len(self.labels))
        for i, label in enumerate(self.labels):
            item = self.horizontalHeaderItem(i)
            if item is None:
                item = qt.QTableWidgetItem(label,
                                           qt.QTableWidgetItem.Type)
                self.setHorizontalHeaderItem(i, item)
            item.setText(label)
            self.resizeColumnToContents(i)

        self.regionlist = []
        self.regiondict = {}

        self.cellClicked[int, int].connect(self.__myslot)
        self.itemSelectionChanged[()].connect(self.__myslot)

    def fillfrommca(self, mcaresult, diag=1):
        line0 = 0
        region = 0
        alreadyforced = 0
        for result in mcaresult:
            region = region + 1
            if result['chisq'] is not None:
                chisq = "%6.2f" % (result['chisq'])
            else:
                chisq = "Fit Error"
            if 1:
                xbegin = "%6g" % (result['xbegin'])
                xend = "%6g" % (result['xend'])
                fitlabel, fitpars, fitsigmas = self.__getfitpar(result)
                if QTVERSION < '4.0.0':
                    qt.QHeader.setLabel(
                        self.horizontalHeader(), 1, "Fit " + fitlabel)
                else:
                    item = self.horizontalHeaderItem(1)
                    item.setText("Fit " + fitlabel)
                i = 0
                for (pos, area, sigma, fwhm) in result['mca_areas']:
                    line0 = line0 + 1
                    if QTVERSION < '4.0.0':
                        nlines = self.numRows()
                        if (line0 > nlines):
                            self.setNumRows(line0)
                    else:
                        nlines = self.rowCount()
                        if (line0 > nlines):
                            self.setRowCount(line0)
                    line = line0 - 1
                    pos = "%6g" % (pos)
                    fitpar = "%6g" % (fitpars[i])
                    if fitlabel == 'Area':
                        sigma = max(sigma, fitsigmas[i])
                    areastr = "%6g" % (area)
                    sigmastr = "%6.3g" % (sigma)
                    fwhm = "%6g" % (fwhm)
                    tregion = "%6g" % (region)
                    fields = [pos, fitpar, areastr, sigmastr,
                              fwhm, chisq, tregion, xbegin, xend]
                    col = 0
                    recolor = 0
                    if fitlabel == 'Area':
                        if diag:
                            if abs(fitpars[i] - area) > (3.0 * sigma):
                                color = qt.QColor(255, 182, 193)
                                recolor = 1
                    for field in fields:
                        key = self.item(line, col)
                        if key is None:
                            key = qt.QTableWidgetItem(field)
                            self.setItem(line, col, key)
                        else:
                            item.setText(field)
                        if recolor:
                            # function introduced in Qt 4.2.0
                            if QTVERSION >= '4.2.0':
                                item.setBackground(qt.QBrush(color))
                        item.setFlags(qt.Qt.ItemIsSelectable |
                                      qt.Qt.ItemIsEnabled)
                        col = col + 1
                    if recolor:
                        if not alreadyforced:
                            alreadyforced = 1
                            self.scrollToItem(self.item(line, 0))
                    i += 1

        for i in range(len(self.labels)):
            self.resizeColumnToContents(i)
        ndict = {}
        ndict['event'] = 'McaTableFilled'
        self.sigMcaTableSignal.emit(ndict)

    def __getfitpar(self, result):
        if result['fitconfig']['fittheory'].find("Area") != -1:
            fitlabel = 'Area'
        elif result['fitconfig']['fittheory'].find("Hypermet") != -1:
            fitlabel = 'Area'
        else:
            fitlabel = 'Height'
        values = []
        sigmavalues = []
        for param in result['paramlist']:
            if param['name'].find('ST_Area') != -1:
                # value and sigmavalue known via fitlabel
                values[-1] = value * (1.0 + param['fitresult'])
                # just an approximation
                sigmavalues[-1] = sigmavalue * (1.0 + param['fitresult'])
            elif param['name'].find('LT_Area') != -1:
                pass
            elif param['name'].find(fitlabel) != -1:
                value = param['fitresult']
                sigmavalue = param['sigma']
                values.append(value)
                sigmavalues.append(sigmavalue)
        return fitlabel, values, sigmavalues

    def __myslot(self, *var):
        ddict = {}
        if len(var) == 0:
            # selection changed event
            # get the current selection
            ddict['event'] = 'McaTableClicked'
            row = self.currentRow()
        else:
            # Header click
            ddict['event'] = 'McaTableRowHeaderClicked'
            row = var[0]
        ccol = self.currentColumn()
        ddict['row'] = row
        ddict['col'] = ccol
        ddict['labelslist'] = self.labels
        if row >= 0:
            col = 0
            for label in self.labels:
                text = str(self.item(row, col).text())
                try:
                    ddict[label] = float(text)
                except:
                    ddict[label] = text
                col += 1
        self.sigMcaTableSignal.emit(ddict)

# FIXME
from PyMca5.PyMcaGui.math.fitting.Parameters import Parameters


class ParametersTab(qt.QTabWidget):
    sigMultiParametersSignal = qt.pyqtSignal(object)

    def __init__(self, parent=None, name="FitParameters"):
        qt.QTabWidget.__init__(self, parent)
        self.setWindowTitle(name)

        # the widgets in the notebook
        self.views = {}
        # the names of the widgets (to have them in order)
        self.tabs = []
        # the widgets/tables themselves
        self.tables = {}
        self.mcatable = None
        self.setContentsMargins(10, 10, 10, 10)
        self.setview(name="Region 1")

    def setview(self, name=None, fitparameterslist=None):
        if name is None:
            name = self.current

        if name in self.tables.keys():
            table = self.tables[name]
        else:
            # create the parameters instance
            self.tables[name] = Parameters(self)
            table = self.tables[name]
            self.tabs.append(name)
            self.views[name] = table
            self.addTab(table, str(name))

        if fitparameterslist is not None:
            table.fillfromfit(fitparameterslist)

        if QTVERSION < '4.0.0':
            self.showPage(self.views[name])
        else:
            self.setCurrentWidget(self.views[name])

        self.current = name

    def renameview(self, oldname=None, newname=None):
        error = 1
        if newname is not None:
            if newname not in self.views.keys():
                if oldname in self.views.keys():
                    parameterlist = self.tables[oldname].fillfitfromtable()
                    self.setview(name=newname,
                                 fitparameterslist=parameterlist)
                    self.removeview(oldname)
                    error = 0
        return error

    def fillfromfit(self, fitparameterslist, current=None):
        if current is None:
            current = self.current

        self.setview(fitparameterslist=fitparameterslist,
                     name=current)

    def fillfitfromtable(self, name=None):  # FIXME: name or view? can't find usage
        if name is None:
            name = self.current

        if hasattr(self.tables[name], 'fillfitfromtable'):
            return self.tables[name].fillfitfromtable()
        else:
            return None

    def removeview(self, view=None):
        error = 1
        if view is None:
            return error

        if view == self.current:
            return error

        if view in self.views.keys():
            self.tabs.remove(view)
            if QTVERSION < '4.0.0':
                self.removePage(self.tables[view])
                self.removePage(self.views[view])
            else:
                index = self.indexOf(self.tables[view])
                self.removeTab(index)
                index = self.indexOf(self.views[view])
                self.removeTab(index)
            del self.tables[view]
            del self.views[view]
            error = 0
        return error

    def removeallviews(self, keep='Fit'):
        for view in list(self.tables.keys()):
            if view != keep:
                self.removeview(view)

    def fillfrommca(self, mcaresult):
        self.removeallviews()
        region = 0

        for result in mcaresult:
            region = region + 1
            self.fillfromfit(result['paramlist'],
                             current='Region %d' % region)
        name = 'MCA'
        if name in self.tables:
            table = self.tables[name]
        else:
            self.tables[name] = McaTable(self)
            table = self.tables[name]
            self.tabs.append(name)
            self.views[name] = table
            self.addTab(table, str(name))
            table.sigMcaTableSignal.connect(self.__forward)
        table.fillfrommca(mcaresult)
        self.setview(name=name)
        return

    def __forward(self, ddict):
        self.sigMultiParametersSignal.emit(ddict)

    def gettext(self, name=None):
        if name is None:
            name = self.current
        table = self.tables[name]
        lemon = ("#%x%x%x" % (255, 250, 205)).upper()
        if QTVERSION < '4.0.0':
            hb = table.horizontalHeader().paletteBackgroundColor()
            hcolor = ("#%x%x%x" % (hb.red(), hb.green(), hb.blue())).upper()
        else:
            if DEBUG:
                print("Actual color to ge got")
            hcolor = ("#%x%x%x" % (230, 240, 249)).upper()
        text = ""
        text += ("<nobr>")
        text += ("<table>")
        text += ("<tr>")
        if QTVERSION < '4.0.0':
            ncols = table.numCols()
        else:
            ncols = table.columnCount()
        for l in range(ncols):
            text += ('<td align="left" bgcolor="%s"><b>' % hcolor)
            if QTVERSION < '4.0.0':
                text += (str(table.horizontalHeader().label(l)))
            else:
                text += (str(table.horizontalHeaderItem(l).text()))
            text += ("</b></td>")
        text += ("</tr>")
        if QTVERSION < '4.0.0':
            nrows = table.numRows()
        else:
            nrows = table.rowCount()
        for r in range(nrows):
            text += ("<tr>")
            if QTVERSION < '4.0.0':
                newtext = str(table.text(r, 0))
            else:
                item = table.item(r, 0)
                newtext = ""
                if item is not None:
                    newtext = str(item.text())
            if len(newtext):
                color = "white"
                b = "<b>"
            else:
                b = ""
                color = lemon
            try:
                # MyQTable item has color defined
                cc = table.item(r, 0).color
                cc = ("#%x%x%x" % (cc.red(), cc.green(), cc.blue())).upper()
                color = cc
            except:
                pass
            for c in range(ncols):
                if QTVERSION < '4.0.0':
                    newtext = str(table.text(r, c))
                else:
                    item = table.item(r, c)
                    newtext = ""
                    if item is not None:
                        newtext = str(item.text())
                if len(newtext):
                    finalcolor = color
                else:
                    finalcolor = "white"
                if c < 2:
                    text += ('<td align="left" bgcolor="%s">%s' %
                             (finalcolor, b))
                else:
                    text += ('<td align="right" bgcolor="%s">%s' %
                             (finalcolor, b))
                text += (newtext)
                if len(b):
                    text += ("</td>")
                else:
                    text += ("</b></td>")
            if QTVERSION < '4.0.0':
                newtext = str(table.text(r, 0))
            else:
                item = table.item(r, 0)
                newtext = ""
                if item is not None:
                    newtext = str(item.text())
            if len(newtext):
                text += ("</b>")
            text += ("</tr>")
            text += ("\n")
        text += ("</table>")
        text += ("</nobr>")
        return text

    def getHTMLText(self, name=None):
        return self.gettext(name)

    if QTVERSION > '4.0.0':
        def getText(self, name=None):
            if name is None:
                name = self.current
            table = self.tables[name]
            text = ""
            ncols = table.columnCount()
            for l in range(ncols):
                text += (str(table.horizontalHeaderItem(l).text())) + "\t"
            text += ("\n")
            nrows = table.rowCount()
            for r in range(nrows):
                for c in range(ncols):
                    newtext = ""
                    if c != 4:
                        item = table.item(r, c)
                        if item is not None:
                            newtext = str(item.text())
                    else:
                        item = table.cellWidget(r, c)
                        if item is not None:
                            newtext = str(item.currentText())
                    text += (newtext) + "\t"
                text += ("\n")
            text += ("\n")
            return text


if __name__ == "__main__":
    import numpy
    from PyMca5 import SpecfitFunctions
    a = SpecfitFunctions.SpecfitFunctions()
    x = numpy.arange(2000).astype(numpy.float)
    p1 = numpy.array([1500, 100., 30.0])
    p2 = numpy.array([1500, 300., 30.0])
    p3 = numpy.array([1500, 500., 30.0])
    p4 = numpy.array([1500, 700., 30.0])
    p5 = numpy.array([1500, 900., 30.0])
    p6 = numpy.array([1500, 1100., 30.0])
    p7 = numpy.array([1500, 1300., 30.0])
    p8 = numpy.array([1500, 1500., 30.0])
    p9 = numpy.array([1500, 1700., 30.0])
    p10 = numpy.array([1500, 1900., 30.0])
    y = a.gauss(p1, x) + 1
    y = y + a.gauss(p2, x)
    y = y + a.gauss(p3, x)
    y = y + a.gauss(p4, x)
    y = y + a.gauss(p5, x)
    #y = y + a.gauss(p6,x)
    #y = y + a.gauss(p7,x)
    #y = y + a.gauss(p8,x)
    #y = y + a.gauss(p9,x)
    #y = y + a.gauss(p10,x)
    y = y / 1000.0
    a = qt.QApplication(sys.argv)
    a.lastWindowClosed.connect(a.quit)
    w = SpecfitGui(config=1, status=1, buttons=1)
    w.setdata(x=x, y=y)
    w.show()
    a.exec_()
