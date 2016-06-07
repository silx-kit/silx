#/*##########################################################################
# Copyright (C) 2004-2014 V.A. Sole, European Synchrotron Radiation Facility
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
__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "06/06/2016"

import sys
from silx.gui import qt
from collections import OrderedDict


class QComboTableItem(qt.QComboBox):
    """:class:`qt.QComboBox` augmented with a ``sigCellChanged`` signal
    to emit a tuple of ``(row, column)`` coordinates when the value is
    changed.

    This signal can be used to locate the modified combo box in a table.

    :param row: Row number of the table cell containing this widget
    :param col: Column number of the table cell containing this widget"""
    sigCellChanged = qt.pyqtSignal(int, int)
    """Signal emitted when this ``QComboBox`` is activated.
    A ``(row, column)`` tuple is passed."""

    def __init__(self, parent=None, row=None, col=None):
        self._row = row
        self._col = col
        qt.QComboBox.__init__(self, parent)
        self.activated[int].connect(self._cellChanged)

    def _cellChanged(self, idx):
        self.sigCellChanged.emit(self._row, self._col)


class QCheckBoxItem(qt.QCheckBox):
    """:class:`qt.QCheckBox` augmented with a ``sigCellChanged`` signal
    to emit a tuple of ``(row, column)`` coordinates when the check box has
    been clicked on.

    This signal can be used to locate the modified check box in a table.

    :param row: Row number of the table cell containing this widget
    :param col: Column number of the table cell containing this widget"""
    sigCellChanged = qt.pyqtSignal(int, int)
    """Signal emitted when this ``QCheckBox`` is clicked.
    A ``(row, column)`` tuple is passed."""

    def __init__(self, parent=None, row=None, col=None):
        self._row = row
        self._col = col
        qt.QCheckBox.__init__(self, parent)
        self.clicked.connect(self._cellChanged)

    def _cellChanged(self):
        self.sigCellChanged.emit(self._row, self._col)


class Parameters(qt.QTableWidget):
    """:class:`qt.QTableWidget` customized to display multi-peak fit results.

    :param parent: Parent widget
    :param labels: Column headers. If ``None``, default headers will be used.
    :type labels: List of strings or None
    :param paramlist: List of fit parameters to be displayed for each fitted
        peak.
    :type paramlist: list[str] or None
    :param allowBackgroundAdd: Enable or disable (default behavior) selecting
        "ADD" in the combobox located in the constraints column.
    :type allowBackgroundAdd: boolean
    """
    def __init__(self, parent=None, allowBackgroundAdd=False, labels=None,
                 paramlist=None):
        qt.QTableWidget.__init__(self, parent)
        self._allowBackgroundAdd = allowBackgroundAdd
        self.setRowCount(1)
        self.setColumnCount(1)
        # Default column headers
        if labels is None:
            labels = ['Parameter', 'Estimation', 'Fit Value', 'Sigma',
                      'Constraints', 'Min/Parame', 'Max/Factor/Delta/']
        
        self.code_options = ["FREE", "POSITIVE", "QUOTED", "FIXED",
                             "FACTOR", "DELTA", "SUM", "IGNORE", "ADD"]
        """Possible values in the combo boxes in the 'Constraints' column.

        The meaning of these values is::

            - FREE: no constraint is applied to the fit parameter
            - POSITIVE: the fit parameters must be positive
            - QUOTED: the fit parameter must be greater than the value in the
                'Min' column and lower than the value in the 'Max' column
            - FIXED: the fit parameter is fixed to be equal to the value in
                the 'Estimation' column
            - FACTOR: the fit parameter must be equal to the value of the fit
                parameter whose name is specified in the 'Parame' column
                multiplied by the value in the 'Factor' column
            - DELTA: the fit parameter must be equal to the value of the fit
                parameter whose name is specified in the 'Parame' column minus
                the value in the 'Delta' column
            - SUM: the fit parameter must be equal to the value of the fit
                parameter whose name is specified in the 'Parame' column plus
                the value in the 'Delta' column
            - IGNORE: ?
            - ADD: ? This choice can be disabled when initializing this widget
        """   # FIXME

        self.__configuring = False
        self.setColumnCount(len(labels))

        for i, label in enumerate(labels):
            item = self.horizontalHeaderItem(i)
            if item is None:
                item = qt.QTableWidgetItem(label,
                                           qt.QTableWidgetItem.Type)
                self.setHorizontalHeaderItem(i, item)
            item.setText(label)

        self.resizeColumnToContents(0)
        self.resizeColumnToContents(1)
        self.resizeColumnToContents(3)
        self.resizeColumnToContents(len(labels) - 1)
        self.resizeColumnToContents(len(labels) - 2)

        # Initialize the table with one line per supplied parameter
        paramlist = paramlist if paramlist is not None else []
        self.parameters = OrderedDict()
        for line, param in enumerate(paramlist):
            self.newparameterline(param, line)

        # connect signal
        self.cellChanged[int, int].connect(self.onCellChanged)

    def newparameterline(self, param, line):
        """Add a line to the :class:`QTableWidget`.

        Each line represents one of the fit parameters for one of
        the fitted peaks.

        :param param: Name of the fit parameter
        :type param: str
        :param line: 0-based line index
        :type line: int
        """
        # get current number of lines
        nlines = self.rowCount()
        self.__configuring = True
        if line >= nlines:
            self.setRowCount(line+1)

        # default configuration for fit parameters
        self.parameters[param] = {'line': line,
                                  'fields': ['name',
                                             'estimation',
                                             'fitresult',
                                             'sigma',
                                             'code',
                                             'val1',
                                             'val2'],
                                  'estimation': '0',
                                  'fitresult': '',
                                  'sigma': '',
                                  'code': 'FREE',
                                  'val1': '',
                                  'val2': '',
                                  'cons1': 0,
                                  'cons2': 0,
                                  'vmin': '0',
                                  'vmax': '1',
                                  'relatedto': '',
                                  'factor': '1.0',
                                  'delta': '0.0',
                                  'sum': '0.0',
                                  'group': '',
                                  'name': param,
                                  'xmin': None,
                                  'xmax': None}
        self.setReadWrite(param, 'estimation')
        self.setReadOnly(param, ['name', 'fitresult', 'sigma', 'val1', 'val2'])

        # the code
        a = []
        for option in self.code_options:
            a.append(option)
        cellWidget = self.cellWidget(line,
                                     self.parameters[param]['fields'].index('code'))
        if cellWidget is None:
            col = self.parameters[param]['fields'].index('code')
            cellWidget = QComboTableItem(self, row=line, col=col)
            cellWidget.addItems(a)
            self.setCellWidget(line, col, cellWidget)
            cellWidget.sigCellChanged[int, int].connect(self.onCellChanged)
        self.parameters[param]['code_item'] = cellWidget
        self.parameters[param]['relatedto_item'] = None
        self.__configuring = False

    def fillTableFromFit(self, fitparameterslist):
        return self.fillfromfit(fitparameterslist)

    def fillfromfit(self, fitparameterslist):
        self.setRowCount(len(fitparameterslist))
        self.parameters = OrderedDict()
        line = 0
        for param in fitparameterslist:
            self.newparameterline(param['name'], line)
            line += 1
        for param in fitparameterslist:
            name = param['name']
            code = str(param['code'])
            if code not in self.code_options:
                code = self.code_options[int(code)]
            val1 = param['cons1']
            val2 = param['cons2']
            estimation = param['estimation']
            group = param['group']
            sigma = param['sigma']
            fitresult = param['fitresult']
            if 'xmin' in param:
                xmin = param['xmin']
            else:
                xmin = None
            if 'xmax' in param:
                xmax = param['xmax']
            else:
                xmax = None
            self.configure(name=name,
                           code=code,
                           val1=val1, val2=val2,
                           estimation=estimation,
                           fitresult=fitresult,
                           sigma=sigma,
                           group=group,
                           xmin=xmin,
                           xmax=xmax)

    def fillFitFromTable(self):
        return self.fillfitfromtable()

    def getConfiguration(self):
        ddict = {}
        ddict['parameters'] = self.fillFitFromTable()
        return ddict

    def setConfiguration(self, ddict):
        self.fillTableFromFit(ddict['parameters'])

    def fillfitfromtable(self):
        fitparameterslist = []
        for param in self.parameters:
            fitparam = {}
            name = param
            estimation, [code, cons1, cons2] = self.cget(name)
            buf = str(self.parameters[param]['fitresult'])
            xmin = self.parameters[param]['xmin']
            xmax = self.parameters[param]['xmax']
            if len(buf):
                fitresult = float(buf)
            else:
                fitresult = 0.0
            buf = str(self.parameters[param]['sigma'])
            if len(buf):
                sigma = float(buf)
            else:
                sigma = 0.0
            buf = str(self.parameters[param]['group'])
            if len(buf):
                group = float(buf)
            else:
                group = 0
            fitparam['name'] = name
            fitparam['estimation'] = estimation
            fitparam['fitresult'] = fitresult
            fitparam['sigma'] = sigma
            fitparam['group'] = group
            fitparam['code'] = code
            fitparam['cons1'] = cons1
            fitparam['cons2'] = cons2
            fitparam['xmin'] = xmin
            fitparam['xmax'] = xmax
            fitparameterslist.append(fitparam)
        return fitparameterslist

    def onCellChanged(self, row, col):
        """Slot called when ``cellChanged`` signal is emitted

        :param row: Row number of the changed cell (0-based index)
        :param col: Column number of the changed cell (0-based index)
        """
        if (col != 4) and (col != -1):
            if row != self.currentRow():
                return
            if col != self.currentColumn():
                return
        if self.__configuring:
            return
        param = list(self.parameters)[row]
        field = self.parameters[param]['fields'][col]
        oldvalue = self.parameters[param][field]
        if col != 4:
            item = self.item(row, col)
            if item is not None:
                newvalue = item.text()
            else:
                newvalue = ''
        else:
            # this is the combobox
            widget = self.cellWidget(row, col)
            newvalue = widget.currentText()
        if self.validate(param, field, oldvalue, newvalue):
            exec("self.configure(name=param,%s=newvalue)" % field)
        else:
            if field == 'code':
                index = self.code_options.index(oldvalue)
                self.__configuring = True
                try:
                    self.parameters[param]['code_item'].setCurrentIndex(index)
                finally:
                    self.__configuring = False
            else:
                exec("self.configure(name=param,%s=oldvalue)" % field)

    def validate(self, param, field, oldvalue, newvalue):
        if field == 'code':
            return self.setcodevalue(param, field, oldvalue, newvalue)
        if field == 'val1' and str(self.parameters[param]['code']) in ['DELTA', 'FACTOR', 'SUM']:
            best, candidates = self.getrelatedcandidates(param)
            if str(newvalue) in candidates:
                return True
            else:
                return False
        else:
            try:
                float(str(newvalue))
            except ValueError:
                return False
        return True

    def setcodevalue(self, workparam, field, oldvalue, newvalue):

        if str(newvalue) == 'FREE':
            self.configure(name=workparam,
                           code=newvalue)
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)
            return 1
        elif str(newvalue) == 'POSITIVE':
            self.configure(name=workparam,
                           code=newvalue)
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)
            return 1
        elif str(newvalue) == 'QUOTED':
            # I have to get the limits
            self.configure(name=workparam,
                           code=newvalue)
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)
            return 1
        elif str(newvalue) == 'FIXED':
            self.configure(name=workparam,
                           code=newvalue)
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)
            return 1
        elif str(newvalue) == 'FACTOR':
            # I should check here that some parameter is set
            best, candidates = self.getrelatedcandidates(workparam)
            if len(candidates) == 0:
                return 0
            self.configure(name=workparam,
                           code=newvalue,
                           relatedto=best)
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)
            return 1
        elif str(newvalue) == 'DELTA':
            # I should check here that some parameter is set
            best, candidates = self.getrelatedcandidates(workparam)
            if len(candidates) == 0:
                return 0
            self.configure(name=workparam,
                           code=newvalue,
                           relatedto=best)
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)
            return 1
        elif str(newvalue) == 'SUM':
            # I should check here that some parameter is set
            best, candidates = self.getrelatedcandidates(workparam)
            if len(candidates) == 0:
                return 0
            self.configure(name=workparam,
                           code=newvalue,
                           relatedto=best)
            if str(oldvalue) == 'IGNORE':
                self.freerestofgroup(workparam)
            return 1
        elif str(newvalue) == 'IGNORE':
            # I should check if the group can be ignored
            # for the time being I just fix all of them to ignore
            group = int(float(str(self.parameters[workparam]['group'])))
            candidates = []
            for param in self.parameters.keys():
                if group == int(float(str(self.parameters[param]['group']))):
                    candidates.append(param)
            # print candidates
            # I should check here if there is any relation to them
            for param in candidates:
                self.configure(name=param,
                               code=newvalue)
            return 1
        elif str(newvalue) == 'ADD':
            group = int(float(str(self.parameters[workparam]['group'])))
            if group == 0:
                if not self._allowBackgroundAdd:
                    # One cannot add a background group
                    return 0
            i = 0
            for param in self.parameters:
                if i <= int(float(str(self.parameters[param]['group']))):
                    i += 1
            if (group == 0) and (i == 1):
                i += 1
            self.addgroup(i, group)
            return 0
        elif str(newvalue) == 'SHOW':
            print(self.cget(workparam))
            return 0
        else:
            print("None of the others!")

    def addgroup(self, newg, gtype):
        line = 0
        newparam = []
        oldparamlist = list(self.parameters)
        for param in oldparamlist:
            line += 1
            paramgroup = int(float(str(self.parameters[param]['group'])))
            if paramgroup == gtype:
                # Try to construct an appropriate name
                # I have to remove any possible trailing number
                # and append the group index
                xmin = self.parameters[param]['xmin']
                xmax = self.parameters[param]['xmax']
                j = len(param) - 1
                while ('0' < param[j]) & (param[j] < '9'):
                    j -= 1
                    if j == -1:
                        break
                if j >= 0:
                    newparam.append(param[0:j + 1] + "%d" % newg)
                else:
                    newparam.append("%d" % newg)
        for param in newparam:
            line += 1
            self.newparameterline(param, line)
        for param in newparam:
            self.configure(name=param, group=newg, xmin=xmin, xmax=xmax)

    def freerestofgroup(self, workparam):
        if workparam in self.parameters.keys():
            group = int(float(str(self.parameters[workparam]['group'])))
            for param in self.parameters.keys():
                if param != workparam:
                    if group == int(float(str(self.parameters[param]['group']))):
                        self.configure(name=param,
                                       code='FREE',
                                       cons1=0,
                                       cons2=0,
                                       val1='',
                                       val2='')

    def getrelatedcandidates(self, workparam):
        best = None
        candidates = []
        for param in self.parameters:
            if param != workparam:
                if str(self.parameters[param]['code']) not in\
                        ['IGNORE', 'FACTOR', 'DELTA', 'SUM']:
                    candidates.append(param)
        # Now get the best from the list
        if candidates == None:
            return best, candidates
        # take the previous one if possible
        if str(self.parameters[workparam]['relatedto']) in candidates:
            best = str(self.parameters[workparam]['relatedto'])
            return best, candidates
        # take the first with similar name
        for param in candidates:
            j = len(param) - 1
            while ('0' <= param[j]) & (param[j] < '9'):
                j -= 1
                if j == -1:
                    break
            if j >= 0:
                try:
                    pos = workparam.index(param[0:j + 1])
                    if pos == 0:
                        best = param
                        return best, candidates
                except:
                    pass
        # take the first
        return candidates[0], candidates

    def setReadOnly(self, parameter, fields):
        editflags = qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled
        self.setfield(parameter, fields, editflags)

    def setReadWrite(self, parameter, fields):
        editflags = qt.Qt.ItemIsSelectable |\
            qt.Qt.ItemIsEnabled |\
            qt.Qt.ItemIsEditable
        self.setfield(parameter, fields, editflags)

    def setfield(self, parameter, fields, EditType):
        if isinstance(parameter, list) or \
           isinstance(parameter, tuple):
            paramlist = parameter
        else:
            paramlist = [parameter]
        if isinstance(fields, list) or \
           isinstance(fields, tuple):
            fieldlist = fields
        else:
            fieldlist = [fields]
        _oldvalue = self.__configuring
        self.__configuring = True
        for param in paramlist:
            if param in self.parameters:
                try:
                    row = list(self.parameters.keys()).index(param)
                except ValueError:
                    row = -1
                if row >= 0:
                    for field in fieldlist:
                        if field in self.parameters[param]['fields']:
                            col = self.parameters[param]['fields'].index(field)
                        if field != 'code':
                            key = field + "_item"
                            item = self.item(row, col)
                            if item is None:
                                item = qt.QTableWidgetItem()
                                item.setText(self.parameters[param][field])
                                self.setItem(row, col, item)
                            else:
                                item.setText(self.parameters[param][field])
                            self.parameters[param][key] = item
                            item.setFlags(EditType)
        self.__configuring = _oldvalue

    def configure(self, name, code=None, **kw):
        error = 0
        paramlist = list(self.parameters.keys())

        if name not in self.parameters:
            return 1

        if code is not None:
            newvalue = str(code)

        for key in kw.keys():
            if key in self.parameters[name]['fields']:
                oldvalue = self.parameters[name][key]
                if len(str(kw[key])):
                    keyDone = False
                    if key == "val1":
                        if str(self.parameters[name]['code']) in\
                                ['DELTA', 'FACTOR', 'SUM']:
                            newvalue = str(kw[key])
                            keyDone = True
                    if not keyDone:
                        newvalue = float(str(kw[key]))
                        if key is 'sigma':
                            newvalue = "%6.3g" % newvalue
                        else:
                            newvalue = "%8g" % newvalue
                else:
                    newvalue = ""
                newvalue = newvalue
                # avoid endless recursivity
                if self.validate(name, key, oldvalue, newvalue):
                    self.parameters[name][key] = newvalue
                else:
                    self.parameters[name][key] = oldvalue
                    error = 1
            elif key in self.parameters[name].keys():
                newvalue = str(kw[key])
                self.parameters[name][key] = newvalue

        if code is not None:
            self.parameters[name]['code'] = code
            for i in range(self.parameters[name]['code_item'].count()):
                if str(code) == str(self.parameters[name]['code_item'].itemText(i)):
                    self.parameters[name]['code_item'].setCurrentIndex(i)
                    break
            if str(code) == 'QUOTED':
                if 'val1' in kw.keys():
                    self.parameters[name][
                        'vmin'] = self.parameters[name]['val1']
                if 'val2' in kw.keys():
                    self.parameters[name][
                        'vmax'] = self.parameters[name]['val2']
            if str(code) == 'DELTA':
                if 'val1'in kw.keys():
                    if kw['val1'] in self.parameters:
                        self.parameters[name]['relatedto'] = kw['val1']
                    else:
                        self.parameters[name]['relatedto'] =\
                            paramlist[int(float(str(kw['val1'])))]
                if 'val2'in kw.keys():
                    self.parameters[name][
                        'delta'] = self.parameters[name]['val2']
            if str(code) == 'SUM':
                if 'val1' in kw.keys():
                    if kw['val1'] in self.parameters:
                        self.parameters[name]['relatedto'] = kw['val1']
                    else:
                        self.parameters[name]['relatedto'] =\
                            paramlist[int(float(str(kw['val1'])))]
                if 'val2' in kw.keys():
                    self.parameters[name][
                        'sum'] = self.parameters[name]['val2']
            if str(code) == 'FACTOR':
                if 'val1'in kw.keys():
                    if kw['val1'] in self.parameters:
                        self.parameters[name]['relatedto'] = kw['val1']
                    else:
                        self.parameters[name]['relatedto'] =\
                            paramlist[int(float(str(kw['val1'])))]
                if 'val2'in kw.keys():
                    self.parameters[name][
                        'factor'] = self.parameters[name]['val2']
        else:
            # Update the proper parameter in case of change in val1 and
            # val2
            if str(self.parameters[name]['code']) == 'QUOTED':
                self.parameters[name][
                    'vmin'] = self.parameters[name]['val1']
                self.parameters[name][
                    'vmax'] = self.parameters[name]['val2']
                # print "vmin =",str(self.parameters[name]['vmin'])
            if str(self.parameters[name]['code']) == 'DELTA':
                self.parameters[name][
                    'relatedto'] = self.parameters[name]['val1']
                self.parameters[name][
                    'delta'] = self.parameters[name]['val2']
            if str(self.parameters[name]['code']) == 'SUM':
                self.parameters[name][
                    'relatedto'] = self.parameters[name]['val1']
                self.parameters[name][
                    'sum'] = self.parameters[name]['val2']
            if str(self.parameters[name]['code']) == 'FACTOR':
                self.parameters[name][
                    'relatedto'] = self.parameters[name]['val1']
                self.parameters[name][
                    'factor'] = self.parameters[name]['val2']

        # Update val1 and val2 according to the parameters
        # and Update the table
        if str(self.parameters[name]['code']) in ['FREE', 'POSITIVE', 'IGNORE', 'FIXED']:
            self.parameters[name]['val1'] = ''
            self.parameters[name]['val2'] = ''
            self.parameters[name]['cons1'] = 0
            self.parameters[name]['cons2'] = 0
            self.setReadWrite(name, 'estimation')
            self.setReadOnly(name, ['fitresult', 'sigma', 'val1', 'val2'])
        elif str(self.parameters[name]['code']) == 'QUOTED':
            self.parameters[name]['val1'] = self.parameters[name]['vmin']
            self.parameters[name]['val2'] = self.parameters[name]['vmax']
            try:
                self.parameters[name]['cons1'] =\
                    float(str(self.parameters[name]['val1']))
            except:
                self.parameters[name]['cons1'] = 0
            try:
                self.parameters[name]['cons2'] =\
                    float(str(self.parameters[name]['val2']))
            except:
                self.parameters[name]['cons2'] = 0
            if self.parameters[name]['cons1'] > self.parameters[name]['cons2']:
                buf = self.parameters[name]['cons1']
                self.parameters[name][
                    'cons1'] = self.parameters[name]['cons2']
                self.parameters[name]['cons2'] = buf
            self.setReadWrite(name, ['estimation', 'val1', 'val2'])
            self.setReadOnly(name, ['fitresult', 'sigma'])
        elif str(self.parameters[name]['code']) == 'FACTOR':
            self.parameters[name][
                'val1'] = self.parameters[name]['relatedto']
            self.parameters[name]['val2'] = self.parameters[name]['factor']
            self.parameters[name]['cons1'] =\
                paramlist.index(str(self.parameters[name]['val1']))
            try:
                self.parameters[name]['cons2'] =\
                    float(str(self.parameters[name]['val2']))
            except ValueError:
                error = 1
                print("Forcing factor to 1")
                self.parameters[name]['cons2'] = 1.0
            self.setReadWrite(name, ['estimation', 'val1', 'val2'])
            self.setReadOnly(name, ['fitresult', 'sigma'])
        elif str(self.parameters[name]['code']) == 'DELTA':
            self.parameters[name][
                'val1'] = self.parameters[name]['relatedto']
            self.parameters[name]['val2'] = self.parameters[name]['delta']
            self.parameters[name]['cons1'] =\
                paramlist.index(str(self.parameters[name]['val1']))
            try:
                self.parameters[name]['cons2'] =\
                    float(str(self.parameters[name]['val2']))
            except ValueError:
                error = 1
                print("Forcing delta to 0")
                self.parameters[name]['cons2'] = 0.0
            self.setReadWrite(name, ['estimation', 'val1', 'val2'])
            self.setReadOnly(name, ['fitresult', 'sigma'])
        elif str(self.parameters[name]['code']) == 'SUM':
            self.parameters[name][
                'val1'] = self.parameters[name]['relatedto']
            self.parameters[name]['val2'] = self.parameters[name]['sum']
            self.parameters[name]['cons1'] =\
                paramlist.index(str(self.parameters[name]['val1']))
            try:
                self.parameters[name]['cons2'] =\
                    float(str(self.parameters[name]['val2']))
            except:
                error = 1
                print("Forcing sum to 0")
                self.parameters[name]['cons2'] = 0.0
            self.setReadWrite(name, ['estimation', 'val1', 'val2'])
            self.setReadOnly(name, ['fitresult', 'sigma'])
        else:
            self.setReadWrite(name, ['estimation', 'val1', 'val2'])
            self.setReadOnly(name, ['fitresult', 'sigma'])
        return error

    def cget(self, param):
        """
        Return tuple estimation,constraints where estimation is the
        value in the estimate field and constraints are the relevant
        constraints according to the active code
        """
        estimation = None
        constraints = None
        if param in self.parameters.keys():
            buf = str(self.parameters[param]['estimation'])
            if len(buf):
                estimation = float(buf)
            else:
                estimation = 0
            self.parameters[param]['code_item']
            if str(self.parameters[param]['code']) in self.code_options:
                code = self.code_options.index(
                    str(self.parameters[param]['code']))
            else:
                code = str(self.parameters[param]['code'])
            cons1 = self.parameters[param]['cons1']
            cons2 = self.parameters[param]['cons2']
            constraints = [code, cons1, cons2]
        return estimation, constraints


def main(args):
    import PyMca5    # FIXME
    from PyMca5 import PyMcaDataDir    # FIXME
    from ..math import specfit
    import numpy
    import os
    app = qt.QApplication(args)
    tab = Parameters(labels=['Parameter', 'Estimation', 'Fit Value', 'Sigma',
                             'Restrains', 'Min/Parame', 'Max/Factor/Delta/'],
                     paramlist=['Height', 'Position', 'FWHM'])
    tab.showGrid()
    tab.configure(name='Height', estimation='1234', group=0)
    tab.configure(name='Position', code='FIXED', group=1)
    tab.configure(name='FWHM', group=1)

    y = numpy.loadtxt(os.path.join(PyMcaDataDir.PYMCA_DATA_DIR,
                      "XRFSpectrum.mca"))    # FIXME
    x = numpy.arange(len(y)) * 0.0502883 - 0.492773
    fit = specfit.Specfit()
    fit.setdata(x=x, y=y)

    fit.importfun(PyMca5.PyMcaMath.fitting.SpecfitFunctions.__file__)    # FIXME

    fit.settheory('Hypermet')
    fit.configure(Yscaling=1.,
                  WeightFlag=1,
                  PosFwhmFlag=1,
                  HeightAreaFlag=1,
                  FwhmPoints=16,
                  PositionFlag=1,
                  HypermetTails=1)
    fit.setbackground('Linear')
    fit.estimate()
    fit.startfit()
    tab.fillfromfit(fit.paramlist)
    tab.show()
    app.lastWindowClosed.connect(app.quit)
    app.exec_()

if __name__ == "__main__":
    main(sys.argv)
