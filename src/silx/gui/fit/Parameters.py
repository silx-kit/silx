# /*##########################################################################
# Copyright (C) 2004-2021 European Synchrotron Radiation Facility
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
"""This module defines a table widget that is specialized in displaying fit
parameter results and associated constraints."""
__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "25/11/2016"

import sys
from collections import OrderedDict

from silx.gui import qt
from silx.gui.widgets.TableWidget import TableWidget


def float_else_zero(sstring):
    """Return converted string to float. If conversion fail, return zero.

    :param sstring: String to be converted
    :return: ``float(sstrinq)`` if ``sstring`` can be converted to float
        (e.g. ``"3.14"``), else ``0``
    """
    try:
        return float(sstring)
    except ValueError:
        return 0


class QComboTableItem(qt.QComboBox):
    """:class:`qt.QComboBox` augmented with a ``sigCellChanged`` signal
    to emit a tuple of ``(row, column)`` coordinates when the value is
    changed.

    This signal can be used to locate the modified combo box in a table.

    :param row: Row number of the table cell containing this widget
    :param col: Column number of the table cell containing this widget"""
    sigCellChanged = qt.Signal(int, int)
    """Signal emitted when this ``QComboBox`` is activated.
    A ``(row, column)`` tuple is passed."""

    def __init__(self, parent=None, row=None, col=None):
        self._row = row
        self._col = col
        qt.QComboBox.__init__(self, parent)
        self.activated[int].connect(self._cellChanged)

    def _cellChanged(self, idx):  # noqa
        self.sigCellChanged.emit(self._row, self._col)


class QCheckBoxItem(qt.QCheckBox):
    """:class:`qt.QCheckBox` augmented with a ``sigCellChanged`` signal
    to emit a tuple of ``(row, column)`` coordinates when the check box has
    been clicked on.

    This signal can be used to locate the modified check box in a table.

    :param row: Row number of the table cell containing this widget
    :param col: Column number of the table cell containing this widget"""
    sigCellChanged = qt.Signal(int, int)
    """Signal emitted when this ``QCheckBox`` is clicked.
    A ``(row, column)`` tuple is passed."""

    def __init__(self, parent=None, row=None, col=None):
        self._row = row
        self._col = col
        qt.QCheckBox.__init__(self, parent)
        self.clicked.connect(self._cellChanged)

    def _cellChanged(self):
        self.sigCellChanged.emit(self._row, self._col)


class Parameters(TableWidget):
    """:class:`TableWidget` customized to display fit results
    and to interact with :class:`FitManager` objects.

    Data and references to cell widgets are kept in a dictionary
    attribute :attr:`parameters`.

    :param parent: Parent widget
    :param labels: Column headers. If ``None``, default headers will be used.
    :type labels: List of strings or None
    :param paramlist: List of fit parameters to be displayed for each fitted
        peak.
    :type paramlist: list[str] or None
    """
    def __init__(self, parent=None, paramlist=None):
        TableWidget.__init__(self, parent)
        self.setContentsMargins(0, 0, 0, 0)

        labels = ['Parameter', 'Estimation', 'Fit Value', 'Sigma',
                  'Constraints', 'Min/Parame', 'Max/Factor/Delta']
        tooltips = ["Fit parameter name",
                    "Estimated value for fit parameter. You can edit this column.",
                    "Actual value for parameter, after fit",
                    "Uncertainty (same unit as the parameter)",
                    "Constraint to be applied to the parameter for fit",
                    "First parameter for constraint (name of another param or min value)",
                    "Second parameter for constraint (max value, or factor/delta)"]

        self.columnKeys = ['name', 'estimation', 'fitresult',
                           'sigma', 'code', 'val1', 'val2']
        """This list assigns shorter keys to refer to columns than the
        displayed labels."""

        self.__configuring = False

        # column headers and associated tooltips
        self.setColumnCount(len(labels))

        for i, label in enumerate(labels):
            item = self.horizontalHeaderItem(i)
            if item is None:
                item = qt.QTableWidgetItem(label,
                                           qt.QTableWidgetItem.Type)
                self.setHorizontalHeaderItem(i, item)

            item.setText(label)
            if tooltips is not None:
                item.setToolTip(tooltips[i])

        # resize columns
        for col_key in ["name", "estimation", "sigma", "val1", "val2"]:
            col_idx = self.columnIndexByField(col_key)
            self.resizeColumnToContents(col_idx)

        # Initialize the table with one line per supplied parameter
        paramlist = paramlist if paramlist is not None else []
        self.parameters = OrderedDict()
        """This attribute stores all the data in an ordered dictionary.
        New data can be added using :meth:`newParameterLine`.
        Existing data can be modified using :meth:`configureLine`

        Keys of the dictionary are:

            -  'name': parameter name
            -  'line': line index for the parameter in the table
            -  'estimation'
            -  'fitresult'
            -  'sigma'
            -  'code': constraint code (one of the elements of
                :attr:`code_options`)
            -  'val1': first parameter related to constraint, formatted
                as a string, as typed in the table
            -  'val2': second parameter related to constraint, formatted
                as a string, as typed in the table
            -  'cons1': scalar representation of 'val1'
                (e.g. when val1 is the name of a fit parameter, cons1
                will be the line index of this parameter)
            -  'cons2': scalar representation of 'val2'
            -  'vmin': equal to 'val1' when 'code' is "QUOTED"
            -  'vmax': equal to 'val2' when 'code' is "QUOTED"
            -  'relatedto': name of related parameter when this parameter
                is constrained to another parameter (same as 'val1')
            -  'factor': same as 'val2' when 'code' is 'FACTOR'
            -  'delta': same as 'val2' when 'code' is 'DELTA'
            -  'sum': same as 'val2' when 'code' is 'SUM'
            -  'group': group index for the parameter
            -  'xmin': data range minimum
            -  'xmax': data range maximum
        """
        for line, param in enumerate(paramlist):
            self.newParameterLine(param, line)

        self.code_options = ["FREE", "POSITIVE", "QUOTED", "FIXED",
                             "FACTOR", "DELTA", "SUM", "IGNORE", "ADD"]
        """Possible values in the combo boxes in the 'Constraints' column.
        """

        # connect signal
        self.cellChanged[int, int].connect(self.onCellChanged)

    def newParameterLine(self, param, line):
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
            self.setRowCount(line + 1)

        # default configuration for fit parameters
        self.parameters[param] = OrderedDict((('line', line),
                                              ('estimation', '0'),
                                              ('fitresult', ''),
                                              ('sigma', ''),
                                              ('code', 'FREE'),
                                              ('val1', ''),
                                              ('val2', ''),
                                              ('cons1', 0),
                                              ('cons2', 0),
                                              ('vmin', '0'),
                                              ('vmax', '1'),
                                              ('relatedto', ''),
                                              ('factor', '1.0'),
                                              ('delta', '0.0'),
                                              ('sum', '0.0'),
                                              ('group', ''),
                                              ('name', param),
                                              ('xmin', None),
                                              ('xmax', None)))
        self.setReadWrite(param, 'estimation')
        self.setReadOnly(param, ['name', 'fitresult', 'sigma', 'val1', 'val2'])

        # Constraint codes
        a = []
        for option in self.code_options:
            a.append(option)

        code_column_index = self.columnIndexByField('code')
        cellWidget = self.cellWidget(line, code_column_index)
        if cellWidget is None:
            cellWidget = QComboTableItem(self, row=line,
                                         col=code_column_index)
            cellWidget.addItems(a)
            self.setCellWidget(line, code_column_index, cellWidget)
            cellWidget.sigCellChanged[int, int].connect(self.onCellChanged)
        self.parameters[param]['code_item'] = cellWidget
        self.parameters[param]['relatedto_item'] = None
        self.__configuring = False

    def columnIndexByField(self, field):
        """

        :param field: Field name (column key)
        :return: Index of the column with this field name
        """
        return self.columnKeys.index(field)

    def fillFromFit(self, fitresults):
        """Fill table with values from a  list of dictionaries
        (see :attr:`silx.math.fit.fitmanager.FitManager.fit_results`)

        :param fitresults: List of parameters as recorded
             in the ``paramlist`` attribute of a :class:`FitManager` object
        :type fitresults: list[dict]
        """
        self.setRowCount(len(fitresults))

        # Reinitialize and fill self.parameters
        self.parameters = OrderedDict()
        for (line, param) in enumerate(fitresults):
            self.newParameterLine(param['name'], line)

        for param in fitresults:
            name = param['name']
            code = str(param['code'])
            if code not in self.code_options:
                # convert code from int to descriptive string
                code = self.code_options[int(code)]
            val1 = param['cons1']
            val2 = param['cons2']
            estimation = param['estimation']
            group = param['group']
            sigma = param['sigma']
            fitresult = param['fitresult']

            xmin = param.get('xmin')
            xmax = param.get('xmax')

            self.configureLine(name=name,
                               code=code,
                               val1=val1, val2=val2,
                               estimation=estimation,
                               fitresult=fitresult,
                               sigma=sigma,
                               group=group,
                               xmin=xmin, xmax=xmax)

    def getConfiguration(self):
        """Return ``FitManager.paramlist`` dictionary
        encapsulated in another dictionary"""
        return {'parameters': self.getFitResults()}

    def setConfiguration(self, ddict):
        """Fill table with values from a ``FitManager.paramlist`` dictionary
        encapsulated in another dictionary"""
        self.fillFromFit(ddict['parameters'])

    def getFitResults(self):
        """Return fit parameters as a list of dictionaries in the format used
        by :class:`FitManager` (attribute ``paramlist``).
        """
        fitparameterslist = []
        for param in self.parameters:
            fitparam = {}
            name = param
            estimation, [code, cons1, cons2] = self.getEstimationConstraints(name)
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
        """Slot called when ``cellChanged`` signal is emitted.
        Checks the validity of the new text in the cell, then calls
        :meth:`configureLine` to update the internal ``self.parameters``
        dictionary.

        :param row: Row number of the changed cell (0-based index)
        :param col: Column number of the changed cell (0-based index)
        """
        if (col != self.columnIndexByField("code")) and (col != -1):
            if row != self.currentRow():
                return
            if col != self.currentColumn():
                return
        if self.__configuring:
            return
        param = list(self.parameters)[row]
        field = self.columnKeys[col]
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
            paramdict = {"name": param, field: newvalue}
            self.configureLine(**paramdict)
        else:
            if field == 'code':
                # New code not valid, try restoring the old one
                index = self.code_options.index(oldvalue)
                self.__configuring = True
                try:
                    self.parameters[param]['code_item'].setCurrentIndex(index)
                finally:
                    self.__configuring = False
            else:
                paramdict = {"name": param, field: oldvalue}
                self.configureLine(**paramdict)

    def validate(self, param, field, oldvalue, newvalue):
        """Check validity of ``newvalue`` when a cell's value is modified.

        :param param: Fit parameter name
        :param field: Column name
        :param oldvalue: Cell value before change attempt
        :param newvalue: New value to be validated
        :return: True if new cell value is valid, else False
        """
        if field == 'code':
            return self.setCodeValue(param, oldvalue, newvalue)
            # FIXME: validate() shouldn't have side effects. Move this bit to configureLine()?
        if field == 'val1' and str(self.parameters[param]['code']) in ['DELTA', 'FACTOR', 'SUM']:
            _, candidates = self.getRelatedCandidates(param)
            # We expect val1 to be a fit parameter name
            if str(newvalue) in candidates:
                return True
            else:
                return False
        # except for code, val1 and name (which is read-only and does not need
        # validation), all fields must always be convertible to float
        else:
            try:
                float(str(newvalue))
            except ValueError:
                return False
        return True

    def setCodeValue(self, param, oldvalue, newvalue):
        """Update 'code' and 'relatedto' fields when code cell is
        changed.

        :param param: Fit parameter name
        :param oldvalue: Cell value before change attempt
        :param newvalue: New value to be validated
        :return: ``True`` if code was successfully updated
        """

        if str(newvalue) in ['FREE', 'POSITIVE', 'QUOTED', 'FIXED']:
            self.configureLine(name=param,
                               code=newvalue)
            if str(oldvalue) == 'IGNORE':
                self.freeRestOfGroup(param)
            return True
        elif str(newvalue) in ['FACTOR', 'DELTA', 'SUM']:
            # I should check here that some parameter is set
            best, candidates = self.getRelatedCandidates(param)
            if len(candidates) == 0:
                return False
            self.configureLine(name=param,
                               code=newvalue,
                               relatedto=best)
            if str(oldvalue) == 'IGNORE':
                self.freeRestOfGroup(param)
            return True

        elif str(newvalue) == 'IGNORE':
            # I should check if the group can be ignored
            # for the time being I just fix all of them to ignore
            group = int(float(str(self.parameters[param]['group'])))
            candidates = []
            for param in self.parameters.keys():
                if group == int(float(str(self.parameters[param]['group']))):
                    candidates.append(param)
            # print candidates
            # I should check here if there is any relation to them
            for param in candidates:
                self.configureLine(name=param,
                                   code=newvalue)
            return True
        elif str(newvalue) == 'ADD':
            group = int(float(str(self.parameters[param]['group'])))
            if group == 0:
                # One cannot add a background group
                return False
            i = 0
            for param in self.parameters:
                if i <= int(float(str(self.parameters[param]['group']))):
                    i += 1
            if (group == 0) and (i == 1):   # FIXME: why +1?
                i += 1
            self.addGroup(i, group)
            return False
        elif str(newvalue) == 'SHOW':
            print(self.getEstimationConstraints(param))
            return False

    def addGroup(self, newg, gtype):
        """Add a fit parameter group with the same fit parameters as an
        existing group.

        This function is called when the user selects "ADD" in the
        "constraints" combobox.

        :param int newg: New group number
        :param int gtype: Group number whose parameters we want to copy

        """
        newparam = []
        # loop through parameters until we encounter group number `gtype`
        for param in list(self.parameters):
            paramgroup = int(float(str(self.parameters[param]['group'])))
            # copy parameter names in group number `gtype`
            if paramgroup == gtype:
                # but replace `gtype` with `newg`
                newparam.append(param.rstrip("0123456789") + "%d" % newg)

                xmin = self.parameters[param]['xmin']
                xmax = self.parameters[param]['xmax']

        # Add new parameters (one table line per parameter) and configureLine each
        # one by updating xmin and xmax to the same values as group `gtype`
        line = len(list(self.parameters))
        for param in newparam:
            self.newParameterLine(param, line)
            line += 1
        for param in newparam:
            self.configureLine(name=param, group=newg, xmin=xmin, xmax=xmax)

    def freeRestOfGroup(self, workparam):
        """Set ``code`` to ``"FREE"`` for all fit parameters belonging to
        the same group as ``workparam``. This is done when the entire group
        of parameters was previously ignored and one of them has his code
        set to something different than ``"IGNORE"``.

        :param workparam: Fit parameter name
        """
        if workparam in self.parameters.keys():
            group = int(float(str(self.parameters[workparam]['group'])))
            for param in self.parameters:
                if param != workparam and\
                        group == int(float(str(self.parameters[param]['group']))):
                    self.configureLine(name=param,
                                       code='FREE',
                                       cons1=0,
                                       cons2=0,
                                       val1='',
                                       val2='')

    def getRelatedCandidates(self, workparam):
        """If fit parameter ``workparam`` has a constraint that involves other
        fit parameters, find possible candidates and try to guess which one
        is the most likely.

        :param workparam: Fit parameter name
        :return: (best_candidate, possible_candidates) tuple
        :rtype: (str, list[str])
        """
        candidates = []
        for param_name in self.parameters:
            if param_name != workparam:
                # ignore parameters that are fixed by a constraint
                if str(self.parameters[param_name]['code']) not in\
                        ['IGNORE', 'FACTOR', 'DELTA', 'SUM']:
                    candidates.append(param_name)
        # take the previous one (before code cell changed) if possible
        if str(self.parameters[workparam]['relatedto']) in candidates:
            best = str(self.parameters[workparam]['relatedto'])
            return best, candidates
        # take the first with same base name (after removing numbers)
        for param_name in candidates:
            basename = param_name.rstrip("0123456789")
            try:
                pos = workparam.index(basename)
                if pos == 0:
                    best = param_name
                    return best, candidates
            except ValueError:
                pass
        # take the first
        return candidates[0], candidates

    def setReadOnly(self, parameter, fields):
        """Make table cells read-only by setting it's flags and omitting
        flag ``qt.Qt.ItemIsEditable``

        :param parameter: Fit parameter names identifying the rows
        :type parameter: str or list[str]
        :param fields: Field names identifying the columns
        :type fields: str or list[str]
        """
        editflags = qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled
        self.setField(parameter, fields, editflags)

    def setReadWrite(self, parameter, fields):
        """Make table cells read-write by setting it's flags including
        flag ``qt.Qt.ItemIsEditable``

        :param parameter: Fit parameter names identifying the rows
        :type parameter: str or list[str]
        :param fields: Field names identifying the columns
        :type fields: str or list[str]
        """
        editflags = qt.Qt.ItemIsSelectable |\
            qt.Qt.ItemIsEnabled |\
            qt.Qt.ItemIsEditable
        self.setField(parameter, fields, editflags)

    def setField(self, parameter, fields, edit_flags):
        """Set text and flags in a table cell.

        :param parameter: Fit parameter names identifying the rows
        :type parameter: str or list[str]
        :param fields: Field names identifying the columns
        :type fields: str or list[str]
        :param edit_flags: Flag combination, e.g::

            qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled |
            qt.Qt.ItemIsEditable
        """
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

        # Set _configuring flag to ignore cellChanged signals in
        # self.onCellChanged
        _oldvalue = self.__configuring
        self.__configuring = True

        # 2D loop through parameter list and field list
        # to update their cells
        for param in paramlist:
            row = list(self.parameters.keys()).index(param)
            for field in fieldlist:
                col = self.columnIndexByField(field)
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
                    item.setFlags(edit_flags)

        # Restore previous _configuring flag
        self.__configuring = _oldvalue

    def configureLine(self, name, code=None, val1=None, val2=None,
                      sigma=None, estimation=None, fitresult=None,
                      group=None, xmin=None, xmax=None, relatedto=None,
                      cons1=None, cons2=None):
        """This function updates values in a line of the table

        :param name: Name of the parameter (serves as unique identifier for
                     a line).
        :param code: Constraint code *FREE, FIXED, POSITIVE, DELTA, FACTOR,
                     SUM, QUOTED, IGNORE*
        :param val1: Constraint 1 (can be the index or name of another
                     parameter for code *DELTA, FACTOR, SUM*, or a min value
                     for code *QUOTED*)
        :param val2: Constraint 2
        :param sigma: Standard deviation for a fit parameter
        :param estimation: Estimated initial value for a fit parameter (used
                           as input to iterative fit)
        :param fitresult: Final result of fit
        :param group: Group number of a fit parameter (peak number when doing
                      multi-peak fitting, as each peak corresponds to a group
                      of several consecutive parameters)
        :param xmin:
        :param xmax:
        :param relatedto: Index or name of another fit parameter
                          to which this parameter is related to (constraints)
        :param cons1: similar meaning to ``val1``, but is always a number
        :param cons2: similar meaning to ``val2``, but is always a number
        :return:
        """
        paramlist = list(self.parameters.keys())

        if name not in self.parameters:
            raise KeyError("'%s' is not in the parameter list" % name)

        # update code first, if specified
        if code is not None:
            code = str(code)
            self.parameters[name]['code'] = code
            # update combobox
            index = self.parameters[name]['code_item'].findText(code)
            self.parameters[name]['code_item'].setCurrentIndex(index)
        else:
            # set code to previous value, used later for setting val1 val2
            code = self.parameters[name]['code']

        # val1 and sigma have special formats
        if val1 is not None:
            fmt = None if self.parameters[name]['code'] in\
                          ['DELTA', 'FACTOR', 'SUM'] else "%8g"
            self._updateField(name, "val1", val1, fmat=fmt)

        if sigma is not None:
            self._updateField(name, "sigma", sigma, fmat="%6.3g")

        # other fields are formatted as "%8g"
        keys_params = (("val2", val2), ("estimation", estimation),
                       ("fitresult", fitresult))
        for key, value in keys_params:
            if value is not None:
                self._updateField(name, key, value, fmat="%8g")

        # the rest of the parameters are treated as strings and don't need
        # validation
        keys_params = (("group", group), ("xmin", xmin),
                       ("xmax", xmax), ("relatedto", relatedto),
                       ("cons1", cons1), ("cons2", cons2))
        for key, value in keys_params:
            if value is not None:
                self.parameters[name][key] = str(value)

        # val1 and val2 have different meanings depending on the code
        if code == 'QUOTED':
            if val1 is not None:
                self.parameters[name]['vmin'] = self.parameters[name]['val1']
            else:
                self.parameters[name]['val1'] = self.parameters[name]['vmin']
            if val2 is not None:
                self.parameters[name]['vmax'] = self.parameters[name]['val2']
            else:
                self.parameters[name]['val2'] = self.parameters[name]['vmax']

            # cons1 and cons2 are scalar representations of val1 and val2
            self.parameters[name]['cons1'] =\
                float_else_zero(self.parameters[name]['val1'])
            self.parameters[name]['cons2'] =\
                float_else_zero(self.parameters[name]['val2'])

            # cons1, cons2 = min(val1, val2), max(val1, val2)
            if self.parameters[name]['cons1'] > self.parameters[name]['cons2']:
                self.parameters[name]['cons1'], self.parameters[name]['cons2'] =\
                    self.parameters[name]['cons2'], self.parameters[name]['cons1']

        elif code in ['DELTA', 'SUM', 'FACTOR']:
            # For these codes, val1 is the fit parameter name on which the
            # constraint depends
            if val1 is not None and val1 in paramlist:
                self.parameters[name]['relatedto'] = self.parameters[name]["val1"]

            elif val1 is not None:
                # val1 could be the index of the fit parameter
                try:
                    self.parameters[name]['relatedto'] = paramlist[int(val1)]
                except ValueError:
                    self.parameters[name]['relatedto'] = self.parameters[name]["val1"]

            elif relatedto is not None:
                # code changed, val1 not specified but relatedto specified:
                # set val1 to relatedto (pre-fill best guess)
                self.parameters[name]["val1"] = relatedto

            # update fields "delta", "sum" or "factor"
            key = code.lower()
            self.parameters[name][key] = self.parameters[name]["val2"]

            # FIXME: val1 is sometimes specified as an index rather than a param name
            self.parameters[name]['val1'] = self.parameters[name]['relatedto']

            # cons1 is the index of the fit parameter in the ordered dictionary
            if self.parameters[name]['val1'] in paramlist:
                self.parameters[name]['cons1'] =\
                    paramlist.index(self.parameters[name]['val1'])

            # cons2 is the constraint value (factor, delta or sum)
            try:
                self.parameters[name]['cons2'] =\
                    float(str(self.parameters[name]['val2']))
            except ValueError:
                self.parameters[name]['cons2'] = 1.0 if code == "FACTOR" else 0.0

        elif code in ['FREE', 'POSITIVE', 'IGNORE', 'FIXED']:
            self.parameters[name]['val1'] = ""
            self.parameters[name]['val2'] = ""
            self.parameters[name]['cons1'] = 0
            self.parameters[name]['cons2'] = 0

        self._updateCellRWFlags(name, code)

    def _updateField(self, name, field, value, fmat=None):
        """Update field in ``self.parameters`` dictionary, if the new value
        is valid.

        :param name: Fit parameter name
        :param field: Field name
        :param value: New value to assign
        :type value: String
        :param fmat: Format string (e.g. "%8g") to be applied if value represents
            a scalar. If ``None``, format is not modified. If ``value`` is an
            empty string, ``fmat`` is ignored.
        """
        if value is not None:
            oldvalue = self.parameters[name][field]
            if fmat is not None:
                newvalue = fmat % float(value) if value != "" else ""
            else:
                newvalue = value
            self.parameters[name][field] = newvalue if\
                self.validate(name, field, oldvalue, newvalue) else\
                oldvalue

    def _updateCellRWFlags(self, name, code=None):
        """Set read-only or read-write flags in a row,
        depending on the constraint code

        :param name: Fit parameter name identifying the row
        :param code: Constraint code, in `'FREE', 'POSITIVE', 'IGNORE',`
            `'FIXED', 'FACTOR', 'DELTA', 'SUM', 'ADD'`
        :return:
        """
        if code in ['FREE', 'POSITIVE', 'IGNORE', 'FIXED']:
            self.setReadWrite(name, 'estimation')
            self.setReadOnly(name, ['fitresult', 'sigma', 'val1', 'val2'])
        else:
            self.setReadWrite(name, ['estimation', 'val1', 'val2'])
            self.setReadOnly(name, ['fitresult', 'sigma'])

    def getEstimationConstraints(self, param):
        """
        Return tuple ``(estimation, constraints)`` where ``estimation`` is the
        value in the ``estimate`` field and ``constraints`` are the relevant
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
    from silx.math.fit import fittheories
    from silx.math.fit import fitmanager
    try:
        from PyMca5 import PyMcaDataDir
    except ImportError:
        raise ImportError("This demo requires PyMca data. Install PyMca5.")
    import numpy
    import os
    app = qt.QApplication(args)
    tab = Parameters(paramlist=['Height', 'Position', 'FWHM'])
    tab.showGrid()
    tab.configureLine(name='Height', estimation='1234', group=0)
    tab.configureLine(name='Position', code='FIXED', group=1)
    tab.configureLine(name='FWHM', group=1)

    y = numpy.loadtxt(os.path.join(PyMcaDataDir.PYMCA_DATA_DIR,
                      "XRFSpectrum.mca"))    # FIXME

    x = numpy.arange(len(y)) * 0.0502883 - 0.492773
    fit = fitmanager.FitManager()
    fit.setdata(x=x, y=y, xmin=20, xmax=150)

    fit.loadtheories(fittheories)

    fit.settheory('ahypermet')
    fit.configure(Yscaling=1.,
                  PositiveFwhmFlag=True,
                  PositiveHeightAreaFlag=True,
                  FwhmPoints=16,
                  QuotedPositionFlag=1,
                  HypermetTails=1)
    fit.setbackground('Linear')
    fit.estimate()
    fit.runfit()
    tab.fillFromFit(fit.fit_results)
    tab.show()
    app.exec()

if __name__ == "__main__":
    main(sys.argv)
