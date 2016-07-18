#/*##########################################################################
# Copyright (C) 2004-2016 European Synchrotron Radiation Facility
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
"""This module defines a table widget that is specialized in displaying
fit parameter results and associated constraints."""
__authors__ = ["V.A. Sole", "P. Knobel"]
__license__ = "MIT"
__date__ = "18/07/2016"

import sys
from collections import OrderedDict

from silx.gui import qt


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
    """:class:`qt.QTableWidget` customized to display fit results
    and to interact with :class:`Specfit` objects.

    Data and references to cell widgets are kept in a dictionary
    attribute :attr:`parameters`.

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
        """

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
        self.parameters[param] = OrderedDict((('line', line),
                                              ('fields', ['name',
                                                          'estimation',
                                                          'fitresult',
                                                          'sigma',
                                                          'code',
                                                          'val1',
                                                          'val2']),
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
        self.set_read_write(param, 'estimation')
        self.set_read_only(param, ['name', 'fitresult', 'sigma', 'val1', 'val2'])

        # Constraint codes
        a = []
        for option in self.code_options:
            a.append(option)

        code_column_index = self.column_index_by_field(param, 'code')
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

    def column_index_by_field(self, param, field):
        """

        :param param: Name of the fit parameter
        :param field: Field name
        :return: Column index of this field
        """
        return self.parameters[param]['fields'].index(field)

    def fillfromfit(self, fitparameterslist):
        """Fill table with values from a ``Specfit.paramlist`` list
        of dictionaries.

        :param fitparameterslist: List of parameters as recorded
             in the ``paramlist`` attribute of a :class:`Specfit` object
        :type fitparameterslist: list[dict]
        """
        self.setRowCount(len(fitparameterslist))

        # Reinitialize and fill self.parameters
        self.parameters = OrderedDict()
        for (line, param) in enumerate(fitparameterslist):
            self.newparameterline(param['name'], line)

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

            # dict.get() returns None if key does not exist
            xmin = param.get('xmin')
            xmax = param.get('xmax')

            self.configure_line(name=name,
                                code=code,
                                val1=val1, val2=val2,
                                estimation=estimation,
                                fitresult=fitresult,
                                sigma=sigma,
                                group=group,
                                xmin=xmin, xmax=xmax)

    def getConfiguration(self):
        """Return ``Specfit.paramlist`` dictionary
        encapsulated in another dictionary"""
        return {'parameters': self.getfitresults()}

    def setConfiguration(self, ddict):
        """Fill table with values from a ``Specfit.paramlist`` dictionary
        encapsulated in another dictionary"""
        self.fillfromfit(ddict['parameters'])

    def getfitresults(self):
        """Return fit parameters as a list of dictionaries in the format used
        by :class:`Specfit` (attribute ``paramlist``).
        """
        fitparameterslist = []
        for param in self.parameters:
            fitparam = {}
            name = param
            estimation, [code, cons1, cons2] = self.get_estimation_constraints(name)
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
        :meth:`configure_line` to update the internal ``self.parameters``
        dictionary.

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
            paramdict = {"name": param, field: newvalue}
            self.configure_line(**paramdict)
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
                self.configure_line(**paramdict)

    def validate(self, param, field, oldvalue, newvalue):
        """Check validity of ``newvalue`` when a cell's value is modified.

        :param param: Fit parameter name
        :param field: Column name
        :param oldvalue: Cell value before change attempt
        :param newvalue: New value to be validated
        :return: True if new cell value is valid, else False
        """
        if field == 'code':
            return self.set_code_value(param, oldvalue, newvalue)
            # FIXME: validate() shouldn't have side effects. Move this bit to configure_line()?
        if field == 'val1' and str(self.parameters[param]['code']) in ['DELTA', 'FACTOR', 'SUM']:
            best, candidates = self.get_related_candidates(param)
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

    def set_code_value(self, param, oldvalue, newvalue):
        """Update 'code' and 'relatedto' fields when code cell is
        changed.

        :param param: Fit parameter name
        :param oldvalue: Cell value before change attempt
        :param newvalue: New value to be validated
        :return: ``True`` if code was successfully updated
        """

        if str(newvalue) in ['FREE', 'POSITIVE', 'QUOTED', 'FIXED']:
            self.configure_line(name=param,
                                code=newvalue)
            if str(oldvalue) == 'IGNORE':
                self.free_rest_of_group(param)
            return True
        elif str(newvalue) in ['FACTOR', 'DELTA', 'SUM']:
            # I should check here that some parameter is set
            best, candidates = self.get_related_candidates(param)
            if len(candidates) == 0:
                return False
            self.configure_line(name=param,
                                code=newvalue,
                                relatedto=best)
            if str(oldvalue) == 'IGNORE':
                self.free_rest_of_group(param)
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
                self.configure_line(name=param,
                                    code=newvalue)
            return True
        elif str(newvalue) == 'ADD':
            group = int(float(str(self.parameters[param]['group'])))
            if group == 0:
                if not self._allowBackgroundAdd:
                    # One cannot add a background group
                    return False
            i = 0
            for param in self.parameters:
                if i <= int(float(str(self.parameters[param]['group']))):
                    i += 1
            if (group == 0) and (i == 1):   # FIXME: why +1?
                i += 1
            self.add_group(i, group)
            return False
        elif str(newvalue) == 'SHOW':
            print(self.get_estimation_constraints(param))
            return False

    def add_group(self, newg, gtype):
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

        # Add new parameters (one table line per parameter) and configure_line each
        # one by updating xmin and xmax to the same values as group `gtype`
        line = len(list(self.parameters))
        for param in newparam:
            line += 1
            self.newparameterline(param, line)
        for param in newparam:
            self.configure_line(name=param, group=newg, xmin=xmin, xmax=xmax)

    def free_rest_of_group(self, workparam):
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
                    self.configure_line(name=param,
                                        code='FREE',
                                        cons1=0,
                                        cons2=0,
                                        val1='',
                                        val2='')

    def get_related_candidates(self, workparam):
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

    def set_read_only(self, parameter, fields):
        """Make table cells read-only by setting it's flags and omitting
        flag ``qt.Qt.ItemIsEditable``

        :param parameter: Fit parameter names identifying the rows
        :type parameter: str or list[str]
        :param fields: Field names identifying the columns
        :type fields: str or list[str]
        """
        editflags = qt.Qt.ItemIsSelectable | qt.Qt.ItemIsEnabled
        self.set_field(parameter, fields, editflags)

    def set_read_write(self, parameter, fields):
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
        self.set_field(parameter, fields, editflags)

    def set_field(self, parameter, fields, edit_flags):
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
                col = self.column_index_by_field(param, field)
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

    def configure_line(self, name, code=None, val1=None, val2=None,
                       sigma=None, estimation=None, fitresult=None,
                       group=None, xmin=None, xmax=None, relatedto=None,
                       cons1=None, cons2=None):
        """This function updates values in a line of the table

        :param name: Name of the parameter (serves as unique identifier for
                     a line.
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
        :param relatedto: Same as val1, index or name of another fit parameter
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
            # update combobox
            index = self.parameters[name]['code_item'].findText(code)
            self.parameters[name]['code_item'].setCurrentIndex(index)
        else:
            code = self.parameters[name]['code']

        # val1 and sigma have special formats
        if val1 is not None:
            fmt = None if self.parameters[name]['code'] in\
                          ['DELTA', 'FACTOR', 'SUM'] else "%8g"
            self._update_field(name, "val1", val1, fmat=fmt)

        if sigma is not None:
            self._update_field(name, "sigma", sigma, fmat="%6.3g")

        # other fields are formatted as "%8g"
        keys_params = (("val2", val2), ("estimation", estimation),
                       ("fitresult", fitresult))
        for key, value in keys_params:
            if value is not None:
                self._update_field(name, key, value, fmat="%8g")

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
            if (val1 is not None and val1 in paramlist) or val1 is None:
                self.parameters[name]['relatedto'] = self.parameters[name]["val1"]

            elif val1 is not None:
                # val1 could be the index of the fit parameter
                try:
                    self.parameters[name]['relatedto'] = paramlist[int(val1)]
                except ValueError:
                    self.parameters[name]['relatedto'] = self.parameters[name]["val1"]

            # update fields "delta", "sum" or "factor"
            key = code.lower()
            self.parameters[name][key] = self.parameters[name]["val2"]

            # FIXME: val1 is sometimes specified as an index rather than a param name
            self.parameters[name]['val1'] = self.parameters[name]['relatedto']

            # cons1 is the index of the fit parameter in the ordered dict
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

        self._update_cell_rw_flags(name, code)

    def _update_field(self, name, field, value, fmat=None):
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

    def _update_cell_rw_flags(self, name, code=None):
        """Set read-only or read-write flags in a row,
        depending on the constraint code

        :param name: Fit parameter name identifying the row
        :param code: Constraint code, in `'FREE', 'POSITIVE', 'IGNORE',`
            `'FIXED', 'FACTOR', 'DELTA', 'SUM', 'ADD'`
        :return:
        """
        if code in ['FREE', 'POSITIVE', 'IGNORE', 'FIXED']:
            self.set_read_write(name, 'estimation')
            self.set_read_only(name, ['fitresult', 'sigma', 'val1', 'val2'])
        else:
            self.set_read_write(name, ['estimation', 'val1', 'val2'])
            self.set_read_only(name, ['fitresult', 'sigma'])

    def get_estimation_constraints(self, param):
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
    from silx.math.fit import fitestimatefunctions
    from silx.math.fit import specfit
    from silx.gui import qt
    from PyMca5 import PyMcaDataDir    # FIXME
    import numpy
    import os
    app = qt.QApplication(args)
    tab = Parameters(labels=['Parameter', 'Estimation', 'Fit Value', 'Sigma',
                             'Restrains', 'Min/Parame', 'Max/Factor/Delta/'],
                     paramlist=['Height', 'Position', 'FWHM'])
    tab.showGrid()
    tab.configure_line(name='Height', estimation='1234', group=0)
    tab.configure_line(name='Position', code='FIXED', group=1)
    tab.configure_line(name='FWHM', group=1)

    y = numpy.loadtxt(os.path.join(PyMcaDataDir.PYMCA_DATA_DIR,
                      "XRFSpectrum.mca"))    # FIXME

    x = numpy.arange(len(y)) * 0.0502883 - 0.492773
    fit = specfit.Specfit()
    fit.setdata(x=x, y=y, xmin=20, xmax=150)

    fit.importfun(fitestimatefunctions.__file__)

    fit.settheory('ahypermet')
    fit.configure(Yscaling=1.,
                  PositiveFwhmFlag=True,
                  PositiveHeightAreaFlag=True,
                  FwhmPoints=16,
                  QuotedPositionFlag=1,
                  HypermetTails=1)
    fit.setbackground('Linear')
    fit.estimate()
    fit.startfit()
    tab.fillfromfit(fit.fit_results)
    tab.show()
    app.exec_()

if __name__ == "__main__":
    main(sys.argv)
