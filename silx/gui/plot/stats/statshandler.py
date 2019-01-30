# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2019 European Synchrotron Radiation Facility
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
"""
"""

__authors__ = ["H. Payno"]
__license__ = "MIT"
__date__ = "05/06/2018"


import logging

from silx.gui import qt
from silx.gui.plot import stats as statsmdl

logger = logging.getLogger(__name__)


class _FloatItem(qt.QTableWidgetItem):
    """Simple QTableWidgetItem allowing ordering on floats"""

    def __init__(self, type=qt.QTableWidgetItem.Type):
        qt.QTableWidgetItem.__init__(self, type=type)

    def __lt__(self, other):
        self_values = self.text().lstrip('(').rstrip(')').split(',')
        other_values = other.text().lstrip('(').rstrip(')').split(',')
        for self_value, other_value in zip(self_values, other_values):
            f_self_value = float(self_value)
            f_other_value = float(other_value)
            if f_self_value != f_other_value:
                return f_self_value < f_other_value
        return False


class StatFormatter(object):
    """
    Class used to apply format on :class:`Stat`

    :param formatter: the formatter. Defined as str.format()
    :param qItemClass: the class inheriting from :class:`QTableWidgetItem`
                       which will be used to display the result of the
                       statistic computation.
    """
    DEFAULT_FORMATTER = '{0:.3f}'

    def __init__(self, formatter=DEFAULT_FORMATTER, qItemClass=_FloatItem):
        self.formatter = formatter
        self.tabWidgetItemClass = qItemClass

    def format(self, val):
        if self.formatter is None or val is None:
            return str(val)
        else:
            return self.formatter.format(val)


class StatsHandler(object):
    """
    Give
    create:

    * Stats object which will manage the statistic computation
    * Associate formatter and :class:`Stat`

    :param statFormatters: Stat and optional formatter.
                           If elements are given as a tuple, elements
                           should be (:class:`Stat`, formatter).
                           Otherwise should be :class:`Stat` elements.
    :rtype: List or tuple
    """

    def __init__(self, statFormatters):
        self.stats = statsmdl.Stats()
        self.formatters = {}
        for elmt in statFormatters:
            stat, formatter = self._processStatArgument(elmt)
            self.add(stat=stat, formatter=formatter)

    @staticmethod
    def _processStatArgument(arg):
        """Process an element of the init arguments

        :param arg: The argument to process
        :return: Corresponding (StatBase, StatFormatter)
        """
        stat, formatter = None, None

        if isinstance(arg, statsmdl.StatBase):
            stat = arg
        else:
            assert len(arg) > 0
            if isinstance(arg[0], statsmdl.StatBase):
                stat = arg[0]
                if len(arg) > 2:
                    raise ValueError('To many argument with %s. At most one '
                                     'argument can be associated with the '
                                     'BaseStat (the `StatFormatter`')
                if len(arg) == 2:
                    assert arg[1] is None or isinstance(arg[1], (StatFormatter, str))
                    formatter = arg[1]
            else:
                if isinstance(arg[0], tuple):
                    if len(arg) > 1:
                        formatter = arg[1]
                    arg = arg[0]

                if type(arg[0]) is not str:
                    raise ValueError('first element of the tuple should be a string'
                                     ' or a StatBase instance')
                if len(arg) == 1:
                    raise ValueError('A function should be associated with the'
                                     'stat name')
                if len(arg) > 3:
                    raise ValueError('Two much argument given for defining statistic.'
                                     'Take at most three arguments (name, function, '
                                     'kinds)')
                if len(arg) == 2:
                    stat = statsmdl.Stat(name=arg[0], fct=arg[1])
                else:
                    stat = statsmdl.Stat(name=arg[0], fct=arg[1], kinds=arg[2])

        return stat, formatter

    def add(self, stat, formatter=None):
        """Add a stat to the list.

        :param StatBase stat:
        :param Union[None,StatFormatter] formatter:
        """
        assert isinstance(stat, statsmdl.StatBase)
        self.stats.add(stat)
        _formatter = formatter
        if type(_formatter) is str:
            _formatter = StatFormatter(formatter=_formatter)
        self.formatters[stat.name] = _formatter

    def format(self, name, val):
        """Apply the format for the `name` statistic and the given value

        :param str name: the name of the associated statistic
        :param val: value before formatting
        :return: formatted value
        """
        if name not in self.formatters:
            logger.warning("statistic %s haven't been registred" % name)
            return val
        else:
            if self.formatters[name] is None:
                return str(val)
            else:
                if isinstance(val, (tuple, list)):
                    res = []
                    [res.append(self.formatters[name].format(_val)) for _val in val]
                    return ', '.join(res)
                else:
                    return self.formatters[name].format(val)

    def calculate(self, item, plot, onlimits):
        """
        compute all statistic registered and return the list of formatted
        statistics result.

        :param item: item for which we want to compute statistics
        :param plot: plot containing the item
        :param onlimits: True if we want to compute statistics on visible data
                         only
        :return: list of formatted statistics (as str)
        :rtype: dict
        """
        res = self.stats.calculate(item, plot, onlimits)
        for resName, resValue in list(res.items()):
            res[resName] = self.format(resName, res[resName])
        return res
