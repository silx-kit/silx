# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
        return float(self.text()) < float(other.text())


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
            helper = _StatHelper(elmt)
            self.add(stat=helper.stat, formatter=helper.statFormatter)

    def add(self, stat, formatter=None):
        assert isinstance(stat, statsmdl.StatBase)
        self.stats.add(stat)
        _formatter = formatter
        if type(_formatter) is str:
            _formatter = StatFormatter(formatter=_formatter)
        self.formatters[stat.name] = _formatter

    def format(self, name, val):
        """
        Apply the format for the `name` statistic and the given value
        :param name: the name of the associated statistic
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
        compute all statistic registred and return the list of formatted
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


class _StatHelper(object):
    """
    Helper class to generated the requested StatBase instance and the
    associated StatFormatter
    """
    def __init__(self, arg):
        self.statFormatter = None
        self.stat = None

        if isinstance(arg, statsmdl.StatBase):
            self.stat = arg
        else:
            assert len(arg) > 0
            if isinstance(arg[0], statsmdl.StatBase):
                self.dealWithStatAndFormatter(arg)
            else:
                _arg = arg
                if isinstance(arg[0], tuple):
                    _arg = arg[0]
                    if len(arg) > 1:
                        self.statFormatter = arg[1]
                self.createStatInstanceAndFormatter(_arg)

    def dealWithStatAndFormatter(self, arg):
        assert isinstance(arg[0], statsmdl.StatBase)
        self.stat = arg[0]
        if len(arg) > 2:
            raise ValueError('To many argument with %s. At most one '
                             'argument can be associated with the '
                             'BaseStat (the `StatFormatter`')
        if len(arg) is 2:
            assert isinstance(arg[1], (StatFormatter, type(None), str))
            self.statFormatter = arg[1]

    def createStatInstanceAndFormatter(self, arg):
        if type(arg[0]) is not str:
            raise ValueError('first element of the tuple should be a string'
                             ' or a StatBase instance')
        if len(arg) is 1:
            raise ValueError('A function should be associated with the'
                             'stat name')
        if len(arg) > 3:
            raise ValueError('Two much argument given for defining statistic.'
                             'Take at most three arguments (name, function, '
                             'kinds)')
        if len(arg) is 2:
            self.stat = statsmdl.Stat(name=arg[0], fct=arg[1])
        else:
            self.stat = statsmdl.Stat(name=arg[0], fct=arg[1], kinds=arg[2])
