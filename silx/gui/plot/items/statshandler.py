# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
__date__ = "07/03/2018"


from silx.gui import qt
from silx.gui.plot.items import stats as statsmdl

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

    def __init__(self, formatter=None, qItemClass=_FloatItem):
        self.formatter = formatter
        self.tabWidgetItemClass = qItemClass

    def format(self, val):
        if self.formatter is None:
            return val
        else:
            self.formatter.format(val)


class StatsHandler(object):
    """
    Give 
    create:

    * Stats object which will manage the statistic computation
    * Associate formatter and :class:`Stat`

    :param _listStatFormatter: Stat and optional formatter.
                               If elements are given as a tuple, elements
                               should be (:class:`Stat`, formatter).
                               Otherwise should be :class:`Stat` elements.
    :rtype: list or tuple
    """

    def __init__(self, _listStatFormatter):
        self.stats = statsmdl.Stats()
        self.formatters = {}
        for elmt in _listStatFormatter:
            if type(elmt) in (tuple, list):
                assert len(elmt) is 2
                self.add(stat=elmt[0], formatter=elmt[1])
            else:
                self.add(stat=elmt)

    def add(self, stat, formatter=None):
        assert isinstance(stat, statsmdl.StatBase)
        self.stats.add(stat)
        _formatter = formatter
        if _formatter is None:
            _formatter = StatFormatter()
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
                return val
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
