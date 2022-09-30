#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2004-2021 European Synchrotron Radiation Facility
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
"""This script is a simple example of how to use the periodic table widgets,
select elements and connect signals.
"""
import sys
from silx.gui import qt
from silx.gui.widgets import PeriodicTable

a = qt.QApplication(sys.argv)
sys.excepthook = qt.exceptionHandler
a.lastWindowClosed.connect(a.quit)

w = qt.QTabWidget()

pt = PeriodicTable.PeriodicTable(w, selectable=True)
pc = PeriodicTable.PeriodicCombo(w)
pl = PeriodicTable.PeriodicList(w)

pt.setSelection(['Fe', 'Si', 'Mt'])
pl.setSelectedElements(['H', 'Be', 'F'])
pc.setSelection("Li")


def change_list(items):
    print("New list selection:", [item.symbol for item in items])


def change_combo(item):
    print("New combo selection:", item.symbol)


def click_table(item):
    print("New table click: %s (%s)" % (item.name, item.subcategory))


def change_table(items):
    print("New table selection:", [item.symbol for item in items])


pt.sigElementClicked.connect(click_table)
pt.sigSelectionChanged.connect(change_table)
pl.sigSelectionChanged.connect(change_list)
pc.sigSelectionChanged.connect(change_combo)

# move combo into container widget to preventing it from filling
# the tab inside TabWidget
comboContainer = qt.QWidget(w)
comboContainer.setLayout(qt.QVBoxLayout())
comboContainer.layout().addWidget(pc)

w.addTab(pt, "PeriodicTable")
w.addTab(pl, "PeriodicList")
w.addTab(comboContainer, "PeriodicCombo")
w.show()

a.exec()
