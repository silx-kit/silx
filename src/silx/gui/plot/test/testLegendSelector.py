# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""Basic tests for PlotWidget"""

__authors__ = ["T. Rueter", "T. Vincent"]
__license__ = "MIT"
__date__ = "15/05/2017"


import logging
import unittest

from silx.gui import qt
from silx.gui.utils.testutils import TestCaseQt
from silx.gui.plot import LegendSelector


_logger = logging.getLogger(__name__)


class TestLegendSelector(TestCaseQt):
    """Basic test for LegendSelector"""

    def testLegendSelector(self):
        """Test copied from __main__ of LegendSelector in PyMca"""
        class Notifier(qt.QObject):
            def __init__(self):
                qt.QObject.__init__(self)
                self.chk = True

            def signalReceived(self, **kw):
                obj = self.sender()
                _logger.info('NOTIFIER -- signal received\n\tsender: %s',
                             str(obj))

        notifier = Notifier()

        legends = ['Legend0',
                   'Legend1',
                   'Long Legend 2',
                   'Foo Legend 3',
                   'Even Longer Legend 4',
                   'Short Leg 5',
                   'Dot symbol 6',
                   'Comma symbol 7']
        colors = [qt.Qt.darkRed, qt.Qt.green, qt.Qt.yellow, qt.Qt.darkCyan,
                  qt.Qt.blue, qt.Qt.darkBlue, qt.Qt.red, qt.Qt.darkYellow]
        symbols = ['o', 't', '+', 'x', 's', 'd', '.', ',']

        win = LegendSelector.LegendListView()
        # win = LegendListContextMenu()
        # win = qt.QWidget()
        # layout = qt.QVBoxLayout()
        # layout.setContentsMargins(0,0,0,0)
        llist = []

        for _idx, (l, c, s) in enumerate(zip(legends, colors, symbols)):
            ddict = {
                'color': qt.QColor(c),
                'linewidth': 4,
                'symbol': s,
            }
            legend = l
            llist.append((legend, ddict))
            # item = qt.QListWidgetItem(win)
            # legendWidget = LegendListItemWidget(l)
            # legendWidget.icon.setSymbol(s)
            # legendWidget.icon.setColor(qt.QColor(c))
            # layout.addWidget(legendWidget)
            # win.setItemWidget(item, legendWidget)

        # win = LegendListItemWidget('Some Legend 1')
        # print(llist)
        model = LegendSelector.LegendModel(legendList=llist)
        win.setModel(model)
        win.setSelectionModel(qt.QItemSelectionModel(model))
        win.setContextMenu()
        # print('Edit triggers: %d'%win.editTriggers())

        # win = LegendListWidget(None, legends)
        # win[0].updateItem(ddict)
        # win.setLayout(layout)
        win.sigLegendSignal.connect(notifier.signalReceived)
        win.show()

        win.clear()
        win.setLegendList(llist)

        self.qWaitForWindowExposed(win)


class TestRenameCurveDialog(TestCaseQt):
    """Basic test for RenameCurveDialog"""

    def testDialog(self):
        """Create dialog, change name and press OK"""
        self.dialog = LegendSelector.RenameCurveDialog(
            None, 'curve1', ['curve1', 'curve2', 'curve3'])
        self.dialog.open()
        self.qWaitForWindowExposed(self.dialog)
        self.keyClicks(self.dialog.lineEdit, 'changed')
        self.mouseClick(self.dialog.okButton, qt.Qt.LeftButton)
        self.qapp.processEvents()
        ret = self.dialog.result()
        self.assertEqual(ret, qt.QDialog.Accepted)
        newName = self.dialog.getText()
        self.assertEqual(newName, 'curve1changed')
        del self.dialog


def suite():
    test_suite = unittest.TestSuite()
    for TestClass in (TestLegendSelector, TestRenameCurveDialog):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestClass))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
