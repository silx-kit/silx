# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Basic tests for :class:`FitWidget`"""

import unittest

from silx.gui.utils.testutils import TestCaseQt

from ... import qt
from .. import FitWidget

from ....math.fit.fittheory import FitTheory
from ....math.fit.fitmanager import FitManager

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "05/12/2016"


class TestFitWidget(TestCaseQt):
    """Basic test for FitWidget"""

    def setUp(self):
        super(TestFitWidget, self).setUp()
        self.fit_widget = FitWidget()
        self.fit_widget.show()
        self.qWaitForWindowExposed(self.fit_widget)

    def tearDown(self):
        self.fit_widget.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.fit_widget.close()
        del self.fit_widget
        super(TestFitWidget, self).tearDown()

    def testShow(self):
        pass

    def testInteract(self):
        self.mouseClick(self.fit_widget, qt.Qt.LeftButton)
        self.keyClick(self.fit_widget, qt.Qt.Key_Enter)
        self.qapp.processEvents()

    def testCustomConfigWidget(self):
        class CustomConfigWidget(qt.QDialog):
            def __init__(self):
                qt.QDialog.__init__(self)
                self.setModal(True)
                self.ok = qt.QPushButton("ok", self)
                self.ok.clicked.connect(self.accept)
                cancel = qt.QPushButton("cancel", self)
                cancel.clicked.connect(self.reject)
                layout = qt.QVBoxLayout(self)
                layout.addWidget(self.ok)
                layout.addWidget(cancel)
                self.output = {"hello": "world"}

        def fitfun(x, a, b):
            return a * x + b

        x = list(range(0, 100))
        y = [fitfun(x_, 2, 3) for x_ in x]

        def conf(**kw):
            return {"spam": "eggs",
                    "hello": "world!"}

        theory = FitTheory(
            function=fitfun,
            parameters=["a", "b"],
            configure=conf)

        fitmngr = FitManager()
        fitmngr.setdata(x, y)
        fitmngr.addtheory("foo", theory)
        fitmngr.addtheory("bar", theory)
        fitmngr.addbgtheory("spam", theory)

        fw = FitWidget(fitmngr=fitmngr)
        fw.associateConfigDialog("spam", CustomConfigWidget(),
                                 theory_is_background=True)
        fw.associateConfigDialog("foo", CustomConfigWidget())
        fw.show()
        self.qWaitForWindowExposed(fw)

        fw.bgconfigdialogs["spam"].accept()
        self.assertTrue(fw.bgconfigdialogs["spam"].result())

        self.assertEqual(fw.bgconfigdialogs["spam"].output,
                         {"hello": "world"})

        fw.bgconfigdialogs["spam"].reject()
        self.assertFalse(fw.bgconfigdialogs["spam"].result())

        fw.configdialogs["foo"].accept()
        self.assertTrue(fw.configdialogs["foo"].result())

        # todo: figure out how to click fw.configdialog.ok to close dialog
        # open dialog
        # self.mouseClick(fw.guiConfig.FunConfigureButton, qt.Qt.LeftButton)
        # clove dialog
        # self.mouseClick(fw.configdialogs["foo"].ok, qt.Qt.LeftButton)
        # self.qapp.processEvents()


def suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestFitWidget))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
