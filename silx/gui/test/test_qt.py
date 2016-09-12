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
"""Basic test of Qt bindings wrapper."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "12/09/2016"


import os.path
import unittest

from silx.testutils import temp_dir
from silx.gui.testutils import TestCaseQt

from silx.gui import qt


class TestQtWrapper(unittest.TestCase):
    """Minimalistic test to check that Qt has been loaded."""

    def testQObject(self):
        """Test that QObject is there."""
        obj = qt.QObject()
        self.assertTrue(obj is not None)


class TestLoadUi(TestCaseQt):
    """Test loadUi function"""

    TEST_UI = """<?xml version="1.0" encoding="UTF-8"?>
    <ui version="4.0">
     <class>MainWindow</class>
     <widget class="QMainWindow" name="MainWindow">
      <property name="geometry">
       <rect>
        <x>0</x>
        <y>0</y>
        <width>293</width>
        <height>296</height>
       </rect>
      </property>
      <property name="windowTitle">
       <string>Test loadUi</string>
      </property>
      <widget class="QWidget" name="centralwidget">
       <widget class="QPushButton" name="pushButton">
        <property name="geometry">
         <rect>
          <x>10</x>
          <y>10</y>
          <width>89</width>
          <height>27</height>
         </rect>
        </property>
        <property name="text">
         <string>Button 1</string>
        </property>
       </widget>
       <widget class="QPushButton" name="pushButton_2">
        <property name="geometry">
         <rect>
          <x>10</x>
          <y>50</y>
          <width>89</width>
          <height>27</height>
         </rect>
        </property>
        <property name="text">
         <string>Button 2</string>
        </property>
       </widget>
      </widget>
      <widget class="QMenuBar" name="menubar">
       <property name="geometry">
        <rect>
         <x>0</x>
         <y>0</y>
         <width>293</width>
         <height>25</height>
        </rect>
       </property>
      </widget>
      <widget class="QStatusBar" name="statusbar"/>
     </widget>
     <resources/>
     <connections/>
    </ui>
    """

    def testLoadUi(self):
        """Create a QMainWindow from an ui file"""
        with temp_dir() as tmp:
            uifile = os.path.join(tmp, "test.ui")

            # write file
            with open(uifile, mode='w') as f:
                f.write(self.TEST_UI)

            class TestMainWindow(qt.QMainWindow):
                def __init__(self, parent=None):
                    super(TestMainWindow, self).__init__(parent)
                    qt.loadUi(uifile, self)

            testMainWindow = TestMainWindow()
            testMainWindow.show()
            self.qWaitForWindowExposed(testMainWindow)

            testMainWindow.setAttribute(qt.Qt.WA_DeleteOnClose)
            testMainWindow.close()


def suite():
    test_suite = unittest.TestSuite()
    for TestCaseCls in (TestQtWrapper, TestLoadUi):
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromTestCase(TestCaseCls))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
