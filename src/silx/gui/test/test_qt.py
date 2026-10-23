# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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
__date__ = "05/12/2016"


import os.path
import subprocess
import sys
import textwrap
import unittest
import pytest

from silx.test.utils import temp_dir
from silx.gui.utils.testutils import TestCaseQt

from silx.gui import qt

try:
    from silx.gui.qt import inspect as qt_inspect
except ImportError:
    qt_inspect = None


class TestQtWrapper(unittest.TestCase):
    """Minimalistic test to check that Qt has been loaded."""

    def testQObject(self):
        """Test that QObject is there."""
        obj = qt.QObject()
        self.assertTrue(obj is not None)


@pytest.mark.skipif(qt.BINDING == "PyQt5", reason="Qt6 only test")
@pytest.mark.parametrize("version", ["6.4.3", "6.5.0", "6.10.0"])
def test_qt6_runtime_version(version):
    """Check the runtime version before the wrapper imports GUI modules."""
    script = textwrap.dedent(
        f"""
        import importlib
        import sys

        core = importlib.import_module("{qt.BINDING}.QtCore")
        core.qVersion = lambda: "{version}"

        if "{version}" == "6.4.3":
            try:
                import silx.gui.qt
            except RuntimeError as error:
                assert str(error) == "Qt v6.4.3 is not supported, please upgrade it."
                assert "{qt.BINDING}.QtGui" not in sys.modules
                assert "{qt.BINDING}.QtWidgets" not in sys.modules
            else:
                raise AssertionError("Unsupported Qt runtime was accepted")
        else:
            from silx.gui import qt
            assert qt.BINDING == "{qt.BINDING}"
            assert qt.qVersion() == "{version}"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(qt.BINDING == "PyQt5", reason="Qt6 only test")
@pytest.mark.parametrize("runtime_version", ["6.4.3", "6.10.0"])
def test_qt6_binding_version(runtime_version):
    """Keep rejecting old bindings, independently of the runtime version."""
    script = textwrap.dedent(
        f"""
        import importlib

        binding = importlib.import_module("{qt.BINDING}")
        core = importlib.import_module("{qt.BINDING}.QtCore")
        core.qVersion = lambda: "{runtime_version}"
        if "{qt.BINDING}" == "PySide6":
            binding.__version__ = "6.4.3"
        else:
            core.PYQT_VERSION = 0x60403
            core.PYQT_VERSION_STR = "6.4.3"

        try:
            import silx.gui.qt
        except RuntimeError as error:
            assert str(error) == "{qt.BINDING} v6.4.3 is not supported, please upgrade it."
        else:
            raise AssertionError("Unsupported Qt binding was accepted")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(qt.BINDING != "PyQt5", reason="PyQt5 only test")
def test_qt5_runtime_version():
    """The Qt6 runtime minimum does not introduce a Qt5 runtime minimum."""
    script = textwrap.dedent(
        """
        from PyQt5 import QtCore

        QtCore.qVersion = lambda: "5.12.0"
        from silx.gui import qt
        assert qt.BINDING == "PyQt5"
        assert qt.qVersion() == "5.12.0"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
       <widget class="Line" name="line">
        <property name="geometry">
         <rect>
          <x>10</x>
          <y>90</y>
          <width>118</width>
          <height>3</height>
         </rect>
        </property>
        <property name="orientation">
         <enum>Qt::Horizontal</enum>
        </property>
       </widget>
       <widget class="Line" name="line_2">
        <property name="geometry">
         <rect>
          <x>150</x>
          <y>20</y>
          <width>3</width>
          <height>61</height>
         </rect>
        </property>
        <property name="orientation">
         <enum>Qt::Vertical</enum>
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
            with open(uifile, mode="w") as f:
                f.write(self.TEST_UI)

            class TestMainWindow(qt.QMainWindow):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    qt.loadUi(uifile, self)

            testMainWindow = TestMainWindow()
            testMainWindow.show()
            self.qWaitForWindowExposed(testMainWindow)

            testMainWindow.setAttribute(qt.Qt.WA_DeleteOnClose)
            testMainWindow.close()


class TestQtInspect(unittest.TestCase):
    """Test functions of silx.gui.qt.inspect module"""

    def test(self):
        """Test functions of silx.gui.qt.inspect module"""
        self.assertIsNotNone(qt_inspect)

        parent = qt.QObject()

        self.assertTrue(qt_inspect.isValid(parent))
        self.assertTrue(qt_inspect.createdByPython(parent))
        self.assertTrue(qt_inspect.ownedByPython(parent))

        obj = qt.QObject(parent)

        self.assertTrue(qt_inspect.isValid(obj))
        self.assertTrue(qt_inspect.createdByPython(obj))
        self.assertFalse(qt_inspect.ownedByPython(obj))

        del parent
        self.assertFalse(qt_inspect.isValid(obj))


@pytest.mark.skipif(qt.BINDING != "PyQt5", reason="PyQt5 only test")
def test_exec_():
    """Test the exec_ is still useable with Qt5 bindings"""
    klasses = [
        # QtWidgets
        qt.QApplication,
        qt.QColorDialog,
        qt.QDialog,
        qt.QErrorMessage,
        qt.QFileDialog,
        qt.QFontDialog,
        qt.QInputDialog,
        qt.QMenu,
        qt.QMessageBox,
        qt.QProgressDialog,
        # QtCore
        qt.QCoreApplication,
        qt.QEventLoop,
        qt.QThread,
    ]
    for klass in klasses:
        assert hasattr(klass, "exec") and callable(klass.exec), (
            "%s.exec missing" % klass.__name__
        )
        assert hasattr(klass, "exec_") and callable(klass.exec_), (
            "%s.exec_ missing" % klass.__name__
        )
