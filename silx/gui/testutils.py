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
"""Helper class to write Qt widget unittests."""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "02/03/2016"


import gc
import logging
import unittest
import time

logging.basicConfig()
_logger = logging.getLogger(__name__)

from silx.gui import qt

if qt.BINDING == 'PySide':
    from PySide.QtTest import QTest
elif qt.BINDING == 'PyQt5':
    from PyQt5.QtTest import QTest
elif qt.BINDING == 'PyQt4':
    from PyQt4.QtTest import QTest
else:
    raise ImportError('Unsupported Qt bindings')

# Qt4/Qt5 compatibility wrapper
if qt.BINDING in ('PySide', 'PyQt4'):
    _logger.info("QTest.qWaitForWindowExposed not available," +
                 "using QTest.qWaitForWindowShown instead.")

    def qWaitForWindowExposed(window, timeout=None):
        """Mimic QTest.qWaitForWindowExposed for Qt4."""
        QTest.qWaitForWindowShown(window)
        return True
else:
    qWaitForWindowExposed = QTest.qWaitForWindowExposed


def qWaitForWindowExposedAndActivate(window, timeout=None):
    """Waits until the window is shown in the screen.

    It also activates the window and raises it.

    See QTest.qWaitForWindowExposed for details.
    """
    if timeout is None:
        result = qWaitForWindowExposed(window)
    else:
        result = qWaitForWindowExposed(window, timeout)

    if result:
        # Makes sure window is active and on top
        window.activateWindow()
        window.raise_()

    return result


# Placeholder for QApplication
_qapp = None


class TestCaseQt(unittest.TestCase):
    """Base class to write test for Qt stuff.

    It creates a QApplication before running the tests.
    WARNING: The QApplication is shared by all tests, which might have side
    effects.

    After each test, this class is checking for widgets remaining alive.
    To allow some widgets to remain alive at the end of a test, set the
    allowedLeakingWidgets attribute to the number of widgets that can remain
    alive at the end of the test.
    With PySide, this test is not run for now as it seems PySide
    is leaking widgets internally.

    All keyboard and mouse event simulation methods call qWait(20) after
    simulating the event (as QTest does on Mac OSX).
    This was introduced to fix issues with continuous integration tests
    running with Xvfb on Linux.
    """

    DEFAULT_TIMEOUT_WAIT = 100
    """Default timeout for qWait"""

    TIMEOUT_WAIT = 0
    """Extra timeout in millisecond to add to qSleep, qWait and
    qWaitForWindowExposed.

    Intended purpose is for debugging, to add extra time to waits in order to
    allow to view the tested widgets.
    """

    @classmethod
    def setUpClass(cls):
        """Makes sure Qt is inited"""
        global _qapp
        if _qapp is None:
            # Makes sure a QApplication exists and do it once for all
            _qapp = qt.QApplication.instance() or qt.QApplication([])

            # Create/delate a QWidget to make sure init of QDesktopWidget
            _dummyWidget = qt.QWidget()
            _dummyWidget.setAttribute(qt.Qt.WA_DeleteOnClose)
            _dummyWidget.show()
            _dummyWidget.close()
            _qapp.processEvents()

    def setUp(self):
        """Get the list of existing widgets."""
        self.allowedLeakingWidgets = 0
        self.__previousWidgets = self.qapp.allWidgets()

    def tearDown(self):
        """Test fixture checking that no more widgets exists."""
        gc.collect()

        widgets = [widget for widget in self.qapp.allWidgets()
                   if widget not in self.__previousWidgets]
        del self.__previousWidgets

        if qt.BINDING == 'PySide':
            return  # Do not test for leaking widgets with PySide

        allowedLeakingWidgets = self.allowedLeakingWidgets
        self.allowedLeakingWidgets = 0

        if widgets and len(widgets) <= allowedLeakingWidgets:
            _logger.info(
                '%s: %d remaining widgets after test' % (self.id(),
                                                         len(widgets)))

        if len(widgets) > allowedLeakingWidgets:
            raise RuntimeError(
                "Test ended with widgets alive: %s" % str(widgets))

    @property
    def qapp(self):
        """The QApplication currently running."""
        return qt.QApplication.instance()

    # Proxy to QTest

    Press = QTest.Press
    """Key press action code"""

    Release = QTest.Release
    """Key release action code"""

    Click = QTest.Click
    """Key click action code"""

    QTest = property(lambda self: QTest,
                     doc="""The Qt QTest class from the used Qt binding.""")

    def keyClick(self, widget, key, modifier=qt.Qt.NoModifier, delay=-1):
        """Simulate clicking a key.

        See QTest.keyClick for details.
        """
        QTest.keyClick(widget, key, modifier, delay)
        self.qWait(20)

    def keyClicks(self, widget, sequence, modifier=qt.Qt.NoModifier, delay=-1):
        """Simulate clicking a sequence of keys.

        See QTest.keyClick for details.
        """
        QTest.keyClicks(widget, sequence, modifier, delay)
        self.qWait(20)

    def keyEvent(self, action, widget, key,
                 modifier=qt.Qt.NoModifier, delay=-1):
        """Sends a Qt key event.

        See QTest.keyEvent for details.
        """
        QTest.keyEvent(action, widget, key, modifier, delay)
        self.qWait(20)

    def keyPress(self, widget, key, modifier=qt.Qt.NoModifier, delay=-1):
        """Sends a Qt key press event.

        See QTest.keyPress for details.
        """
        QTest.keyPress(widget, key, modifier, delay)
        self.qWait(20)

    def keyRelease(self, widget, key, modifier=qt.Qt.NoModifier, delay=-1):
        """Sends a Qt key release event.

        See QTest.keyRelease for details.
        """
        QTest.keyRelease(widget, key, modifier, delay)
        self.qWait(20)

    def mouseClick(self, widget, button, modifier=None, pos=None, delay=-1):
        """Simulate clicking a mouse button.

        See QTest.mouseClick for details.
        """
        if modifier is None:
            modifier = qt.Qt.KeyboardModifiers()
        pos = qt.QPoint(pos[0], pos[1]) if pos is not None else qt.QPoint()
        QTest.mouseClick(widget, button, modifier, pos, delay)
        self.qWait(20)

    def mouseDClick(self, widget, button, modifier=None, pos=None, delay=-1):
        """Simulate double clicking a mouse button.

        See QTest.mouseDClick for details.
        """
        if modifier is None:
            modifier = qt.Qt.KeyboardModifiers()
        pos = qt.QPoint(pos[0], pos[1]) if pos is not None else qt.QPoint()
        QTest.mouseDClick(widget, button, modifier, pos, delay)
        self.qWait(20)

    def mouseMove(self, widget, pos=None, delay=-1):
        """Simulate moving the mouse.

        See QTest.mouseMove for details.
        """
        pos = qt.QPoint(pos[0], pos[1]) if pos is not None else qt.QPoint()
        QTest.mouseMove(widget, pos, delay)
        self.qWait(20)

    def mousePress(self, widget, button, modifier=None, pos=None, delay=-1):
        """Simulate pressing a mouse button.

        See QTest.mousePress for details.
        """
        if modifier is None:
            modifier = qt.Qt.KeyboardModifiers()
        pos = qt.QPoint(pos[0], pos[1]) if pos is not None else qt.QPoint()
        QTest.mousePress(widget, button, modifier, pos, delay)
        self.qWait(20)

    def mouseRelease(self, widget, button, modifier=None, pos=None, delay=-1):
        """Simulate releasing a mouse button.

        See QTest.mouseRelease for details.
        """
        if modifier is None:
            modifier = qt.Qt.KeyboardModifiers()
        pos = qt.QPoint(pos[0], pos[1]) if pos is not None else qt.QPoint()
        QTest.mouseRelease(widget, button, modifier, pos, delay)
        self.qWait(20)

    def qSleep(self, ms):
        """Sleep for ms milliseconds, blocking the execution of the test.

        See QTest.qSleep for details.
        """
        QTest.qSleep(ms + self.TIMEOUT_WAIT)

    def qWait(self, ms=None):
        """Waits for ms milliseconds, events will be processed.

        See QTest.qWait for details.
        """
        if ms is None:
            ms = self.DEFAULT_TIMEOUT_WAIT

        if qt.BINDING == 'PySide':
            # PySide has no qWait, provide a replacement
            timeout = int(ms)
            endTimeMS = int(time.time() * 1000) + timeout
            while timeout > 0:
                self.qapp.processEvents(qt.QEventLoop.AllEvents,
                                        maxtime=timeout)
                timeout = endTimeMS - int(time.time() * 1000)
        else:
            QTest.qWait(ms + self.TIMEOUT_WAIT)

    def qWaitForWindowExposed(self, window, timeout=None):
        """Waits until the window is shown in the screen.

        See QTest.qWaitForWindowExposed for details.
        """
        result = qWaitForWindowExposedAndActivate(window, timeout)

        if self.TIMEOUT_WAIT:
            QTest.qWait(self.TIMEOUT_WAIT)

        return result


def getQToolButtonFromAction(action):
    """Return a QToolButton corresponding to a QAction.

    :param QAction action: The QAction from which to get QToolButton.
    :return: A QToolButton associated to action or None.
    """
    for widget in action.associatedWidgets():
        if isinstance(widget, qt.QToolButton):
            return widget
    return None
