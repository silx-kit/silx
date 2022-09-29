#!/usr/bin/env python
# /*##########################################################################
#
# Copyright (c) 2012 University of North Carolina at Chapel Hill, Luke Campagnola
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
"""This module contains utils relative to qt Signal
"""

from silx.gui import qt
import weakref
from time import time
from silx.gui.utils import concurrent

__all__ = ['SignalProxy']
__authors__ = ['L. Campagnola', 'M. Liberty']
__license__ = "MIT"


class SignalProxy(qt.QObject):
    """
    This peace of code come from pyqtgraph
    Object which collects rapid-fire signals and condenses them
    into a single signal or a rate-limited stream of signals.
    Used, for example, to prevent a SpinBox from generating multiple
    signals when the mouse wheel is rolled over it.

    Emits sigDelayed after input signals have stopped for a certain period of time.
    """

    sigDelayed = qt.Signal(object)

    def __init__(self, signal, delay=0.3, rateLimit=0, slot=None):
        """Initialization arguments:
        signal - a bound Signal or pyqtSignal instance
        delay - Time (in seconds) to wait for signals to stop before emitting (default 0.3s)
        slot - Optional function to connect sigDelayed to.
        rateLimit - (signals/sec) if greater than 0, this allows signals to stream out at a
                    steady rate while they are being received.
        """

        qt.QObject.__init__(self)
        signal.connect(self.signalReceived)
        self.signal = signal
        self.delay = delay
        self.rateLimit = rateLimit
        self.args = None
        self.timer = qt.QTimer()
        self.timer.timeout.connect(self.flush)
        self.blockSignal = False
        self.slot = weakref.ref(slot)
        self.lastFlushTime = None
        if slot is not None:
            self.sigDelayed.connect(slot)

    def setDelay(self, delay):
        self.delay = delay

    def signalReceived(self, *args):
        """Received signal. Cancel previous timer and store args to be forwarded later."""
        if self.blockSignal:
            return
        self.args = args
        if self.rateLimit == 0:
            concurrent.submitToQtMainThread(self.timer.stop)
            concurrent.submitToQtMainThread(self.timer.start, (self.delay * 1000) + 1)
        else:
            now = time()
            if self.lastFlushTime is None:
                leakTime = 0
            else:
                lastFlush = self.lastFlushTime
                leakTime = max(0, (lastFlush + (1.0 / self.rateLimit)) - now)

            concurrent.submitToQtMainThread(self.timer.stop)
            concurrent.submitToQtMainThread(self.timer.start, (min(leakTime, self.delay) * 1000) + 1)
            # self.timer.stop()
            # self.timer.start((min(leakTime, self.delay) * 1000) + 1)

    def flush(self):
        """If there is a signal queued up, send it now."""
        if self.args is None or self.blockSignal:
            return False
        args, self.args = self.args, None
        concurrent.submitToQtMainThread(self.timer.stop)
        self.lastFlushTime = time()
        # self.emit(self.signal, *self.args)
        concurrent.submitToQtMainThread(self.sigDelayed.emit, args)
        # self.sigDelayed.emit(args)
        return True

    def disconnect(self):
        self.blockSignal = True
        try:
            self.signal.disconnect(self.signalReceived)
        except:
            pass
        try:
            self.sigDelayed.disconnect(self.slot)
        except:
            pass


if __name__ == '__main__':
    app = qt.QApplication([])
    win = qt.QMainWindow()
    spin = qt.QSpinBox()
    win.setCentralWidget(spin)
    win.show()


    def fn(*args):
        print("Raw signal:", args)


    def fn2(*args):
        print("Delayed signal:", args)


    spin.valueChanged.connect(fn)
    # proxy = proxyConnect(spin, QtCore.SIGNAL('valueChanged(int)'), fn)
    proxy = SignalProxy(spin.valueChanged, delay=0.5, slot=fn2)
