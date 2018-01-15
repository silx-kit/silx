# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
"""ThreadPoolPushButton module
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "13/10/2016"

import logging
from .. import qt
from .WaitingPushButton import WaitingPushButton


_logger = logging.getLogger(__name__)


class _Wrapper(qt.QRunnable):
    """Wrapper to allow to call a function into a `QThreadPool` and
    sending signals during the life cycle of the object"""

    def __init__(self, signalHolder, function, args, kwargs):
        """Constructor"""
        super(_Wrapper, self).__init__()
        self.__signalHolder = signalHolder
        self.__callable = function
        self.__args = args
        self.__kwargs = kwargs

    def run(self):
        holder = self.__signalHolder
        holder.started.emit()
        try:
            result = self.__callable(*self.__args, **self.__kwargs)
            holder.succeeded.emit(result)
        except Exception as e:
            module = self.__callable.__module__
            name = self.__callable.__name__
            _logger.error("Error while executing callable %s.%s.", module, name, exc_info=True)
            holder.failed.emit(e)
        finally:
            holder.finished.emit()
        holder._sigReleaseRunner.emit(self)

    def autoDelete(self):
        """Returns true to ask the QThreadPool to manage the life cycle of
        this QRunner."""
        return True


class ThreadPoolPushButton(WaitingPushButton):
    """
    ThreadPoolPushButton provides a simple push button to execute
    a threaded task with user feedback when the task is running.

    The task can be defined with the method `setCallable`. It takes a python
    function and arguments as parameters.

    WARNING: This task is run in a separate thread.

    Everytime the button is pushed a new runner is created to execute the
    function with defined arguments. An animated waiting icon is displayed
    to show the activity. By default the button is disabled when an execution
    is requested. This behaviour can be disabled by using
    `setDisabledWhenWaiting`.

    When the button is clicked a `beforeExecuting` signal is sent from the
    Qt main thread. Then the task is started in a thread pool and the following
    signals are emitted from the thread pool. Right before calling the
    registered callable, the widget emits a `started` signal.
    When the task ends, its result is emitted by the `succeeded` signal, but
    if it fails the signal `failed` is emitted with the resulting exception.
    At the end, the `finished` signal is emitted.

    The task can be programatically executed by using `executeCallable`.

    >>> # Compute a value
    >>> import math
    >>> button = ThreadPoolPushButton(text="Compute 2^16")
    >>> button.setCallable(math.pow, 2, 16)
    >>> button.succeeded.connect(print) # python3

    .. image:: img/ThreadPoolPushButton.png

    >>> # Compute a wrong value
    >>> import math
    >>> button = ThreadPoolPushButton(text="Compute sqrt(-1)")
    >>> button.setCallable(math.sqrt, -1)
    >>> button.failed.connect(print) # python3
    """

    def __init__(self, parent=None, text=None, icon=None):
        """Constructor

        :param str text: Text displayed on the button
        :param qt.QIcon icon: Icon displayed on the button
        :param qt.QWidget parent: Parent of the widget
        """
        WaitingPushButton.__init__(self, parent=parent, text=text, icon=icon)
        self.__callable = None
        self.__args = None
        self.__kwargs = None
        self.__runnerCount = 0
        self.__runnerSet = set([])
        self.clicked.connect(self.executeCallable)
        self.finished.connect(self.__runnerFinished)
        self._sigReleaseRunner.connect(self.__releaseRunner)

    beforeExecuting = qt.Signal()
    """Signal emitted just before execution of the callable by the main Qt
    thread. In synchronous mode (direct mode), it can be used to define
    dynamically `setCallable`, or to execute something in the Qt thread before
    the execution, or both."""

    started = qt.Signal()
    """Signal emitted from the thread pool when the defined callable is
    started.

    WARNING: This signal is emitted from the thread performing the task, and
    might be received after the registered callable has been called. If you
    want to perform some initialisation or set the callable to run, use the
    `beforeExecuting` signal instead.
    """

    finished = qt.Signal()
    """Signal emitted from the thread pool when the defined callable is
    finished"""

    succeeded = qt.Signal(object)
    """Signal emitted from the thread pool when the callable exit with a
    success.

    The parameter of the signal is the result returned by the callable.
    """

    failed = qt.Signal(object)
    """Signal emitted emitted from the thread pool when the callable raises an
    exception.

    The parameter of the signal is the raised exception.
    """

    _sigReleaseRunner = qt.Signal(object)
    """Callback to release runners"""

    def __runnerStarted(self):
        """Called when a runner is started.

        Count the number of executed tasks to change the state of the widget.
        """
        self.__runnerCount += 1
        if self.__runnerCount > 0:
            self.wait()

    def __runnerFinished(self):
        """Called when a runner is finished.

        Count the number of executed tasks to change the state of the widget.
        """
        self.__runnerCount -= 1
        if self.__runnerCount <= 0:
            self.stopWaiting()

    @qt.Slot()
    def executeCallable(self):
        """Execute the defined callable in QThreadPool.

        First emit a `beforeExecuting` signal.
        If callable is not defined, nothing append.
        If a callable is defined, it will be started
        as a new thread using the `QThreadPool` system. At start of the thread
        the `started` will be emitted. When the callable returns a result it
        is emitted by the `succeeded` signal. If the callable fail, the signal
        `failed` is emitted with the resulting exception. Then the `finished`
        signal is emitted.
        """
        self.beforeExecuting.emit()
        if self.__callable is None:
            return
        self.__runnerStarted()
        runner = self._createRunner(self.__callable, self.__args, self.__kwargs)
        qt.silxGlobalThreadPool().start(runner)
        self.__runnerSet.add(runner)

    def __releaseRunner(self, runner):
        self.__runnerSet.remove(runner)

    def hasPendingOperations(self):
        return len(self.__runnerSet) > 0

    def _createRunner(self, function, args, kwargs):
        """Create a QRunnable from a callable object.

        :param callable function: A callable Python object.
        :param List args: List of arguments to call the function.
        :param dict kwargs: Dictionary of arguments used to call the function.
        :rtpye: qt.QRunnable
        """
        runnable = _Wrapper(self, function, args, kwargs)
        return runnable

    def setCallable(self, function, *args, **kwargs):
        """Define a callable which will be executed on QThreadPool everytime
        the button is clicked.

        To retrieve the results, connect to the `succeeded` signal.

        WARNING: The callable will be called in a separate thread.

        :param callable function: A callable Python object
        :param List args: List of arguments to call the function.
        :param dict kwargs: Dictionary of arguments used to call the function.
        """
        self.__callable = function
        self.__args = args
        self.__kwargs = kwargs
