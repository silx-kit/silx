# /*##########################################################################
#
# Copyright (c) 2018 European Synchrotron Radiation Facility
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
"""This module allows to run a function in Qt main thread from another thread
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "09/03/2018"


from concurrent.futures import Future

from .. import qt


class _QtExecutor(qt.QObject):
    """Executor of tasks in Qt main thread"""

    __sigSubmit = qt.Signal(Future, object, tuple, dict)
    """Signal used to run tasks."""

    def __init__(self):
        super(_QtExecutor, self).__init__(parent=None)

        # Makes sure the executor lives in the main thread
        app = qt.QApplication.instance()
        assert app is not None
        mainThread = app.thread()
        if self.thread() != mainThread:
            self.moveToThread(mainThread)

        self.__sigSubmit.connect(self.__run)

    def submit(self, fn, *args, **kwargs):
        """Submit fn(*args, **kwargs) to Qt main thread

        :param callable fn: Function to call in main thread
        :return: Future object to retrieve result
        :rtype: concurrent.future.Future
        """
        future = Future()
        self.__sigSubmit.emit(future, fn, args, kwargs)
        return future

    def __run(self, future, fn, args, kwargs):
        """Run task in Qt main thread

        :param concurrent.future.Future future:
        :param callable fn: Function to run
        :param tuple args: Arguments
        :param dict kwargs: Keyword arguments
        """
        if not future.set_running_or_notify_cancel():
            return

        try:
            result = fn(*args, **kwargs)
        except BaseException as e:
            future.set_exception(e)
        else:
            future.set_result(result)


_executor = None
"""QObject running the tasks in main thread"""


def submitToQtMainThread(fn, *args, **kwargs):
    """Run fn(args, kwargs) in Qt's main thread.

    If not called from the main thread, this is run asynchronously.

    :param callable fn: Function to call in main thread.
    :return: A future object to retrieve the result
    :rtype: concurrent.future.Future
    """
    global _executor
    if _executor is None:  # Lazy-loading
        _executor = _QtExecutor()

    return _executor.submit(fn, *args, **kwargs)
