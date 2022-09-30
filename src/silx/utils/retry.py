# /*##########################################################################
# Copyright (C) 2016-2017 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""
This module provides utility methods for retrying methods until they
no longer fail.
"""

__authors__ = ["W. de Nolf"]
__license__ = "MIT"
__date__ = "05/02/2020"


import time
import inspect
from functools import wraps
from contextlib import contextmanager
import multiprocessing
from queue import Empty


RETRY_PERIOD = 0.01


class RetryTimeoutError(TimeoutError):
    pass


class RetryError(RuntimeError):
    pass


def _default_retry_on_error(e):
    """
    :param BaseException e:
    :returns bool:
    """
    return isinstance(e, RetryError)


@contextmanager
def _handle_exception(retry_state):
    try:
        yield
    except BaseException as e:
        retry_on_error = retry_state.get("retry_on_error")
        if retry_on_error is not None and retry_on_error(e):
            retry_state["exception"] = e
        else:
            raise


def _retry_loop(retry_timeout=None, retry_period=None, retry_on_error=None):
    """Iterator which is endless or ends with an RetryTimeoutError.
    It yields a dictionary which can be used to influence the loop.

    :param num retry_timeout:
    :param num retry_period: sleep before retry
    :param callable or None retry_on_error: checks whether an exception is
                                            eligible for retry
    """
    has_timeout = retry_timeout is not None
    if has_timeout:
        t0 = time.time()
    else:
        t0 = None
    retry_state = {"t0": t0, "exception": None, "retry_on_error": retry_on_error}
    while True:
        yield retry_state
        if retry_period is not None:
            time.sleep(retry_period)
        if has_timeout and (time.time() - t0) > retry_timeout:
            err_msg = "%s seconds" % retry_timeout
            cause = retry_state.get("exception")
            raise RetryTimeoutError(err_msg) from cause


def retry(
    retry_timeout=None, retry_period=None, retry_on_error=_default_retry_on_error
):
    """Decorator for a method that needs to be executed until it not longer
    fails or until `retry_on_error` returns False.

    The decorator arguments can be overriden by using them when calling the
    decorated method.

    Generator functions are required to have a `start_index` argument which allows
    the method to start iterating from the last failure when called on retry.

    :param num retry_timeout:
    :param num retry_period: sleep before retry
    :param callable or None retry_on_error: checks whether an exception is
                                            eligible for retry
    """

    if retry_period is None:
        retry_period = RETRY_PERIOD

    def decorator(method):
        if inspect.isgeneratorfunction(method):
            if "start_index" not in inspect.signature(method).parameters:
                raise TypeError(
                    "The generator function '%s' needs a `start_index` named argument because it is wrapped with the `retry` decorator."
                    % method.__name__
                )

            @wraps(method)
            def wrapper(*args, **kw):
                _retry_timeout = kw.pop("retry_timeout", retry_timeout)
                _retry_period = kw.pop("retry_period", retry_period)
                _retry_on_error = kw.pop("retry_on_error", retry_on_error)
                start_index = kw.pop("start_index", 0)
                if start_index is None:
                    start_index = 0
                for retry_state in _retry_loop(
                    retry_timeout=_retry_timeout,
                    retry_period=_retry_period,
                    retry_on_error=_retry_on_error,
                ):
                    with _handle_exception(retry_state):
                        oretry_on_error = retry_state["retry_on_error"]
                        for result in method(*args, start_index=start_index, **kw):
                            start_index += 1
                            retry_state["retry_on_error"] = None
                            # any exception here will NOT cause a retry
                            yield result
                            # restart the retry loop
                            if retry_state["t0"] is not None:
                                retry_state["t0"] = time.time()
                            retry_state["retry_on_error"] = oretry_on_error
                        return

        else:

            @wraps(method)
            def wrapper(*args, **kw):
                _retry_timeout = kw.pop("retry_timeout", retry_timeout)
                _retry_period = kw.pop("retry_period", retry_period)
                _retry_on_error = kw.pop("retry_on_error", retry_on_error)
                for retry_state in _retry_loop(
                    retry_timeout=_retry_timeout,
                    retry_period=_retry_period,
                    retry_on_error=_retry_on_error,
                ):
                    with _handle_exception(retry_state):
                        return method(*args, **kw)

        return wrapper

    return decorator


def retry_contextmanager(
    retry_timeout=None, retry_period=None, retry_on_error=_default_retry_on_error
):
    """Decorator to make a context manager from a method that needs to be
    entered until it no longer fails or until `retry_on_error` returns False.

    The decorator arguments can be overriden by using them when calling the
    decorated method.

    :param num retry_timeout:
    :param num retry_period: sleep before retry
    :param callable or None retry_on_error: checks whether an exception is
                                            eligible for retry
    """

    if retry_period is None:
        retry_period = RETRY_PERIOD

    def decorator(method):
        @wraps(method)
        def wrapper(*args, **kw):
            _retry_timeout = kw.pop("retry_timeout", retry_timeout)
            _retry_period = kw.pop("retry_period", retry_period)
            _retry_on_error = kw.pop("retry_on_error", retry_on_error)
            for retry_state in _retry_loop(
                retry_timeout=_retry_timeout,
                retry_period=_retry_period,
                retry_on_error=_retry_on_error,
            ):
                with _handle_exception(retry_state):
                    ctx = method(*args, **kw)
                    result = next(ctx)
                    retry_state["retry_on_error"] = None
                    yield result
                    try:
                        next(ctx)
                    except StopIteration:
                        return
                    else:
                        raise RuntimeError(str(method) + " should only yield once")

        return contextmanager(wrapper)

    return decorator


def _subprocess_main(queue, method, retry_on_error, *args, **kw):
    try:
        result = method(*args, **kw)
    except BaseException as e:
        if retry_on_error(e):
            # As the traceback gets lost, make sure the top-level
            # exception is RetryError
            e = RetryError(str(e))
        queue.put(e)
    else:
        queue.put(result)


def retry_in_subprocess(
    retry_timeout=None, retry_period=None, retry_on_error=_default_retry_on_error
):
    """Same as `retry` but it also retries segmentation faults.

    As subprocesses are spawned, you cannot use this decorator with the "@" syntax
    because the decorated method needs to be an attribute of a module:

    .. code-block:: python

        def _method(*args, **kw):
            ...

        method = retry_in_subprocess()(_method)

    :param num retry_timeout:
    :param num retry_period: sleep before retry
    :param callable or None retry_on_error: checks whether an exception is
                                            eligible for retry
    """

    if retry_period is None:
        retry_period = RETRY_PERIOD

    def decorator(method):
        @wraps(method)
        def wrapper(*args, **kw):
            _retry_timeout = kw.pop("retry_timeout", retry_timeout)
            _retry_period = kw.pop("retry_period", retry_period)
            _retry_on_error = kw.pop("retry_on_error", retry_on_error)

            ctx = multiprocessing.get_context("spawn")

            def start_subprocess():
                queue = ctx.Queue(maxsize=1)
                p = ctx.Process(
                    target=_subprocess_main,
                    args=(queue, method, retry_on_error) + args,
                    kwargs=kw,
                )
                p.start()
                return p, queue

            def stop_subprocess(p):
                try:
                    p.kill()
                except AttributeError:
                    p.terminate()
                p.join()

            p, queue = start_subprocess()
            try:
                for retry_state in _retry_loop(
                    retry_timeout=_retry_timeout, retry_on_error=_retry_on_error
                ):
                    with _handle_exception(retry_state):
                        if not p.is_alive():
                            p, queue = start_subprocess()
                        try:
                            result = queue.get(block=True, timeout=_retry_period)
                        except Empty:
                            pass
                        except ValueError:
                            pass
                        else:
                            if isinstance(result, BaseException):
                                stop_subprocess(p)
                                raise result
                            else:
                                return result
            finally:
                stop_subprocess(p)

        return wrapper

    return decorator
