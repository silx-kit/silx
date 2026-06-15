# /*##########################################################################
#
# Copyright (c) 2024 European Synchrotron Radiation Facility
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
import os
import numpy
from packaging.version import Version

NP_OPTIONAL_COPY = False if Version(numpy.version.version).major < 2 else None


def nfs_cache_refresh(dirname: str) -> None:
    """Perform ls -l dirname operation to refresh NFS cache"""
    try:
        with os.scandir(dirname) as it:
            for entry in it:
                # This forces a fresh stat call, similar to "ls -l"
                _ = entry.stat()
    except Exception:
        pass


class IgnoreArgPartial:
    """
    Alternative implementation of ``functools.partial`` but the partial
    function ignores any arguments. This was done because the signature
    is not correct which causes PyQt to pass arguments when it shouldn't.

    This class was introduced as a workaround for segfaults observed with
    ``functools.partial`` in Qt/Python object destruction and garbage
    collection scenarios since Python 3.13.

    This is possible caused by the partial keeping a reference to the
    callstack in which the partial funcion was created.
    """

    __slots__ = ("_func", "_args", "_kwargs")

    def __init__(self, func, *args, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *_, **__):
        return self._func(*self._args, **self._kwargs)

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"{self._func!r}, "
            f"args={self._args!r}, "
            f"kwargs={self._kwargs!r})"
        )
