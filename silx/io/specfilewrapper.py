# coding: utf-8
#/*##########################################################################
# Copyright (C) 2016 European Synchrotron Radiation Facility
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
#############################################################################*/
"""This module provides a to backward compatibility layer with previous
specfile wrapper for :mod:`specfile`"""

from silx.io.specfile import SpecFile, Scan, MCA

class Specfile(SpecFile):
    def __init__(self, filename):
        SpecFile.__init__(self, filename)

    def __getitem__(self, key):
        """Get scan by 0-based index

        :param key: 0-based scan index
        :type key: int

        :return: Scan
        """
        if not isinstance(key, int):
            raise TypeError("Scan index must be an integer")

        return myscandata(scanobj=SpecFile.__getitem__(self, key))

        scan_index = key
        # allow negative index, like lists
        if scan_index < 0:
            scan_index = len(self) + scan_index

        if not 0 <= scan_index < len(self):
            msg = "Scan index must be in range 0-%d" % (len(self) - 1)
            raise IndexError(msg)

        return myscandata(self, scan_index)

    def select(self, key):
        """Get scan by ``n.m`` key

        :param key: ``"s.o"`` (scan number, scan order)
        :type key: str
        :return: Scan
        """
        msg = "Key must be a string 'N.M' with N being the scan"
        msg += " number and M the order (eg '2.3')."

        if not hasattr(key, "lower") or not "." in key:
            raise TypeError(msg)

        try:
            (number, order) = map(int, key.split("."))
            scan_index = self.index(number, order)
        except (ValueError, IndexError):
            # self.index can raise an index error
            # int() can raise a value error
            raise KeyError(msg + "\nValid keys: '" +
                           "', '".join(self.keys()) + "'")
        except AttributeError:
            # e.g. "AttrErr: 'float' object has no attribute 'split'"
            raise TypeError(msg)

        if not 0 <= scan_index < len(self):
            msg = "Scan index must be in range 0-%d" % (len(self) - 1)
            raise IndexError(msg)

        return myscandata(self, scan_index)

    def scanno(self):
        """Return the number of scans in the SpecFile

        This is an alias for :meth:`__len__`, for compatibility with the old
        specfile wrapper API.
        """
        return len(self)

    def allmotors(self, scan_index=0):
        """motor_names(scan_index=0)

        This is an alias for :meth:`motor_names`, for compatibility with
        the old specfile wrapper API.
        """
        return self.motor_names(scan_index)


class myscandata(Scan):
    pass

