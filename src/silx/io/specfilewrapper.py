# /*#########################################################################
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
"""This module provides a backward compatibility layer with the legacy
specfile wrapper.

If you are starting a new project, please consider using :mod:`silx.io.specfile`
instead of this module.

If you want to use this module for an existing project that used the old
wrapper through PyMca, you can try replacing::

    from PyMca5.PyMcaIO import specfilewrapper

with::

    from silx.io import specfilewrapper

There might still be differences between this module and the old
wrapper, due to differences in the underlying implementation.
Any of these differences that break your code should be reported on
https://github.com/silx-kit/silx/issues

The documentation mentions only the methods and attributes that are different
from the ones in :class:`silx.io.specfile.SpecFile` and
:class:`silx.io.specfile.Scan`. You should refer to the documentation of these
base classes for more information.
"""
from silx.io.specfile import SpecFile, Scan

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "15/09/2016"


def _format_number_list(number_list):
    """Return string representation of a list of integers,
    using ``,`` as a separator and ``:`` as a range separator.
    """
    ret = ""
    first_in_range = number_list[0]
    last_in_range = number_list[0]
    previous = number_list[0]
    for number in number_list[1:]:
        if number - previous != 1:
            # reached end of range
            if last_in_range > first_in_range:
                ret += "%d:%d," % (first_in_range, last_in_range)
            # passed isolated number
            else:
                ret += "%d," % previous
            # reinitialize range
            first_in_range = number
            last_in_range = number
        else:
            # still inside a continuous range
            last_in_range = number

        previous = number

    # last number
    if last_in_range > first_in_range:
        ret += "%d:%d" % (first_in_range, last_in_range)
    else:
        ret += "%d" % previous

    return ret


class Specfile(SpecFile):
    """
    This class is a subclass of :class:`silx.io.specfile.SpecFile`.

    It redefines following methods:

        - :meth:`__getitem__`: returns a :class:`scandata` object instead of
          a :class:`silx.io.specfile.Scan` object
        - :meth:`list`: returns a string representation of a list instead of a
          list of integers

    Following methods are added:

        - :meth:`select`
        - :meth:`scanno`
        - :meth:`allmotors`
        - :meth:`epoch`
        - :meth:`title`
    """
    def __init__(self, filename):
        SpecFile.__init__(self, filename)

    def __getitem__(self, key):
        """Get scan by 0-based index

        :param key: 0-based scan index
        :type key: int

        :return: Scan
        :rtype: :class:`scandata`
        """
        if not isinstance(key, int):
            raise TypeError("Scan index must be an integer")

        scan_index = key
        # allow negative index, like lists
        if scan_index < 0:
            scan_index += len(self)

        if not 0 <= scan_index < len(self):
            msg = "Scan index must be in range 0-%d" % (len(self) - 1)
            raise IndexError(msg)

        return scandata(self, scan_index)

    def list(self):
        """Return a string representation of a list of scan numbers.

        The scans numbers are listed in the order in which they appear
        in the file. Continuous ranges of scan numbers are represented
        as ``first:last``.

        For instance, let's assume our specfile contains following scans:
        *1, 2, 3, 4, 5, 684, 685, 687, 688, 689, 700, 688, 688*.
        This method will then return::

            "1:5,684:685,687:689,700,688,688"
        """
        number_list = SpecFile.list(self)
        return _format_number_list(number_list)

    def select(self, key):
        """Get scan by ``n.m`` key

        :param key: ``"s.o"`` (scan number, scan order)
        :type key: str
        :return: Scan
        :rtype: :class:`scandata`
        """
        msg = "Key must be a string 'N.M' with N being the scan"
        msg += " number and M the order (eg '2.3')."

        if not hasattr(key, "lower") or "." not in key:
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

        return scandata(self, scan_index)

    def scanno(self):
        """Return the number of scans in the SpecFile

        This is an alias for :meth:`__len__`, for compatibility with the old
        specfile wrapper API.
        """
        return len(self)

    def allmotors(self, scan_index=0):
        """
        This is an alias for :meth:`motor_names`, for compatibility with
        the old specfile wrapper API.
        """
        return self.motor_names(scan_index)

    def epoch(self):
        """:return: Epoch, from last word on file header line *#E*
        :rtype: int
        :raise: ValueError if *#E* line not found in header or last
            word on *#E* cannot be converted to type *int*"""
        fh = self.file_header()
        for line in fh:
            if line.startswith("#E "):
                return int(line.split()[-1])
        raise ValueError("No #E header found in specfile")

    def title(self):
        """:return: Title, from second field on *#C* header line (field are
            strings separated by two spaces)
        :rtype: str
        :raise: ValueError if *#C* line not found in header or line is empty"""
        fh = self.file_header()
        for line in fh:
            if line.startswith("#C "):
                line1 = line.lstrip("#C ")
                return line1.split("  ")[0]
        raise ValueError("No #C header found in specfile")

    # # these functions exist in the old API but don't seem to be
    # # used, and are not easy to implement
    # def show(self):
    #     raise NotImplementedError
    #
    # def user(self):
    #     raise NotImplementedError
    #
    # def update(self):
    #     raise NotImplementedError


# PEP8 violation in class name is to respect old API
class scandata(Scan):  # noqa
    """
    This class is a subclass of :class:`silx.io.specfile.Scan`.

    It redefines following methods/attributes:

        - :meth:`data` becomes a method returning an array, instead of just
          an array
        - :meth:`mca`: becomes a method returning an array, instead of
          a :class:`silx.io.specfile.MCA` object
        - :meth:`header`: becomes a method returning a list of **scan**
          header lines (or a list of a single header line, if a key is
          specified), instead of just a list of all header lines

    Following methods are added:

        - :meth:`allmotors`
        - :meth:`allmotorpos`
        - :meth:`alllabels`
        - :meth:`cols`
        - :meth:`lines`
        - :meth:`command`
        - :meth:`date`
        - :meth:`datacol`
        - :meth:`dataline`
        - :meth:`fileheader`
        - :meth:`nbmca`
    """
    def __init__(self, specfile, scan_index):
        Scan.__init__(self, specfile, scan_index)

    def allmotors(self):
        """Return a list of all motor names (identical to
        :attr:`motor_names`).
        """
        return self.motor_names

    def allmotorpos(self):
        """Return a list of all motor positions (identical to
        :attr:`motor_positions`).
        """
        return self.motor_positions

    def alllabels(self):
        """
        Return a list of all labels (:attr:`labels`).
        """
        return self.labels

    def cols(self):
        """Return the number of data columns (number of detectors)"""
        return super(scandata, self).data.shape[1]

    def command(self):
        """Return the command called for this scan (``#S`` header line)"""
        return self._specfile.command(self._index)

    def data(self):
        """Return the data in this scan as a 2D numpy array.

        The first index corresponds to the columns/detectors in the original
        file, and the second index is the row index from the original file.
        Indices are 0-based.

        For instance, this is how you access the 18th data sample for the 3rd
        detector (assuming ``sc`` is your scan object):

            >>> scdata = sc.data()
            >>> data_sample = scdata[2, 17]"""
        return super(scandata, self).data

    def datacol(self, col):
        """Return a data column (all data for one detector)

        :param col: column number (1-based index)"""
        return super(scandata, self).data[col - 1, :]

    def dataline(self, line):
        """Return a data line (one sample for all detectors)

        :param line: line number (1-based index)"""
        return super(scandata, self).data[:, line - 1]

    def date(self):
        """Return the date from the scan header line ``#D``"""
        return self._specfile.date(self._index)

    def fileheader(self, key=''):  # noqa
        """Return a list of file header lines"""
        # key is there for compatibility
        return self.file_header

    def header(self, key=""):
        """Return a list of scan header lines if no key is specified.
        If a valid key is specified, return a list of a single header line.

        :param key: Header key (e.g. ``S, N, L, @CALIB``…)
            If ``key`` is an empty string, return complete list of scan header
            lines.
            If ``key`` does not match any header line, return empty list.
        :return: List of scan header lines
        :rtype: List[str]
        """
        if key.strip() == "":
            return self.scan_header
        if self.record_exists_in_hdr(key):
            prefix = "#" + key + " "
            # there is no leading @ in self.mca_header_dict keys
            key_mca_dict = key.lstrip("@") if key.startswith("@") else None
            if key_mca_dict in self.mca_header_dict:
                return [prefix + self.mca_header_dict[key_mca_dict]]
            elif key in self.scan_header_dict:
                return [prefix + self.scan_header_dict[key]]
            elif key in self.file_header_dict:
                return [prefix + self.file_header_dict[key]]
        elif self.record_exists_in_hdr("@" + key):
            # in case key is a mca header key without the @
            if key in self.mca_header_dict:
                prefix = "#@" + key + " "
                return [prefix + self.mca_header_dict[key]]
        return []

    def lines(self):
        """Return the number of data lines (number of data points per
        detector)"""
        return super(scandata, self).data.shape[0]

    def mca(self, number):
        """Return one MCA spectrum

        :param number: MCA number (1-based index)
        :rtype: 1D numpy array"""
        # in the base class, mca is an object that can be indexed (but 0-based)
        return super(scandata, self).mca[number - 1]

    def nbmca(self):
        """Return number of MCAs in this scan"""
        return len(super(scandata, self).mca)
