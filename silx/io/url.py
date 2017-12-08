# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
"""URL module"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "08/12/2017"


import sys


class DataUrl(object):
    """Non-mutable object to parse and reach information identifying a link to
    a data.

    It supports:
    - path to file and path inside file to the data
    - data slicing
    - fabio or silx access to the data
    - absolute and relative file access

    >>> # fabio access using absolute path
    >>> DataUrl("fabio:///data/image.edf::[2]")
    >>> DataUrl("fabio:///C:/data/image.edf::[2]")

    >>> # silx access using absolute path
    >>> DataUrl("silx:///data/image.h5::/data/dataset[1,5]")
    >>> DataUrl("silx:///data/image.edf::/scan_0/detector/data")
    >>> DataUrl("silx:///C:/data/image.edf::/scan_0/detector/data")

    >>> # Relative path access
    >>> DataUrl("silx:./image.h5")
    >>> DataUrl("fabio:./image.edf")
    >>> DataUrl("silx:image.h5")
    >>> DataUrl("fabio:image.edf")

    >>> # Is also support parsing of file access for convenience
    >>> DataUrl("./foo/bar/image.edf")
    >>> DataUrl("C:/data/")
    """

    def __init__(self, path=None, file_path=None, data_path=None, slicing=None, scheme=None):
        self.__is_valid = False
        if path is not None:
            self.__parse_from_path(path)
        else:
            self.__file_path = file_path
            self.__data_path = data_path
            self.__slice = slicing
            self.__scheme = scheme
            self.__path = None
            self.__check_validity()

    def __check_validity(self):
        """Check the validity of the attributes."""
        if self.__file_path in ["", "/"]:
            self.__is_valid = self.__data_path is None and self.__slice is None
        else:
            self.__is_valid = self.__file_path is not None

        if self.__scheme not in [None, "silx", "fabio"]:
            self.__is_valid = False

        if self.__scheme == "fabio":
            self.__is_valid = self.__is_valid and self.__data_path is None
        elif self.__scheme == "silx":
            slice_implies_data = (self.__data_path is None and self.__slice is None) or self.__data_path is not None
            self.__is_valid = self.__is_valid and slice_implies_data

    def __parse_from_path(self, path):
        """Parse the path and initialize attributes.

        :param str path: Path representing the URL.
        """
        def str_to_slice(string):
            if string == "...":
                return Ellipsis
            elif string == ":":
                return slice(None)
            else:
                return int(string)

        elements = path.split("::", 1)
        self.__path = path

        scheme_and_filepath = elements[0].split(":", 1)
        if len(scheme_and_filepath) == 2:
            if len(scheme_and_filepath[0]) <= 2:
                # Windows driver
                self.__scheme = None
                file_path = elements[0]
            else:
                self.__scheme = scheme_and_filepath[0]
                file_path = scheme_and_filepath[1]
        else:
            self.__scheme = None
            file_path = scheme_and_filepath[0]

        if file_path.startswith("///"):
            # absolute path
            file_path = file_path[3:]
            if len(file_path) > 2 and (file_path[1] == ":" or file_path[2] == ":"):
                # Windows driver
                pass
            else:
                file_path = "/" + file_path
        self.__file_path = file_path

        self.__slice = None
        self.__data_path = None
        if len(elements) == 1:
            pass
        else:
            selector = elements[1]
            selectors = selector.split("[", 1)
            data_path = selectors[0]
            if len(data_path) == 0:
                data_path = None
            self.__data_path = data_path

            if len(selectors) == 2:
                slicing = selectors[1].split("]", 1)
                if len(slicing) < 2 or slicing[1] != "":
                    self.__is_valid = False
                    return
                slicing = slicing[0].split(",")
                try:
                    slicing = tuple(str_to_slice(s) for s in slicing)
                    self.__slice = slicing
                except ValueError:
                    self.__is_valid = False
                    return

        self.__check_validity()

    def is_valid(self):
        """Returns true if the URL is valid. Else attributes can be None.

        :rtype: bool
        """
        return self.__is_valid

    def path(self):
        """Returns the string representing the URL.

        :rtype: str
        """
        if self.__path is not None:
            return self.__path

        def slice_to_string(slicing):
            if slicing == Ellipsis:
                return "..."
            elif slicing == slice(None):
                return ":"
            elif isinstance(slicing, int):
                return str(slicing)
            else:
                raise TypeError("Unexpected slicing type. Found %s" % type(slicing))

        path = ""
        selector = ""
        if self.__file_path is not None:
            path += self.__file_path
        if self.__data_path is not None:
            selector += self.__data_path
        if self.__slice is not None:
            selector += "[%s]" % ",".join([slice_to_string(s) for s in self.__slice])

        if selector != "":
            path = path + "::" + selector

        if self.__scheme is not None:
            if self.is_absolute():
                if path.startswith("/"):
                    path = self.__scheme + "://" + path
                else:
                    path = self.__scheme + ":///" + path
            else:
                path = self.__scheme + ":" + path

        return path

    def is_absolute(self):
        """Returns true if the file path is an absolute path.

        :rtype: bool
        """
        file_path = self.file_path()
        if len(file_path) > 0:
            if file_path[0] == "/":
                return True
        if len(file_path) > 2:
            # Windows
            if file_path[1] == ":" or file_path[2] == ":":
                return True
        elif len(file_path) > 1:
            # Windows
            if file_path[1] == ":":
                return True
        return False

    def file_path(self):
        """Returns the path to the file containing the data.

        :rtype: str
        """
        return self.__file_path

    def data_path(self):
        """Returns the path inside the file to the data.

        :rtype: str
        """
        return self.__data_path

    def slice(self):
        """Returns the slicing applyed to the data.

        It is a tuple containing numbers, slice or ellipses.

        :rtype: Tuple[int, slice, Ellipse]
        """
        return self.__slice

    def scheme(self):
        """Returns the scheme. It can be None if no scheme is specified.

        :rtype: Union[str, None]
        """
        return self.__scheme
