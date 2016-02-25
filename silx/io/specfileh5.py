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
"""

"""
from .specfile import SpecFile

__authors__ = ["P. Knobel"]
__license__ = "MIT"
__date__ = "25/02/2016"


class SpecFileH5:
    def __init__(self, filename):
        self._sf = SpecFile(filename)

    def __getitem__(self, key):
        """Fetch the requested piece of data or header. If key points to
        an intermediate directory/group which is not actual data (e.g. "/",
        or "/1.3/"), return a list of pathes to its subdirectories.

        :param key: Path to data (e.g. "/3.1/measurement/column_name",
            or "/1.1/instrument/positioners/motor_name", or "/1.2/title")
        :type key: str
        """
        try:
            split_path = key.split("/")
        except AttributeError:
            raise TypeError("key must be a string")

        if not key.startswith("/"):
            raise KeyError("key must be a HDF5 like key, starting with /")
        # first element must be an empty string if key started with "/"
        else:
            # remove empty string
            split_path.pop(0)

        if key == "/":
            return ["/" + scan_key for scan_key in self._sf.keys()]

        # next element should be a scan index such as "1.1"
        scan_key = split_path.pop(0)
        scan = self._sf[scan_key]

        parent_path = "/" + scan_key + "/"

        # if there is nothing remaining in split_path, list available subfolders
        if not split_path:
            possible_sub_dirs = ["title", "start_time", "instrument", "measurement"]
            return [parent_path + item for item in possible_sub_dirs]

        next_level = split_path.pop(0)

        if next_level == "title":
            return scan.scan_header['S']
        elif next_level == "start_time":
            # TODO: format date
            return scan.scan_header['D']
        elif next_level == "instrument":
            if not split_path:
                return ["/" + scan_key + "/instrument/positioners"]
            else:
                next_level = split_path.pop(0)
                if next_level == "positioners":
                    if not split_path:
                        return ["/" + scan_key + "/instrument/positioners/" + motor
                                for motor in scan.motor_names]
                    # request for a specific motor position
                    elif len(split_path) == 1:
                        return scan.motor_position_by_name(split_path.pop())
                    else:
                        raise KeyError("invalid key: " + key)
                else:
                    raise KeyError("invalid key: " + key)
        elif next_level == "measurement":
            raise NotImplementedError()
            # TODO: implement
