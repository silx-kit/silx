# /*##########################################################################
#
# Copyright (c) 2019 European Synchrotron Radiation Facility
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
"""Utils function relative to files
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "19/09/2016"

import os.path
import glob

def expand_filenames(filenames):
    """
    Takes a list of paths and expand it into a list of files.

    :param List[str] filenames: list of filenames or path with wildcards
    :rtype: List[str]
    :return: list of existing filenames or non-existing files
        (which was provided as input)
    """
    result = []
    for filename in filenames:
        if os.path.exists(filename):
            result.append(filename)
        elif glob.has_magic(filename):
            expanded_filenames = glob.glob(filename)
            if expanded_filenames:
                result += expanded_filenames
            else:  # Cannot expand, add as is
                result.append(filename)
        else:
            result.append(filename)
    return result
