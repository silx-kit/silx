#!/usr/bin/env python3
# /*##########################################################################
#
# Copyright (c) 2015-2022 European Synchrotron Radiation Facility
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
"""Unique place where the version number is defined.

provides:
* version = "1.2.3" or "1.2.3-beta4"
* version_info = named tuple (1,2,3,"beta",4)
* hexversion: 0x010203B4
* strictversion = "1.2.3b4
* debianversion = "1.2.3~beta4"
* calc_hexversion: the function to transform a version_tuple into an integer

This is called hexversion since it only really looks meaningful when viewed as the
result of passing it to the built-in hex() function.
The version_info value may be used for a more human-friendly encoding of the same information.

The hexversion is a 32-bit number with the following layout:
Bits (big endian order)     Meaning
1-8     PY_MAJOR_VERSION (the 2 in 2.1.0a3)
9-16     PY_MINOR_VERSION (the 1 in 2.1.0a3)
17-24     PY_MICRO_VERSION (the 0 in 2.1.0a3)
25-28     PY_RELEASE_LEVEL (0xA for alpha, 0xB for beta, 0xC for release candidate and 0xF for final)
29-32     PY_RELEASE_SERIAL (the 3 in 2.1.0a3, zero for final releases)

Thus 2.1.0a3 is hexversion 0x020100a3.

"""

__authors__ = ["Jérôme Kieffer"]
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "30/09/2020"
__status__ = "production"
__docformat__ = 'restructuredtext'
__all__ = ["date", "version_info", "strictversion", "hexversion", "debianversion",
           "calc_hexversion"]

RELEASE_LEVEL_VALUE = {"dev": 0,
                       "alpha": 10,
                       "beta": 11,
                       "candidate": 12,
                       "final": 15}

PRERELEASE_NORMALIZED_NAME = {"dev": "a",
                              "alpha": "a",
                              "beta": "b",
                              "candidate": "rc"}

MAJOR = 1
MINOR = 1
MICRO = 2
RELEV = "final"  # <16
SERIAL = 0  # <16

date = __date__

from collections import namedtuple
_version_info = namedtuple("version_info", ["major", "minor", "micro", "releaselevel", "serial"])

version_info = _version_info(MAJOR, MINOR, MICRO, RELEV, SERIAL)

strictversion = version = debianversion = "%d.%d.%d" % version_info[:3]
if version_info.releaselevel != "final":
    _prerelease = PRERELEASE_NORMALIZED_NAME[version_info[3]]
    version += "-%s%s" % (_prerelease, version_info[-1])
    debianversion += "~adev%i" % version_info[-1] if RELEV == "dev" else "~%s%i" % (_prerelease, version_info[-1])
    strictversion += _prerelease + str(version_info[-1])


def calc_hexversion(major=0, minor=0, micro=0, releaselevel="dev", serial=0):
    """Calculate the hexadecimal version number from the tuple version_info:

    :param major: integer
    :param minor: integer
    :param micro: integer
    :param relev: integer or string
    :param serial: integer
    :return: integer always increasing with revision numbers
    """
    try:
        releaselevel = int(releaselevel)
    except ValueError:
        releaselevel = RELEASE_LEVEL_VALUE.get(releaselevel, 0)

    hex_version = int(serial)
    hex_version |= releaselevel * 1 << 4
    hex_version |= int(micro) * 1 << 8
    hex_version |= int(minor) * 1 << 16
    hex_version |= int(major) * 1 << 24
    return hex_version


hexversion = calc_hexversion(*version_info)

if __name__ == "__main__":
    print(version)
