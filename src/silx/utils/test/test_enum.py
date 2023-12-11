# /*##########################################################################
#
# Copyright (c) 2019-2023 European Synchrotron Radiation Facility
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
"""Tests of Enum class with extra class methods"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "29/04/2019"


import pytest
from silx.utils.enum import Enum


def test_enum_methods():
    """Test Enum"""

    class Success(Enum):
        A = 1
        B = "B"

    assert Success.members() == (Success.A, Success.B)
    assert Success.names() == ("A", "B")
    assert Success.values() == (1, "B")

    assert Success.from_value(1) == Success.A
    assert Success.from_value("B") == Success.B
    with pytest.raises(ValueError):
        Success.from_value(3)
