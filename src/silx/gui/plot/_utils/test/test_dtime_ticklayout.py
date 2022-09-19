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

__authors__ = ["P. Kenter"]
__license__ = "MIT"
__date__ = "06/04/2018"


import datetime as dt
import pytest


from silx.gui.plot._utils.dtime_ticklayout import calcTicks, DtUnit, SECONDS_PER_YEAR


def testSmallMonthlySpacing():
    """Tests a range that did result in a spacing of less than 1 month.
    It is impossible to add fractional month so the unit must be in days
    """
    from dateutil import parser

    d1 = parser.parse("2017-01-03 13:15:06.000044")
    d2 = parser.parse("2017-03-08 09:16:16.307584")
    _ticks, _units, spacing = calcTicks(d1, d2, nTicks=4)

    assert spacing == DtUnit.DAYS


def testNoCrash():
    """Creates many combinations of and number-of-ticks and end-dates;
    tests that it doesn't give an exception and returns a reasonable number
    of ticks.
    """
    d1 = dt.datetime(2017, 1, 3, 13, 15, 6, 44)

    value = 100e-6  # Start at 100 micro sec range.

    while value <= 200 * SECONDS_PER_YEAR:

        d2 = d1 + dt.timedelta(microseconds=value * 1e6)  # end date range

        for numTicks in range(2, 12):
            ticks, _, _ = calcTicks(d1, d2, numTicks)

            margin = 2.5
            assert (
                numTicks / margin <= len(ticks) <= numTicks * margin
            ), "Condition {} <= {} <= {} failed for # ticks={} and d2={}:".format(
                numTicks / margin, len(ticks), numTicks * margin, numTicks, d2
            )

        value = value * 1.5  # let date period grow exponentially


@pytest.mark.parametrize(
    "dMin, dMax",
    [
        (dt.datetime(1, 1, 1), dt.datetime(400, 1, 1)),
        (dt.datetime(4000, 1, 1), dt.datetime(9999, 1, 1)),
        (dt.datetime(1, 1, 1), dt.datetime(9999, 12, 23)),
    ],
)
def testCalcTicksOutOfBoundTicks(dMin, dMax):
    """Test tick generation with values leading to out-of-bound ticks"""
    ticks, _, unit = calcTicks(dMin, dMax, nTicks=5)
    assert len(ticks) != 0
    assert unit == DtUnit.YEARS
