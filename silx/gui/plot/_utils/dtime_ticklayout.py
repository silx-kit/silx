# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2014-2017 European Synchrotron Radiation Facility
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
"""This module implements date-time labels layout on graph axes."""

from __future__ import absolute_import, division, unicode_literals

__authors__ = ["P. Kenter"]
__license__ = "MIT"
__date__ = "04/04/2018"


import datetime as dt
import enum
import logging
import math

from dateutil.relativedelta import relativedelta

_logger = logging.getLogger(__name__)


MICROSECONDS_PER_SECOND = 1000000
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY
SECONDS_PER_MONTH_AVERAGE = SECONDS_PER_YEAR / 12 # Seconds per average month


@enum.unique
class DtUnit(enum.Enum):
    YEARS = 0
    MONTHS = 1
    DAYS = 2
    HOURS = 3
    MINUTES = 4
    SECONDS = 5
    MICRO_SECONDS = 6  # a fraction of a second


def getDateElement(dateTime, unit):

    if unit == DtUnit.YEARS:
        return dateTime.year
    elif unit == DtUnit.MONTHS:
        return dateTime.month
    elif unit == DtUnit.DAYS:
        return dateTime.day
    elif unit == DtUnit.HOURS:
        return dateTime.hour
    elif unit == DtUnit.MINUTES:
        return dateTime.minute
    elif unit == DtUnit.SECONDS:
        return dateTime.second
    elif unit == DtUnit.MICRO_SECONDS:
        return dateTime.microsecond
    else:
        raise ValueError("Unexpected DtUnit: {}".format(unit))


def setDateElement(dateTime, value, unit):
    """ Returns a copy of dateTime with the tickStep unit set to value

        :param datetime.datetime dateTime: date time object
        :param int value: value to set
        :param DtUnit unit: unit
        :return: datetime.datetime
    """
    intValue = int(value)
    _logger.debug("setDateElement({}, {} (int={}), {})"
                  .format(dateTime, value, intValue, unit))

    year = dateTime.year
    month = dateTime.month
    day = dateTime.day
    hour = dateTime.hour
    minute = dateTime.minute
    second = dateTime.second
    microsecond = dateTime.microsecond

    if unit == DtUnit.YEARS:
        year = intValue
    elif unit == DtUnit.MONTHS:
        month = intValue
    elif unit == DtUnit.DAYS:
        day = intValue
    elif unit == DtUnit.HOURS:
        hour = intValue
    elif unit == DtUnit.MINUTES:
        minute = intValue
    elif unit == DtUnit.SECONDS:
        second = intValue
    elif unit == DtUnit.MICRO_SECONDS:
        microsecond = intValue
    else:
        raise ValueError("Unexpected DtUnit: {}".format(unit))

    _logger.debug("creating date time {}"
        .format((year, month, day, hour, minute, second, microsecond)))

    return dt.datetime(year, month, day, hour, minute, second, microsecond)



def roundToElement(dateTime, unit):
    """ Returns a copy of dateTime with the

        :param datetime.datetime dateTime: date time object
        :param DtUnit unit: unit
        :return: datetime.datetime
    """
    year = dateTime.year
    month = dateTime.month
    day = dateTime.day
    hour = dateTime.hour
    minute = dateTime.minute
    second = dateTime.second
    microsecond = dateTime.microsecond

    if unit.value < DtUnit.YEARS.value:
        pass # Never round years
    if unit.value < DtUnit.MONTHS.value:
        month = 1
    if unit.value < DtUnit.DAYS.value:
        day = 1
    if unit.value < DtUnit.HOURS.value:
        hour = 0
    if unit.value < DtUnit.MINUTES.value:
        minute = 0
    if unit.value < DtUnit.SECONDS.value:
        second = 0
    if unit.value < DtUnit.MICRO_SECONDS.value:
        microsecond = 0

    return dt.datetime(year, month, day, hour, minute, second, microsecond)


def addValueToDate(dateTime, value, unit):

    #logger.debug("addValueToDate({}, {}, {})".format(dateTime, value, unit))

    if unit == DtUnit.YEARS:
        intValue = int(value) # floats not implemented in relativeDelta(years)
        return dateTime + relativedelta(years=intValue)
    elif unit == DtUnit.MONTHS:
        intValue = int(value) # floats not implemented in relativeDelta(mohths)
        return dateTime + relativedelta(months=intValue)
    elif unit == DtUnit.DAYS:
        return dateTime + relativedelta(days=value)
    elif unit == DtUnit.HOURS:
        return dateTime + relativedelta(hours=value)
    elif unit == DtUnit.MINUTES:
        return dateTime + relativedelta(minutes=value)
    elif unit == DtUnit.SECONDS:
        return dateTime + relativedelta(seconds=value)
    elif unit == DtUnit.MICRO_SECONDS:
        return dateTime + relativedelta(microseconds=value)
    else:
        raise ValueError("Unexpected DtUnit: {}".format(unit))


def bestUnit(durationInSeconds):
    """ Gets the best tick spacing given a duration in seconds.

    :param durationInSeconds: time span duration in seconds
    :return: DtUnit enumeration.
    """

    # Based on; https://stackoverflow.com/a/2144398/
    # If the duration is longer than two years the tick spacing will be in
    # years. Else, if the duration is longer than two months, the spacing will
    # be in months, Etcetera.
    #
    # This factor differs per unit. As a baseline it is 2, but for instance,
    # for Months this needs to be higher (3>), This because it is impossible to
    # have partial months so the tick spacing is always at least 1 month. A
    # duration of two months would result in two ticks, which is too few.
    # months would then results

    if durationInSeconds > SECONDS_PER_YEAR * 3:
        return (durationInSeconds / SECONDS_PER_YEAR, DtUnit.YEARS)
    elif durationInSeconds > SECONDS_PER_MONTH_AVERAGE * 3:
        return (durationInSeconds / SECONDS_PER_MONTH_AVERAGE, DtUnit.MONTHS)
    elif durationInSeconds > SECONDS_PER_DAY * 2:
        return (durationInSeconds / SECONDS_PER_DAY, DtUnit.DAYS)
    elif durationInSeconds > SECONDS_PER_HOUR * 2:
        return (durationInSeconds / SECONDS_PER_HOUR, DtUnit.HOURS)
    elif durationInSeconds > SECONDS_PER_MINUTE * 2:
        return (durationInSeconds / SECONDS_PER_MINUTE, DtUnit.MINUTES)
    elif durationInSeconds > 1 * 2:
        return (durationInSeconds, DtUnit.SECONDS)
    else:
        return (durationInSeconds * MICROSECONDS_PER_SECOND,
                DtUnit.MICRO_SECONDS)




def niceNumGeneric(value, niceFractions, isRound=False):

    if value == 0:
        return value

    #niceFractions = [1.0, 2.0, 5.0, 10.]
    roundFractions = list(niceFractions)

    if isRound:
        # Take the average with the next element. The last remains the same.
        for i in range(len(roundFractions) - 1):
            roundFractions[i] = (niceFractions[i] + niceFractions[i+1]) / 2

    _logger.debug("Used fractions ({}): {}".format(isRound, roundFractions))

    highest = niceFractions[-1]
    value = float(value)

    expvalue = math.floor(math.log(value, highest))
    frac = value/pow(highest, expvalue)

    for niceFrac, roundFrac in zip(niceFractions, roundFractions):
        if frac <= roundFrac:
            return niceFrac * pow(highest, expvalue)

    # should not come here
    assert False, "should not come here"



NICE_DATE_VALUES = {
    DtUnit.YEARS: [1, 2, 5, 10],
    DtUnit.MONTHS: [1, 2, 3, 4, 6, 12],
    DtUnit.DAYS: [1, 2, 3, 7, 14, 28],
    DtUnit.HOURS: [1, 2, 3, 4, 6, 12],
    DtUnit.MINUTES: [1, 2, 3, 5, 10, 15, 30],
    DtUnit.SECONDS: [1, 2, 3, 5, 10, 15, 30],
    DtUnit.MICRO_SECONDS : [1.0, 2.0, 5.0, 10.0], # floats for microsec
}


# Format strings per unit (to use in datetime.strftime)
DATE_FORMAT_STRINGS = {
    DtUnit.YEARS: "%Y",
    DtUnit.MONTHS: "%Y-%m",
    DtUnit.DAYS: "%Y-%m-%d",
    DtUnit.HOURS: "%H:%M",
    DtUnit.MINUTES: "%H:%M",
    DtUnit.SECONDS: "%H:%M:%S",
    DtUnit.MICRO_SECONDS : "%S.%f", # floats for microsec
}


def niceDateTimeElement(value, unit, isRound=False):

    niceValues = NICE_DATE_VALUES[unit]
    elemValue = niceNumGeneric(value, niceValues, isRound=isRound)

    if unit == DtUnit.YEARS or unit == DtUnit.MONTHS:
        elemValue = max(1, int(elemValue))

    return elemValue


def findStartDate(dMin, dMax, nTicks):
    """ Rounds a date down to the nearest nice number of ticks"""

    assert dMax > dMin, "dMin ({}) should come before dMax ({})".format(dMin, dMax)
    delta = dMax - dMin
    lengthSec = delta.total_seconds()
    _logger.debug("findStartDate: {}, {} (duration = {} sec, {} days)"
                  .format(dMin, dMax, lengthSec, lengthSec / SECONDS_PER_DAY))

    length, unit = bestUnit(delta.total_seconds())
    niceLength = niceDateTimeElement(length, unit)

    _logger.debug("Length: {:8.3f} {} (nice = {})"
                  .format(length, unit.name, niceLength))

    niceSpacing = niceDateTimeElement(niceLength / nTicks, unit, isRound=True)

    _logger.debug("Spacing: {:8.3f} {} (nice = {})"
                  .format(niceLength / nTicks, unit.name, niceSpacing))

    dVal = getDateElement(dMin, unit)

    if unit == DtUnit.MONTHS: # TODO: better rounding?
        niceVal = math.floor((dVal-1) / niceSpacing) * niceSpacing + 1
    elif unit == DtUnit.DAYS:
        niceVal = math.floor((dVal-1) / niceSpacing) * niceSpacing + 1
    else:
        niceVal = math.floor(dVal / niceSpacing) * niceSpacing

    _logger.debug("StartValue: dVal = {}, niceVal: {} ({})"
                  .format(dVal, niceVal, unit.name))

    startDate = roundToElement(dMin, unit)
    startDate = setDateElement(startDate, niceVal, unit)

    return startDate, niceSpacing, unit


def dateRange(dMin, dMax, tickStep, unit, includeFirstBeyond = False):

    if unit == DtUnit.YEARS or unit == DtUnit.MONTHS or unit == DtUnit.MICRO_SECONDS:

        # Month and years will be converted to integers
        assert int(tickStep) > 0, "Integer value or tickstep is 0"
    else:
        assert tickStep > 0, "tickstep is 0"

    dateTime = dMin
    while dateTime <= dMax:
        yield dateTime
        dateTime = addValueToDate(dateTime, tickStep, unit)

    if includeFirstBeyond:
        yield dateTime



def calcTicks(dMin, dMax, nTicks):
    """Returns tick positions.

    :param datetime dMin: The min value on the axis
    :param datetime dMax: The max value on the axis
    :param int nTicks: The target number of ticks. The actual number of found
        ticks may differ.
    :returns: (list of datetimes, DtUnit) tuple
    """
    _logger.debug("Calc calcTicks({}, {}, nTicks={})"
                  .format(dMin, dMax, nTicks))

    startDate, niceSpacing, unit = findStartDate(dMin, dMax, nTicks)

    result = []
    for d in dateRange(startDate, dMax, niceSpacing, unit,
                       includeFirstBeyond=True):
        result.append(d)

    assert result[0] <= dMin, \
        "First nice date ({}) should be <= dMin {}".format(result[0], dMin)

    assert result[-1] >= dMax, \
        "Last nice date ({}) should be >= dMax {}".format(result[-1], dMax)

    return result, unit


def calcTicksAdaptive(dMin, dMax, axisLength, tickDensity):

    # At least 2 ticks
    nticks = max(2, int(round(tickDensity * axisLength)))
    return  calcTicks(dMin, dMax, nticks)




