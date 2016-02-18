# /*#########################################################################
#
# The PyMca X-Ray Fluorescence Toolkit
#
# Copyright (c) 2004-2015 European Synchrotron Radiation Facility
#
# This file is part of the PyMca X-ray Fluorescence Toolkit developed at
# the ESRF by the Software group.
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
__author__ = "T. Vincent - ESRF Data Analysis"
__contact__ = "thomas.vincent@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__doc__ = """
Functions to prepare events to be sent to backend callback.
"""


# import ######################################################################

import numpy as np


# signals #####################################################################

def prepareDrawingSignal(event, type_, points, parameters=None):
    if parameters is None:
        parameters = {}

    eventDict = {}
    eventDict['event'] = event
    eventDict['type'] = type_
    points = np.array(points, dtype=np.float32)
    points.shape = -1, 2
    eventDict['points'] = points
    eventDict['xdata'] = points[:, 0]
    eventDict['ydata'] = points[:, 1]
    if type_ in ('rectangle',):
        eventDict['x'] = eventDict['xdata'].min()
        eventDict['y'] = eventDict['ydata'].min()
        eventDict['width'] = eventDict['xdata'].max() - eventDict['x']
        eventDict['height'] = eventDict['ydata'].max() - eventDict['y']
    eventDict['parameters'] = parameters.copy()
    return eventDict


def prepareMouseSignal(eventType, button, xData, yData, xPixel, yPixel):
    assert eventType in ('mouseMoved', 'mouseClicked', 'mouseDoubleClicked')
    assert button in (None, 'left', 'right')

    return {'event': eventType,
            'x': xData,
            'y': yData,
            'xpixel': xPixel,
            'ypixel': yPixel,
            'button': button}


def prepareHoverSignal(label, type_, posData, posPixel, draggable, selectable):
    return {'event': 'hover',
            'label': label,
            'type': type_,
            'x': posData[0],
            'y': posData[1],
            'xpixel': posPixel[0],
            'ypixel': posPixel[1],
            'draggable': draggable,
            'selectable': selectable}


def prepareMarkerSignal(eventType, button, label, type_,
                        draggable, selectable,
                        posDataMarker,
                        posPixelCursor=None, posDataCursor=None):
    if eventType == 'markerClicked':
        assert posPixelCursor is not None
        assert posDataCursor is None

        posDataCursor = list(posDataMarker)
        if hasattr(posDataCursor[0], "__len__"):
            posDataCursor[0] = posDataCursor[0][-1]
        if hasattr(posDataCursor[1], "__len__"):
            posDataCursor[1] = posDataCursor[1][-1]

    elif eventType == 'markerMoving':
        assert posPixelCursor is not None
        assert posDataCursor is not None

    elif eventType == 'markerMoved':
        assert posPixelCursor is None
        assert posDataCursor is None

        posDataCursor = posDataMarker
    else:
        raise NotImplementedError("Unknown event type {0}".format(eventType))

    eventDict = {'event': eventType,
                 'button': button,
                 'label': label,
                 'type': type_,
                 'x': posDataCursor[0],
                 'y': posDataCursor[1],
                 'xdata': posDataMarker[0],
                 'ydata': posDataMarker[1],
                 'draggable': draggable,
                 'selectable': selectable}

    if eventType in ('markerMoving', 'markerClicked'):
        eventDict['xpixel'] = posPixelCursor[0]
        eventDict['ypixel'] = posPixelCursor[1]

    return eventDict


def prepareImageSignal(button, label, type_, col, row,
                       x, y, xPixel, yPixel):
    return {'event': 'imageClicked',
            'button': button,
            'label': label,
            'type': type_,
            'col': col,
            'row': row,
            'x': x,
            'y': y,
            'xpixel': xPixel,
            'ypixel': yPixel}


def prepareCurveSignal(button, label, type_, xData, yData,
                       x, y, xPixel, yPixel):
    return {'event': 'curveClicked',
            'button': button,
            'label': label,
            'type': type_,
            'xdata': xData,
            'ydata': yData,
            'x': x,
            'y': y,
            'xpixel': xPixel,
            'ypixel': yPixel}


def prepareLimitsChangedSignal(sourceObj, xRange, yRange, y2Range):
    return {'event': 'limitsChanged',
            'source': id(sourceObj),
            'xdata': xRange,
            'ydata': yRange,
            'y2data': y2Range}
