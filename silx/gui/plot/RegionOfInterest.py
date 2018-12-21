# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2018 European Synchrotron Radiation Facility
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
"""
Widget to handle regions of interest (:class:`ROI`) on curves displayed in a
:class:`PlotWindow`.

This widget is meant to work with :class:`PlotWindow`.
"""

__authors__ = ["V.A. Sole", "T. Vincent", "H. Payno"]
__license__ = "MIT"
__date__ = "21/12/2018"


class _BaseRegionOfInterest(qt.QObject):
    """Base class of all the Region Of Interest object"""

    sigRegionChanged = qt.Signal()
    """Signal emitted when the ROI is edited"""

    def __init__(self, parent, name):
        self._name = name
        self._editable = False

    def setName(self, name):
        """
        Set the name of the :class:`ROI`

        :param str name:
        """
        self._name = name
        self.sigChanged.emit()

    def getName(self):
        """

        :return str: name of the :class:`ROI`
        """
        return self._name

    @property
    def editable(self):
        raise NotImplemented('')

    def isEditable(self):
        """Returns whether the ROI is editable by the user or not.

        :rtype: bool
        """
        return self._editable

    def setEditable(self, editable):
        """Set whether the ROI can be changed interactively.

        :param bool editable: True to allow edition by the user,
           False to disable.
        """
        self._editable = editable

_indexNextROI = 0


class ROI(_BaseRegionOfInterest):
    """The Region Of Interest is defined by:

    - A name
    - A type. The type is the label of the x axis. This can be used to apply or
    not some ROI to a curve and do some post processing.
    - The x coordinate of the left limit (fromdata)
    - The x coordinate of the right limit (todata)

    :param str: name of the ROI
    :param fromdata: left limit of the roi
    :param todata: right limit of the roi
    :param type: type of the ROI
    """

    def __init__(self, name, fromdata=None, todata=None, type_=None):
        _BaseRegionOfInterest.__init__(self, parent=None, name=name)
        global _indexNextROI
        self._id = _indexNextROI
        _indexNextROI += 1

        self._fromdata = fromdata
        self._todata = todata
        self._type = type_ or 'Default'

    def setName(self, name):
        _BaseRegionOfInterest.setName(self, name)
        sigRegionChanged.emit()

    def getID(self):
        """

        :return int: the unique ID of the ROI
        """
        return self._id

    def setType(self, type_):
        """

        :param str type_:
        """
        self._type = type_
        self.sigRegionChanged.emit()

    def getType(self):
        """

        :return str: the type of the ROI.
        """
        return self._type

    def setFrom(self, frm):
        """

        :param frm: set x coordinate of the left limit
        """
        self._fromdata = frm
        self.sigRegionChanged.emit()

    def getFrom(self):
        """

        :return: x coordinate of the left limit
        """
        return self._fromdata

    def setTo(self, to):
        """

        :param to: x coordinate of the right limit
        """
        self._todata = to
        self.sigRegionChanged.emit()

    def getTo(self):
        """

        :return: x coordinate of the right limit
        """
        return self._todata

    def getMiddle(self):
        """

        :return: middle position between 'from' and 'to' values
        """
        return 0.5 * (self.getFrom() + self.getTo())

    def toDict(self):
        """

        :return: dict containing the roi parameters
        """
        ddict = {
            'type': self._type,
            'name': self._name,
            'from': self._fromdata,
            'to': self._todata,
        }
        if hasattr(self, '_extraInfo'):
            ddict.update(self._extraInfo)
        return ddict

    @staticmethod
    def _fromDict(dic):
        assert 'name' in dic
        roi = ROI(name=dic['name'])
        roi._extraInfo = {}
        for key in dic:
            if key == 'from':
                roi.setFrom(dic['from'])
            elif key == 'to':
                roi.setTo(dic['to'])
            elif key == 'type':
                roi.setType(dic['type'])
            else:
                roi._extraInfo[key] = dic[key]

        return roi

    def isICR(self):
        """

        :return: True if the ROI is the `ICR`
        """
        return self._name == 'ICR'

    def computeRawAndNetCounts(self, curve):
        """Compute the Raw and net counts in the ROI for the given curve.

        - Raw count: Points values sum of the curve in the defined Region Of
           Interest.

          .. image:: img/rawCounts.png

        - Net count: Raw counts minus background

          .. image:: img/netCounts.png

        :param CurveItem curve:
        :return tuple: rawCount, netCount
        """
        assert isinstance(curve, Curve) or curve is None

        if curve is None:
            return None, None

        x = curve.getXData(copy=False)
        y = curve.getYData(copy=False)

        idx = numpy.nonzero((self._fromdata <= x) &
                            (x <= self._todata))[0]
        if len(idx):
            xw = x[idx]
            yw = y[idx]
            rawCounts = yw.sum(dtype=numpy.float)
            deltaX = xw[-1] - xw[0]
            deltaY = yw[-1] - yw[0]
            if deltaX > 0.0:
                slope = (deltaY / deltaX)
                background = yw[0] + slope * (xw - xw[0])
                netCounts = (rawCounts -
                             background.sum(dtype=numpy.float))
            else:
                netCounts = 0.0
        else:
            rawCounts = 0.0
            netCounts = 0.0
        return rawCounts, netCounts

    def computeRawAndNetArea(self, curve):
        """Compute the Raw and net counts in the ROI for the given curve.

        - Raw area: integral of the curve between the min ROI point and the
           max ROI point to the y = 0 line.

          .. image:: img/rawAreas.png

        - Net area: Raw counts minus background

          .. image:: img/netAreas.png

        :param CurveItem curve:
        :return tuple: rawArea, netArea
        """
        assert isinstance(curve, Curve) or curve is None

        if curve is None:
            return None, None

        x = curve.getXData(copy=False)
        y = curve.getYData(copy=False)

        y = y[(x >= self._fromdata) & (x <= self._todata)]
        x = x[(x >= self._fromdata) & (x <= self._todata)]

        if x.size is 0:
            return 0.0, 0.0

        rawArea = numpy.trapz(y, x=x)
        # to speed up and avoid an intersection calculation we are taking the
        # closest index to the ROI
        closestXLeftIndex = (numpy.abs(x - self.getFrom())).argmin()
        closestXRightIndex = (numpy.abs(x - self.getTo())).argmin()
        yBackground = y[closestXLeftIndex], y[closestXRightIndex]
        background = numpy.trapz(yBackground, x=x)
        netArea = rawArea - background
        return rawArea, netArea


class _RoiMarkerHandler(object):
    """Used to deal with ROI markers used in ROITable"""
    def __init__(self, roi, plot):
        assert roi and isinstance(roi, ROI)
        assert plot

        self._roi = weakref.ref(roi)
        self._plot = weakref.ref(plot)
        self.draggable = False if roi.isICR() else True
        self._color = 'blue' if roi.isICR() else 'black'
        self._displayMidMarker = True
        self._visible = True

    @property
    def plot(self):
        return self._plot()

    def clear(self):
        if self.plot and self.roi:
            self.plot.removeMarker(self._markerID('min'))
            self.plot.removeMarker(self._markerID('max'))
            self.plot.removeMarker(self._markerID('middle'))

    @property
    def roi(self):
        return self._roi()

    def setVisible(self, visible):
        if visible != self._visible:
            self._visible = visible
            self.updateMarkers()

    def showMiddleMarker(self, visible):
        self._displayMidMarker = visible
        self.getMarker('middle').setVisible(self._displayMidMarker)

    def updateMarkers(self):
        if self.roi is None:
            return
        self._updateMinMarkerPos()
        self._updateMaxMarkerPos()
        self._updateMiddleMarkerPos()

    def _updateMinMarkerPos(self):
        self.getMarker('min').setPosition(x=self.roi.getFrom(), y=None)
        self.getMarker('min').setVisible(self._visible)

    def _updateMaxMarkerPos(self):
        self.getMarker('max').setPosition(x=self.roi.getTo(), y=None)
        self.getMarker('max').setVisible(self._visible)

    def _updateMiddleMarkerPos(self):
        self.getMarker('middle').setPosition(x=self.roi.getMiddle(), y=None)
        self.getMarker('middle').setVisible(self._displayMidMarker and self._visible)

    def getMarker(self, markerType):
        if self.plot is None:
            return None
        assert markerType in ('min', 'max', 'middle')
        if self.plot._getMarker(self._markerID(markerType)) is None:
            assert self.roi
            if markerType == 'min':
                val = self.roi.getFrom()
            elif markerType == 'max':
                val = self.roi.getTo()
            else:
                val = self.roi.getMiddle()

            _color = self._color
            if markerType == 'middle':
                _color = 'yellow'
            self.plot.addXMarker(val,
                                 legend=self._markerID(markerType),
                                 text=self.getMarkerName(markerType),
                                 color=_color,
                                 draggable=self.draggable)
        return self.plot._getMarker(self._markerID(markerType))

    def _markerID(self, markerType):
        assert markerType in ('min', 'max', 'middle')
        assert self.roi
        return '_'.join((str(self.roi.getID()), markerType))

    def getMarkerName(self, markerType):
        assert markerType in ('min', 'max', 'middle')
        assert self.roi
        return ' '.join((self.roi.getName(), markerType))

    def updateTexts(self):
        self.getMarker('min').setText(self.getMarkerName('min'))
        self.getMarker('max').setText(self.getMarkerName('max'))
        self.getMarker('middle').setText(self.getMarkerName('middle'))

    def changePosition(self, markerID, x):
        assert self.hasMarker(markerID)
        markerType = self._getMarkerType(markerID)
        assert markerType is not None
        if self.roi is None:
            return
        if markerType == 'min':
            self.roi.setFrom(x)
            self._updateMiddleMarkerPos()
        elif markerType == 'max':
            self.roi.setTo(x)
            self._updateMiddleMarkerPos()
        else:
            delta = x - 0.5 * (self.roi.getFrom() + self.roi.getTo())
            self.roi.setFrom(self.roi.getFrom() + delta)
            self.roi.setTo(self.roi.getTo() + delta)
            self._updateMinMarkerPos()
            self._updateMaxMarkerPos()

    def hasMarker(self, marker):
        return marker in (self._markerID('min'),
                          self._markerID('max'),
                          self._markerID('middle'))

    def _getMarkerType(self, markerID):
        if markerID.endswith('_min'):
            return 'min'
        elif markerID.endswith('_max'):
            return 'max'
        elif markerID.endswith('_middle'):
            return 'middle'
        else:
            return None
