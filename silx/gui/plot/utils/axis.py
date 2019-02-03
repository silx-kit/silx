# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""This module contains utils class for axes management.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "20/11/2018"

import functools
import logging
from contextlib import contextmanager
import weakref
import silx.utils.weakref as silxWeakref
from silx.gui.plot.items.axis import Axis, XAxis, YAxis

try:
    from ...qt.inspect import isValid as _isQObjectValid
except ImportError:  # PySide(1) fallback
    def _isQObjectValid(obj):
        return True


_logger = logging.getLogger(__name__)


class SyncAxes(object):
    """Synchronize a set of plot axes together.

    It is created with the expected axes and starts to synchronize them.

    It can be customized to synchronize limits, scale, and direction of axes
    together. By default everything is synchronized.

    The API :meth:`start` and :meth:`stop` can be used to enable/disable the
    synchronization while this object is still alive.

    If this object is destroyed the synchronization stop.

    .. versionadded:: 0.6
    """

    def __init__(self, axes,
                 syncLimits=True,
                 syncScale=True,
                 syncDirection=True,
                 syncCenter=False,
                 syncZoom=False,
                 filterHiddenPlots=False
                 ):
        """
        Constructor

        :param list(Axis) axes: A list of axes to synchronize together
        :param bool syncLimits: Synchronize axes limits
        :param bool syncScale: Synchronize axes scale
        :param bool syncDirection: Synchronize axes direction
        :param bool syncCenter: Synchronize the center of the axes in the center
            of the plots
        :param bool syncZoom: Synchronize the zoom of the plot
        :param bool filterHiddenPlots: True to avoid updating hidden plots.
            Default: False.
        """
        object.__init__(self)

        def implies(x, y): return bool(y ** x)

        assert(implies(syncZoom, not syncLimits))
        assert(implies(syncCenter, not syncLimits))
        assert(implies(syncLimits, not syncCenter))
        assert(implies(syncLimits, not syncZoom))

        self.__filterHiddenPlots = filterHiddenPlots
        self.__locked = False
        self.__axisRefs = []
        self.__syncLimits = syncLimits
        self.__syncScale = syncScale
        self.__syncDirection = syncDirection
        self.__syncCenter = syncCenter
        self.__syncZoom = syncZoom
        self.__callbacks = None
        self.__lastMainAxis = None

        for axis in axes:
            self.addAxis(axis)

        self.start()

    def start(self):
        """Start synchronizing axes together.

        The first axis is used as the reference for the first synchronization.
        After that, any changes to any axes will be used to synchronize other
        axes.
        """
        if self.isSynchronizing():
            raise RuntimeError("Axes already synchronized")
        self.__callbacks = {}

        axes = self.__getAxes()

        # register callback for further sync
        for axis in axes:
            self.__connectAxes(axis)
        self.synchronize()

    def isSynchronizing(self):
        """Returns true if events are connected to the axes to synchronize them
        all together

        :rtype: bool
        """
        return self.__callbacks is not None

    def __connectAxes(self, axis):
        refAxis = weakref.ref(axis)
        callbacks = []
        if self.__syncLimits:
            # the weakref is needed to be able ignore self references
            callback = silxWeakref.WeakMethodProxy(self.__axisLimitsChanged)
            callback = functools.partial(callback, refAxis)
            sig = axis.sigLimitsChanged
            sig.connect(callback)
            callbacks.append(("sigLimitsChanged", callback))
        elif self.__syncCenter and self.__syncZoom:
            # the weakref is needed to be able ignore self references
            callback = silxWeakref.WeakMethodProxy(self.__axisCenterAndZoomChanged)
            callback = functools.partial(callback, refAxis)
            sig = axis.sigLimitsChanged
            sig.connect(callback)
            callbacks.append(("sigLimitsChanged", callback))
        elif self.__syncZoom:
            raise NotImplementedError()
        elif self.__syncCenter:
            # the weakref is needed to be able ignore self references
            callback = silxWeakref.WeakMethodProxy(self.__axisCenterChanged)
            callback = functools.partial(callback, refAxis)
            sig = axis.sigLimitsChanged
            sig.connect(callback)
            callbacks.append(("sigLimitsChanged", callback))
        if self.__syncScale:
            # the weakref is needed to be able ignore self references
            callback = silxWeakref.WeakMethodProxy(self.__axisScaleChanged)
            callback = functools.partial(callback, refAxis)
            sig = axis.sigScaleChanged
            sig.connect(callback)
            callbacks.append(("sigScaleChanged", callback))
        if self.__syncDirection:
            # the weakref is needed to be able ignore self references
            callback = silxWeakref.WeakMethodProxy(self.__axisInvertedChanged)
            callback = functools.partial(callback, refAxis)
            sig = axis.sigInvertedChanged
            sig.connect(callback)
            callbacks.append(("sigInvertedChanged", callback))

        if self.__filterHiddenPlots:
            # the weakref is needed to be able ignore self references
            callback = silxWeakref.WeakMethodProxy(self.__axisVisibilityChanged)
            callback = functools.partial(callback, refAxis)
            plot = axis._getPlot()
            plot.sigVisibilityChanged.connect(callback)
            callbacks.append(("sigVisibilityChanged", callback))

        self.__callbacks[refAxis] = callbacks

    def __disconnectAxes(self, axis):
        if axis is not None and _isQObjectValid(axis):
            ref = weakref.ref(axis)
            callbacks = self.__callbacks.pop(ref)
            for sigName, callback in callbacks:
                if sigName == "sigVisibilityChanged":
                    obj = axis._getPlot()
                else:
                    obj = axis
                if obj is not None:
                    sig = getattr(obj, sigName)
                    sig.disconnect(callback)

    def addAxis(self, axis):
        """Add a new axes to synchronize.

        :param ~silx.gui.plot.items.Axis axis: The axis to synchronize
        """
        self.__axisRefs.append(weakref.ref(axis))
        if self.isSynchronizing():
            self.__connectAxes(axis)
            # This could be done faster as only this axis have to be fixed
            self.synchronize()

    def removeAxis(self, axis):
        """Remove an axis from the synchronized axes.

        :param ~silx.gui.plot.items.Axis axis: The axis to remove
        """
        ref = weakref.ref(axis)
        self.__axisRefs.remove(ref)
        if self.isSynchronizing():
            self.__disconnectAxes(axis)

    def synchronize(self, mainAxis=None):
        """Synchronize programatically all the axes.

        :param ~silx.gui.plot.items.Axis mainAxis:
            The axis to take as reference (Default: the first axis).
        """
        # sync the current state
        axes = self.__getAxes()
        if len(axes) == 0:
            return

        if mainAxis is None:
            mainAxis = axes[0]

        refMainAxis = weakref.ref(mainAxis)
        if self.__syncLimits:
            self.__axisLimitsChanged(refMainAxis, *mainAxis.getLimits())
        elif self.__syncCenter and self.__syncZoom:
            self.__axisCenterAndZoomChanged(refMainAxis, *mainAxis.getLimits())
        elif self.__syncCenter:
            self.__axisCenterChanged(refMainAxis, *mainAxis.getLimits())
        if self.__syncScale:
            self.__axisScaleChanged(refMainAxis, mainAxis.getScale())
        if self.__syncDirection:
            self.__axisInvertedChanged(refMainAxis, mainAxis.isInverted())

    def stop(self):
        """Stop the synchronization of the axes"""
        if not self.isSynchronizing():
            raise RuntimeError("Axes not synchronized")
        for ref in list(self.__callbacks.keys()):
            axis = ref()
            self.__disconnectAxes(axis)
        self.__callbacks = None

    def __del__(self):
        """Destructor"""
        # clean up references
        if self.__callbacks is not None:
            self.stop()

    def __getAxes(self):
        """Returns list of existing axes.

        :rtype: List[Axis]
        """
        axes = [ref() for ref in self.__axisRefs]
        return [axis for axis in axes if axis is not None]

    @contextmanager
    def __inhibitSignals(self):
        self.__locked = True
        yield
        self.__locked = False

    def __axesToUpdate(self, changedAxis):
        for axis in self.__getAxes():
            if axis is changedAxis:
                continue
            if self.__filterHiddenPlots:
                plot = axis._getPlot()
                if not plot.isVisible():
                    continue
            yield axis

    def __axisVisibilityChanged(self, changedAxis, isVisible):
        if not isVisible:
            return
        if self.__locked:
            return
        changedAxis = changedAxis()
        if self.__lastMainAxis is None:
            self.__lastMainAxis = self.__axisRefs[0]
        mainAxis = self.__lastMainAxis
        mainAxis = mainAxis()
        self.synchronize(mainAxis=mainAxis)
        # force back the main axis
        self.__lastMainAxis = weakref.ref(mainAxis)

    def __getAxesCenter(self, axis, vmin, vmax):
        """Returns the value displayed in the center of this axis range.

        :rtype: float
        """
        scale = axis.getScale()
        if scale == Axis.LINEAR:
            center = (vmin + vmax) * 0.5
        else:
            raise NotImplementedError("Log scale not implemented")
        return center

    def __getRangeInPixel(self, axis):
        """Returns the size of the axis in pixel"""
        bounds = axis._getPlot().getPlotBoundsInPixels()
        # bounds: left, top, width, height
        if isinstance(axis, XAxis):
            return bounds[2]
        elif isinstance(axis, YAxis):
            return bounds[3]
        else:
            assert(False)

    def __getLimitsFromCenter(self, axis, pos, pixelSize=None):
        """Returns the limits to apply to this axis to move the `pos` into the
        center of this axis.

        :param Axis axis:
        :param float pos: Position in the center of the computed limits
        :param Union[None,float] pixelSize: Pixel size to apply to compute the
            limits. If `None` the current pixel size is applyed.
        """
        scale = axis.getScale()
        if scale == Axis.LINEAR:
            if pixelSize is None:
                # Use the current pixel size of the axis
                limits = axis.getLimits()
                valueRange = limits[0] - limits[1]
                a = pos - valueRange * 0.5
                b = pos + valueRange * 0.5
            else:
                pixelRange = self.__getRangeInPixel(axis)
                a = pos - pixelRange * 0.5 * pixelSize
                b = pos + pixelRange * 0.5 * pixelSize

        else:
            raise NotImplementedError("Log scale not implemented")
        if a > b:
            return b, a
        return a, b

    def __axisLimitsChanged(self, changedAxis, vmin, vmax):
        if self.__locked:
            return
        self.__lastMainAxis = changedAxis
        changedAxis = changedAxis()
        with self.__inhibitSignals():
            for axis in self.__axesToUpdate(changedAxis):
                axis.setLimits(vmin, vmax)

    def __axisCenterAndZoomChanged(self, changedAxis, vmin, vmax):
        if self.__locked:
            return
        self.__lastMainAxis = changedAxis
        changedAxis = changedAxis()
        with self.__inhibitSignals():
            center = self.__getAxesCenter(changedAxis, vmin, vmax)
            pixelRange = self.__getRangeInPixel(changedAxis)
            if pixelRange == 0:
                return
            pixelSize = (vmax - vmin) / pixelRange
            for axis in self.__axesToUpdate(changedAxis):
                vmin, vmax = self.__getLimitsFromCenter(axis, center, pixelSize)
                axis.setLimits(vmin, vmax)

    def __axisCenterChanged(self, changedAxis, vmin, vmax):
        if self.__locked:
            return
        self.__lastMainAxis = changedAxis
        changedAxis = changedAxis()
        with self.__inhibitSignals():
            center = self.__getAxesCenter(changedAxis, vmin, vmax)
            for axis in self.__axesToUpdate(changedAxis):
                vmin, vmax = self.__getLimitsFromCenter(axis, center)
                axis.setLimits(vmin, vmax)

    def __axisScaleChanged(self, changedAxis, scale):
        if self.__locked:
            return
        self.__lastMainAxis = changedAxis
        changedAxis = changedAxis()
        with self.__inhibitSignals():
            for axis in self.__axesToUpdate(changedAxis):
                axis.setScale(scale)

    def __axisInvertedChanged(self, changedAxis, isInverted):
        if self.__locked:
            return
        self.__lastMainAxis = changedAxis
        changedAxis = changedAxis()
        with self.__inhibitSignals():
            for axis in self.__axesToUpdate(changedAxis):
                axis.setInverted(isInverted)
