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
__date__ = "23/02/2018"

import functools
import logging
from contextlib import contextmanager
import weakref
import silx.utils.weakref as silxWeakref

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

    def __init__(self, axes, syncLimits=True, syncScale=True, syncDirection=True):
        """
        Constructor

        :param list(Axis) axes: A list of axes to synchronize together
        :param bool syncLimits: Synchronize axes limits
        :param bool syncScale: Synchronize axes scale
        :param bool syncDirection: Synchronize axes direction
        """
        object.__init__(self)
        self.__locked = False
        self.__axisRefs = []
        self.__syncLimits = syncLimits
        self.__syncScale = syncScale
        self.__syncDirection = syncDirection
        self.__callbacks = None

        for axis in axes:
            self.__axisRefs.append(weakref.ref(axis))

        self.start()

    def start(self):
        """Start synchronizing axes together.

        The first axis is used as the reference for the first synchronization.
        After that, any changes to any axes will be used to synchronize other
        axes.
        """
        if self.__callbacks is not None:
            raise RuntimeError("Axes already synchronized")
        self.__callbacks = {}

        axes = self.__getAxes()
        if len(axes) == 0:
            raise RuntimeError('No axis to synchronize')

        # register callback for further sync
        for axis in axes:
            refAxis = weakref.ref(axis)
            callbacks = []
            if self.__syncLimits:
                # the weakref is needed to be able ignore self references
                callback = silxWeakref.WeakMethodProxy(self.__axisLimitsChanged)
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

            self.__callbacks[refAxis] = callbacks

        # sync the current state
        mainAxis = axes[0]
        refMainAxis = weakref.ref(mainAxis)
        if self.__syncLimits:
            self.__axisLimitsChanged(refMainAxis, *mainAxis.getLimits())
        if self.__syncScale:
            self.__axisScaleChanged(refMainAxis, mainAxis.getScale())
        if self.__syncDirection:
            self.__axisInvertedChanged(refMainAxis, mainAxis.isInverted())

    def stop(self):
        """Stop the synchronization of the axes"""
        if self.__callbacks is None:
            raise RuntimeError("Axes not synchronized")
        for ref, callbacks in self.__callbacks.items():
            axis = ref()
            if axis is not None and _isQObjectValid(axis):
                for sigName, callback in callbacks:
                    sig = getattr(axis, sigName)
                    sig.disconnect(callback)
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

    def __otherAxes(self, changedAxis):
        for axis in self.__getAxes():
            if axis is changedAxis:
                continue
            yield axis

    def __axisLimitsChanged(self, changedAxis, vmin, vmax):
        if self.__locked:
            return
        changedAxis = changedAxis()
        with self.__inhibitSignals():
            for axis in self.__otherAxes(changedAxis):
                axis.setLimits(vmin, vmax)

    def __axisScaleChanged(self, changedAxis, scale):
        if self.__locked:
            return
        changedAxis = changedAxis()
        with self.__inhibitSignals():
            for axis in self.__otherAxes(changedAxis):
                axis.setScale(scale)

    def __axisInvertedChanged(self, changedAxis, isInverted):
        if self.__locked:
            return
        changedAxis = changedAxis()
        with self.__inhibitSignals():
            for axis in self.__otherAxes(changedAxis):
                axis.setInverted(isInverted)
