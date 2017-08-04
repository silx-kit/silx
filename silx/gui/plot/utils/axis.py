# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
__date__ = "04/08/2017"

import functools
import logging
from contextlib import contextmanager
from silx.utils import weakref

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
        self.__axes = []
        self.__locked = False
        self.__syncLimits = syncLimits
        self.__syncScale = syncScale
        self.__syncDirection = syncDirection
        self.__callbacks = []

        self.__axes.extend(axes)
        self.start()

    def start(self):
        """Start synchronizing axes together.

        The first axis is used as the reference for the first synchronization.
        After that, any changes to any axes will be used to synchronize other
        axes.
        """
        if len(self.__callbacks) != 0:
            raise RuntimeError("Axes already synchronized")

        # register callback for further sync
        for axis in self.__axes:
            if self.__syncLimits:
                # the weakref is needed to be able ignore self references
                callback = weakref.WeakMethodProxy(self.__axisLimitsChanged)
                callback = functools.partial(callback, axis)
                sig = axis.sigLimitsChanged
                sig.connect(callback)
                self.__callbacks.append((sig, callback))
            if self.__syncScale:
                # the weakref is needed to be able ignore self references
                callback = weakref.WeakMethodProxy(self.__axisScaleChanged)
                callback = functools.partial(callback, axis)
                sig = axis.sigScaleChanged
                sig.connect(callback)
                self.__callbacks.append((sig, callback))
            if self.__syncDirection:
                # the weakref is needed to be able ignore self references
                callback = weakref.WeakMethodProxy(self.__axisInvertedChanged)
                callback = functools.partial(callback, axis)
                sig = axis.sigInvertedChanged
                sig.connect(callback)
                self.__callbacks.append((sig, callback))

        # sync the current state
        mainAxis = self.__axes[0]
        if self.__syncLimits:
            self.__axisLimitsChanged(mainAxis, *mainAxis.getLimits())
        if self.__syncScale:
            self.__axisScaleChanged(mainAxis, mainAxis.getScale())
        if self.__syncDirection:
            self.__axisInvertedChanged(mainAxis, mainAxis.isInverted())

    def stop(self):
        """Stop the synchronization of the axes"""
        if len(self.__callbacks) == 0:
            raise RuntimeError("Axes not synchronized")
        for sig, callback in self.__callbacks:
            sig.disconnect(callback)
        self.__callbacks = []

    def __del__(self):
        """Destructor"""
        # clean up references
        if len(self.__callbacks) != 0:
            self.stop()

    @contextmanager
    def __inhibitSignals(self):
        self.__locked = True
        yield
        self.__locked = False

    def __otherAxes(self, changedAxis):
        for axis in self.__axes:
            if axis is changedAxis:
                continue
            yield axis

    def __axisLimitsChanged(self, changedAxis, vmin, vmax):
        if self.__locked:
            return
        with self.__inhibitSignals():
            for axis in self.__otherAxes(changedAxis):
                axis.setLimits(vmin, vmax)

    def __axisScaleChanged(self, changedAxis, scale):
        if self.__locked:
            return
        with self.__inhibitSignals():
            for axis in self.__otherAxes(changedAxis):
                axis.setScale(scale)

    def __axisInvertedChanged(self, changedAxis, isInverted):
        if self.__locked:
            return
        with self.__inhibitSignals():
            for axis in self.__otherAxes(changedAxis):
                axis.setInverted(isInverted)
