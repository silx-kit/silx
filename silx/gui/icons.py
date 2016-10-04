# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Set of icons for buttons.

Use :func:`getQIcon` to create Qt QIcon from the name identifying an icon.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "15/09/2016"


import logging
import weakref
from . import qt
from ..resources import resource_filename
from ..utils import weakref as silxweakref


_logger = logging.getLogger(__name__)
"""Module logger"""


_cached_icons = weakref.WeakValueDictionary()
"""Cache loaded icons in a weak structure"""


_supported_formats = None
"""Order of file format extension to check"""


_process_working = None
"""Cache an AnimatedIcon for working process"""


def getWaitIcon():
    """Returns a cached version of the waiting AnimatedIcon.

    :rtype: AnimatedIcon
    """
    global _process_working
    if _process_working is None:
        _process_working = AnimatedIcon("process-working")
    return _process_working


class AnimatedIcon(qt.QObject):
    """Store a looping QMovie to provide icons for each frames.
    Provides an event with the new icon everytime the movie frame
    is updated."""

    def __init__(self, filename, parent=None):
        """Constructor

        :param str filename: An icon name to an animated format
        :param qt.QObject parent: Parent of the QObject
        :raises: ValueError when name is not known
        """
        qt.QObject.__init__(self, parent)

        qfile = getQFile(filename)
        self.__movie = qt.QMovie(qfile.fileName(), qt.QByteArray(), parent)
        self.__movie.setCacheMode(qt.QMovie.CacheAll)
        self.__movie.frameChanged.connect(self.__frameChanged)

        self.__targets = silxweakref.WeakList()
        self.__currentIcon = None
        self.__cacheIcons = {}

        self.__movie.jumpToFrame(0)
        self.__updateIconAtFrame(0)

    iconChanged = qt.Signal(qt.QIcon)
    """Signal sent with a QIcon everytime the animation changed."""

    def __frameChanged(self, frameId):
        """Callback everytime the QMovie frame change
        :param int frameId: Current frame id
        """
        self.__updateIconAtFrame(frameId)

    def __updateIconAtFrame(self, frameId):
        """
        Update the current stored QIcon

        :param int frameId: Current frame id
        """
        if frameId in self.__cacheIcons:
            self.__currentIcon = self.__cacheIcons[frameId]
        else:
            self.__currentIcon = qt.QIcon(self.__movie.currentPixmap())
            self.__cacheIcons[frameId] = self.__currentIcon
        self.iconChanged.emit(self.__currentIcon)

    def register(self, obj):
        """Register an object to the AnimatedIcon.
        If no object are registred, the animation is paused.
        Object are stored in a weaked list.

        :param object obj: An object
        """
        if obj not in self.__targets:
            self.__targets.append(obj)
        self.__updateMovie()

    def unregister(self, obj):
        """Remove the object from the registration.
        If no object are registred the animation is paused.

        :param object obj: A registered object
        """
        if obj in self.__targets:
            self.__targets.remove(obj)
        self.__updateMovie()

    def isRegistered(self, obj):
        """Returns true if the object is registred in the AnimatedIcon.

        :param object obj: An object
        """
        return obj in self.__targets

    def __updateMovie(self):
        """Update the movie play according to internal stat of the
        AnimatedIcon."""
        # FIXME it should take care of the item count of the registred list
        self.__movie.setPaused(len(self.__targets) == 0)

    def currentIcon(self):
        """Returns the icon of the current frame.

        :rtype: qt.QIcon
        """
        return self.__currentIcon


def getQIcon(name):
    """Create a QIcon from its name.

    :param str name: Name of the icon, in one of the defined icons
                     in this module.
    :return: Corresponding QIcon
    :raises: ValueError when name is not known
    """
    if name not in _cached_icons:
        qfile = getQFile(name)
        icon = qt.QIcon(qfile.fileName())
        _cached_icons[name] = icon
    else:
        icon = _cached_icons[name]
    return icon


def getQPixmap(name):
    """Create a QPixmap from its name.

    :param str name: Name of the icon, in one of the defined icons
                     in this module.
    :return: Corresponding QPixmap
    :raises: ValueError when name is not known
    """
    qfile = getQFile(name)
    return qt.QPixmap(qfile.fileName())


def getQFile(name):
    """Create a QFile from an icon name. Filename is found
    according to supported Qt formats.

    :param str name: Name of the icon, in one of the defined icons
                     in this module.
    :return: Corresponding QFile
    :rtype: qt.QFile
    :raises: ValueError when name is not known
    """
    global _supported_formats
    if _supported_formats is None:
        _supported_formats = []
        supported_formats = qt.supportedImageFormats()
        order = ["mng", "gif", "svg", "png", "jpg"]
        for format_ in order:
            if format_ in supported_formats:
                _supported_formats.append(format_)
        if len(_supported_formats) == 0:
            _logger.error("No format supported for icons")
        else:
            _logger.debug("Format %s supported", ", ".join(_supported_formats))

    for format_ in _supported_formats:
        format_ = str(format_)
        filename = resource_filename('gui/icons/%s.%s' % (name, format_))
        qfile = qt.QFile(filename)
        if qfile.exists():
            return qfile
    raise ValueError('Not an icon name: %s' % name)
