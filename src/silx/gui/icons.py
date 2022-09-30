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
__date__ = "07/01/2019"


import os
import logging
import weakref
from . import qt
import silx.resources
from silx.utils import weakref as silxweakref
from silx.utils.deprecation import deprecated


_logger = logging.getLogger(__name__)
"""Module logger"""


_cached_icons = None
"""Cache loaded icons in a weak structure"""


def getIconCache():
    """Get access to all cached icons

    :rtype: dict
    """
    global _cached_icons
    if _cached_icons is None:
        _cached_icons = weakref.WeakValueDictionary()
        # Clean up the cache before leaving the application
        # See https://github.com/silx-kit/silx/issues/1771
        qt.QApplication.instance().aboutToQuit.connect(cleanIconCache)
    return _cached_icons


def cleanIconCache():
    """Clean up the icon cache"""
    _logger.debug("Clean up icon cache")
    _cached_icons.clear()


_supported_formats = None
"""Order of file format extension to check"""


class AbstractAnimatedIcon(qt.QObject):
    """Store an animated icon.

    It provides an event containing the new icon everytime it is updated."""

    def __init__(self, parent=None):
        """Constructor

        :param qt.QObject parent: Parent of the QObject
        :raises: ValueError when name is not known
        """
        qt.QObject.__init__(self, parent)

        self.__targets = silxweakref.WeakList()
        self.__currentIcon = None

    iconChanged = qt.Signal(qt.QIcon)
    """Signal sent with a QIcon everytime the animation changed."""

    def register(self, obj):
        """Register an object to the AnimatedIcon.
        If no object are registred, the animation is paused.
        Object are stored in a weaked list.

        :param object obj: An object
        """
        if obj not in self.__targets:
            self.__targets.append(obj)
        self._updateState()

    def unregister(self, obj):
        """Remove the object from the registration.
        If no object are registred the animation is paused.

        :param object obj: A registered object
        """
        if obj in self.__targets:
            self.__targets.remove(obj)
        self._updateState()

    def hasRegistredObjects(self):
        """Returns true if any object is registred.

        :rtype: bool
        """
        return len(self.__targets)

    def isRegistered(self, obj):
        """Returns true if the object is registred in the AnimatedIcon.

        :param object obj: An object
        :rtype: bool
        """
        return obj in self.__targets

    def currentIcon(self):
        """Returns the icon of the current frame.

        :rtype: qt.QIcon
        """
        return self.__currentIcon

    def _updateState(self):
        """Update the object according to the connected objects."""
        pass

    def _setCurrentIcon(self, icon):
        """Store the current icon and emit a `iconChanged` event.

        :param qt.QIcon icon: The current icon
        """
        self.__currentIcon = icon
        self.iconChanged.emit(self.__currentIcon)


class MovieAnimatedIcon(AbstractAnimatedIcon):
    """Store a looping QMovie to provide icons for each frames.
    Provides an event with the new icon everytime the movie frame
    is updated."""

    def __init__(self, filename, parent=None):
        """Constructor

        :param str filename: An icon name to an animated format
        :param qt.QObject parent: Parent of the QObject
        :raises: ValueError when name is not known
        """
        AbstractAnimatedIcon.__init__(self, parent)

        qfile = getQFile(filename)
        self.__movie = qt.QMovie(qfile.fileName(), qt.QByteArray(), parent)
        self.__movie.setCacheMode(qt.QMovie.CacheAll)
        self.__movie.frameChanged.connect(self.__frameChanged)
        self.__cacheIcons = {}

        self.__movie.jumpToFrame(0)
        self.__updateIconAtFrame(0)

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
            icon = self.__cacheIcons[frameId]
        else:
            icon = qt.QIcon(self.__movie.currentPixmap())
            self.__cacheIcons[frameId] = icon
        self._setCurrentIcon(icon)

    def _updateState(self):
        """Update the movie play according to internal stat of the
        AnimatedIcon."""
        self.__movie.setPaused(not self.hasRegistredObjects())


class MultiImageAnimatedIcon(AbstractAnimatedIcon):
    """Store a looping QMovie to provide icons for each frames.
    Provides an event with the new icon everytime the movie frame
    is updated."""

    def __init__(self, filename, parent=None):
        """Constructor

        :param str filename: An icon name to an animated format
        :param qt.QObject parent: Parent of the QObject
        :raises: ValueError when name is not known
        """
        AbstractAnimatedIcon.__init__(self, parent)

        self.__frames = []
        for i in range(100):
            try:
                frame_filename = os.sep.join((filename, ("%02d" %i)))
                frame_file = getQFile(frame_filename)
            except ValueError:
                break
            try:
                icon = qt.QIcon(frame_file.fileName())
            except ValueError:
                break
            self.__frames.append(icon)

        if len(self.__frames) == 0:
            raise ValueError("Animated icon '%s' do not exists" % filename)

        self.__frameId = -1
        self.__timer = qt.QTimer(self)
        self.__timer.timeout.connect(self.__increaseFrame)
        self.__updateIconAtFrame(0)

    def __increaseFrame(self):
        """Callback called every timer timeout to change the current frame of
        the animation
        """
        frameId = (self.__frameId + 1) % len(self.__frames)
        self.__updateIconAtFrame(frameId)

    def __updateIconAtFrame(self, frameId):
        """
        Update the current stored QIcon

        :param int frameId: Current frame id
        """
        self.__frameId = frameId
        icon = self.__frames[frameId]
        self._setCurrentIcon(icon)

    def _updateState(self):
        """Update the object to wake up or sleep it according to its use."""
        if self.hasRegistredObjects():
            if not self.__timer.isActive():
                self.__timer.start(100)
        else:
            if self.__timer.isActive():
                self.__timer.stop()


class AnimatedIcon(MovieAnimatedIcon):
    """Store a looping QMovie to provide icons for each frames.
    Provides an event with the new icon everytime the movie frame
    is updated.

    It may not be available anymore for the silx release 0.6.

    .. deprecated:: 0.5
       Use :class:`MovieAnimatedIcon` instead.
    """

    @deprecated
    def __init__(self, filename, parent=None):
        MovieAnimatedIcon.__init__(self, filename, parent=parent)


def getWaitIcon():
    """Returns a cached version of the waiting AbstractAnimatedIcon.

    :rtype: AbstractAnimatedIcon
    """
    return getAnimatedIcon("process-working")


def getAnimatedIcon(name):
    """Create an AbstractAnimatedIcon from a resource name.

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx".

    If no prefix are specified, the file with be returned from the silx
    resource directory with a specific path "gui/icons".

    See also :func:`silx.resources.register_resource_directory`.

    Try to load a mng or a gif file, then try to load a multi-image animated
    icon.

    In Qt5 mng or gif are not used, because the transparency is not very well
    managed.

    :param str name: Name of the icon, in one of the defined icons
                     in this module.
    :return: Corresponding AbstractAnimatedIcon
    :raises: ValueError when name is not known
    """
    key = name + "__anim"
    cached_icons = getIconCache()
    if key not in cached_icons:

        qtMajorVersion = int(qt.qVersion().split(".")[0])
        icon = None

        # ignore mng and gif in Qt5
        if qtMajorVersion != 5:
            try:
                icon = MovieAnimatedIcon(name)
            except ValueError:
                icon = None

        if icon is None:
            try:
                icon = MultiImageAnimatedIcon(name)
            except ValueError:
                icon = None

        if icon is None:
            raise ValueError("Not an animated icon name: %s", name)

        cached_icons[key] = icon
    else:
        icon = cached_icons[key]
    return icon


def getQIcon(name):
    """Create a QIcon from its name.

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx".

    If no prefix are specified, the file with be returned from the silx
    resource directory with a specific path "gui/icons".

    See also :func:`silx.resources.register_resource_directory`.

    :param str name: Name of the icon, in one of the defined icons
                     in this module.
    :return: Corresponding QIcon
    :raises: ValueError when name is not known
    """
    cached_icons = getIconCache()
    if name not in cached_icons:
        qfile = getQFile(name)
        icon = qt.QIcon(qfile.fileName())
        cached_icons[name] = icon
    else:
        icon = cached_icons[name]
    return icon


def getQPixmap(name):
    """Create a QPixmap from its name.

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx".

    If no prefix are specified, the file with be returned from the silx
    resource directory with a specific path "gui/icons".

    See also :func:`silx.resources.register_resource_directory`.

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

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx".

    If no prefix are specified, the file with be returned from the silx
    resource directory with a specific path "gui/icons".

    See also :func:`silx.resources.register_resource_directory`.

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
        filename = silx.resources._resource_filename('%s.%s' % (name, format_),
                                                     default_directory=os.path.join('gui', 'icons'))
        qfile = qt.QFile(filename)
        if qfile.exists():
            return qfile
        _logger.debug("File '%s' not found.", filename)
    raise ValueError('Not an icon name: %s' % name)
