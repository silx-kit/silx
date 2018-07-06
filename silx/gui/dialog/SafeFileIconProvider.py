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
"""
This module contains :class:`SafeIconProvider`.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "31/10/2017"

import sys
import logging
from silx.gui import qt


_logger = logging.getLogger(__name__)


class SafeFileIconProvider(qt.QFileIconProvider):
    """
    This class reimplement :class:`qt.QFileIconProvider` to avoid blocking
    access to the file system.

    It avoid to use `qt.QFileInfo.absoluteFilePath` or
    `qt.QFileInfo.canonicalPath` to reach drive icons which are known to
    freeze the file system using network drives.

    Computer root, and drive root paths are filtered. Other paths are not
    filtered while it is anyway needed to synchronoze a drive to accesss to it.
    """

    WIN32_DRIVE_UNKNOWN = 0
    """The drive type cannot be determined."""
    WIN32_DRIVE_NO_ROOT_DIR = 1
    """The root path is invalid; for example, there is no volume mounted at the
    specified path."""
    WIN32_DRIVE_REMOVABLE = 2
    """The drive has removable media; for example, a floppy drive, thumb drive,
    or flash card reader."""
    WIN32_DRIVE_FIXED = 3
    """The drive has fixed media; for example, a hard disk drive or flash
    drive."""
    WIN32_DRIVE_REMOTE = 4
    """The drive is a remote (network) drive."""
    WIN32_DRIVE_CDROM = 5
    """The drive is a CD-ROM drive."""
    WIN32_DRIVE_RAMDISK = 6
    """The drive is a RAM disk."""

    def __init__(self):
        qt.QFileIconProvider.__init__(self)
        self.__filterDirAndFiles = False
        if sys.platform == "win32":
            self._windowsTypes = {}
            item = "Drive", qt.QStyle.SP_DriveHDIcon
            self._windowsTypes[self.WIN32_DRIVE_UNKNOWN] = item
            item = "Invalid root", qt.QStyle.SP_DriveHDIcon
            self._windowsTypes[self.WIN32_DRIVE_NO_ROOT_DIR] = item
            item = "Removable", qt.QStyle.SP_DriveNetIcon
            self._windowsTypes[self.WIN32_DRIVE_REMOVABLE] = item
            item = "Drive", qt.QStyle.SP_DriveHDIcon
            self._windowsTypes[self.WIN32_DRIVE_FIXED] = item
            item = "Remote", qt.QStyle.SP_DriveNetIcon
            self._windowsTypes[self.WIN32_DRIVE_REMOTE] = item
            item = "CD-ROM", qt.QStyle.SP_DriveCDIcon
            self._windowsTypes[self.WIN32_DRIVE_CDROM] = item
            item = "RAM disk", qt.QStyle.SP_DriveHDIcon
            self._windowsTypes[self.WIN32_DRIVE_RAMDISK] = item

    def __windowsDriveTypeId(self, info):
        try:
            import ctypes
            path = info.filePath()
            dtype = ctypes.cdll.kernel32.GetDriveTypeW(path)
        except Exception:
            _logger.warning("Impossible to identify drive %s" % path)
            _logger.debug("Backtrace", exc_info=True)
            return self.WIN32_DRIVE_UNKNOWN
        return dtype

    def __windowsDriveIcon(self, info):
        dtype = self.__windowsDriveTypeId(info)
        default = self._windowsTypes[self.WIN32_DRIVE_UNKNOWN]
        driveInfo = self._windowsTypes.get(dtype, default)
        style = qt.QApplication.instance().style()
        icon = style.standardIcon(driveInfo[1])
        return icon

    def __windowsDriveType(self, info):
        dtype = self.__windowsDriveTypeId(info)
        default = self._windowsTypes[self.WIN32_DRIVE_UNKNOWN]
        driveInfo = self._windowsTypes.get(dtype, default)
        return driveInfo[0]

    def icon(self, info):
        if isinstance(info, qt.QFileIconProvider.IconType):
            # It's another C++ method signature:
            # QIcon QFileIconProvider::icon(QFileIconProvider::IconType type)
            return super(SafeFileIconProvider, self).icon(info)
        style = qt.QApplication.instance().style()
        path = info.filePath()
        if path in ["", "/"]:
            # That's the computer root on Windows or Linux
            result = style.standardIcon(qt.QStyle.SP_ComputerIcon)
        elif sys.platform == "win32" and path[-2] == ":":
            # That's a drive on Windows
            result = self.__windowsDriveIcon(info)
        elif self.__filterDirAndFiles:
            if info.isDir():
                result = style.standardIcon(qt.QStyle.SP_DirIcon)
            else:
                result = style.standardIcon(qt.QStyle.SP_FileIcon)
        else:
            result = qt.QFileIconProvider.icon(self, info)
        return result

    def type(self, info):
        path = info.filePath()
        if path in ["", "/"]:
            # That's the computer root on Windows or Linux
            result = "Computer"
        elif sys.platform == "win32" and path[-2] == ":":
            # That's a drive on Windows
            result = self.__windowsDriveType(info)
        elif self.__filterDirAndFiles:
            if info.isDir():
                result = "Directory"
            else:
                result = info.suffix()
        else:
            result = qt.QFileIconProvider.type(self, info)
        return result
