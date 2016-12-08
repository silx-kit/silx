# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
Patches for Mac OS X
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "30/11/2016"


def patch_QUrl_toLocalFile():
    """Apply a monkey-patch on qt.QUrl to allow to reach filename when the URL
    come from a MIME data from a file drop. Without, `QUrl.toLocalName` with
    some version of Mac OS X returns a path which looks like
    `/.file/id=180.112`.

    Qt5 is or will be patch, but Qt4 and PySide are not.

    This fix uses the file URL and use an subprocess with an
    AppleScript. The script convert the URI into a posix path.
    The interpreter (osascript) is available on default OS X installs.

    See https://bugreports.qt.io/browse/QTBUG-40449
    """
    from ._qt import QUrl
    import subprocess

    def QUrl_toLocalFile(self):
        path = QUrl._oldToLocalFile(self)
        if not path.startswith("/.file/id="):
            return path

        url = self.toString()
        script = 'get posix path of my posix file \"%s\" -- kthxbai' % url
        try:
            p = subprocess.Popen(["osascript", "-e", script], stdout=subprocess.PIPE)
            out, _err = p.communicate()
            if p.returncode == 0:
                return out.strip()
        except OSError:
            pass
        return path

    QUrl._oldToLocalFile = QUrl.toLocalFile
    QUrl.toLocalFile = QUrl_toLocalFile
