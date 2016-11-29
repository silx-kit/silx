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
"""Utils function relative to HTML
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "19/09/2016"


def escape(string, quote=True):
    """Returns a string where HTML metacharacters are properly escaped.

    Compatibility layer to avoid incompatibilities between Python versions,
    Qt versions and Qt bindings.

    >>> import silx.utils.html
    >>> silx.utils.html.escape("<html>")
    >>> "&lt;html&gt;"

    .. note:: Since Python 3.3 you can use the `html` module. For previous
        version, it is provided by `sgi` module.
    .. note:: Qt4 provides it with `Qt.escape` while Qt5 provide it with
        `QString.toHtmlEscaped`. But `QString` is not exposed by `PyQt` or
        `PySide`.

    :param str string: Human readable string.
    :param bool quote: Escape quote and double quotes (default behaviour).
    :returns: Valid HTML syntax to display the input string.
    :rtype: str
    """
    string = string.replace("&", "&amp;")  # must be done first
    string = string.replace("<", "&lt;")
    string = string.replace(">", "&gt;")
    if quote:
        string = string.replace("'", "&apos;")
        string = string.replace("\"", "&quot;")
    return string
