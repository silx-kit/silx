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
__date__ = "12/05/2016"


from . import qt
from ..resources import resource_filename


def getQIcon(name):
    """Create a QIcon from its name.

    :param str name: Name of the icon, in one of the defined icons
                     in this module.
    :return: Corresponding QIcon
    :raises: ValueError when name is not known
    """
    try:
        filename = resource_filename('gui/icons/%s.png' % name)
    except ValueError:
        raise ValueError('Not an icon name: %s' % name)

    return qt.QIcon(filename)


def getQPixmap(name):
    """Create a QPixmap from its name.

    :param str name: Name of the icon, in one of the defined icons
                     in this module.
    :return: Corresponding QPixmap
    :raises: ValueError when name is not known
    """
    try:
        filename = resource_filename('icons/%s.png')
    except ValueError:
        raise ValueError('Not an icon name: %s' % name)

    return qt.QPixmap(filename)
