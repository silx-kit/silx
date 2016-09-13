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
"""This package provides a set of utilitary class and function used by the
package `silx.gui.hdf5` package.
"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "13/09/2016"


import os
import logging
from .. import qt

_logger = logging.getLogger(__name__)

try:
    import h5py
except ImportError as e:
    _logger.error("Module %s requires h5py", __name__)
    raise e


def load_file_as_h5py(filename):
    """
    Load a file as an h5py.File object

    :param str filename: A filename
    :raises: IOError if the file can't be loaded as an h5py.File like object
    :rtype: h5py.File
    """
    if not os.path.isfile(filename):
        raise IOError("Filename '%s' must be a file path" % filename)

    if h5py.is_hdf5(filename):
        return h5py.File(filename)

    try:
        from ..io import spech5
        return spech5.SpecH5(filename)
    except ImportError:
        _logger.debug("spech5 can't be loaded.", filename, exc_info=True)
    except IOError:
        _logger.debug("File '%s' can't be read as spec file.", filename, exc_info=True)

    try:
        from silx.io import fabioh5
        return fabioh5.File(filename)
    except ImportError:
        _logger.debug("fabioh5 can't be loaded.", filename, exc_info=True)
    except Exception:
        _logger.debug("File '%s' can't be read as fabio file.", filename, exc_info=True)

    raise IOError("Format of filename '%s' is not supported" % filename)


class Hdf5ContextMenuEvent(object):
    """Hold information provided to context menu callbacks."""

    def __init__(self, source, menu, hoveredObject):
        self.__source = source
        self.__menu = menu
        self.__hoveredObject = hoveredObject

    def source(self):
        """Source of the event

        :rtype: Hdf5TreeView
        """
        return self.__source

    def menu(self):
        """Menu which will be displayed

        :rtype: qt.QMenu
        """
        return self.__menu

    def hoveredObject(self):
        """Item content overed by the mouse when the context menu was
        requested

        :rtype: h5py.File or h5py.Dataset or h5py.Group
        """
        return self.__menu


def htmlFromDict(dictionary):
    """Generate a readable HTML from a dictionary

    :param dict dictionary: A Dictionary
    :rtype: str
    """
    result = "<html><ul>"
    for key, value in dictionary.items():
        result += "<li><b>%s</b>: %s</li>" % (key, value)
    result += "</ul></html>"
    return result


class Hdf5NodeMimeData(qt.QMimeData):
    """Mimedata class to identify an internal drag and drop of a Hdf5Node."""

    MIME_TYPE = "application/x-internal-h5py-node"

    def __init__(self, node=None):
        qt.QMimeData.__init__(self)
        self.__node = node
        self.setData(self.MIME_TYPE, "")

    def node(self):
        return self.__node
