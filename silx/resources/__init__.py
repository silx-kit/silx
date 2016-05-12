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
"""Access project's data and documentation files.

All access to data and documentation files MUST be made through the functions
of this modules to ensure access accross different software distribution
schemes (i.e., Linux packaging, zipped package)
"""

__authors__ = ["V.A. Sole", "Thomas Vincent"]
__license__ = "MIT"
__date__ = "12/05/2016"


import os
import pkg_resources


# This should only be used for Linux packaging:
# Patch this variable if data is not installed the source directory
_RESOURCES_DIR = None

# This should only be used for Linux packaging:
# Patch this variable if documentation is not installed in a doc/ subfolder
# of the resources directory
_RESOURCES_DOC_DIR = None


def resource_filename(resource):
    """Return filename corresponding to resource.

    resource can be the name of either a file or a directory.
    The existence of the resource is not checked.

    :param str resource: Resource path relative to resource directory
                         using '/' path separator.
    :return: Absolute resource path in the file system
    """
    if _RESOURCES_DIR is not None:
        return os.path.join(_RESOURCES_DIR, *resource.split('/'))
    else:
        return pkg_resources.resource_filename(__name__, resource)


def doc_filename(resource):
    """Return filename corresponding to documentation resource.

    resource can be the name of either a file or a directory.
    The existence of the resource is not checked.

    :param str resource: Resource path relative to documentation directory
                         using '/'-separated path.
    :return: Absolute resource path in the file system
    """
    if _RESOURCES_DOC_DIR is not None:
        return os.path.join(_RESOURCES_DOC_DIR, *resource.split('/'))
    else:
        return resource_filename('doc/' + resource)
