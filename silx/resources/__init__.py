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
of this modules to ensure access accross different distribution schemes:

- Installing from source or from wheel
- Installing package as a zip (through the use of pkg_resources)
- Linux packaging willing to install data files (and doc files) in
  alternative folders. In this case, this file must be patched.
- Frozen fat binary application using silx (forzen with cx_Freeze or py2app).
  This needs special care for the resource files in the setup:

  - With cx_Freeze, add silx/resources to include_files:

    .. code-block:: python

       import silx.resources
       silx_include_files = (os.path.dirname(silx.resources.__file__),
                             os.path.join('silx', 'resources'))
       setup(...
             options={'build_exe': {'include_files': [silx_include_files]}}
             )

  - With py2app, add silx in the packages list of the py2app options:

    .. code-block:: python

       setup(...
             options={'py2app': {'packages': ['silx']}}
             )
"""

__authors__ = ["V.A. Sole", "Thomas Vincent"]
__license__ = "MIT"
__date__ = "12/05/2016"


import os
import sys

# pkg_resources is useful when this package is stored in a zip
# When pkg_resources is not available, the resources dir defaults to the
# directory containing this module.
try:
    import pkg_resources
except ImportError:
    pkg_resources = None


# For packaging purpose, patch this variable to use an alternative directory
# E.g., replace with _RESOURCES_DIR = '/usr/share/silx/data'
_RESOURCES_DIR = None

# For packaging purpose, patch this variable to use an alternative directory
# E.g., replace with _RESOURCES_DIR = '/usr/share/silx/doc'
# Not in use, uncomment when functionnality is needed
# _RESOURCES_DOC_DIR = None

# cx_Freeze forzen support
# See http://cx-freeze.readthedocs.io/en/latest/faq.html#using-data-files
if getattr(sys, 'frozen', False):
    # Running in a frozen application:
    # We expect resources to be located either in a silx/resources/ dir
    # relative to the executable or within this package.
    _dir = os.path.join(os.path.dirname(sys.executable), 'silx', 'resources')
    if os.path.isdir(_dir):
        _RESOURCES_DIR = _dir


def resource_filename(resource):
    """Return filename corresponding to resource.

    resource can be the name of either a file or a directory.
    The existence of the resource is not checked.

    :param str resource: Resource path relative to resource directory
                         using '/' path separator.
    :return: Absolute resource path in the file system
    """
    # Not in use, uncomment when functionnality is needed
    # If _RESOURCES_DOC_DIR is set, use it to get resources in doc/ subflodler
    # from an alternative directory.
    # if _RESOURCES_DOC_DIR is not None and (resource is 'doc' or
    #         resource.startswith('doc/')):
    #     # Remove doc folder from resource relative path
    #     return os.path.join(_RESOURCES_DOC_DIR, *resource.split('/')[1:])

    if _RESOURCES_DIR is not None:  # if set, use this directory
        return os.path.join(_RESOURCES_DIR, *resource.split('/'))
    elif pkg_resources is None:  # Fallback if pkg_resources is not available
        return os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            *resource.split('/'))
    else:  # Preferred way to get resources as it supports zipfile package
        return pkg_resources.resource_filename(__name__, resource)
