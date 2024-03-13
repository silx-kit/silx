# /*##########################################################################
#
# Copyright (c) 2016-2023 European Synchrotron Radiation Facility
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
of this modules to ensure access across different distribution schemes:

- Installing from source or from wheel
- Installing package as a zip
- Linux packaging willing to install data files (and doc files) in
  alternative folders. In this case, this file must be patched.
- Frozen fat binary application using silx (frozen with cx_Freeze or py2app).
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
from __future__ import annotations

__authors__ = ["V.A. Sole", "Thomas Vincent", "J. Kieffer"]
__license__ = "MIT"
__date__ = "08/03/2019"


import atexit
import contextlib
import functools
import logging
import os
import sys
from typing import NamedTuple, Optional

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources


logger = logging.getLogger(__name__)


# For packaging purpose, patch this variable to use an alternative directory
# E.g., replace with _RESOURCES_DIR = '/usr/share/silx/data'
_RESOURCES_DIR = None

# For packaging purpose, patch this variable to use an alternative directory
# E.g., replace with _RESOURCES_DIR = '/usr/share/silx/doc'
# Not in use, uncomment when functionality is needed
# _RESOURCES_DOC_DIR = None

# cx_Freeze frozen support
# See http://cx-freeze.readthedocs.io/en/latest/faq.html#using-data-files
if getattr(sys, "frozen", False):
    # Running in a frozen application:
    # We expect resources to be located either in a silx/resources/ dir
    # relative to the executable or within this package.
    _dir = os.path.join(os.path.dirname(sys.executable), "silx", "resources")
    if os.path.isdir(_dir):
        _RESOURCES_DIR = _dir


class _ResourceDirectory(NamedTuple):
    """Store a source of resources"""

    package_name: str
    forced_path: Optional[str] = None


_SILX_DIRECTORY = _ResourceDirectory(package_name=__name__, forced_path=_RESOURCES_DIR)

_RESOURCE_DIRECTORIES = {}
_RESOURCE_DIRECTORIES["silx"] = _SILX_DIRECTORY


def register_resource_directory(
    name: str, package_name: str, forced_path: Optional[str] = None
):
    """Register another resource directory to the available list.

    By default only the directory "silx" is available.

    .. versionadded:: 0.6

    :param name: Name of the resource directory. It is used on the resource
        name to specify the resource directory to use. The resource
        "silx:foo.png" will use the "silx" resource directory.
    :param package_name: Python name of the package containing resources.
        For example "silx.resources".
    :param forced_path: Path containing the resources. If specified
        neither `importlib` nor `package_name` will be used
        For example "silx.resources".
    :raises ValueError: If the resource directory name already exists.
    """
    if name in _RESOURCE_DIRECTORIES:
        raise ValueError("Resource directory name %s already exists" % name)
    resource_directory = _ResourceDirectory(
        package_name=package_name, forced_path=forced_path
    )
    _RESOURCE_DIRECTORIES[name] = resource_directory


def list_dir(resource: str) -> list[str]:
    """List the content of a resource directory.

    Result are not prefixed by the resource name.

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx". See also :func:`register_resource_directory`.

    :param resource: Name of the resource directory to list
    :return: list of name contained in the directory
    """
    resource_directory, resource_name = _get_package_and_resource(resource)

    if resource_directory.forced_path is not None:
        # if set, use this directory
        path = resource_filename(resource)
        return os.listdir(path)

    path = importlib_resources.files(resource_directory.package_name) / resource_name
    return [entry.name for entry in path.iterdir()]


def is_dir(resource: str) -> bool:
    """True is the resource is a resource directory.

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx". See also :func:`register_resource_directory`.

    :param resource: Name of the resource
    """
    path = resource_filename(resource)
    return os.path.isdir(path)


def exists(resource: str) -> bool:
    """True is the resource exists.

    :param resource: Name of the resource
    """
    path = resource_filename(resource)
    return os.path.exists(path)


def _get_package_and_resource(
    resource: str, default_directory: Optional[str] = None
) -> tuple[_ResourceDirectory, str]:
    """
    Return the resource directory class and a cleaned resource name without
    prefix.

    :param resource: Name of the resource with resource prefix.
    :param default_directory: If the resource is not prefixed, the resource
        will be searched on this default directory of the silx resource
        directory.
    :raises ValueError: If the resource name uses an unregistred resource
        directory name
    """
    if ":" in resource:
        prefix, resource = resource.split(":", 1)
    else:
        prefix = "silx"
        if default_directory is not None:
            resource = f"{default_directory}/{resource}"
    if prefix not in _RESOURCE_DIRECTORIES:
        raise ValueError("Resource '%s' uses an unregistred prefix", resource)
    resource_directory = _RESOURCE_DIRECTORIES[prefix]
    return resource_directory, resource


def resource_filename(resource: str) -> str:
    """Return filename corresponding to resource.

    The existence of the resource is not checked.

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx". See also :func:`register_resource_directory`.

    :param resource: Resource path relative to resource directory
                     using '/' path separator. It can be either a file or
                     a directory.
    :raises ValueError: If the resource name uses an unregistred resource
        directory name
    :return: Absolute resource path in the file system
    """
    return _resource_filename(resource, default_directory=None)


# Manage resource files life-cycle
_file_manager = contextlib.ExitStack()
atexit.register(_file_manager.close)


@functools.lru_cache(maxsize=None)
def _get_resource_filename(package: str, resource: str) -> str:
    """Returns path to requested resource in package

    :param package: Name of the package in which to look for the resource
    :param resource: Resource path relative to package using '/' path separator
    :return: Abolute resource path in the file system
    """
    # Caching prevents extracting the resource twice
    file_context = importlib_resources.as_file(
        importlib_resources.files(package) / resource
    )
    path = _file_manager.enter_context(file_context)
    return str(path.absolute())


def _resource_filename(resource: str, default_directory: Optional[str] = None) -> str:
    """Return filename corresponding to resource.

    The existence of the resource is not checked.

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx". See also :func:`register_resource_directory`.

    :param resource: Resource path relative to resource directory
                     using '/' path separator. It can be either a file or
                     a directory.
    :param default_directory: If the resource is not prefixed, the resource
        will be searched on this default directory of the silx resource
        directory. It should only be used internally by silx.
    :return: Absolute resource path in the file system
    """
    resource_directory, resource_name = _get_package_and_resource(
        resource, default_directory=default_directory
    )

    if resource_directory.forced_path is not None:
        # if set, use this directory
        base_dir = resource_directory.forced_path
        resource_path = os.path.join(base_dir, *resource_name.split("/"))
        return resource_path

    return _get_resource_filename(resource_directory.package_name, resource_name)


# Expose ExternalResources for compatibility (since silx 0.11)
from ..utils.ExternalResources import ExternalResources
