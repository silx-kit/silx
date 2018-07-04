# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2018 European Synchrotron Radiation Facility
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
- Installing package as a zip (through the use of pkg_resources)
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

__authors__ = ["V.A. Sole", "Thomas Vincent", "J. Kieffer"]
__license__ = "MIT"
__date__ = "15/02/2018"


import os
import sys
import threading
import json
import getpass
import logging
import tempfile
import unittest
import importlib
from silx.third_party import six
logger = logging.getLogger(__name__)


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
# Not in use, uncomment when functionality is needed
# _RESOURCES_DOC_DIR = None

# cx_Freeze frozen support
# See http://cx-freeze.readthedocs.io/en/latest/faq.html#using-data-files
if getattr(sys, 'frozen', False):
    # Running in a frozen application:
    # We expect resources to be located either in a silx/resources/ dir
    # relative to the executable or within this package.
    _dir = os.path.join(os.path.dirname(sys.executable), 'silx', 'resources')
    if os.path.isdir(_dir):
        _RESOURCES_DIR = _dir


class _ResourceDirectory(object):
    """Store a source of resources"""

    def __init__(self, package_name, package_path=None, forced_path=None):
        if forced_path is None:
            if package_path is None:
                if pkg_resources is None:
                    # In this case we have to compute the package path
                    # Else it will not be used
                    module = importlib.import_module(package_name)
                    package_path = os.path.abspath(os.path.dirname(module.__file__))
        self.package_name = package_name
        self.package_path = package_path
        self.forced_path = forced_path


_SILX_DIRECTORY = _ResourceDirectory(
    package_name=__name__,
    package_path=os.path.abspath(os.path.dirname(__file__)),
    forced_path=_RESOURCES_DIR)

_RESOURCE_DIRECTORIES = {}
_RESOURCE_DIRECTORIES["silx"] = _SILX_DIRECTORY


def register_resource_directory(name, package_name, forced_path=None):
    """Register another resource directory to the available list.

    By default only the directory "silx" is available.

    .. versionadded:: 0.6

    :param str name: Name of the resource directory. It is used on the resource
        name to specify the resource directory to use. The resource
        "silx:foo.png" will use the "silx" resource directory.
    :param str package_name: Python name of the package containing resources.
        For example "silx.resources".
    :param str forced_path: Path containing the resources. If specified
        `pkg_resources` nor `package_name` will be used
        For example "silx.resources".
    :raises ValueError: If the resource directory name already exists.
    """
    if name in _RESOURCE_DIRECTORIES:
        raise ValueError("Resource directory name %s already exists" % name)
    resource_directory = _ResourceDirectory(
        package_name=package_name,
        forced_path=forced_path)
    _RESOURCE_DIRECTORIES[name] = resource_directory


def list_dir(resource):
    """List the content of a resource directory.

    Result are not prefixed by the resource name.

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx". See also :func:`register_resource_directory`.

    :param str resource: Name of the resource directory to list
    :return: list of name contained in the directory
    :rtype: List
    """
    resource_directory, resource_name = _get_package_and_resource(resource)

    if resource_directory.forced_path is not None:
        # if set, use this directory
        path = resource_filename(resource)
        return os.listdir(path)
    elif pkg_resources is None:
        # Fallback if pkg_resources is not available
        path = resource_filename(resource)
        return os.listdir(path)
    else:
        # Preferred way to get resources as it supports zipfile package
        package_name = resource_directory.package_name
        return pkg_resources.resource_listdir(package_name, resource_name)


def is_dir(resource):
    """True is the resource is a resource directory.

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx". See also :func:`register_resource_directory`.

    :param str resource: Name of the resource
    :rtype: bool
    """
    path = resource_filename(resource)
    return os.path.isdir(path)


def exists(resource):
    """True is the resource exists.

    :param str resource: Name of the resource
    :rtype: bool
    """
    path = resource_filename(resource)
    return os.path.exists(path)


def _get_package_and_resource(resource, default_directory=None):
    """
    Return the resource directory class and a cleaned resource name without
    prefix.

    :param str: resource: Name of the resource with resource prefix.
    :param str default_directory: If the resource is not prefixed, the resource
        will be searched on this default directory of the silx resource
        directory.
    :rtype: tuple(_ResourceDirectory, str)
    :raises ValueError: If the resource name uses an unregistred resource
        directory name
    """
    if ":" in resource:
        prefix, resource = resource.split(":", 1)
    else:
        prefix = "silx"
        if default_directory is not None:
            resource = os.path.join(default_directory, resource)
    if prefix not in _RESOURCE_DIRECTORIES:
        raise ValueError("Resource '%s' uses an unregistred prefix", resource)
    resource_directory = _RESOURCE_DIRECTORIES[prefix]
    return resource_directory, resource


def resource_filename(resource):
    """Return filename corresponding to resource.

    The existence of the resource is not checked.

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx". See also :func:`register_resource_directory`.

    :param str resource: Resource path relative to resource directory
                         using '/' path separator. It can be either a file or
                         a directory.
    :raises ValueError: If the resource name uses an unregistred resource
        directory name
    :return: Absolute resource path in the file system
    :rtype: str
    """
    return _resource_filename(resource, default_directory=None)


def _resource_filename(resource, default_directory=None):
    """Return filename corresponding to resource.

    The existence of the resource is not checked.

    The resource name can be prefixed by the name of a resource directory. For
    example "silx:foo.png" identify the resource "foo.png" from the resource
    directory "silx". See also :func:`register_resource_directory`.

    :param str resource: Resource path relative to resource directory
                         using '/' path separator. It can be either a file or
                         a directory.
    :param str default_directory: If the resource is not prefixed, the resource
        will be searched on this default directory of the silx resource
        directory. It should only be used internally by silx.
    :return: Absolute resource path in the file system
    :rtype: str
    """
    resource_directory, resource_name = _get_package_and_resource(resource,
                                                                  default_directory=default_directory)

    if resource_directory.forced_path is not None:
        # if set, use this directory
        base_dir = resource_directory.forced_path
        resource_path = os.path.join(base_dir, *resource_name.split('/'))
        return resource_path
    elif pkg_resources is None:
        # Fallback if pkg_resources is not available
        base_dir = resource_directory.package_path
        resource_path = os.path.join(base_dir, *resource_name.split('/'))
        return resource_path
    else:
        # Preferred way to get resources as it supports zipfile package
        package_name = resource_directory.package_name
        return pkg_resources.resource_filename(package_name, resource_name)


class ExternalResources(object):
    """Utility class which allows to download test-data from www.silx.org
    and manage the temporary data during the tests.

    """

    def __init__(self, project,
                 url_base,
                 env_key=None,
                 timeout=60):
        """Constructor of the class

        :param str project: name of the project, like "silx"
        :param str url_base: base URL for the data, like "http://www.silx.org/pub"
        :param str env_key: name of the environment variable which contains the
                            test_data directory, like "SILX_DATA".
                            If None (default), then the name of the
                            environment variable is built from the project argument:
                            "<PROJECT>_DATA".
                            The environment variable is optional: in case it is not set,
                            a directory in the temporary folder is used.
        :param timeout: time in seconds before it breaks
        """
        self.project = project
        self._initialized = False
        self.sem = threading.Semaphore()
        self.env_key = env_key or (self.project.upper() + "_DATA")
        self.url_base = url_base
        self.all_data = set()
        self.timeout = timeout
        self.data_home = None

    def _initialize_data(self):
        """Initialize for downloading test data"""
        if not self._initialized:
            with self.sem:
                if not self._initialized:

                    self.data_home = os.environ.get(self.env_key)
                    if self.data_home is None:
                        self.data_home = os.path.join(tempfile.gettempdir(),
                                                      "%s_testdata_%s" % (self.project, getpass.getuser()))
                    if not os.path.exists(self.data_home):
                        os.makedirs(self.data_home)
                    self.testdata = os.path.join(self.data_home, "all_testdata.json")
                    if os.path.exists(self.testdata):
                        with open(self.testdata) as f:
                            self.all_data = set(json.load(f))
                    self._initialized = True

    def clean_up(self):
        pass

    def getfile(self, filename):
        """Downloads the requested file from web-server available
        at https://www.silx.org/pub/silx/

        :param: relative name of the image.
        :return: full path of the locally saved file.
        """
        logger.debug("ExternalResources.getfile('%s')", filename)

        if not self._initialized:
            self._initialize_data()

        if not os.path.exists(self.data_home):
            os.makedirs(self.data_home)

        fullfilename = os.path.abspath(os.path.join(self.data_home, filename))

        if not os.path.isfile(fullfilename):
            logger.debug("Trying to download image %s, timeout set to %ss",
                         filename, self.timeout)
            dictProxies = {}
            if "http_proxy" in os.environ:
                dictProxies['http'] = os.environ["http_proxy"]
                dictProxies['https'] = os.environ["http_proxy"]
            if "https_proxy" in os.environ:
                dictProxies['https'] = os.environ["https_proxy"]
            if dictProxies:
                proxy_handler = six.moves.urllib.request.ProxyHandler(dictProxies)
                opener = six.moves.urllib.request.build_opener(proxy_handler).open
            else:
                opener = six.moves.urllib.request.urlopen

            logger.debug("wget %s/%s", self.url_base, filename)
            try:
                data = opener("%s/%s" % (self.url_base, filename),
                              data=None, timeout=self.timeout).read()
                logger.info("Image %s successfully downloaded.", filename)
            except six.moves.urllib.error.URLError:
                raise unittest.SkipTest("network unreachable.")

            try:
                with open(fullfilename, "wb") as outfile:
                    outfile.write(data)
            except IOError:
                raise IOError("unable to write downloaded \
                    data to disk at %s" % self.data_home)

            if not os.path.isfile(fullfilename):
                raise RuntimeError(
                    "Could not automatically \
                    download test images %s!\n \ If you are behind a firewall, \
                    please set both environment variable http_proxy and https_proxy.\
                    This even works under windows ! \n \
                    Otherwise please try to download the images manually from \n%s/%s"
                    % (filename, self.url_base, filename))

        if filename not in self.all_data:
            self.all_data.add(filename)
            image_list = list(self.all_data)
            image_list.sort()
            try:
                with open(self.testdata, "w") as fp:
                    json.dump(image_list, fp, indent=4)
            except IOError:
                logger.debug("Unable to save JSON list")

        return fullfilename

    def getdir(self, dirname):
        """Downloads the requested tarball from the server
        https://www.silx.org/pub/silx/
        and unzips it into the data directory

        :param: relative name of the image.
        :return: list of files with their full path.
        """
        lodn = dirname.lower()
        if (lodn.endswith("tar") or lodn.endswith("tgz") or
            lodn.endswith("tbz2") or lodn.endswith("tar.gz") or
                lodn.endswith("tar.bz2")):
            import tarfile
            engine = tarfile.TarFile.open
        elif lodn.endswith("zip"):
            import zipfile
            engine = zipfile.ZipFile
        else:
            raise RuntimeError("Unsupported archive format. Only tar and zip "
                               "are currently supported")
        full_path = self.getfile(dirname)
        root = os.path.dirname(full_path)
        with engine(full_path, mode="r") as fd:
            fd.extractall(self.data_home)
            if lodn.endswith("zip"):
                result = [os.path.join(root, i) for i in fd.namelist()]
            else:
                result = [os.path.join(root, i) for i in fd.getnames()]
        return result

    def download_all(self, imgs=None):
        """Download all data needed for the test/benchmarks

        :param imgs: list of files to download, by default all
        :return: list of path with all files
        """
        if not self._initialized:
            self._initialize_data()
        if not imgs:
            imgs = self.all_data
        res = []
        for fn in imgs:
            logger.info("Downloading from silx.org: %s", fn)
            res.append(self.getfile(fn))
        return res
