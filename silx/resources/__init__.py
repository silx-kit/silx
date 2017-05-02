# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2017 European Synchrotron Radiation Facility
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
__date__ = "02/05/2017"


import os
import sys
import threading
import json
import getpass
import logging
import tempfile
import unittest
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


def resource_filename(resource):
    """Return filename corresponding to resource.

    resource can be the name of either a file or a directory.
    The existence of the resource is not checked.

    :param str resource: Resource path relative to resource directory
                         using '/' path separator.
    :return: Absolute resource path in the file system
    """
    # Not in use, uncomment when functionality is needed
    # If _RESOURCES_DOC_DIR is set, use it to get resources in doc/ subfoldler
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


class ExternalResources(object):
    """Utility class which allows to download test-data from www.silx.org
    and manage the temporary data during the tests.

    """

    def __init__(self, project,
                 url_base,
                 env_key=None,
                 timeout=60):
        """Constructor of the class

        :param project: name of the project, like "silx"
        :param url_base: base URL for the data, like "http://www.silx.org/pub"
        :param env_key: name of the environment variable which contains the
                        test_data directory like "SILX_DATA"
        :param timeout: time in seconds before it breaks
        """
        self.project = project
        self._initialized = False
        self._tempdir = None
        self.sem = threading.Semaphore()
        self.env_key = env_key
        self.url_base = url_base
        self.all_data = set()
        self.timeout = timeout

    def _initialize_tmpdir(self):
        """Initialize the temporary directory"""
        if not self._tempdir:
            with self.sem:
                if not self._tempdir:
                    self._tempdir = tempfile.mkdtemp("_" + getpass.getuser(),
                                                     self.project + "_")

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

    @property
    def tempdir(self):
        if not self._tempdir:
            self._initialize_tmpdir()
        return self._tempdir

    def clean_up(self):
        """Removes the temporary directory (and all its content !)"""
        with self.sem:
            if not self._tempdir:
                return
            if not os.path.isdir(self._tempdir):
                return
            for root, dirs, files in os.walk(self._tempdir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self._tempdir)
            self._tempdir = None

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
                raise RuntimeError("Could not automatically \
                download test images %s!\n \ If you are behind a firewall, \
                please set both environment variable http_proxy and https_proxy.\
                This even works under windows ! \n \
                Otherwise please try to download the images manually from \n%s/%s"\
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
        :return: full path of the locally saved file.
        """
        lodn = dirname.lower()
        if (lodn.endswith("tar") or lodn.endswith("tgz") or
            lodn.endswith("tbz2") or lodn.endswith("tar.gz") or
                lodn.endswith("tar.bz2")):
            import tarfile
            engine = tarfile.TarFile
        elif lodn.endswith("zip"):
            import zipfile
            engine = zipfile.ZipFile
        else:
            raise RuntimeError("Unsupported archive format. Only tar and zip "
                               "are currently supported")
        full_path = self.getfile(dirname)
        with engine.open(full_path) as fd:
            fd.extractall(self.data_home)
        return full_path

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
