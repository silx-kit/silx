# /*##########################################################################
#
# Copyright (c) 2016-2021 European Synchrotron Radiation Facility
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
"""Helper to access to external resources.
"""

__authors__ = ["Thomas Vincent", "J. Kieffer"]
__license__ = "MIT"
__date__ = "21/12/2021"


import os
import threading
import json
import logging
import tempfile
import unittest
import urllib.request
import urllib.error
import hashlib
from collections import OrderedDict
logger = logging.getLogger(__name__)


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
        self.hash = hashlib.sha256
        self.env_key = env_key or (self.project.upper() + "_TESTDATA")
        self.url_base = url_base
        self.all_data = {}
        self.timeout = timeout
        self._data_home = None

    @property
    def data_home(self):
        """Returns the data_home path and make sure it exists in the file
        system."""
        if self._data_home is not None:
            return self._data_home

        data_home = os.environ.get(self.env_key)
        if data_home is None:
            try:
                import getpass
                name = getpass.getuser()
            except Exception:
                if "getlogin" in dir(os):
                    name = os.getlogin()
                elif "USER" in os.environ:
                    name = os.environ["USER"]
                elif "USERNAME" in os.environ:
                    name = os.environ["USERNAME"]
                else:
                    name = "uid" + str(os.getuid())

            basename = "%s_testdata_%s" % (self.project, name)
            data_home = os.path.join(tempfile.gettempdir(), basename)
        if not os.path.exists(data_home):
            os.makedirs(data_home)
        self._data_home = data_home
        return data_home

    def get_hash(self, filename=None, data=None):
        "Calculate and return the hash of a file or a bunch of data"
        if data is None and filename is None:
            return
        h = self.hash()
        if filename is not None:
            fullfilename = os.path.join(self.data_home, filename) 
            if os.path.exists(fullfilename):
                with open(fullfilename, "rb") as fd:
                    data = fd.read()
            else:
                raise RuntimeError(f"Filename {fullfilename} does not exist !")
        h.update(data)
        return h.hexdigest()

    def _initialize_data(self):
        """Initialize for downloading test data"""
        if not self._initialized:
            with self.sem:
                if not self._initialized:
                    self.testdata = os.path.join(self.data_home, "all_testdata.json")
                    if os.path.exists(self.testdata):
                        with open(self.testdata) as f:
                            jdata = json.load(f)
                        if isinstance(jdata, dict):
                            self.all_data = jdata
                        else: 
                            #recalculate the hash only if the data was stored as a list
                            self.all_data = {k: self.get_hash(k) for k in jdata}
                            self.save_json()
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
                proxy_handler = urllib.request.ProxyHandler(dictProxies)
                opener = urllib.request.build_opener(proxy_handler).open
            else:
                opener = urllib.request.urlopen

            logger.debug("wget %s/%s", self.url_base, filename)
            try:
                data = opener("%s/%s" % (self.url_base, filename),
                              data=None, timeout=self.timeout).read()
                logger.info("Image %s successfully downloaded.", filename)
            except urllib.error.URLError:
                raise unittest.SkipTest("network unreachable.")

            if not os.path.isdir(os.path.dirname(fullfilename)):
                # Create sub-directory if needed
                os.makedirs(os.path.dirname(fullfilename))

            try:
                with open(fullfilename, mode="wb") as outfile:
                    outfile.write(data)
            except IOError:
                raise IOError("unable to write downloaded \
                    data to disk at %s" % self.data_home)

            if not os.path.isfile(fullfilename):
                raise RuntimeError(
                    """Could not automatically download test images %s!
                    If you are behind a firewall, please set both environment variable http_proxy and https_proxy.
                    This even works under windows !
                    Otherwise please try to download the images manually from
                    %s/%s""" % (filename, self.url_base, filename))
            else:
                self.all_data[filename] = self.get_hash(data=data)
                self.save_json()

        else:
            h = self.hash()
            with open(fullfilename, mode="rb") as fd:
                h.update(fd.read())
            if h.hexdigest() != self.all_data[filename]:
                logger.warning(f"Detected corruped file {fullfilename}")
                self.all_data.pop(filename)
                os.unlink(fullfilename)
                return self.getfile(filename)
            
        return fullfilename

    def save_json(self):
        image_list = list(self.all_data.keys())
        image_list.sort()
        dico = OrderedDict([(i, self.all_data[i]) for i in image_list])
        try:
            with open(self.testdata, "w") as fp:
                json.dump(dico, fp, indent=4)
        except IOError:
            logger.info("Unable to save JSON dict")

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
        with engine(full_path, mode="r") as fd:
            output = os.path.join(self.data_home, dirname + "__content")
            fd.extractall(output)
            if lodn.endswith("zip"):
                result = [os.path.join(output, i) for i in fd.namelist()]
            else:
                result = [os.path.join(output, i) for i in fd.getnames()]
        return result

    def get_file_and_repack(self, filename):
        """
        Download the requested file, decompress and repack it to bz2 and gz.

        :param str filename: name of the image.
        :rtype: str
        :return: full path of the locally saved file
        """
        if not self._initialized:
            self._initialize_data()
        if filename not in self.all_data:
            self.all_data[filename] = self.get_hash(filename)
            seld.save_json()
        baseimage = os.path.basename(filename)
        logger.info("UtilsTest.getimage('%s')" % baseimage)

        if not os.path.exists(self.data_home):
            os.makedirs(self.data_home)
        fullimagename = os.path.abspath(os.path.join(self.data_home, baseimage))

        if baseimage.endswith(".bz2"):
            bzip2name = baseimage
            basename = baseimage[:-4]
            gzipname = basename + ".gz"
        elif baseimage.endswith(".gz"):
            gzipname = baseimage
            basename = baseimage[:-3]
            bzip2name = basename + ".bz2"
        else:
            basename = baseimage
            gzipname = baseimage + "gz2"
            bzip2name = basename + ".bz2"

        fullimagename_gz = os.path.abspath(os.path.join(self.data_home, gzipname))
        fullimagename_raw = os.path.abspath(os.path.join(self.data_home, basename))
        fullimagename_bz2 = os.path.abspath(os.path.join(self.data_home, bzip2name))

        # The files are recreated from the bz2 file
        if not os.path.isfile(fullimagename_bz2):
            self.getfile(bzip2name)
            if not os.path.isfile(fullimagename_bz2):
                raise RuntimeError(
                    """Could not automatically download test images %s!
                    If you are behind a firewall, please set the environment variable http_proxy.
                    Otherwise please try to download the images manually from
                    %s""" % (self.url_base, filename))

        try:
            import bz2
        except ImportError:
            raise RuntimeError("bz2 library is needed to decompress data")
        try:
            import gzip
        except ImportError:
            gzip = None

        raw_file_exists = os.path.isfile(fullimagename_raw)
        gz_file_exists = os.path.isfile(fullimagename_gz)
        if not raw_file_exists or not gz_file_exists:
            with open(fullimagename_bz2, "rb") as f:
                data = f.read()
            decompressed = bz2.decompress(data)

            if not raw_file_exists:
                try:
                    with open(fullimagename_raw, "wb") as fullimage:
                        fullimage.write(decompressed)
                except IOError:
                    raise IOError("unable to write decompressed \
                    data to disk at %s" % self.data_home)

            if not gz_file_exists:
                if gzip is None:
                    raise RuntimeError("gzip library is expected to recompress data")
                try:
                    gzip.open(fullimagename_gz, "wb").write(decompressed)
                except IOError:
                    raise IOError("unable to write gzipped \
                    data to disk at %s" % self.data_home)

        return fullimagename

    def download_all(self, imgs=None):
        """Download all data needed for the test/benchmarks

        :param imgs: list of files to download, by default all
        :return: list of path with all files
        """
        if not self._initialized:
            self._initialize_data()
        if not imgs:
            imgs = self.all_data.keys()
        res = []
        for fn in imgs:
            logger.info("Downloading from silx.org: %s", fn)
            res.append(self.getfile(fn))
        return res
