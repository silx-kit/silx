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
"""Helper to access to external resources."""

__authors__ = ["Thomas Vincent", "J. Kieffer"]
__license__ = "MIT"
__date__ = "12/05/2026"


import hashlib
import json
import logging
import os
import sys
import tarfile
import threading
import tempfile
import unittest
import urllib.request
import urllib.error
import zipfile
import filelock

logger = logging.getLogger(__name__)


class ExternalResources:
    """Utility class which allows to download test-data from www.silx.org
    and manage the temporary data during the tests.

    """

    def __init__(self, project, url_base, env_key=None, timeout=60, data_home=None):
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
        :param data_home: Directory in which the data will be downloaded
        """
        self.project = project
        self._initialized = False
        self.sem = threading.Semaphore()
        self.hash = hashlib.sha256
        self.env_key = env_key or (self.project.upper() + "_TESTDATA")
        self.url_base = url_base
        self.all_data = {}
        self.timeout = timeout
        self._data_home = data_home
        self.testdata = ""
        self.lock = None

    @property
    def lockfile(self):
        """Returns the lockfile path."""
        return self.testdata + ".lock"

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

            basename = f"{self.project}_testdata_{name}"
            data_home = os.path.join(tempfile.gettempdir(), basename)
        if not os.path.exists(data_home):
            try:
                os.makedirs(data_home)
            except OSError as exc:
                raise RuntimeError(
                    f"Unable to create data directory {data_home} ! ({exc})"
                )
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
                    self.lock = filelock.FileLock(self.lockfile, timeout=10)
                    self.all_data = self.load_json()
                    self._initialized = True

    def getfile(self, filename):
        """Downloads the requested file from web-server available
        at https://www.silx.org/pub/silx/

        :param: relative name of the file.
        :return: full path of the locally saved file.
        """
        logger.debug("ExternalResources.getfile('%s')", filename)

        if not self._initialized:
            self._initialize_data()

        fullfilename = os.path.abspath(os.path.join(self.data_home, filename))

        if os.path.isfile(fullfilename):
            h = self.hash()
            with open(fullfilename, mode="rb") as fd:
                h.update(fd.read())
            if filename not in self.all_data:
                dico = self.load_json()
                dico.update(self.all_data)
                self.all_data = dico
                if filename not in self.all_data:
                    logger.error(
                        f"Filename {filename} not present in all_data:{os.linesep}{self.all_data}"
                    )
            if h.hexdigest() != self.all_data[filename]:
                logger.warning(f"Detected corruped file {fullfilename}")
                self.all_data.pop(filename)
                os.unlink(fullfilename)
                return self.getfile(filename)

        else:
            logger.debug(
                "Trying to download file %s, timeout set to %ss",
                filename,
                self.timeout,
            )
            dictProxies = {}
            if "http_proxy" in os.environ:
                dictProxies["http"] = os.environ["http_proxy"]
                dictProxies["https"] = os.environ["http_proxy"]
            if "https_proxy" in os.environ:
                dictProxies["https"] = os.environ["https_proxy"]
            if dictProxies:
                proxy_handler = urllib.request.ProxyHandler(dictProxies)
                opener = urllib.request.build_opener(proxy_handler).open
            else:
                opener = urllib.request.urlopen

            logger.debug("wget %s/%s", self.url_base, filename)
            try:
                data = opener(
                    f"{self.url_base}/{filename}", data=None, timeout=self.timeout
                ).read()
                logger.info("File %s successfully downloaded.", filename)
            except urllib.error.URLError:
                raise unittest.SkipTest("network unreachable.")

            if not os.path.isdir(os.path.dirname(fullfilename)):
                # Create sub-directory if needed
                os.makedirs(os.path.dirname(fullfilename))

            try:
                with open(fullfilename, mode="wb") as outfile:
                    outfile.write(data)
            except OSError:
                raise OSError(f"unable to write downloaded \
                    data to disk at {fullfilename}")

            if not os.path.isfile(fullfilename):
                raise RuntimeError("""Could not automatically download test files %s!
                    If you are behind a firewall, please set both environment variable
                     http_proxy and https_proxy.
                    This even works under windows !
                    Otherwise please try to download the files manually from
                    %s/%s""" % (filename, self.url_base, filename))
            else:
                self.all_data[filename] = self.get_hash(data=data)
                self.save_json()

        return fullfilename

    def load_json(self) -> dict:
        """Loads the JSON file containing the list of files and their hashes"""
        all_data = {}
        if self.testdata and os.path.exists(self.testdata):
            try:
                with self.lock:
                    with open(self.testdata) as f:
                        jdata = json.load(f)
            except filelock.Timeout:
                logger.error("Unable to lock JSON file")
                jdata = {}
            if isinstance(jdata, dict):
                all_data = jdata
            else:
                # recalculate the hash only if the data was stored as a list
                all_data = {k: self.get_hash(k) for k in jdata}
        return all_data

    def save_json(self):
        """Saves the JSON file containing the list of files and their hashes"""
        dico = self.load_json()
        dico.update(self.all_data)
        file_list = list(dico.keys())
        file_list.sort()
        dico = {i: dico[i] for i in file_list}  # reorder items

        try:
            with self.lock:
                with open(self.testdata, "w") as fp:
                    json.dump(dico, fp, indent=4)
        except filelock.Timeout:
            logger.error("Unable to lock JSON file")
        except OSError:
            logger.error("Unable to save JSON dict")

    def getdir(self, dirname):
        """Downloads the requested tarball from the server
        https://www.silx.org/pub/silx/
        and unzips it into the data directory

        :param: relative name of the file.
        :return: list of files with their full path.
        """
        lodn = dirname.lower()
        full_path = self.getfile(dirname)
        output = os.path.join(self.data_home, dirname + "__content")

        if lodn.endswith(("tar", "tgz", "tbz2", "tar.gz", "tar.bz2")):
            with tarfile.TarFile.open(full_path, mode="r") as fd:
                # Avoid unsafe filter deprecation warning during mode change
                if (3, 12) <= sys.version_info < (3, 14):
                    fd.extraction_filter = tarfile.data_filter
                fd.extractall(output)
                return [os.path.join(output, i) for i in fd.getnames()]

        if lodn.endswith("zip"):
            with zipfile.ZipFile(full_path, mode="r") as fd:
                fd.extractall(output)
                return [os.path.join(output, i) for i in fd.namelist()]

        raise RuntimeError(
            "Unsupported archive format. Only tar and zip " "are currently supported"
        )

    def get_file_and_repack(self, filename):
        """
        Download the requested file, decompress and repack it to bz2 and gz.

        :param str filename: name of the file.
        :rtype: str
        :return: full path of the locally saved file
        """
        if not self._initialized:
            self._initialize_data()
        if filename not in self.all_data:
            self.all_data[filename] = self.get_hash(filename)
            self.save_json()
        basefilename = os.path.basename(filename)

        if not os.path.exists(self.data_home):
            os.makedirs(self.data_home)
        fullfilename = os.path.abspath(os.path.join(self.data_home, basefilename))

        if basefilename.endswith(".bz2"):
            bzip2name = basefilename
            basename = basefilename[:-4]
            gzipname = basename + ".gz"
        elif basefilename.endswith(".gz"):
            gzipname = basefilename
            basename = basefilename[:-3]
            bzip2name = basename + ".bz2"
        else:
            basename = basefilename
            gzipname = basefilename + "gz2"
            bzip2name = basename + ".bz2"

        fullfilename_gz = os.path.abspath(os.path.join(self.data_home, gzipname))
        fullfilename_raw = os.path.abspath(os.path.join(self.data_home, basename))
        fullfilename_bz2 = os.path.abspath(os.path.join(self.data_home, bzip2name))

        # The files are recreated from the bz2 file
        if not os.path.isfile(fullfilename_bz2):
            self.getfile(bzip2name)
            if not os.path.isfile(fullfilename_bz2):
                raise RuntimeError("""Could not automatically download test files %s!
                    If you are behind a firewall, please set the environment variable
                     http_proxy.
                    Otherwise please try to download the files manually from
                    %s""" % (self.url_base, filename))

        try:
            import bz2
        except ImportError:
            raise RuntimeError("bz2 library is needed to decompress data")
        try:
            import gzip
        except ImportError:
            gzip = None

        raw_file_exists = os.path.isfile(fullfilename_raw)
        gz_file_exists = os.path.isfile(fullfilename_gz)
        if not raw_file_exists or not gz_file_exists:
            with open(fullfilename_bz2, "rb") as f:
                data = f.read()
            decompressed = bz2.decompress(data)

            if not raw_file_exists:
                try:
                    with open(fullfilename_raw, "wb") as fullfile:
                        fullfile.write(decompressed)
                except OSError:
                    raise OSError("unable to write decompressed \
                    data to disk at %s" % self.data_home)

            if not gz_file_exists:
                if gzip is None:
                    raise RuntimeError("gzip library is expected to recompress data")
                try:
                    gzip.open(fullfilename_gz, "wb").write(decompressed)
                except OSError:
                    raise OSError("unable to write gzipped \
                    data to disk at %s" % self.data_home)

        return fullfilename

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
