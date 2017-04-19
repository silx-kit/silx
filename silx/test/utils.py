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
"""Utilities for writing tests.

- :class:`ParametricTestCase` provides a :meth:`TestCase.subTest` replacement
  for Python < 3.4
- :class:`TestLogging` with context or the :func:`test_logging` decorator
  enables testing the number of logging messages of different levels.
- :func:`temp_dir` provides a with context to create/delete a temporary
  directory.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "19/04/2017"

PACKAGE = "silx"
DATA_KEY = "SILX_DATA"

import os
import contextlib
import functools
import logging
import numpy
import shutil
import sys
import tempfile
import unittest
import threading
import json
import getpass

logger = logging.getLogger("silx.test.utils")


# Parametric Test Base Class ##################################################

if sys.hexversion >= 0x030400F0:  # Python >= 3.4
    class ParametricTestCase(unittest.TestCase):
        pass

else:
    class ParametricTestCase(unittest.TestCase):
        """TestCase with subTest support for Python < 3.4.

        Add subTest method to support parametric tests.
        API is the same, but behavior differs:
        If a subTest fails, the following ones are not run.
        """

        _subtest_msg = None  # Class attribute to provide a default value

        @contextlib.contextmanager
        def subTest(self, msg=None, **params):
            """Use as unittest.TestCase.subTest method in Python >= 3.4."""
            # Format arguments as: '[msg] (key=value, ...)'
            param_str = ', '.join(['%s=%s' % (k, v) for k, v in params.items()])
            self._subtest_msg = '[%s] (%s)' % (msg or '', param_str)
            yield
            self._subtest_msg = None

        def shortDescription(self):
            short_desc = super(ParametricTestCase, self).shortDescription()
            if self._subtest_msg is not None:
                # Append subTest message to shortDescription
                short_desc = ' '.join(
                    [msg for msg in (short_desc, self._subtest_msg) if msg])

            return short_desc if short_desc else None


# Test logging messages #######################################################

class TestLogging(logging.Handler):
    """Context checking the number of logging messages from a specified Logger.

    It disables propagation of logging message while running.

    This is meant to be used as a with statement, for example:

    >>> with TestLogging(logger, error=2, warning=0):
    >>>     pass  # Run tests here expecting 2 ERROR and no WARNING from logger
    ...

    :param logger: Name or instance of the logger to test.
                   (Default: root logger)
    :type logger: str or :class:`logging.Logger`
    :param int critical: Expected number of CRITICAL messages.
                         Default: Do not check.
    :param int error: Expected number of ERROR messages.
                      Default: Do not check.
    :param int warning: Expected number of WARNING messages.
                        Default: Do not check.
    :param int info: Expected number of INFO messages.
                     Default: Do not check.
    :param int debug: Expected number of DEBUG messages.
                      Default: Do not check.
    :param int notset: Expected number of NOTSET messages.
                       Default: Do not check.
    :raises RuntimeError: If the message counts are the expected ones.
    """

    def __init__(self, logger=None, critical=None, error=None,
                 warning=None, info=None, debug=None, notset=None):
        if logger is None:
            logger = logging.getLogger()
        elif not isinstance(logger, logging.Logger):
            logger = logging.getLogger(logger)
        self.logger = logger

        self.records = []

        self.count_by_level = {
            logging.CRITICAL: critical,
            logging.ERROR: error,
            logging.WARNING: warning,
            logging.INFO: info,
            logging.DEBUG: debug,
            logging.NOTSET: notset
        }

        super(TestLogging, self).__init__()

    def __enter__(self):
        """Context (i.e., with) support"""
        self.records = []  # Reset recorded LogRecords
        self.logger.addHandler(self)
        self.logger.propagate = False

    def __exit__(self, exc_type, exc_value, traceback):
        """Context (i.e., with) support"""
        self.logger.removeHandler(self)
        self.logger.propagate = True

        for level, expected_count in self.count_by_level.items():
            if expected_count is None:
                continue

            # Number of records for the specified level_str
            count = len([r for r in self.records if r.levelno == level])
            if count != expected_count:  # That's an error
                # Resend record logs through logger as they where masked
                # to help debug
                for record in self.records:
                    self.logger.handle(record)
                raise RuntimeError(
                    'Expected %d %s logging messages, got %d' % (
                        expected_count, logging.getLevelName(level), count))

    def emit(self, record):
        """Override :meth:`logging.Handler.emit`"""
        self.records.append(record)


def test_logging(logger=None, critical=None, error=None,
                 warning=None, info=None, debug=None, notset=None):
    """Decorator checking number of logging messages.

    Propagation of logging messages is disabled by this decorator.

    In case the expected number of logging messages is not found, it raises
    a RuntimeError.

    >>> class Test(unittest.TestCase):
    ...     @test_logging('module_logger_name', error=2, warning=0)
    ...     def test(self):
    ...         pass  # Test expecting 2 ERROR and 0 WARNING messages

    :param logger: Name or instance of the logger to test.
                   (Default: root logger)
    :type logger: str or :class:`logging.Logger`
    :param int critical: Expected number of CRITICAL messages.
                         Default: Do not check.
    :param int error: Expected number of ERROR messages.
                      Default: Do not check.
    :param int warning: Expected number of WARNING messages.
                        Default: Do not check.
    :param int info: Expected number of INFO messages.
                     Default: Do not check.
    :param int debug: Expected number of DEBUG messages.
                      Default: Do not check.
    :param int notset: Expected number of NOTSET messages.
                       Default: Do not check.
    """
    def decorator(func):
        test_context = TestLogging(logger, critical, error,
                                   warning, info, debug, notset)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with test_context:
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator


class _UtilsTest(object):
    """Utility class which allows to download test-data from www.silx.org 
    and manage the temporary data during the tests.

    """
    options = None
    timeout = 60  # timeout in seconds for downloading images
    url_base = "http://www.silx.org/pub/silx/"
    
    testdata = None
    data_home = None
    all_data = set()

    def __init__(self):
        """Constructor of the class
        """
        self._initialized = False
        self._tempdir = None
        self.sem = threading.Semaphore()

    def initialize_tmpdir(self):
        """Initialize the temporary directory"""
        if not self._tempdir:
            with self.sem:
                if not self._tempdir:
                    self._tempdir = tempfile.mkdtemp("_" + getpass.getuser(), PACKAGE + "_")

    def initialize_data(self):
        """Initialize for downloading test data"""
        if not self._initialized:
            with self.sem:
                if not self._initialized:

                    self.data_home = os.environ.get(DATA_KEY)
                    if self.data_home is None:
                        self.data_home = os.path.join(tempfile.gettempdir(), "%s_testdata_%s" % (PACKAGE, getpass.getuser()))
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
            self.initialize_tmpdir()
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
            self._tempdir=None
                    
    def getfile(self, filename):
        """Downloads the requested file from web-server available at 
        https://www.silx.org/pub/silx/

        :param: relative name of the image.
        :return: full path of the locally saved file.
        """

        if not self._initialized:
            from ..third_party.six.moves.urllib.request import urlopen, ProxyHandler, build_opener
            from ..third_party.six.moves.urllib.error import URLError
            self.initialize_data()

        if filename not in self.all_data:
            self.all_data.add(filename)
            image_list = list(self.all_data)
            image_list.sort()
            try:
                with open(self.testdata, "w") as fp:
                    json.dump(image_list, fp, indent=4)
            except IOError:
                logger.debug("Unable to save JSON list")
        logger.info("UtilsTest.getimage('%s')", filename)
        if not os.path.exists(self.data_home):
            os.makedirs(self.data_home)

        fullfilename = os.path.abspath(os.path.join(self.data_home, filename))
        if not os.path.isfile(fullfilename):
            logger.info("Trying to download image %s, timeout set to %ss",
                        filename, self.timeout)
            dictProxies = {}
            if "http_proxy" in os.environ:
                dictProxies['http'] = os.environ["http_proxy"]
                dictProxies['https'] = os.environ["http_proxy"]
            if "https_proxy" in os.environ:
                dictProxies['https'] = os.environ["https_proxy"]
            if dictProxies:
                proxy_handler = ProxyHandler(dictProxies)
                opener = build_opener(proxy_handler).open
            else:
                opener = urlopen

            logger.info("wget %s/%s", self.url_base, filename)
            try:
                data = opener("%s/%s" % (self.url_base, filename),
                              data=None, timeout=self.timeout).read()
                logger.info("Image %s successfully downloaded.", filename)
            except URLError:
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
                Otherwise please try to download the images manually from \n%s/%s" % (filename, self.url_base, filename))

        return fullfilename

    def getdir(self, dirname):
        """Downloads the requested tarball from the server and un-zips it into  
        https://www.silx.org/pub/silx/

        :param: relative name of the image.
        :return: full path of the locally saved file.

        """
        if not(dirname.endswith("tar") or dirname.endswith("tgz") or dirname.endswith("tbz2") 
               or dirname.endswith("tar.gz") or dirname.endswith("tar.bz2")):
            dirname = dirname+".tar"
        dirname =  self.getfile(dirname)      
        #TODO: unzip content of tar in directory
        
    def download_all(self, imgs=None):
        """Download all data needed for the test/benchmarks

        :param imgs: list of files to download
        """
        if not self._initialized:
            self.initialize_data()
        if not imgs:
            imgs = self.all_data
        for fn in imgs:
            print("Downloading from silx.org: %s" % fn)
            self.getimage(fn)


utilstest = _UtilsTest()
"This is the instance to be used. Singleton like feature provided by module"

# Temporary directory context #################################################

@contextlib.contextmanager
def temp_dir():
    """with context providing a temporary directory.

    >>> import os.path
    >>> with temp_dir() as tmp:
    ...     print(os.path.isdir(tmp))  # Use tmp directory
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir)


# Synthetic data and random noise #############################################
def add_gaussian_noise(y, stdev=1., mean=0.):
    """Add random gaussian noise to synthetic data.

    :param ndarray y: Array of synthetic data
    :param float mean: Mean of the gaussian distribution of noise.
    :param float stdev: Standard deviation of the gaussian distribution of
        noise.
    :return: Array of data with noise added
    """
    noise = numpy.random.normal(mean, stdev, size=y.size)
    noise.shape = y.shape
    return y + noise


def add_poisson_noise(y):
    """Add random noise from a poisson distribution to synthetic data.

    :param ndarray y: Array of synthetic data
    :return: Array of data with noise added
    """
    yn = numpy.random.poisson(y)
    yn.shape = y.shape
    return yn


def add_relative_noise(y, max_noise=5.):
    """Add relative random noise to synthetic data. The maximum noise level
    is given in percents.

    An array of noise in the interval [-max_noise, max_noise] (continuous
    uniform distribution) is generated, and applied to the data the
    following way:

    :math:`yn = y * (1. + noise / 100.)`

    :param ndarray y: Array of synthetic data
    :param float max_noise: Maximum percentage of noise
    :return: Array of data with noise added
    """
    noise = max_noise * (2 * numpy.random.random(size=y.size) - 1)
    noise.shape = y.shape
    return y * (1. + noise / 100.)
