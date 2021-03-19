# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2019 European Synchrotron Radiation Facility
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

- :func:`temp_dir` provides a with context to create/delete a temporary
  directory.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "03/01/2019"


import sys
import contextlib
import os
import numpy
import shutil
import tempfile
from ..resources import ExternalResources


utilstest = ExternalResources(project="silx",
                              url_base="http://www.silx.org/pub/silx/",
                              env_key="SILX_DATA",
                              timeout=60)
"This is the instance to be used. Singleton-like feature provided by module"


class _TestOptions(object):

    def __init__(self):
        self.WITH_QT_TEST = True
        """Qt tests are included"""

        self.WITH_QT_TEST_REASON = ""
        """Reason for Qt tests are disabled if any"""

        self.WITH_OPENCL_TEST = True
        """OpenCL tests are included"""

        self.WITH_GL_TEST = True
        """OpenGL tests are included"""

        self.WITH_GL_TEST_REASON = ""
        """Reason for OpenGL tests are disabled if any"""

        self.TEST_LOW_MEM = False
        """Skip tests using too much memory"""

    def configure(self, parsed_options=None):
        """Configure the TestOptions class from the command line arguments and the
        environment variables
        """
        if parsed_options is not None and not parsed_options.gui:
            self.WITH_QT_TEST = False
            self.WITH_QT_TEST_REASON = "Skipped by command line"
        elif os.environ.get('WITH_QT_TEST', 'True') == 'False':
            self.WITH_QT_TEST = False
            self.WITH_QT_TEST_REASON = "Skipped by WITH_QT_TEST env var"
        elif sys.platform.startswith('linux') and not os.environ.get('DISPLAY', ''):
            self.WITH_QT_TEST = False
            self.WITH_QT_TEST_REASON = "DISPLAY env variable not set"

        if (parsed_options is not None and not parsed_options.opencl) or os.environ.get('SILX_OPENCL', 'True') == 'False':
            self.WITH_OPENCL_TEST = False
            # That's an easy way to skip OpenCL tests
            # It disable the use of OpenCL on the full silx project
            os.environ['SILX_OPENCL'] = "False"

        if parsed_options is not None and not parsed_options.opengl:
            self.WITH_GL_TEST = False
            self.WITH_GL_TEST_REASON = "Skipped by command line"
        elif os.environ.get('WITH_GL_TEST', 'True') == 'False':
            self.WITH_GL_TEST = False
            self.WITH_GL_TEST_REASON = "Skipped by WITH_GL_TEST env var"
        elif sys.platform.startswith('linux') and not os.environ.get('DISPLAY', ''):
            self.WITH_GL_TEST = False
            self.WITH_GL_TEST_REASON = "DISPLAY env variable not set"
        else:
            try:
                import OpenGL
            except ImportError:
                self.WITH_GL_TEST = False
                self.WITH_GL_TEST_REASON = "OpenGL package not available"

        if (parsed_options is not None and parsed_options.low_mem) or os.environ.get('SILX_TEST_LOW_MEM', 'True') == 'False':
            self.TEST_LOW_MEM = True

        if self.WITH_QT_TEST:
            try:
                from silx.gui import qt
            except ImportError:
                pass
            else:
                if sys.platform == "win32" and qt.qVersion() == "5.9.2":
                    self.SKIP_TEST_FOR_ISSUE_936 = True

    def add_parser_argument(self, parser):
        """Add extrat arguments to the test argument parser

        :param ArgumentParser parser: An argument parser
        """

        parser.add_argument("-x", "--no-gui", dest="gui", default=True,
                            action="store_false",
                            help="Disable the test of the graphical use interface")
        parser.add_argument("-g", "--no-opengl", dest="opengl", default=True,
                            action="store_false",
                            help="Disable tests using OpenGL")
        parser.add_argument("-o", "--no-opencl", dest="opencl", default=True,
                            action="store_false",
                            help="Disable the test of the OpenCL part")
        parser.add_argument("-l", "--low-mem", dest="low_mem", default=False,
                            action="store_true",
                            help="Disable test with large memory consumption (>100Mbyte")


test_options = _TestOptions()
"""Singleton providing configuration information for all the tests"""


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
