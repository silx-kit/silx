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
"""Utilities for writing tests.

- :func:`temp_dir` provides a with context to create/delete a temporary
  directory.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "17/01/2018"


import contextlib
import numpy
import shutil
import tempfile
from ..resources import ExternalResources


utilstest = ExternalResources(project="silx",
                              url_base="http://www.silx.org/pub/silx/",
                              env_key="SILX_DATA",
                              timeout=60)
"This is the instance to be used. Singleton-like feature provided by module"

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
