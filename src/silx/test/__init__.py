# /*##########################################################################
#
# Copyright (c) 2015-2022 European Synchrotron Radiation Facility
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
"""This package provides test of the root modules
"""

import logging


try:
    import pytest
except ImportError:
    logging.getLogger(__name__).error(
        "pytest is required to run the tests, please install it.")
    raise

def run_tests(module: str='silx', verbosity: int=0, args=()):
    """Run tests

    :param module: Name of the silx module to test (default: 'silx')
    :param verbosity: Requested level of verbosity
    :param args: List of extra arguments to pass to `pytest`
    """
    return pytest.main([
        '--pyargs',
        module,
        '--verbosity',
        str(verbosity),
        '-o python_files=["test/test*.py","test/Test*.py"]',
        '-o python_classes=["Test"]',
        '-o python_functions=["test"]',
    ] + list(args))
