#!/usr/bin/env python3
# /*##########################################################################
#
# Copyright (c) 2015-2025 European Synchrotron Radiation Facility
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
"""Run the tests of the project.

This script expects a suite function in <project_package>.test,
which returns a unittest.TestSuite.

Test coverage dependencies: coverage, lxml.
"""

__authors__ = ["Jérôme Kieffer", "Thomas Vincent"]
__date__ = "14/04/2025"
__license__ = "MIT"

import sys
import logging
import os
import sys
import sysconfig
from argparse import ArgumentParser
from pathlib import Path


# Capture all default warnings
logging.captureWarnings(True)
import warnings

warnings.simplefilter("default")

logger = logging.getLogger("run_tests")
logger.setLevel(logging.WARNING)

logger.info("Python %s %s", sys.version, tuple.__itemsize__ * 8)
if sys.version_info.major < 3 or sys.version_info.minor < 10:
    logger.error("SILX requires at least Python3.10")

try:
    import resource
except ImportError:
    resource = None
    logger.warning("resource module missing")

try:
    import importlib
    importer = importlib.import_module
except ImportError:

    def importer(name):
        module = __import__(name)
        # returns the leaf module, instead of the root module
        subnames = name.split(".")
        subnames.pop(0)
        for subname in subnames:
            module = getattr(module, subname)
            return module

try:
    import numpy
except Exception as error:
    logger.warning("Numpy missing: %s", error)
else:
    logger.info("Numpy %s", numpy.version.version)

try:
    import h5py
except Exception as error:
    logger.warning("h5py missing: %s", error)
else:
    logger.info("h5py %s", h5py.version.version)


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
from bootstrap import get_project_name, build_project
PROJECT_NAME = get_project_name(PROJECT_DIR)
logger.info("Project name: %s", PROJECT_NAME)

def is_debug_python():
    """Returns true if the Python interpreter is in debug mode."""
    if sysconfig.get_config_var("Py_DEBUG"):
        return True

    return hasattr(sys, "gettotalrefcount")


# Prevent importing from source directory
if os.path.dirname(os.path.abspath(__file__)) == os.path.abspath(sys.path[0]):
    removed_from_sys_path = sys.path.pop(0)
    logger.info("Patched sys.path, removed: '%s'", removed_from_sys_path)


def get_test_options(project_module):
    """Returns the test options if available, else None"""
    module_name = project_module.__name__ + '.test.utilstest'
    logger.info('Import %s', module_name)
    try:
        test_utils = importer(module_name)
    except ImportError:
        logger.warning("No module named '%s'. No test options available.", module_name)
        return None

    test_options = getattr(test_utils, "test_options", None)
    return test_options


if "-i" in sys.argv or "--installed" in sys.argv:
    for bad_path in (".", os.getcwd(), PROJECT_DIR):
        if bad_path in sys.path:
            sys.path.remove(bad_path)
    try:
        module = importer(PROJECT_NAME)
    except Exception:
        logger.error("Cannot run tests on installed version: %s not installed or raising error.",
                     PROJECT_NAME)
        raise
    else:
        print("Running tests on system-wide installed project")
else:
    build_dir = build_project(PROJECT_NAME, PROJECT_DIR)
    sys.path.insert(0, build_dir)
    logger.warning("Patched sys.path, added: '%s'", build_dir)
    module = importer(PROJECT_NAME)


epilog = """Environment variables:
WITH_QT_TEST=False to disable graphical tests
PYFAI_OPENCL=False to disable OpenCL tests.
PYFAI_LOW_MEM: set to True to skip all tests >100Mb
WITH_GL_TEST=False to disable tests using OpenGL
"""
parser = ArgumentParser(description='Run the tests.',
                        epilog=epilog)

test_options = get_test_options(module)
"""Contains extra configuration for the tests."""
if test_options is not None:
    test_options.add_parser_argument(parser)

default_test_name = f"{PROJECT_NAME}.test.suite"
parser.add_argument("test_name", nargs='*',
                    default=(default_test_name,),
                    help="Test names to run (Default: %s)" % default_test_name)

parser.add_argument("--installed",
                    action="store_true", dest="installed", default=False,
                    help=("Test the installed version instead of" +
                          "building from the source"))
parser.add_argument("-c", "--coverage", dest="coverage",
                    action="store_true", default=False,
                    help=("Report code coverage" +
                          "(requires 'coverage' and 'lxml' module)"))
parser.add_argument("-m", "--memprofile", dest="memprofile",
                    action="store_true", default=False,
                    help="Report memory profiling")
parser.add_argument("-v", "--verbose", default=0,
                    action="count", dest="verbose",
                    help="Increase verbosity. Option -v prints additional " +
                         "INFO messages. Use -vv for full verbosity, " +
                         "including debug messages and test help strings.")
parser.add_argument("--qt-binding", dest="qt_binding", default=None,
                    help="Force using a Qt binding, from 'PyQt4', 'PyQt5', or 'PySide'")

options = parser.parse_args()
sys.argv = [sys.argv[0]]

test_verbosity = 1
use_buffer = True
if options.verbose == 1:
    logging.root.setLevel(logging.INFO)
    logger.info("Set log level: INFO")
    test_verbosity = 2
    use_buffer = False
elif options.verbose > 1:
    logging.root.setLevel(logging.DEBUG)
    logger.info("Set log level: DEBUG")
    test_verbosity = 2
    use_buffer = False

if options.coverage:
    logger.info("Running test-coverage")
    import coverage
    omits = ["*test*", "*third_party*", "*/setup.py",
             # temporary test modules (silx.math.fit.test.test_fitmanager)
             "*customfun.py", ]
    try:
        coverage_class = coverage.Coverage
    except AttributeError:
        coverage_class = coverage.coverage
    print(f"|{PROJECT_NAME}|")
    cov = coverage_class(include=[f"*/{PROJECT_NAME}/*"],
                         omit=omits)
    cov.start()

if options.qt_binding:
    binding = options.qt_binding.lower()
    if binding == "pyqt4":
        logger.info("Force using PyQt4")
        import PyQt4.QtCore  # noqa
    elif binding == "pyqt5":
        logger.info("Force using PyQt5")
        import PyQt5.QtCore  # noqa
    elif binding == "pyside":
        logger.info("Force using PySide")
        import PySide.QtCore  # noqa
    elif binding == "pyside2":
        logger.info("Force using PySide2")
        import PySide2.QtCore  # noqa
    else:
        raise ValueError("Qt binding '%s' is unknown" % options.qt_binding)

PROJECT_VERSION = getattr(module, 'version', '')
PROJECT_PATH = module.__path__[0]


if __name__ == "__main__":  # Needed for multiprocessing support on Windows

    project_module = module
    PROJECT_PATH = str(Path(project_module.__path__[0]).resolve())
    print("PROJECT_PATH:", PROJECT_PATH)
    sys.path.insert(0, PROJECT_PATH)

    def normalize_option(option):
        option_parts = option.split(os.path.sep)
        if option_parts == ["src", "silx"]:
            return PROJECT_PATH
        if option_parts[:2] == ["src", "silx"]:
            return os.path.join(PROJECT_PATH, *option_parts[2:])
        return option

    test_module = importlib.import_module(f"{PROJECT_NAME}.test")
    print(test_module)
    sys.exit(0)
    #     test_module.run_tests(
    #         module=None,
    #         args=[normalize_option(p) for p in sys.argv[1:] if p != "--installed"],
    #     )
    # )
