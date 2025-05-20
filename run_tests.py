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
__date__ = "05/05/2025"
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
    module_name = project_module.__name__ + '.test.utils'
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
SILX_OPENCL=False to disable OpenCL tests.
WITH_HIGH_MEM_TEST: set to True to enable all tests >100Mb
WITH_GL_TEST=False to disable tests using OpenGL
"""
parser = ArgumentParser(description='Run the tests.',
                        epilog=epilog)

test_options = get_test_options(module)
"""Contains extra configuration for the tests."""
if test_options is not None:
    test_options.add_parser_argument(parser)

parser.add_argument("test_name", nargs='*',
                    default=(PROJECT_NAME,),
                    help=f"Test names to run (Default: {PROJECT_NAME})")

parser.add_argument("-i", "--installed",
                    action="store_true", dest="installed", default=False,
                    help="Test the installed version instead of"
                          "building from the source and testing the development version")
parser.add_argument("--no-gui",
                    action="store_false", dest="gui", default=True,
                    help="Disable the test of the graphical use interface")
parser.add_argument("--no-opengl",
                    action="store_false", dest="opengl", default=True,
                    help="Disable tests using OpenGL")
parser.add_argument("--no-opencl",
                    action="store_false", dest="opencl", default=True,
                    help="Disable tests using OpenCL")
parser.add_argument("--high-mem",
                    action="store_true", dest="high_mem", default=False,
                    help="Enable tests requiring large amounts of data (>100Mb)")
parser.add_argument("-v", "--verbose", default=0,
                    action="count", dest="verbose",
                    help="Increase verbosity. Option -v prints additional " +
                         "INFO messages. Use -vv for full verbosity, " +
                         "including debug messages and test help strings.")
parser.add_argument("--qt-binding", dest="qt_binding", default=None,
                    help="Force using a Qt binding, from 'PyQt5', 'PyQt6', or 'PySide6'")

options = parser.parse_args()

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

if options.qt_binding:
    binding = options.qt_binding.lower()
    if binding == "pyqt5":
        logger.info("Force using PyQt5")
        import PyQt5.QtCore  # noqa
    elif binding == "pyqt6":
        logger.info("Force using PyQt6")
        import PyQt6.QtCore  # noqa
    elif binding == "pyside6":
        logger.info("Force using PySide6")
        import PySide6.QtCore  # noqa
    else:
        raise ValueError("Qt binding '%s' is unknown" % options.qt_binding)

PROJECT_VERSION = getattr(module, 'version', '')
PROJECT_PATH = module.__path__[0]


if __name__ == "__main__":  # Needed for multiprocessing support on Windows

    project_module = module
    PROJECT_PATH = str(Path(project_module.__path__[0]).resolve())
    print(f"PROJECT_PATH: {PROJECT_PATH}")
    sys.path.insert(0, PROJECT_PATH)

    # corresponds to options to pass back to pytest ...
    pytest_options = []
    if options.qt_binding:
        pytest_options.append(f"--qt-binding={options.qt_binding}")
    if options.gui is False:
        pytest_options.append("--no-gui")
    if options.opencl is False:
        pytest_options.append("--no-opencl")
    if options.opengl is False:
        pytest_options.append("--no-opengl")
    if options.high_mem is True:
        pytest_options.append("--high-mem")

    def path2module(option):
        if option.endswith(".py"):
            option=option[:-3]
        option_parts = option.split(os.path.sep)
        if option_parts == ["src", PROJECT_NAME]:
            option_parts = [PROJECT_NAME]
        elif len(option_parts)==1:
            pass
        elif option_parts[:2] == ["src", PROJECT_NAME]:
            option_parts = option_parts[1:]
        return ".".join(i for i in option_parts if i)

    modules = [path2module(p) for p in options.test_name]
    test_module = importlib.import_module(f"{PROJECT_NAME}.test")
    # print(modules)
    # print(pytest_options)
    rc = test_module.run_tests(
            modules=modules,
            args=pytest_options)
    sys.exit(rc)

