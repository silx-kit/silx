#!/usr/bin/env python3
# coding: utf8
# /*##########################################################################
#
# Copyright (c) 2015-2021 European Synchrotron Radiation Facility
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
__date__ = "30/09/2020"
__license__ = "MIT"

import distutils.util
import logging
import os
import subprocess
import sys
import importlib


# Capture all default warnings
logging.captureWarnings(True)
import warnings
warnings.simplefilter('default')

logger = logging.getLogger("run_tests")
logger.setLevel(logging.WARNING)

logger.info("Python %s %s", sys.version, tuple.__itemsize__ * 8)


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


def get_project_name(root_dir):
    """Retrieve project name by running python setup.py --name in root_dir.

    :param str root_dir: Directory where to run the command.
    :return: The name of the project stored in root_dir
    """
    logger.debug("Getting project name in %s", root_dir)
    p = subprocess.Popen([sys.executable, "setup.py", "--name"],
                         shell=False, cwd=root_dir, stdout=subprocess.PIPE)
    name, _stderr_data = p.communicate()
    logger.debug("subprocess ended with rc= %s", p.returncode)
    return name.split()[-1].decode('ascii')


def is_debug_python():
    """Returns true if the Python interpreter is in debug mode."""
    try:
        import sysconfig
    except ImportError:  # pragma nocover
        # Python < 2.7
        import distutils.sysconfig as sysconfig

    if sysconfig.get_config_var("Py_DEBUG"):
        return True

    return hasattr(sys, "gettotalrefcount")


def build_project(name, root_dir):
    """Run python setup.py build for the project.

    Build directory can be modified by environment variables.

    :param str name: Name of the project.
    :param str root_dir: Root directory of the project
    :return: The path to the directory were build was performed
    """
    platform = distutils.util.get_platform()
    architecture = "lib.%s-%i.%i" % (platform,
                                     sys.version_info[0], sys.version_info[1])
    if is_debug_python():
        architecture += "-pydebug"

    if os.environ.get("PYBUILD_NAME") == name:
        # we are in the debian packaging way
        home = os.environ.get("PYTHONPATH", "").split(os.pathsep)[-1]
    elif os.environ.get("BUILDPYTHONPATH"):
        home = os.path.abspath(os.environ.get("BUILDPYTHONPATH", ""))
    else:
        home = os.path.join(root_dir, "build", architecture)

    logger.warning("Building %s to %s", name, home)
    p = subprocess.Popen([sys.executable, "setup.py", "build"],
                         shell=False, cwd=root_dir)
    logger.debug("subprocess ended with rc= %s", p.wait())

    if os.path.isdir(home):
        return home
    alt_home = os.path.join(os.path.dirname(home), "lib")
    if os.path.isdir(alt_home):
        return alt_home


def import_project_module(project_name, project_dir):
    """Import project module, from the system of from the project directory"""
    if "--installed" in sys.argv:
        try:
            module = importlib.import_module(project_name)
        except Exception:
            logger.error("Cannot run tests on installed version: %s not installed or raising error.",
                         project_name)
            raise
    else:  # Use built source
        build_dir = build_project(project_name, project_dir)
        if build_dir is None:
            logging.error("Built project is not available !!! investigate")
        sys.path.insert(0, build_dir)
        logger.warning("Patched sys.path, added: '%s'", build_dir)
        module = importlib.import_module(project_name)
    return module


if __name__ == "__main__":  # Needed for multiprocessing support on Windows
    import pytest

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_NAME = get_project_name(PROJECT_DIR)
    logger.info("Project name: %s", PROJECT_NAME)

    project_module = import_project_module(PROJECT_NAME, PROJECT_DIR)
    PROJECT_VERSION = getattr(project_module, 'version', '')
    PROJECT_PATH = project_module.__path__[0]

    def normalize_option(option):
        option_parts = option.split(os.path.sep)
        if option_parts == ["src", "silx"]:
            return PROJECT_PATH
        if option_parts[:2] == ["src", "silx"]:
            return os.path.join(PROJECT_PATH, *option_parts[2:])
        return option

    args = [normalize_option(p) for p in sys.argv[1:] if p != "--installed"]

    # Run test on PROJECT_PATH if nothing is specified
    without_options = [a for a in args if not a.startswith("-")]
    if len(without_options) == 0:
        args += [PROJECT_PATH]

    argv = ["--rootdir", PROJECT_PATH] + args
    sys.exit(pytest.main(argv))
