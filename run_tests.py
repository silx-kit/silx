#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2015-2018 European Synchrotron Radiation Facility
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
__date__ = "02/03/2018"
__license__ = "MIT"

import distutils.util
import logging
import os
import subprocess
import sys
import time
import unittest
import collections
from argparse import ArgumentParser


class StreamHandlerUnittestReady(logging.StreamHandler):
    """The unittest class TestResult redefine sys.stdout/err to capture
    stdout/err from tests and to display them only when a test fail.
    This class allow to use unittest stdout-capture by using the last sys.stdout
    and not a cached one.
    """

    def emit(self, record):
        """
        :type record: logging.LogRecord
        """
        self.stream = sys.stderr
        super(StreamHandlerUnittestReady, self).emit(record)

    def flush(self):
        pass


def createBasicHandler():
    """Create the handler using the basic configuration"""
    hdlr = StreamHandlerUnittestReady()
    fs = logging.BASIC_FORMAT
    dfs = None
    fmt = logging.Formatter(fs, dfs)
    hdlr.setFormatter(fmt)
    return hdlr


# Use an handler compatible with unittests, else use_buffer is not working
logging.root.addHandler(createBasicHandler())

# Capture all default warnings
logging.captureWarnings(True)
import warnings
warnings.simplefilter('default')

logger = logging.getLogger("run_tests")
logger.setLevel(logging.WARNING)

logger.info("Python %s %s", sys.version, tuple.__itemsize__ * 8)


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


class TextTestResultWithSkipList(unittest.TextTestResult):
    """Override default TextTestResult to display list of skipped tests at the
    end
    """

    def printErrors(self):
        unittest.TextTestResult.printErrors(self)
        # Print skipped tests at the end
        self.printGroupedList("SKIPPED", self.skipped)

    def printGroupedList(self, flavour, errors):
        grouped = collections.OrderedDict()

        for test, err in errors:
            if err in grouped:
                grouped[err] = grouped[err] + [test]
            else:
                grouped[err] = [test]

        for err, tests in grouped.items():
            self.stream.writeln(self.separator1)
            for test in tests:
                self.stream.writeln("%s: %s" % (flavour, self.getDescription(test)))
            self.stream.writeln(self.separator2)
            self.stream.writeln("%s" % err)


class ProfileTextTestResult(unittest.TextTestRunner.resultclass):

    def __init__(self, *arg, **kwarg):
        unittest.TextTestRunner.resultclass.__init__(self, *arg, **kwarg)
        self.logger = logging.getLogger("memProf")
        self.logger.setLevel(min(logging.INFO, logging.root.level))
        self.logger.handlers.append(logging.FileHandler("profile.log"))

    def startTest(self, test):
        unittest.TextTestRunner.resultclass.startTest(self, test)
        if resource:
            self.__mem_start = \
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.__time_start = time.time()

    def stopTest(self, test):
        unittest.TextTestRunner.resultclass.stopTest(self, test)
        # see issue 311. For other platform, get size of ru_maxrss in "man getrusage"
        if sys.platform == "darwin":
            ratio = 1e-6
        else:
            ratio = 1e-3
        if resource:
            memusage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss -
                        self.__mem_start) * ratio
        else:
            memusage = 0
        self.logger.info("Time: %.3fs \t RAM: %.3f Mb\t%s",
                         time.time() - self.__time_start,
                         memusage, test.id())


def report_rst(cov, package, version="0.0.0", base=""):
    """
    Generate a report of test coverage in RST (for Sphinx inclusion)

    :param cov: test coverage instance
    :param str package: Name of the package
    :param str base: base directory of modules to include in the report
    :return: RST string
    """
    import tempfile
    fd, fn = tempfile.mkstemp(suffix=".xml")
    os.close(fd)
    cov.xml_report(outfile=fn)

    from lxml import etree
    xml = etree.parse(fn)
    classes = xml.xpath("//class")

    line0 = "Test coverage report for %s" % package
    res = [line0, "=" * len(line0), ""]
    res.append("Measured on *%s* version %s, %s" %
               (package, version, time.strftime("%d/%m/%Y")))
    res += ["",
            ".. csv-table:: Test suite coverage",
            '   :header: "Name", "Stmts", "Exec", "Cover"',
            '   :widths: 35, 8, 8, 8',
            '']
    tot_sum_lines = 0
    tot_sum_hits = 0

    for cl in classes:
        name = cl.get("name")
        fname = cl.get("filename")
        if not os.path.abspath(fname).startswith(base):
            continue
        lines = cl.find("lines").getchildren()
        hits = [int(i.get("hits")) for i in lines]

        sum_hits = sum(hits)
        sum_lines = len(lines)

        cover = 100.0 * sum_hits / sum_lines if sum_lines else 0

        if base:
            name = os.path.relpath(fname, base)

        res.append('   "%s", "%s", "%s", "%.1f %%"' %
                   (name, sum_lines, sum_hits, cover))
        tot_sum_lines += sum_lines
        tot_sum_hits += sum_hits
    res.append("")
    res.append('   "%s total", "%s", "%s", "%.1f %%"' %
               (package, tot_sum_lines, tot_sum_hits,
                100.0 * tot_sum_hits / tot_sum_lines if tot_sum_lines else 0))
    res.append("")
    return os.linesep.join(res)


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
    # Prevent importing from source directory
    if (os.path.dirname(os.path.abspath(__file__)) == os.path.abspath(sys.path[0])):
        removed_from_sys_path = sys.path.pop(0)
        logger.info("Patched sys.path, removed: '%s'", removed_from_sys_path)

    if "--installed" in sys.argv:
        try:
            module = importer(project_name)
        except ImportError:
            raise ImportError(
                "%s not installed: Cannot run tests on installed version" %
                PROJECT_NAME)
    else:  # Use built source
        build_dir = build_project(project_name, project_dir)
        if build_dir is None:
            logging.error("Built project is not available !!! investigate")
        sys.path.insert(0, build_dir)
        logger.warning("Patched sys.path, added: '%s'", build_dir)
        module = importer(project_name)
    return module


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


if __name__ == "__main__":  # Needed for multiprocessing support on Windows
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_NAME = get_project_name(PROJECT_DIR)
    logger.info("Project name: %s", PROJECT_NAME)

    project_module = import_project_module(PROJECT_NAME, PROJECT_DIR)
    PROJECT_VERSION = getattr(project_module, 'version', '')
    PROJECT_PATH = project_module.__path__[0]


    test_options = get_test_options(project_module)
    """Contains extra configuration for the tests."""


    epilog = """Environment variables:
    WITH_QT_TEST=False to disable graphical tests
    SILX_OPENCL=False to disable OpenCL tests
    SILX_TEST_LOW_MEM=True to disable tests taking large amount of memory
    GPU=False to disable the use of a GPU with OpenCL test
    WITH_GL_TEST=False to disable tests using OpenGL
    """
    parser = ArgumentParser(description='Run the tests.',
                            epilog=epilog)

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
    if test_options is not None:
        test_options.add_parser_argument(parser)

    default_test_name = "%s.test.suite" % PROJECT_NAME
    parser.add_argument("test_name", nargs='*',
                        default=(default_test_name,),
                        help="Test names to run (Default: %s)" % default_test_name)
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
            cov = coverage.Coverage(omit=omits)
        except AttributeError:
            cov = coverage.coverage(omit=omits)
        cov.start()

    if options.qt_binding:
        binding = options.qt_binding.lower()
        if binding == "pyqt4":
            logger.info("Force using PyQt4")
            if sys.version < "3.0.0":
                try:
                    import sip
                    sip.setapi("QString", 2)
                    sip.setapi("QVariant", 2)
                except Exception:
                    logger.warning("Cannot set sip API")
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

    # Run the tests
    runnerArgs = {}
    runnerArgs["verbosity"] = test_verbosity
    runnerArgs["buffer"] = use_buffer
    if options.memprofile:
        runnerArgs["resultclass"] = ProfileTextTestResult
    else:
        runnerArgs["resultclass"] = TextTestResultWithSkipList
    runner = unittest.TextTestRunner(**runnerArgs)

    logger.warning("Test %s %s from %s",
                   PROJECT_NAME, PROJECT_VERSION, PROJECT_PATH)

    test_module_name = PROJECT_NAME + '.test'
    logger.info('Import %s', test_module_name)
    test_module = importer(test_module_name)
    test_suite = unittest.TestSuite()

    if test_options is not None:
        # Configure the test options according to the command lines and the the environment
        test_options.configure(options)
    else:
        logger.warning("No test options available.")


    if not options.test_name:
        # Do not use test loader to avoid cryptic exception
        # when an error occur during import
        project_test_suite = getattr(test_module, 'suite')
        test_suite.addTest(project_test_suite())
    else:
        test_suite.addTest(
            unittest.defaultTestLoader.loadTestsFromNames(options.test_name))

    # Display the result when using CTRL-C
    unittest.installHandler()

    result = runner.run(test_suite)

    if result.wasSuccessful():
        exit_status = 0
    else:
        exit_status = 1


    if options.coverage:
        cov.stop()
        cov.save()
        with open("coverage.rst", "w") as fn:
            fn.write(report_rst(cov, PROJECT_NAME, PROJECT_VERSION, PROJECT_PATH))

    sys.exit(exit_status)
