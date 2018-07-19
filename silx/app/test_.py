# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
# ############################################################################*/
"""Launch unittests of the library"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "12/01/2018"

import sys
import argparse
import logging
import unittest


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
for h in logging.root.handlers:
    logging.root.removeHandler(h)
logging.root.addHandler(createBasicHandler())
logging.captureWarnings(True)

_logger = logging.getLogger(__name__)
"""Module logger"""


class TextTestResultWithSkipList(unittest.TextTestResult):
    """Override default TextTestResult to display list of skipped tests at the
    end
    """

    def printErrors(self):
        unittest.TextTestResult.printErrors(self)
        # Print skipped tests at the end
        self.printErrorList("SKIPPED", self.skipped)


def main(argv):
    """
    Main function to launch the unittests as an application

    :param argv: Command line arguments
    :returns: exit status
    """
    from silx.test import utils

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", default=0,
                        action="count", dest="verbose",
                        help="Increase verbosity. Option -v prints additional " +
                             "INFO messages. Use -vv for full verbosity, " +
                             "including debug messages and test help strings.")
    parser.add_argument("--qt-binding", dest="qt_binding", default=None,
                        help="Force using a Qt binding: 'PyQt5' or 'PySide2'")
    utils.test_options.add_parser_argument(parser)

    options = parser.parse_args(argv[1:])

    test_verbosity = 1
    use_buffer = True
    if options.verbose == 1:
        logging.root.setLevel(logging.INFO)
        _logger.info("Set log level: INFO")
        test_verbosity = 2
        use_buffer = False
    elif options.verbose > 1:
        logging.root.setLevel(logging.DEBUG)
        _logger.info("Set log level: DEBUG")
        test_verbosity = 2
        use_buffer = False

    if options.qt_binding:
        binding = options.qt_binding.lower()
        if binding == "pyqt4":
            _logger.info("Force using PyQt4")
            import PyQt4.QtCore  # noqa
        elif binding == "pyqt5":
            _logger.info("Force using PyQt5")
            import PyQt5.QtCore  # noqa
        elif binding == "pyside":
            _logger.info("Force using PySide")
            import PySide.QtCore  # noqa
        elif binding == "pyside2":
            _logger.info("Force using PySide2")
            import PySide2.QtCore  # noqa
        else:
            raise ValueError("Qt binding '%s' is unknown" % options.qt_binding)

    # Configure test options
    utils.test_options.configure(options)

    # Run the tests
    runnerArgs = {}
    runnerArgs["verbosity"] = test_verbosity
    runnerArgs["buffer"] = use_buffer
    runner = unittest.TextTestRunner(**runnerArgs)
    runner.resultclass = TextTestResultWithSkipList

    # Display the result when using CTRL-C
    unittest.installHandler()

    import silx.test
    test_suite = unittest.TestSuite()
    test_suite.addTest(silx.test.suite())
    result = runner.run(test_suite)

    if result.wasSuccessful():
        exit_status = 0
    else:
        exit_status = 1
    return exit_status
