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
"""Module testing silx.app.view"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "07/06/2018"


import os
import shutil
import sys
import tempfile
import unittest
import logging
import subprocess
import pytest

from .. import main
from silx import __main__ as silx_main

_logger = logging.getLogger(__name__)


@pytest.mark.usefixtures("qapp")
class TestLauncher(unittest.TestCase):
    """Test command line parsing"""

    def testHelp(self):
        # option -h must cause a raise SystemExit or a return 0
        try:
            parser = main.createParser()
            parser.parse_args(["view", "--help"])
            result = 0
        except SystemExit as e:
            result = e.args[0]
        self.assertEqual(result, 0)

    def testWrongOption(self):
        try:
            parser = main.createParser()
            parser.parse_args(["view", "--foo"])
            self.fail()
        except SystemExit as e:
            result = e.args[0]
        self.assertNotEqual(result, 0)

    def testWrongFile(self):
        try:
            parser = main.createParser()
            result = parser.parse_args(["view", "__file.not.found__"])
            result = 0
        except SystemExit as e:
            result = e.args[0]
        self.assertEqual(result, 0)

    def executeAsScript(self, filename, *args):
        """Execute a command line.

        Log output as debug in case of bad return code.
        """
        env = self.createTestEnv()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy file to temporary dir to avoid import from current dir.
            script = os.path.join(tmpdir, "launcher.py")
            shutil.copyfile(filename, script)
            command_line = [sys.executable, script] + list(args)

            _logger.info("Execute: %s", " ".join(command_line))
            p = subprocess.Popen(
                command_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
            )
            out, err = p.communicate()
            _logger.info("Return code: %d", p.returncode)
            try:
                out = out.decode("utf-8")
            except UnicodeError:
                pass
            try:
                err = err.decode("utf-8")
            except UnicodeError:
                pass

            if p.returncode != 0:
                _logger.info("stdout:")
                _logger.info("%s", out)
                _logger.info("stderr:")
                _logger.info("%s", err)
            else:
                _logger.debug("stdout:")
                _logger.debug("%s", out)
                _logger.debug("stderr:")
                _logger.debug("%s", err)
            self.assertEqual(p.returncode, 0)

    def createTestEnv(self):
        """
        Returns an associated environment with a working project.
        """
        env = {str(k): str(v) for k, v in os.environ.items()}
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        return env

    def testExecuteViewHelp(self):
        """Test if the main module is well connected.

        Uses subprocess to avoid to parasite the current environment.
        """
        self.executeAsScript(main.__file__, "--help")

    def testExecuteSilxViewHelp(self):
        """Test if the main module is well connected.

        Uses subprocess to avoid to parasite the current environment.
        """
        self.executeAsScript(silx_main.__file__, "view", "--help")
