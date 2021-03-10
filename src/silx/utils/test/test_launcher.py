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
"""Tests for html module"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "17/01/2018"


import sys
import unittest
from silx.utils.testutils import ParametricTestCase
from .. import launcher


class CallbackMock():

    def __init__(self, result=None):
        self._execute_count = 0
        self._execute_argv = None
        self._result = result

    def execute(self, argv):
        self._execute_count = self._execute_count + 1
        self._execute_argv = argv
        return self._result

    def __call__(self, argv):
        return self.execute(argv)


class TestLauncherCommand(unittest.TestCase):
    """Tests for launcher class."""

    def testEnv(self):
        command = launcher.LauncherCommand("foo")
        old = sys.argv
        params = ["foo", "bar"]
        with command.get_env(params):
            self.assertEqual(params, sys.argv)
        self.assertEqual(sys.argv, old)

    def testEnvWhileException(self):
        command = launcher.LauncherCommand("foo")
        old = sys.argv
        params = ["foo", "bar"]
        try:
            with command.get_env(params):
                raise RuntimeError()
        except RuntimeError:
            pass
        self.assertEqual(sys.argv, old)

    def testExecute(self):
        params = ["foo", "bar"]
        callback = CallbackMock(result=42)
        command = launcher.LauncherCommand("foo", function=callback)
        status = command.execute(params)
        self.assertEqual(callback._execute_count, 1)
        self.assertEqual(callback._execute_argv, params)
        self.assertEqual(status, 42)


class TestModuleCommand(ParametricTestCase):

    def setUp(self):
        module_name = "silx.utils.test.test_launcher_command"
        command = launcher.LauncherCommand("foo", module_name=module_name)
        self.command = command

    def testHelp(self):
        status = self.command.execute(["--help"])
        self.assertEqual(status, 0)

    def testException(self):
        try:
            self.command.execute(["exception"])
            self.fail()
        except RuntimeError:
            pass

    def testCall(self):
        status = self.command.execute([])
        self.assertEqual(status, 0)

    def testError(self):
        status = self.command.execute(["error"])
        self.assertEqual(status, -1)


class TestLauncher(ParametricTestCase):
    """Tests for launcher class."""

    def testCallCommand(self):
        callback = CallbackMock(result=42)
        runner = launcher.Launcher(prog="prog")
        command = launcher.LauncherCommand("foo", function=callback)
        runner.add_command(command=command)
        status = runner.execute(["prog", "foo", "param1", "param2"])
        self.assertEqual(status, 42)
        self.assertEqual(callback._execute_argv, ["prog foo", "param1", "param2"])
        self.assertEqual(callback._execute_count, 1)

    def testAddCommand(self):
        runner = launcher.Launcher(prog="prog")
        module_name = "silx.utils.test.test_launcher_command"
        runner.add_command("foo", module_name=module_name)
        status = runner.execute(["prog", "foo"])
        self.assertEqual(status, 0)

    def testCallHelpOnCommand(self):
        callback = CallbackMock(result=42)
        runner = launcher.Launcher(prog="prog")
        command = launcher.LauncherCommand("foo", function=callback)
        runner.add_command(command=command)
        status = runner.execute(["prog", "--help", "foo"])
        self.assertEqual(status, 42)
        self.assertEqual(callback._execute_argv, ["prog foo", "--help"])
        self.assertEqual(callback._execute_count, 1)

    def testCallHelpOnCommand2(self):
        callback = CallbackMock(result=42)
        runner = launcher.Launcher(prog="prog")
        command = launcher.LauncherCommand("foo", function=callback)
        runner.add_command(command=command)
        status = runner.execute(["prog", "help", "foo"])
        self.assertEqual(status, 42)
        self.assertEqual(callback._execute_argv, ["prog foo", "--help"])
        self.assertEqual(callback._execute_count, 1)

    def testCallHelpOnUnknownCommand(self):
        callback = CallbackMock(result=42)
        runner = launcher.Launcher(prog="prog")
        command = launcher.LauncherCommand("foo", function=callback)
        runner.add_command(command=command)
        status = runner.execute(["prog", "help", "foo2"])
        self.assertEqual(status, -1)

    def testNotAvailableCommand(self):
        callback = CallbackMock(result=42)
        runner = launcher.Launcher(prog="prog")
        command = launcher.LauncherCommand("foo", function=callback)
        runner.add_command(command=command)
        status = runner.execute(["prog", "foo2", "param1", "param2"])
        self.assertEqual(status, -1)
        self.assertEqual(callback._execute_count, 0)

    def testCallHelp(self):
        callback = CallbackMock(result=42)
        runner = launcher.Launcher(prog="prog")
        command = launcher.LauncherCommand("foo", function=callback)
        runner.add_command(command=command)
        status = runner.execute(["prog", "help"])
        self.assertEqual(status, 0)
        self.assertEqual(callback._execute_count, 0)

    def testCommonCommands(self):
        runner = launcher.Launcher()
        tests = [
            ["prog"],
            ["prog", "--help"],
            ["prog", "--version"],
            ["prog", "help", "--help"],
            ["prog", "help", "help"],
        ]
        for arguments in tests:
            with self.subTest(args=tests):
                status = runner.execute(arguments)
                self.assertEqual(status, 0)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    test_suite = unittest.TestSuite()
    test_suite.addTest(loader(TestLauncherCommand))
    test_suite.addTest(loader(TestLauncher))
    test_suite.addTest(loader(TestModuleCommand))
    return test_suite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
