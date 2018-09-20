#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2017 European Synchrotron Radiation Facility
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
"""This module define silx application available throug the silx launcher.
"""

__authors__ = ["P. Knobel", "V. Valls"]
__license__ = "MIT"
__date__ = "03/04/2017"


import sys
import importlib
import contextlib
import argparse
import logging


_logger = logging.getLogger(__name__)


class LauncherCommand(object):
    """Description of a command"""

    def __init__(self, name, description=None, module_name=None, function=None):
        """
        Constructor

        :param str name: Name of the command
        :param str description: Description of the command
        :param str module_name: Module name to execute
        :param callable function: Python function to execute
        """
        self.name = name
        self.module_name = module_name
        if description is None:
            description = "A command"
        self.description = description
        self.function = function

    def get_module(self):
        """Returns the python module to execute. If any.

        :rtype: module
        """
        try:
            module = importlib.import_module(self.module_name)
            return module
        except ImportError:
            msg = "Error while reaching module '%s'"
            _logger.error(msg, self.module_name, exc_info=True)
            return None

    def get_function(self):
        """Returns the main function to execute.

        :rtype: callable
        """
        if self.function is not None:
            return self.function
        else:
            module = self.get_module()
            if module is None:
                _logger.error("Impossible to load module name '%s'" % self.module_name)
                return None

            # reach the 'main' function
            if not hasattr(module, "main"):
                raise TypeError("Module expect to have a 'main' function")
            else:
                main = getattr(module, "main")
            return main

    @contextlib.contextmanager
    def get_env(self, argv):
        """Fix the environement before and after executing the command.

        :param list argv: The list of arguments (the first one is the name of
            the application and command)
        :rtype: int
        """

        # fix the context
        old_argv = sys.argv
        sys.argv = argv

        try:
            yield
        finally:
            # clean up the context
            sys.argv = old_argv

    def execute(self, argv):
        """Execute the command.

        :param list[str] argv: The list of arguments (the first one is the
            name of the application and command)
        :rtype: int
        :returns: The execution status
        """
        with self.get_env(argv):
            func = self.get_function()
            if func is None:
                _logger.error("Imposible to execute the command '%s'" % self.name)
                return -1
            try:
                status = func(argv)
            except SystemExit as e:
                # ArgumentParser likes to finish with a sys.exit
                status = e.args[0]
            return status


class Launcher(object):
    """
    Manage launch of module.

    Provides an API to describe available commands and feature to display help
    and execute the commands.
    """

    def __init__(self,
                 prog=None,
                 usage=None,
                 description=None,
                 epilog=None,
                 version=None):
        """
        :param str prog: Name of the program. If it is not defined it uses the
            first argument of `sys.argv`
        :param str usage: Custom the string explaining the usage. Else it is
            autogenerated.
        :param str description: Description of the application displayed after the
            usage.
        :param str epilog: Custom the string displayed at the end of the help.
        :param str version: Define the version of the application.
        """
        if prog is None:
            prog = sys.argv[0]
        self.prog = prog
        self.usage = usage
        self.description = description
        self.epilog = epilog
        self.version = version
        self._commands = {}

        help_command = LauncherCommand(
            "help",
            description="Show help of the following command",
            function=self.execute_help)
        self.add_command(command=help_command)

    def add_command(self, name=None, module_name=None, description=None, command=None):
        """
        Add a command to the launcher.

        See also `LauncherCommand`.

        :param str name: Name of the command
        :param str module_name: Module to execute
        :param str description: Description of the command
        :param LauncherCommand command: A `LauncherCommand`
        """
        if command is not None:
            assert(name is None and module_name is None and description is None)
        else:
            command = LauncherCommand(
                name=name,
                description=description,
                module_name=module_name)
        self._commands[command.name] = command

    def print_help(self):
        """Print the help to stdout.
        """
        usage = self.usage
        if usage is None:
            usage = "usage: {0.prog} [--version|--help] <command> [<args>]"
        description = self.description
        epilog = self.epilog
        if epilog is None:
            epilog = "See '{0.prog} help <command>' to read about a specific subcommand"

        print(usage.format(self))
        print("")
        if description is not None:
            print(description)
            print("")
        print("The {0.prog} commands are:".format(self))
        commands = sorted(self._commands.keys())
        for command in commands:
            command = self._commands[command]
            print("   {:10s} {:s}".format(command.name, command.description))
        print("")
        print(epilog.format(self))

    def execute_help(self, argv):
        """Execute the help command.

        :param list[str] argv: The list of arguments (the first one is the
            name of the application with the help command)
        :rtype: int
        :returns: The execution status
        """
        description = "Display help information about %s" % self.prog
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument(
            'command',
            default=None,
            nargs=argparse.OPTIONAL,
            help='Command in which aving help')

        try:
            options = parser.parse_args(argv[1:])
        except SystemExit as e:
            # ArgumentParser likes to finish with a sys.exit
            return e.args[0]

        command_name = options.command
        if command_name is None:
            self.print_help()
            return 0

        if command_name not in self._commands:
            print("Unknown command: %s", command_name)
            self.print_help()
            return -1

        command = self._commands[command_name]
        prog = "%s %s" % (self.prog, command.name)
        return command.execute([prog, "--help"])

    def execute(self, argv=None):
        """Execute the launcher.

        :param list[str] argv: The list of arguments (the first one is the
            name of the application)
        :rtype: int
        :returns: The execution status
        """
        if argv is None:
            argv = sys.argv

        if len(argv) <= 1:
            self.print_help()
            return 0

        command_name = argv[1]

        if command_name == "--version":
            print("%s version %s" % (self.prog, str(self.version)))
            return 0

        if command_name in ["--help", "-h"]:
            # Special help command
            if len(argv) == 2:
                self.print_help()
                return 0
            else:
                command_name = argv[2]
                command_argv = argv[2:] + ["--help"]
                command_argv[0] = "%s %s" % (self.prog, command_argv[0])
        else:
            command_argv = argv[1:]
            command_argv[0] = "%s %s" % (self.prog, command_argv[0])

        if command_name not in self._commands:
            print("Unknown command: %s" % command_name)
            self.print_help()
            return -1

        command = self._commands[command_name]
        return command.execute(command_argv)
