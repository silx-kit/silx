#!/usr/bin/env python
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2016 European Synchrotron Radiation Facility
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
"""This program is mainly used to execute application modules provided
by the library.
"""

__authors__ = ["P. Knobel", "V. Valls"]
__license__ = "MIT"
__date__ = "30/03/2017"


import sys
import importlib
import logging


logging.basicConfig()
_logger = logging.getLevelName("silx.launcher")


default_apps = {
    'help': ("silx.app.help", "Show help of the following command"),
    'view': ("silx.app.view", "Browse a data file with a GUI"),
}
"""List available applications and a short description"""


def get_module_from_name(module_name):
    """
    Get a module object from it's name

    :param str module_name: Name of the module to load
    :returns: Python module
    """
    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        _logger.debug("Error while reaching module ''%s", module_name, exc_info=True)
        return None


def bootstrap_module(module, argv):
    """Execute the module

    It must contains a main function taking an argument list as single
    parameter.

    :param module: module A python module
    :param argv: list[string] argv Argument to use for the main function
    :rtype: int
    """

    # fix the context
    old_argv = sys.argv
    sys.argv = argv

    try:
        # reach the 'main' function
        if not hasattr(module, "main"):
            raise TypeError("Module excpect to have a 'main' function")
        else:
            main = getattr(module, "main")

        # run main() function
        status = main(argv)
        return status

    finally:
        # clean up the context
        sys.argv = old_argv


def print_help():
    """Print the help into stdout.
    """
    print("usage: silx [--version] <command> [<args>]")
    print("")
    print("The silx commands are:")
    commands = sorted(default_apps.keys())
    for command in commands:
        _module_name, description = default_apps[command]
        print("   {:10s} {:s}".format(command, description))
    print("")
    print("See 'silx help <command>' to read about a specific subcommand")


def main():
    """Main function of the launcher

    :rtype: int
    """
    if len(sys.argv) <= 1:
        print_help()
        return 0

    command = sys.argv[1]

    if command == "--version":
        import silx._version
        print("silx version %s" % silx._version.version)
        return 0

    if command == "help":
        # Special help command
        if len(sys.argv) == 2:
            print_help()
            return 0
        else:
            command = sys.argv[2]
            args = sys.argv[2:] + ["--help"]
            args[0] = "silx " + args[0]
    else:
        args = sys.argv[1:]
        args[0] = "silx " + args[0]

    if command not in default_apps:
        print("Unknown command: %s", command)
        print_help()
        return -1

    module_name = default_apps[command][0]
    module = get_module_from_name(module_name)
    if module is None:
        raise Exception("Module name '%s' unknown" % module_name)
    status = bootstrap_module(module, args)
    return status


if __name__ == "__main__":
    status = main()
    sys.exit(status)
