#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bootstrap helps you to test scripts without installing them
by patching your PYTHONPATH on the fly

example: ./bootstrap.py ipython
"""

__authors__ = ["Frédéric-Emmanuel Picca", "Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__date__ = "07/04/2017"


import sys
import os
import distutils.util
import subprocess
import logging

logging.basicConfig()
logger = logging.getLogger("bootstrap")


def _distutils_dir_name(dname="lib"):
    """
    Returns the name of a distutils build directory
    """
    platform = distutils.util.get_platform()
    architecture = "%s.%s-%i.%i" % (dname, platform,
                                    sys.version_info[0], sys.version_info[1])
    return architecture


def _distutils_scripts_name():
    """Return the name of the distrutils scripts sirectory"""
    f = "scripts-{version[0]}.{version[1]}"
    return f.format(version=sys.version_info)


def _get_available_scripts(path):
    res = []
    try:
        res = " ".join([s.rstrip('.py') for s in os.listdir(path)])
    except OSError:
        res = ["no script available, did you ran "
               "'python setup.py build' before bootstrapping ?"]
    return res


if sys.version_info[0] >= 3:  # Python3
    def execfile(fullpath, globals=None, locals=None):
        "Python3 implementation for execfile"
        with open(fullpath) as f:
            try:
                data = f.read()
            except UnicodeDecodeError:
                raise SyntaxError("Not a Python script")
            code = compile(data, fullpath, 'exec')
            exec(code, globals, locals)


def run_file(filename, argv):
    """
    Execute a script trying first to use execfile, then a subprocess

    :param str filename: Script to execute
    :param list[str] argv: Arguments passed to the filename
    """
    full_args = [filename]
    full_args.extend(argv)

    try:
        logger.info("Execute target using exec")
        # execfile is considered as a local call.
        # Providing globals() as locals will force to feed the file into
        # globals() (for examples imports).
        # Without this any function call from the executed file loses imports
        try:
            old_argv = sys.argv
            sys.argv = full_args
            logger.info("Patch the sys.argv: %s", sys.argv)
            logger.info("Executing %s.main()", filename)
            print("########### EXECFILE ###########")
            execfile(filename, globals(), globals())
        finally:
            sys.argv = old_argv
    except SyntaxError as error:
        logger.error(error)
        logger.info("Execute target using subprocess")
        env = os.environ.copy()
        env.update({"PYTHONPATH": LIBPATH + os.pathsep + os.environ.get("PYTHONPATH", ""),
                    "PATH": SCRIPTSPATH + os.pathsep + os.environ.get("PATH", "")})
        print("########### SUBPROCESS ###########")
        run = subprocess.Popen(full_args, shell=False, env=env)
        run.wait()


def find_executable_file(script_name):
    """Find a filename from a script name.

    Check the script name as file path, then checks files from the 'scripts'
    directory, then search the script from the PATH environment variable.

    :param str script_name: Name of the script
    :returns: An absolute file path, else None if nothing found.
    """
    if os.path.isfile(script_name):
        return os.path.abspath(script_name)

    # search the file from the script path
    path = os.path.join(SCRIPTSPATH, script_name)
    if os.path.isfile(path):
        return path

    # search the file from env PATH
    for dirname in os.environ.get("PATH", "").split(os.pathsep):
        path = os.path.join(dirname, script_name)
        if os.path.isfile(path):
            return path

    return None


home = os.path.dirname(os.path.abspath(__file__))
SCRIPTSPATH = os.path.join(home, 'build', _distutils_scripts_name())
LIBPATH = os.path.join(home, 'build', _distutils_dir_name('lib'))
cwd = os.getcwd()
os.chdir(home)
build = subprocess.Popen([sys.executable, "setup.py", "build"],
                         shell=False, cwd=os.path.dirname(os.path.abspath(__file__)))
logger.info("Build process ended with rc= %s", build.wait())
os.chdir(cwd)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.warning("usage: ./bootstrap.py <script>\n")
        logger.warning("Available scripts : %s\n" %
                        _get_available_scripts(SCRIPTSPATH))
        script = None
    else:
        script = sys.argv[1]

    if script:
        logger.info("Executing %s from source checkout", script)
    else:
        logging.info("Running iPython by default")
    sys.path.insert(0, LIBPATH)
    logger.info("Patched sys.path with %s", LIBPATH)

    if script:
        argv = sys.argv[2:]
        fullpath = find_executable_file(script)
        if fullpath is not None:
            run_file(fullpath, argv)
        else:
            logger.error("Script %s not found", script)
    else:
        logger.info("Patch the sys.argv: %s", sys.argv)
        sys.path.insert(2, "")
        try:
            from IPython import embed
        except Exception as err:
            logger.error("Unable to execute iPython, using normal Python")
            logger.error(err)
            import code
            code.interact()
        else:
            embed()
