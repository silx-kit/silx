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
__date__ = "20/12/2016"


import sys
import os
import distutils.util
import subprocess
import logging
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


def runfile(fname):
    try:
        logger.info("Execute target using exec")
        # execfile is considered as a local call.
        # Providing globals() as locals will force to feed the file into
        # globals() (for examples imports).
        # Without this any function call from the executed file loses imports
        execfile(fname, globals(), globals())
    except SyntaxError as error:
        logger.error(error)
        logger.info("Execute target using subprocess")
        env = os.environ.copy()
        env.update({"PYTHONPATH": LIBPATH + os.pathsep + os.environ.get("PYTHONPATH", ""),
                    "PATH": SCRIPTSPATH + os.pathsep + os.environ.get("PATH", "")})
        run = subprocess.Popen(sys.argv, shell=False, env=env)
        run.wait()

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
        logging.warning("usage: ./bootstrap.py <script>\n")
        logging.warning("Available scripts : %s\n" %
                        _get_available_scripts(SCRIPTSPATH))
        script = None
    else:
        script = sys.argv[1]

    if script:
        logger.info("Executing %s from source checkout", script)
    else:
        logging.info("Running iPython by default")
    sys.path.insert(0, LIBPATH)
    logger.info("01. Patched sys.path with %s", LIBPATH)

    sys.path.insert(0, SCRIPTSPATH)
    logger.info("02. Patched sys.path with %s", SCRIPTSPATH)

    if script:
        sys.argv = sys.argv[1:]
        logger.info("03. patch the sys.argv: %s", sys.argv)
        logger.info("04. Executing %s.main()", script)
        fullpath = os.path.join(SCRIPTSPATH, script)
        if os.path.exists(fullpath):
            runfile(fullpath)
        else:
            if os.path.exists(script):
                runfile(script)
            else:
                for dirname in os.environ.get("PATH", "").split(os.pathsep):
                    fullpath = os.path.join(dirname, script)
                    if os.path.exists(fullpath):
                        runfile(fullpath)
                        break
    else:
        logger.info("03. patch the sys.argv: %s", sys.argv)
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
