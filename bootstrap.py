#!/usr/bin/env python3
"""
Bootstrap helps you to test scripts without installing them
by patching your PYTHONPATH on the fly

example: ./bootstrap.py ipython
"""

__authors__ = ["Frédéric-Emmanuel Picca", "Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__date__ = "30/09/2020"

import argparse
import logging
import os
import subprocess
import sys
import sysconfig

logging.basicConfig()
logger = logging.getLogger("bootstrap")


def is_debug_python():
    """Returns true if the Python interpreter is in debug mode."""
    if sysconfig.get_config_var("Py_DEBUG"):
        return True

    return hasattr(sys, "gettotalrefcount")


def _setuptools_dir_name(dname="lib"):
    """
    Returns the name of a setuptools build directory
    """
    platform = sysconfig.get_platform()
    architecture = "%s.%s-%i.%i" % (dname, platform,
                                    sys.version_info[0], sys.version_info[1])
    if is_debug_python():
        architecture += "-pydebug"
    return architecture


def _setuptools_scripts_name():
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
            module_globals = globals().copy()
            module_globals['__file__'] = filename
            execfile(filename, module_globals, module_globals)
        finally:
            sys.argv = old_argv
    except SyntaxError as error:
        logger.error(error)
        logger.info("Execute target using subprocess")
        env = os.environ.copy()
        env.update({"PYTHONPATH": LIBPATH + os.pathsep + os.environ.get("PYTHONPATH", ""),
                    "PATH": os.environ.get("PATH", "")})
        print("########### SUBPROCESS ###########")
        run = subprocess.Popen(full_args, shell=False, env=env)
        run.wait()


def run_entry_point(entry_point, argv):
    """
    Execute an entry_point using the current python context
    (http://setuptools.readthedocs.io/en/latest/setuptools.html#automatic-script-creation)

    :param str entry_point: A string identifying a function from a module
        (NAME = PACKAGE.MODULE:FUNCTION [EXTRA])
    """
    import importlib
    elements = entry_point.split("=")
    target_name = elements[0].strip()
    elements = elements[1].split(":")
    module_name = elements[0].strip()
    # Take care of entry_point optional "extra" requirements declaration
    function_name = elements[1].split()[0].strip()

    logger.info("Execute target %s (function %s from module %s) using importlib", target_name, function_name, module_name)
    full_args = [target_name]
    full_args.extend(argv)
    try:
        old_argv = sys.argv
        sys.argv = full_args
        print("########### IMPORTLIB ###########")
        module = importlib.import_module(module_name)
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            func()
        else:
            logger.info("Function %s not found", function_name)
    finally:
        sys.argv = old_argv


def find_executable(target):
    """Find a filename from a script name.

    - Check the script name as file path,
    - Then checks if the name is a target of the setup.py
    - Then search the script from the PATH environment variable.

    :param str target: Name of the script
    :returns: Returns a tuple: kind, name.
    """
    if os.path.isfile(target):
        return ("path", os.path.abspath(target))

    # search the file from setup.py
    import setup
    config = setup.get_project_configuration()
    # scripts from project configuration
    if "scripts" in config:
        for script_name in config["scripts"]:
            if os.path.basename(script) == target:
                return ("path", os.path.abspath(script_name))
    # entry-points from project configuration
    if "entry_points" in config:
        for kind in config["entry_points"]:
            for entry_point in config["entry_points"][kind]:
                elements = entry_point.split("=")
                name = elements[0].strip()
                if name == target:
                    return ("entry_point", entry_point)

    # search the file from env PATH
    for dirname in os.environ.get("PATH", "").split(os.pathsep):
        path = os.path.join(dirname, target)
        if os.path.isfile(path):
            return ("path", path)

    return None, None


def main(argv):
    parser = argparse.ArgumentParser(
        prog="bootstrap", usage="./bootstrap.py <script>", description=__doc__)
    parser.add_argument("script", nargs=argparse.REMAINDER)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-m", nargs=argparse.REMAINDER, dest='module',
        help="run library module as a script (terminates option list)")
    group.add_argument(
        "-j", "--jupyter", action='store_true',
        help="Start jupyter notebook rather than IPython console")
    options = parser.parse_args()

    if options.jupyter:
        if options.script:
            logger.error("-j, --jupyter is mutually exclusive with other options")
            parser.print_help()
            return

        logger.info("Start Jupyter notebook")
        from notebook.notebookapp import main as notebook_main
        os.environ["PYTHONPATH"] = LIBPATH + os.pathsep + os.environ.get("PYTHONPATH", "")
        notebook_main(argv=[])

    elif options.script:
        logger.info("Executing %s from source checkout", options.script)
        script = options.script[0]
        argv = options.script[1:]
        kind, target = find_executable(script)
        if kind == "path":
            run_file(target, argv)
        elif kind == "entry_point":
            run_entry_point(target, argv)
        else:
            logger.error("Script %s not found", options.script)

    elif options.module:
        logging.info("Running module %s", options.module)
        import runpy
        module = options.module[0]
        try:
            old = sys.argv
            sys.argv = [None] + options.module[1:]
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        finally:
            sys.argv = old

    else:
        logging.info("Running IPython by default")
        try:
            from IPython import start_ipython
        except Exception as err:
            logger.error("Unable to execute iPython, using normal Python")
            logger.error(err)
            import code
            code.interact()
        else:
            start_ipython(argv=[])


if __name__ == "__main__":
    home = os.path.dirname(os.path.abspath(__file__))
    LIBPATH = os.path.join(home, 'build', _setuptools_dir_name('lib'))
    cwd = os.getcwd()
    os.chdir(home)
    build = subprocess.Popen(
        [sys.executable, "setup.py", "build", "--build-lib", LIBPATH],
        shell=False,
    )
    build_rc = build.wait()
    if not os.path.exists(LIBPATH):
        logger.warning("`lib` directory does not exist, trying common Python3 lib")
        LIBPATH = os.path.join(os.path.split(LIBPATH)[0], "lib")
    os.chdir(cwd)

    if build_rc == 0:
        logger.info("Build process ended.")
    else:
        logger.error("Build process ended with rc=%s", build_rc)
        sys.exit(-1)

    sys.path.insert(0, LIBPATH)
    logger.info("Patched sys.path with %s", LIBPATH)

    main(sys.argv)

