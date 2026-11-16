#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap helps you to test scripts without installing them
by patching your PYTHONPATH on the fly

example: ./bootstrap.py ipython
         ./bootstrap.py -c "import silx; print(silx.version)"
"""

__authors__ = ["Frédéric-Emmanuel Picca", "Jérôme Kieffer"]
__contact__ = "jerome.kieffer@esrf.eu"
__license__ = "MIT"
__date__ = "16/11/2026"

import sys
import os
import argparse
import json
import subprocess
import logging

if sys.version_info[:2] < (3, 11):
    import tomli
else:
    import tomllib as tomli
logging.basicConfig()
logger = logging.getLogger("bootstrap")


def get_project_name(root_dir):
    """Retrieve project name by running python setup.py --name in root_dir.

    :param str root_dir: Directory where to run the command.
    :return: The name of the project stored in root_dir
    """
    logger.debug("Getting project name in %s", root_dir)
    with open(os.path.join(root_dir, "pyproject.toml")) as f:
        pyproject = tomli.loads(f.read())
    return pyproject.get("project", {}).get("name")


def find_install_dir(build, name):
    """Locate the directory (in the *build* tree) in which meson installed the package.

    The layout depends on the platform, the distribution and the meson/python
    installation: `lib/python3.13/site-packages` (Linux, macOS), `lib64/...`
    (Fedora), `lib/python3/dist-packages` (Debian), `Lib/site-packages`
    (Windows)... so the path is not guessed but asked to meson.
    If this fails, the build tree is searched for the package.

    :param str build: meson build directory, installed with `--destdir <build>`
    :param str name: name of the package (directory holding `__init__.py`)
    :return: the directory containing the package, or None if not found
    """
    marker = os.path.join(name, "__init__.py")
    candidates = []

    # 1. Ask meson where the package is installed (relative to the prefix)
    try:
        out = subprocess.run(
            ["meson", "introspect", "--installed", build],
            capture_output=True,
            text=True,
            check=True,
            env=os.environ,
        ).stdout
        for installed in json.loads(out).values():
            installed = installed.replace("\\", "/")
            if installed.endswith("/" + marker.replace(os.sep, "/")):
                # like meson does with destdir: the drive letter is dropped
                relative = os.path.splitdrive(installed)[1].lstrip("/")
                candidates.append(os.path.join(build, *relative.split("/")[:-2]))
    except (OSError, subprocess.CalledProcessError, ValueError) as err:
        logger.warning("`meson introspect` failed (%s), searching the build tree", err)

    # 2. Fallback: search the build tree, in a deterministic way
    if not candidates:
        for root, dirs, files in os.walk(build):
            dirs.sort()
            if root[len(build) :].count(os.sep) >= 5:
                dirs[:] = []
            if os.path.basename(root) == name and "__init__.py" in files:
                candidates.append(os.path.dirname(root))

    candidates = [
        i
        for i in dict.fromkeys(os.path.abspath(i) for i in candidates)
        if os.path.isfile(os.path.join(i, marker))
    ]
    if len(candidates) > 1:
        logger.warning(
            "Several installations of %s found: %s. Using the first one.",
            name,
            candidates,
        )
    return candidates[0] if candidates else None


def build_project(name, root_dir):
    """Build locally the project using meson

    :param str name: Name of the project.
    :param str root_dir: Root directory of the project
    :return: The path to the directory in which the package was installed
        (to be added to `sys.path`)
    """
    extra = []
    build_dir = f"build_py{sys.version_info[0]}{sys.version_info[1]}"
    build = os.path.join(root_dir, build_dir)
    if not (os.path.isdir(build) and os.path.isdir(os.path.join(build, name))):
        p = subprocess.Popen(
            ["meson", "setup", build_dir], shell=False, cwd=root_dir, env=os.environ
        )
        p.wait()
    p = subprocess.Popen(
        ["meson", "configure", "--prefix", "/"] + extra,
        shell=False,
        cwd=build,
        env=os.environ,
    )
    p.wait()
    p = subprocess.Popen(
        ["meson", "install", "--destdir", "."], shell=False, cwd=build, env=os.environ
    )
    logger.debug("meson install ended with rc= %s", p.wait())

    home = None
    if os.environ.get("PYBUILD_NAME") == name:
        # we are in the debian packaging way
        home = os.environ.get("PYTHONPATH", "").split(os.pathsep)[-1]
    elif os.environ.get("BUILDPYTHONPATH"):
        home = os.path.abspath(os.environ.get("BUILDPYTHONPATH", ""))
    if home:
        if not os.path.isfile(os.path.join(home, name, "__init__.py")):
            logger.warning(
                "The package %s is not in %s (PYBUILD_NAME/PYTHONPATH/BUILDPYTHONPATH)",
                name,
                home,
            )
    else:
        home = find_install_dir(build, name)
        if home is None:
            raise RuntimeError(
                f"Unable to find the installed package '{name}' in {build}: "
                "the build or `meson install` failed, see the messages above. "
                "Set BUILDPYTHONPATH to the directory containing the package to override."
            )

    logger.warning("Building %s to %s", name, home)

    return home


def execfile(fullpath, globals=None, locals=None):
    "Python3 implementation for execfile"
    with open(fullpath) as f:
        try:
            data = f.read()
        except UnicodeDecodeError:
            raise SyntaxError("Not a Python script")
        code = compile(data, fullpath, "exec")
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
            module_globals["__file__"] = filename
            execfile(filename, module_globals, module_globals)
        finally:
            sys.argv = old_argv
    except SyntaxError as error:
        logger.error(error)
        logger.info("Execute target using subprocess")
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": LIBPATH + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "PATH": os.environ.get("PATH", ""),
            }
        )
        print("########### SUBPROCESS ###########")
        run = subprocess.Popen(full_args, shell=False, env=env)
        run.wait()


def run_entry_point(target_name, entry_point, argv):
    """
    Execute an entry_point using the current python context

    :param str entry_point: A string identifying a function from a module
        (NAME = PACKAGE.MODULE:FUNCTION)
    :param argv: list of arguments
    """
    import importlib

    elements = entry_point.split(":")
    module_name = elements[0].strip()
    function_name = elements[1].strip()

    logger.info(
        "Execute target %s (function %s from module %s) using importlib",
        target_name,
        function_name,
        module_name,
    )
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
    :returns: Returns a tuple: (kind, name) or (kind, name, entry_point).
    """
    if os.path.isfile(target):
        return ("path", os.path.abspath(target))

    # search the executable in pyproject.toml
    with open(os.path.join(PROJECT_DIR, "pyproject.toml")) as f:
        pyproject = tomli.loads(f.read())

    scripts = {}
    scripts.update(pyproject.get("project", {}).get("scripts", {}))
    scripts.update(pyproject.get("project", {}).get("gui-scripts", {}))

    for script, entry_point in scripts.items():
        if script == target:
            # print(script, entry_point)
            return ("entry_point", target, entry_point)
    return None, None


def main(argv):
    parser = argparse.ArgumentParser(
        prog="bootstrap", usage="./bootstrap.py <script>", description=__doc__
    )
    parser.add_argument("script", nargs=argparse.REMAINDER)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-m",
        nargs=argparse.REMAINDER,
        dest="module",
        help="run library module as a script (terminates option list)",
    )
    group.add_argument(
        "-c",
        action="append",
        dest="commands",
        metavar="COMMAND",
        help="execute the Python command(s) from the source checkout, like `python -c`. "
        "Can be repeated: commands run in order and share the same namespace. "
        "Extra arguments end up in sys.argv[1:]",
    )
    options = parser.parse_args()

    if options.commands:
        logger.info("Running %d command(s) from source checkout", len(options.commands))
        old = sys.argv
        sys.argv = ["-c"] + options.script
        namespace = {"__name__": "__main__"}
        try:
            for command in options.commands:
                exec(compile(command, "<string>", "exec"), namespace)
        finally:
            sys.argv = old

    elif options.script:
        logger.info("Executing %s from source checkout", options.script)
        script = options.script[0]
        argv = options.script[1:]
        res = find_executable(script)
        kind = res[0]
        if kind == "path":
            run_file(res[1], argv)
        elif kind == "entry_point":
            run_entry_point(res[1], res[2], argv)
        else:
            logger.error("Script %s not found", options.script)

    elif options.module:
        logger.info("Running module %s", options.module)
        import runpy

        module = options.module[0]
        try:
            old = sys.argv
            sys.argv = [None] + options.module[1:]
            runpy.run_module(module, run_name="__main__", alter_sys=True)
        finally:
            sys.argv = old

    else:
        logger.info("Running IPython by default")
        try:
            from IPython import start_ipython
        except Exception as err:
            logger.error("Unable to execute iPython, using normal Python")
            logger.error(err)
            import code

            code.interact()
        else:
            start_ipython(argv=[])


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = get_project_name(PROJECT_DIR)
logger.info("Project name: %s", PROJECT_NAME)


if __name__ == "__main__":
    home = os.path.dirname(os.path.abspath(__file__))
    LIBPATH = build_project(PROJECT_NAME, PROJECT_DIR)

    sys.path.insert(0, LIBPATH)
    logger.info("Patched sys.path with %s", LIBPATH)

    main(sys.argv)
