#!/usr/bin/env python3
# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016-2023 European Synchrotron Radiation Facility
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
"""Build man pages of the project's entry points

The project's package MUST be installed in the current Python environment.
"""

import logging
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterator, Tuple

from setuptools.config import read_configuration


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROJECT_PATH = Path(__file__).parent.parent


def entry_points(project_path: Path) -> Iterator[Tuple[str, str, str]]:
    config = read_configuration(project_path / "setup.cfg")
    entry_points_config = config.get("options", {}).get("entry_points", {})
    print(entry_points_config)

    for group in ("console_scripts", "gui_scripts"):
        for entry in entry_points_config.get(group, ()):
            print(entry)
            match = re.fullmatch(
                r"(?P<name>\w+)\s*=\s*(?P<module>[^:]+):(?P<object>\w*)",
                entry,
            )
            if match is not None:
                yield match["name"], match["module"], match["object"]


def get_synopsis(module_name: str) -> str:
    """Execute Python commands to retrieve the synopsis for help2man"""
    commands = (
        "import logging",
        "logging.basicConfig(level=logging.ERROR)",
        f"import {module_name}",
        f"print({module_name}.__doc__)",
    )
    result = subprocess.run(
        [sys.executable, "-c", "; ".join(commands)],
        capture_output=True,
    )
    if result.returncode:
        logger.warning("Error while getting synopsis for module '%s'.", module_name)
        return None
    synopsis = result.stdout.decode("utf-8").strip()
    if synopsis == "None":
        return None
    return synopsis


def main(project_path: Path, out_path: Path):
    out_path.mkdir(parents=True, exist_ok=True)

    eps = tuple(entry_points(project_path))
    if not eps:
        raise RuntimeError("No entry points found!")

    for target_name, module_name, function_name in eps:
        logger.info(f"Build man for entry-point target '{target_name}'")
        python_command = [
            sys.executable,
            "-c",
            f'"import {module_name}; {module_name}.{function_name}()"',
        ]

        help2man_command = [
            "help2man",
            "-N",
            " ".join(python_command),
            "-o",
            str(out_path / f"{target_name}.1"),
        ]

        synopsis = get_synopsis(module_name)
        if synopsis:
            help2man_command += ["-n", synopsis]

        result = subprocess.run(help2man_command)
        if result.returncode != 0:
            logger.error(f"Error while generating man file for target '{target_name}'.")
            for argument in ("--help", "--version"):
                test_command = python_command + [argument]
                logger.info(f"Running: {test_command}")
                result = subprocess.run(test_command)
                logger.info(f"\tReturn code: {result.returncode}")
            raise RuntimeError(f"Fail to generate '{target_name}' man documentation")


if __name__ == "__main__":
    main(project_path=PROJECT_PATH, out_path=PROJECT_PATH / "build" / "man")
