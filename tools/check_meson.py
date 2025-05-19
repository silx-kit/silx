#!/usr/bin/env python3
"""
Script to check if all source files are actually registered in meson
"""
from functools import cache
from pathlib import Path
import os
import subprocess
import sys
import zipfile


EXCLUDED_FILES = {
    "src/silx/io/specfile/src/sfwrite.c",
    "src/silx/math/histogramnd/README",
    "src/silx/math/histogramnd/src/histogramnd_template.c",
}
EXCLUDED_FILES = set(map(Path, EXCLUDED_FILES))
"""Files to ignore during the check"""


HEADER_SUFFIXES = ".h", ".hpp", ".pxd"

SOURCE_SUFFIXES = ".c", ".cpp", ".pyx"


def get_repository_files(root: str) -> set[Path]:
    """list of all files in the git repo

    :param root: root of the project
    """
    process = subprocess.run(["git", "ls-files"], cwd=root, stdout=subprocess.PIPE)
    if process.returncode != 0:
        print("`git` command failed")
    res = process.stdout.decode()
    return set(Path(i.strip()) for i in res.split(os.linesep))


def get_wheel_files(wheel_filename: str) -> set[Path]:
    """Returns the files installed by the given wheel"""
    with zipfile.ZipFile(wheel_filename) as zipf:
        return set(
            path
            for path in map(Path, zipf.namelist())
            if not path.parts[0].endswith(".dist-info")
        )


@cache
def read_file(filepath: Path) -> str:
    return filepath.read_text()


def main(repository_root: str, wheel_filename: str) -> bool:
    """Check that all relevant files under version control are in the wheel

    :param repository_root: path of the root of the project
    :param wheel_filename: Path of th wheel to check
    """
    isvalid = True

    meson_files = set()
    source_files = set()
    installable_files = set()
    for filepath in get_repository_files(repository_root):
        if filepath in EXCLUDED_FILES:
            print(f"File excluded from the check: {filepath}")
            continue
        if filepath.suffix in HEADER_SUFFIXES:
            continue

        if filepath.name == "meson.build":
            meson_files.add(filepath)
        elif filepath.suffix in SOURCE_SUFFIXES:
            source_files.add(filepath)
        elif filepath.is_relative_to("src"):
            installable_files.add(filepath.relative_to("src"))
        elif filepath.is_relative_to("examples"):
            # Special case: examples/ is installed as silx.examples
            installable_files.add(Path("silx").joinpath(filepath))

    wheel_extension_files = set()
    wheel_files = set()
    for filepath in get_wheel_files(wheel_filename):
        if filepath.suffix in (".so", ".pyd"):
            wheel_extension_files.add(filepath)
        else:
            wheel_files.add(filepath)

    missing_files = installable_files - wheel_files
    if missing_files:
        print("ERROR: The following files are missing from the wheel:")
        print(*map(str, missing_files))
        isvalid = False

    unexpected_files = wheel_files - installable_files
    if unexpected_files:
        print("ERROR: The following files are not expected to be in the wheel:")
        print(*map(str, unexpected_files))
        isvalid = False

    for filepath in source_files:
        for ancestor in filepath.parents:
            meson_file = ancestor / "meson.build"
            if meson_file in meson_files:
                relative_path = filepath.relative_to(ancestor)
                if f"'{str(relative_path)}'" not in read_file(meson_file):
                    print(f"ERROR: File {relative_path} is not in {meson_file}")
                    isvalid = False
                break
        else:
            print(f"ERROR: No meson.build file for {str(filepath)}")
            isvalid = False

    return isvalid


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(0 if main(root, sys.argv[1]) else 1)
