#!/usr/bin/env python3
"""
Script to check if all source files are actually registered in meson
"""
from functools import cache
import os
import subprocess
import sys


EXCLUDED_FILES = [
    "src/silx/io/specfile/src/sfwrite.c",
    "src/silx/math/histogramnd/README",
    "src/silx/math/histogramnd/src/histogramnd_template.c",
]
"""Files to ignore during the check"""


def get_all_files(root):
    """list of all files in the git repo:

    :param root: root of the project
    :return: list of files in the repo
    """
    process = subprocess.run(["git","ls-files"], cwd=root, stdout=subprocess.PIPE)
    if process.returncode !=0:
        print("`git` command failed")
    res = process.stdout.decode()
    return  [i.strip() for i in res.split(os.linesep)]


@cache
def read_file(filename: str):
    with open(filename) as fd:
        return fd.read()


def main(root):
    """print out all python filenames not properly registered in meson.build

    :param root: path of the root of the project
    :return: number of missing files/directories.
    """
    cnt = 0
    src = []
    meson = []

    for line in get_all_files(root):
        if line.endswith("meson.build"):
            meson.append(line)
        else:
            extension = os.path.splitext(line)[1]
            if line.startswith("src") and extension not in (".h", ".hpp", ".pxd"):
                src.append(line)

    for filename in src:
        if filename in EXCLUDED_FILES:
            print(f"File excluded from the check: {filename}")
            continue

        filename_parts = filename.split(os.sep)
        for index in range(len(filename_parts) - 1, 0, -1):
            base_folder = os.path.join(*filename_parts[:index])
            unix_file_path = "/".join(filename_parts[index:])

            mesonfile = os.path.join(base_folder, "meson.build")
            if mesonfile in meson:
                break
        else:
            print(f"ERROR: File `{filename}` is in a directory without `meson.build` file")
            cnt += 1
            continue

        if f"'{unix_file_path}'" not in read_file(mesonfile):
            print(f"ERROR: File `{unix_file_path}` is not listed in `{mesonfile}` file")
            cnt += 1

    print(f"Analyzed {len(src)} source files in {len(meson)} submodules and found {cnt} issues to investigate")
    return cnt


if __name__=="__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main(root))
