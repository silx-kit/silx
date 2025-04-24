#!/usr/bin/env python3
"""
Script to check if all python sources files are actually registered in meson
"""
import os
import subprocess
import sys




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

def main(root):
    """print out all python filenames not properly registered in meson.build

    :param root: path of the root of the project
    :return: number of missing files/directories.
    """
    cnt = 0
    src = []
    meson = []

    for line in get_all_files(root):
        if line.endswith(".py") and line.startswith("src"):
            src.append(line)
        elif line.endswith("meson.build"):
            meson.append(line)
    mesoncache = {}
    for f in src:
        base, fn = os.path.split(f)
        mesonfile = os.path.join(base,"meson.build")
        if mesonfile in meson:
            if mesonfile not in mesoncache:
                with open(mesonfile) as fd:
                    mesoncache[mesonfile] = fd.read()
        else:
            print(f"File `{f}` is in a directory without `meson.build` file")
            cnt += 1
            continue
        mesondata = mesoncache[mesonfile]
        if fn not in mesondata:
            print(f"File `{f}` is not listed in `{mesonfile}` file")
            cnt += 1
    print(f"Analyzed {len(src)} python source files in {len(meson)} submodules and found {cnt} issues to investigate")
    return cnt


if __name__=="__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main(root))
