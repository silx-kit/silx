#!/usr/bin/env python3

import os
import subprocess
cnt = 0
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
process = subprocess.run(["git","ls-files"], cwd=root, stdout=subprocess.PIPE)
if process.returncode !=0:
    print("`git` command failed")
res = process.stdout.decode()
src = []
meson = []
for i in res.split(os.linesep):
    line = i.strip()
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
