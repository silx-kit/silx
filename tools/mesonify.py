#!/usr/bin/python3
"""
Tool to bootstrap the structure of meson-python project.
Adds all Python files by default.
Does NOT update missing files. Manual edition is needed.
"""

import os


def mesonify(where, top=None):
    if top is None:
        top = os.path.join(where, "src")
    # print(where, "top: ", top)
    txt = []
    pyfiles = []
    ndir = 0
    for smth in os.listdir(where):
        path = os.path.join(where, smth)
        if os.path.isdir(path):
            txt.append(f"subdir('{smth}')")
            mesonify(path, top)
            ndir += 1
        elif smth.endswith(".py"):
            pyfiles.append(smth)
    if ndir:
        txt.append("")

    if pyfiles:
        pyfiles.sort()
        txt.append("")
        txt.append("py.install_sources([")
        for f in pyfiles:
            txt.append(f"    '{f}',")
        txt.append("],")
        if len(path) > len(top):
            txt.append(
                f"subdir: '{where[len(top)+1:]}',  # Folder relative to site-packages to install to"
            )
        txt.append(")")
        txt.append("")

    if txt:
        dst = os.path.join(where, "meson.build")
        if os.path.exists(dst):
            print(f"Meson file `{dst}` already exist, not overwriting it!")
        else:
            print(f"Generating Meson file `{dst}`.")
            with open(dst, "w") as w:
                w.write("\n".join(txt))


if __name__ == "__main__":
    base = os.path.abspath(os.path.join(__file__, "..", "..", "src"))
    print("Start working at `{base}`")
    mesonify(base, base)
