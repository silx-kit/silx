#!/usr/bin/python3

import os

def mesonify(where, top=None):    
    if top==None:
        top = os.path.join(where,"src")
    # print(where, "top: ", top)
    txt = []
    pyfiles = []
    ndir = 0
    for smth in os.listdir(where):
        path = os.path.join(where,smth)
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
        txt.append("pure: false,    # Will be installed next to binaries")
        if len(path)>len(top):
            txt.append(f"subdir: '{where[len(top)+1:]}',  # Folder relative to site-packages to install to")
        txt.append(")")
        txt.append("")

    dst = os.path.join(where,"meson.build")
    print(dst)
    with open(dst,"w") as w:
        w.write("\n".join(txt))

if __name__ == "__main__":
    base = os.path.abspath(os.path.join(__file__, "..", "..", "src"))
    print(base)
    mesonify(base, base)
