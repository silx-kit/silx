#!/usr/bin/env python
"""Print information about python."""

__authors__ = ["Jérôme Kieffer"]
__date__ = "09/09/2016"
__license__ = "MIT"

import sys
import platform
import subprocess


print("Python %s bits" % (tuple.__itemsize__ * 8))
print(f"       maxsize: {sys.maxsize}\t maxunicode: {sys.maxunicode}")
print(sys.version)
print(" ")

print("Platform: " + platform.platform())
print("- Machine: " + platform.machine())
print(" ")

from sysconfig import get_config_vars

print("Config: " + str(get_config_vars("CONFIG_ARGS")))
print("")

try:
    import numpy
except ImportError:
    print("Numpy not installed")
else:
    print("Numpy %s" % numpy.version.version)
    print("      include %s" % numpy.get_include())
    print("      options %s" % numpy.get_printoptions())
print("")
try:
    import pyopencl
except Exception as error:
    print("Unable to import pyopencl: %s" % error)
else:
    print("PyOpenCL platform:")
    try:
        cl_platforms = pyopencl.get_platforms()
    except pyopencl.LogicError:
        print("The module pyOpenCL has been imported but get_platforms failed")
    else:
        for p in cl_platforms:
            print("  %s" % p)
            for d in p.get_devices():
                print(f"    {d} max_workgroup_size is {d.max_work_group_size}")
try:
    from silx.opencl import ocl
except Exception:
    print("Unable to import silx")
else:
    print("PyOpenCL platform as seen by silx:")
    if ocl:
        for p in ocl.platforms:
            print("  %s:" % p)
            for d in p.devices:
                print(f"    {d} max_workgroup_size is {d.max_work_group_size}")


for binding_name in ("PyQt5", "PySide6", "PyQt6"):
    # Check Qt version in subprocess to avoid issues with importing multiple Qt bindins
    cmd = [
        sys.executable,
        "-c",
        "import {0}.QtCore; print({0}.QtCore.qVersion())".format(binding_name),
    ]
    try:
        version = subprocess.check_output(cmd, timeout=4).decode("ascii").rstrip("\n")
    except subprocess.CalledProcessError:
        print(f"{binding_name}: Not available")
    else:
        print(f"{binding_name}: Qt version {version}")
