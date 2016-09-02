#!/usr/bin/env python
# coding: utf-8
"""Print information about python."""

__authors__ = ["Jérôme Kieffer"]
__date__ = "02/09/2016"
__license__ = "MIT"


import sys

print("Python %s bits" % (tuple.__itemsize__ * 8))
print("       maxsize: %s\t maxunicode: %s" % (sys.maxsize, sys.maxunicode))
print(sys.version)
print(" ")

try:
    from distutils.sysconfig import get_config_vars
except ImportError:
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
    for p in pyopencl.get_platforms():
        print("  %s" % p)
        for d in p.get_devices():
            print("    %s" % d)
