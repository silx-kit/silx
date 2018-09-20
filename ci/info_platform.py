#!/usr/bin/env python
# coding: utf-8
"""Print information about python."""

__authors__ = ["Jérôme Kieffer"]
__date__ = "09/09/2016"
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
            print("    %s max_workgroup_size is %s" % (d, d.max_work_group_size))
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
                print("    %s max_workgroup_size is %s" % (d, d.max_work_group_size))

have_qt_binding = False

try:
    import PyQt5.QtCore
    have_qt_binding = True
    print("Qt (from PyQt5): %s" % PyQt5.QtCore.qVersion())
except ImportError:
    pass

try:
    import PyQt4.QtCore
    have_qt_binding = True
    print("Qt (from PyQt4): %s" % PyQt4.QtCore.qVersion())
except ImportError:
    pass

try:
    import PySide2.QtCore
    have_qt_binding = True
    print("Qt (from PySide2): %s" % PySide2.QtCore.qVersion())
except ImportError:
    pass

try:
    import PySide.QtCore
    have_qt_binding = True
    print("Qt (from PySide): %s" % PySide.QtCore.qVersion())
except ImportError:
    pass

if not have_qt_binding:
    print("No Qt binding")

try:
    import sip
    print("SIP: %s" % sip.SIP_VERSION_STR)
except ImportError:
    pass
