# coding: utf-8
from __future__ import absolute_import, print_function, division

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "28/01/2016"

import os
project = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
try:
    from ._version import __date__ as date
    from ._version import version, version_info, hexversion, strictversion
except ImportError:
    raise RuntimeError("Do NOT use %s from its sources: build it and use the built version" % project)
