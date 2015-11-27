# coding: utf-8
from __future__ import absolute_import, print_function, division

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "27/11/2015"

try:
    from ._version import __date__ as date
    from ._version import version, version_info, hexversion, strictversion
except ImportError:
    raise RuntimeError("Do NOT use silx from its sources: build it and use the built version")
