# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2021 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/
"""PyQt6 backward compatibility patching"""

__authors__ = ["Thomas VINCENT"]
__license__ = "MIT"
__date__ = "02/09/2021"

import enum
import logging

import PyQt6.sip
from PyQt6.QtCore import Qt


_logger = logging.getLogger(__name__)


def patch_enums(*modules):
    """Patch PyQt6 modules to provide backward compatibility of enum values

    :param modules: Modules to patch (e.g., PyQt6.QtCore).
    """
    for module in modules:
        for clsName in dir(module):
            cls = getattr(module, clsName, None)
            if isinstance(cls, PyQt6.sip.wrappertype) and clsName.startswith('Q'):
                for qenumName in dir(cls):
                    if qenumName[0].isupper():
                        qenum = getattr(cls, qenumName, None)
                        if isinstance(qenum, enum.EnumMeta):
                            if qenum is getattr(cls.__mro__[1], qenumName, None):
                                continue  # Only handle it once
                            for item in qenum:
                                # Special cases to avoid overrides and mimic PySide6
                                if clsName == 'QColorSpace' and qenumName in (
                                        'Primaries', 'TransferFunction'):
                                    break
                                if qenumName in ('DeviceType', 'PointerType'):
                                    break

                                setattr(cls, item.name, item)
