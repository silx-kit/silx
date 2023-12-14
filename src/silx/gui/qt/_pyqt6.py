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


_logger = logging.getLogger(__name__)


def patch_enums(*modules):
    """Patch PyQt6 modules to provide backward compatibility of enum values

    :param modules: Modules to patch (e.g., PyQt6.QtCore).
    """
    for module in modules:
        for clsName in dir(module):
            cls = getattr(module, clsName, None)
            if not isinstance(cls, PyQt6.sip.wrappertype) or not clsName.startswith(
                "Q"
            ):
                continue

            for qenumName in dir(cls):
                if not qenumName[0].isupper():
                    continue
                # Special cases to avoid overrides and mimic PySide6
                if clsName == "QColorSpace" and qenumName in (
                    "Primaries",
                    "TransferFunction",
                ):
                    continue
                if qenumName in ("DeviceType", "PointerType"):
                    continue

                qenum = getattr(cls, qenumName)
                if not isinstance(qenum, enum.EnumMeta):
                    continue

                if any(
                    map(
                        lambda ancestor: isinstance(ancestor, PyQt6.sip.wrappertype)
                        and qenum is getattr(ancestor, qenumName, None),
                        cls.__mro__[1:],
                    )
                ):
                    continue  # Only handle it once in case of inheritance

                for name, value in qenum.__members__.items():
                    setattr(cls, name, value)
