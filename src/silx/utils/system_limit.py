# /*##########################################################################
#
# Copyright (c) 2019 European Synchrotron Radiation Facility
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
"""Utils function relative to system limit"""

__authors__ = ["M. Ruyer"]
__license__ = "MIT"
__date__ = "05/06/2026"

import logging

_logger = logging.getLogger()


def increase_max_opened_files():
    """Use max opened files hard limit as soft limit"""
    try:
        import resource
    except ImportError:
        _logger.debug("No resource module available")
    else:
        if hasattr(resource, "RLIMIT_NOFILE"):
            try:
                hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
                resource.setrlimit(resource.RLIMIT_NOFILE, (hard_nofile, hard_nofile))
            except (ValueError, OSError):
                _logger.warning("Failed to retrieve and set the max opened files limit")
            else:
                _logger.debug("Set max opened files to %d", hard_nofile)
