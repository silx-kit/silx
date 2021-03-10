# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2015-2019 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import absolute_import, print_function, division

"""Provide convenient URL for silx-kit projects."""

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/01/2019"


from ... import _version as version

BASE_DOC_URL = None
"""This could be patched by project packagers."""

_DEFAULT_BASE_DOC_URL = "http://www.silx.org/pub/doc/silx/{silx_doc_version}/{subpath}"
"""Identify the base URL of the project documentation.

It supportes string replacement:

- `{major}` the major version
- `{minor}` the minor version
- `{micro}` the micro version
- `{relev}` the status of the version (dev, final, rc).
- `{silx_doc_version}` is used to map the documentation stored at www.silx.org
- `{subpath}` is the subpart of the URL pointing to a specific page of the
    documentation. It is mandatory.
"""


def getDocumentationUrl(subpath):
    """Returns the URL to the documentation"""

    if version.RELEV == "final":
        # Released verison will point to a specific documentation
        silx_doc_version = "%d.%d.%d" % (version.MAJOR, version.MINOR, version.MICRO)
    else:
        # Dev versions will point to a single 'dev' documentation
        silx_doc_version = "dev"

    keyworks = {
        "silx_doc_version": silx_doc_version,
        "major": version.MAJOR,
        "minor": version.MINOR,
        "micro": version.MICRO,
        "relev": version.RELEV,
        "subpath": subpath}
    template = BASE_DOC_URL
    if template is None:
        template = _DEFAULT_BASE_DOC_URL
    return template.format(**keyworks)
