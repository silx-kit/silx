# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
"""Script to update icons.rst file according to icons available in resources.
"""

__authors__ = ["Thomas Vincent"]
__license__ = "MIT"
__date__ = "30/06/2016"


import os
import glob


ICONS_RST_FILENAME = os.path.join(os.path.dirname(__file__), 'icons.rst')


ICONS_DIR = os.path.join(
    os.path.dirname(__file__),
    '..', '..', '..', '..', 'silx', 'resources', 'gui', 'icons', '*.png')


ICONS_RST_HEADER = """
.. AUTOMATICALLY GENERATED FILE DO NOT EDIT
   Use %s script instead

.. currentmodule:: silx.gui

:mod:`icons`: Set of icons
--------------------------

.. automodule:: silx.gui.icons
   :members:

Available icons
+++++++++++++++

.. list-table::
   :widths: 1 4
   :header-rows: 1

   * - Icon
     - Name
""" % os.path.basename(__file__)


def main():
    """Write icons.rst file"""
    icons = glob.glob(ICONS_DIR)
    icons = [os.path.relpath(f, os.path.dirname(__file__)) for f in icons]
    icons = sorted(icons)

    icons_table = '\n'.join(
        '   * - |%s|\n     - %s' % (os.path.basename(f)[:-4],
                                    os.path.basename(f)[:-4]) for f in icons)

    icon_definitions = '\n'.join(
        '.. |%s| image:: %s' % (os.path.basename(f)[:-4], f) for f in icons)

    content = ICONS_RST_HEADER + icons_table + '\n\n' + icon_definitions + '\n'

    # Write to file
    with open(ICONS_RST_FILENAME, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    main()
