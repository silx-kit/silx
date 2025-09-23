How to add an icon to silx
==========================

Icons are stored in the `src/silx/resources/gui/icons <https://github.com/silx-kit/silx/tree/main/src/silx/resources/gui/icons>`_ folder in both SVG and PNG format.

Create a SVG icon 
-----------------

Use `inkscape`_ to create a SVG icon with the following constraints:

- The SVG `viewBox` should be a 32x32 square
- There should be no embed images (png or jpeg)
- No external resources such as fonts should be used: Convert text to paths using `inkscape`_'s "Path/Object to Path" menu.

Save the icon as "Optimized SVG" without compression.
Then, optimize the file with the `tools/optimize_svg.sh <https://github.com/silx-kit/silx/blob/main/tools/optimize_svg.sh>`_ script.

Create a PNG icon
-----------------

The `tools/export_svg.sh <https://github.com/silx-kit/silx/blob/main/tools/export_svg.sh>`_ script converts SVG files to PNG files with the same name::

  tools/export_svg.sh myicon.svg

Make sure that the produced PNG file:

- has a transparent background
- has a size of 32x32 pixels

.. note::

  It is also possible to export the SVG file as a PNG file using `inkscape`_'s "File/Export..." menu.

Add the icon files to silx
--------------------------

Add both files to the `src/silx/resources/gui/icons <https://github.com/silx-kit/silx/tree/main/src/silx/resources/gui/icons>`_ folder.

Run the `tools/update_icons_rst.py <https://github.com/silx-kit/silx/blob/main/tools/update_icons_rst.py>`_ script to update the `documentation page <http://www.silx.org/doc/silx/latest/modules/gui/icons.html#available-icons>`_.


.. _inkscape: https://inkscape.org/
