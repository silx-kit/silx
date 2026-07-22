How to add an icon to silx
==========================

Icons are stored in the `src/silx/resources/gui/icons <https://github.com/silx-kit/silx/tree/main/src/silx/resources/gui/icons>`_ folder in both SVG and PNG format.

There are three steps to add an icon:

1. Create a SVG icon
2. Export it as a PNG
3. Add the files to silx

Create a SVG icon 
-----------------

* Use `inkscape`_ to create a SVG icon with the following constraints:

  - The SVG `viewBox` should be a 32x32 square
  - There should be no embed images (png or jpeg)
  - No external resources such as fonts should be used: Convert text to paths using `inkscape`_'s "Path/Object to Path" menu.

* Save the icon as "Optimized SVG" without compression.
* Patch the SVG file to support dark color theme:
  In silx, when the dark color theme is active, ``class="dark"`` is added to the ``<svg>`` opening tag before rendering the icons.
  The ``"dark"`` class can be used to custom the SVG style through a ``<style>`` tag, for example:

  - Set ``fill`` (default: ``black``) of all SVG primitives:

    .. code-block:: xml
    
       <?xml version="1.0" encoding="UTF-8"?>
       <svg version="1.1" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
       <style>
       .dark * {
           fill: #ffffff;
       }
       </style>
       <rect x="3" y="3" width="26" height="26" rx="2"/>
       </svg>

  - Set the value of ``currentColor`` (default: ``black``) and use it where needed, e.g. for the ``stroke``:

    .. code-block:: xml

       <?xml version="1.0" encoding="UTF-8"?>
       <svg version="1.1" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
       <style>
       .dark * {
           color: #ffffff;
       }
       </style>
       <rect x="3" y="3" width="26" height="26" rx="2" fill="none" stroke="currentColor" stroke-width="2px"/>
       </svg>

Look at SVG icons in ``src/silx/resources/gui/icons/`` for more examples.

Export it as a PNG
------------------

The `tools/export_svg.sh <https://github.com/silx-kit/silx/blob/main/tools/export_svg.sh>`_ script converts SVG files to PNG files with the same name::

  tools/export_svg.sh myicon.svg

Make sure that the produced PNG file:

- has a transparent background
- has a size of 32x32 pixels

.. note::

  It is also possible to export the SVG file as a PNG file using `inkscape`_'s "File/Export..." menu.

Add the files to silx
---------------------

* Add both files to the `src/silx/resources/gui/icons <https://github.com/silx-kit/silx/tree/main/src/silx/resources/gui/icons>`_ folder. Both the SVG and PNG should be added to Git.
* Test that the new icon supports the dark color theme by using the ``examples/icons.py`` script to display all icons and change the operating system color theme.
* Run the `tools/update_icons_rst.py <https://github.com/silx-kit/silx/blob/main/tools/update_icons_rst.py>`_ script to update the `documentation page <https://silx.readthedocs.io/en/latest/modules/gui/icons.html#available-icons>`_.


.. _inkscape: https://inkscape.org/
