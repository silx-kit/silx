# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017 European Synchrotron Radiation Facility
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
"""Sphinx extension creating a link to an archive.

This Sphinx extension adds the possibility to create a zip archive and
include it in the generated HTML documentation.

This extension provides an *archive* directive.

Usage:

.. archive:: <relative_path/to/directory/to/compress>
   :filename: <name_of_the_archive_file, default: last_folder_of_directory.zip>
   :filter: <Space-separated list of included file patterns, default *.*>
   :basedir: <Name of the base directory in the archive, default: filename>

Examples:

To create a example.zip archive of the ../../example/ folder:

.. archive:: ../../examples/

To get more control on the name of the archive and its content:

.. archive:: ../../examples/
   :filename: myproject_examples.zip
   :filter: *.py *.png
   :basedir: myproject_examples

WARNING: The content of this directory is not checked for outdated documents.
"""
__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "01/09/2017"


import fnmatch
import logging
import os
import os.path
import posixpath
import shutil
import tempfile

from docutils.parsers.rst import directives, Directive
import docutils.nodes


_logger = logging.getLogger(__name__)


# docutils directive

class ArchiveDirective(Directive):
    """Add a link to download an archive

    This directive add an :class:`archive` node to the document tree.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        'filename': directives.unchanged,
        'filter': directives.unchanged,
        'basedir': directives.unchanged,
    }

    def run(self):
        node = archive('')

        # Get rst source file containing this directive
        source_file = self.state_machine.get_source_and_line()[0]
        if source_file is None:
            raise RuntimeError('Cannot get rst source file path')

        # Build input path from rst source file and directive argument
        input_path = self.arguments[0]
        if not input_path.startswith('/'):  # Argument is a relative path
            input_path = os.path.abspath(
                os.path.join(os.path.dirname(source_file), input_path))
        node['input_path'] = input_path

        default_basedir = os.path.basename(input_path)
        node['basedir'] = self.options.get('basedir', default_basedir)
        node['filename'] = self.options.get('filename',
                                            '.'.join((default_basedir, 'zip')))

        node['filter'] = self.options.get('filter', '*.*')

        return [node]


# archive doctuils node

class archive(docutils.nodes.General, docutils.nodes.Element, docutils.nodes.Inline):
    """archive node created by :class:`ArchiveDirective`"""
    pass


def visit_archive_html(self, node):
    """Node visitor to translate :class:`archive` nodes to HTML.

    :param self: Sphinx HTML Writter
    :param node: The :class:`archive` node to translate to HTML
    :raise: SkipNode as depart is not implemented
    """
    filename = node['filename']
    input_path = node['input_path']

    # Create a temporary folder to create archive content
    tmp_dir = tempfile.mkdtemp()

    # Copy selected content to temporary folder
    base_dir = os.path.join(tmp_dir, node['basedir'])

    def ignore(src, names):
        patterns = node['filter'].split()
        ignored_names = []
        for name in names:
            for pattern in patterns:
                if fnmatch.fnmatch(name, pattern):
                    break
            else:
                ignored_names.append(name)
        return ignored_names

    shutil.copytree(input_path, base_dir, ignore=ignore)

    # Compress temporary folder to zip
    output_filename = os.path.join(
        self.builder.outdir, '_downloads', filename)
    root, ext = os.path.splitext(output_filename)
    assert ext == '.zip'
    shutil.make_archive(root, 'zip', tmp_dir, node['basedir'])

    # Clean-up temporary folder
    shutil.rmtree(tmp_dir)

    # Generate HTML
    relative_path = posixpath.join(self.builder.dlpath, filename)
    self.body.append('<a href="%s">%s</a>' % (relative_path, filename))
    raise docutils.nodes.SkipNode


def visit_skip(self, node):
    """No-op node visitor"""
    raise docutils.nodes.SkipNode


# Extension setup

def setup(app):
    """Sphinx extension registration"""
    app.add_node(archive,
                 html=(visit_archive_html, None),
                 latex=(visit_skip, None))

    app.add_directive('archive', ArchiveDirective)

    return {'version': '0.1'}
