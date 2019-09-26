# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2004-2019 European Synchrotron Radiation Facility
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
"""RST directive to include snapshot of a Qt application in Sphinx doc.

Configuration variable in conf.py:

- snapshotqt_image_type: image file extension (default 'png').
- snapshotqt_script_dir: relative path of the root directory for scripts from
  the documentation source directory (i.e., the directory of conf.py)
  (default: '..').
"""
from __future__ import absolute_import

__authors__ = ["H. Payno", "T. Vincent"]
__license__ = "MIT"
__date__ = "07/12/2018"

import os
import logging
import sys
from docutils.parsers.rst.directives.images import Image
from docutils.parsers.rst import directives

# from docutils.par
# note: conf.py is patching the PATH so this will be the 'current' qt version

home = os.path.abspath(os.path.join(__file__, "..", "..", "..", '..'))


if not os.environ.get('DIRECTIVE_SNAPSHOT_QT') == 'True':
    """
    In case we don't wan't to regenerate screenshot, simply apply Figure
    directive
    """
    class SnapshotQtDirective(Image):
        option_spec = Image.option_spec.copy()
        option_spec['script'] = directives.unchanged
        has_content = True

        def run(self):
            self.options['figwidth'] = 'image'
            self.content = []

            # Create an image filename from arguments
            return Image.run(self)

    def makescreenshot(*args, **kwargs):
        raise RuntimeError('not defined without env variable SILX_GENERATE_SCREENSHOT set to True')

    def setup(app):
        app.add_config_value('snapshotqt_image_type', 'png', 'env')
        app.add_config_value('snapshotqt_script_dir', '..', 'env')
        app.add_directive('snapshotqt', SnapshotQtDirective)
        return {'version': '0.1'}

else:
    from silx.gui import qt

    logging.basicConfig()
    _logger = logging.getLogger(__name__)

    # RST directive ###############################################################

    class SnapshotQtDirective(Image):
        """Image of a Qt application snapshot.

        Directive Type: "snapshotqt"
        Doctree Elements: As for figure
        Directive Arguments: One or more, required (script URI + script arguments).
        Directive Options: Possible.
        Directive Content: Interpreted as the figure caption and optional legend.

        A "snapshotqt" is a rst `figure
        <http://docutils.sourceforge.net/docs/ref/rst/directives.html#figure>`_
        that is generated from a Python script that uses Qt.

        The path of the script to take a snapshot is relative to
        the path given in conf.py 'snapshotqt_script_dir' value.

        ::

            .. snapshotqt: img/demo.py
               :align: center
               :height: 5cm

               source code


        you can also define a snapshot from a script, using the :script: option
        .. note:: on this path are given from the project root level

        ::
            .. snapshotqt: img/demo.py
               :align: center
               :height: 5cm
               :script: myscript.py
        """
        option_spec = Image.option_spec.copy()
        option_spec['script'] = directives.unchanged
        has_content = True

        def run(self):
            assert len(self.arguments) > 0
            # Run script stored in arguments and replace by snapshot filename
            script = self.options.pop('script', None)
            env = self.state.document.settings.env

            image_ext = env.config.snapshotqt_image_type.lower()
            script_name = self.arguments[0].replace(image_ext, 'py')
            output_script = os.path.join(env.app.outdir, script_name)

            image_file_source_path = env.relfn2path(self.arguments[0])[0]
            image_file_source_path = os.path.join(home, env.srcdir, image_file_source_path)

            def createNeededDirs(_dir):
                parentDir = os.path.dirname(_dir)
                if parentDir not in ('', os.sep):
                    createNeededDirs(parentDir)
                if os.path.exists(_dir) is False:
                    os.mkdir(_dir)

            createNeededDirs(os.path.dirname(output_script))

            has_source_code = not (self.content is None or len(self.content) is 0)
            if has_source_code:
                with open(output_script, 'w') as _file:
                    _file.write("# from silx.gui import qt\n")
                    _file.write("# app = qt.QApplication([])\n")
                    for _line in self.content:
                        _towrite = _line.lstrip(' ')
                        if not _towrite.startswith(':'):
                            _file.write(_towrite + '\n')
                    _file.write("app.exec_()")
                self.content = []
                if script is not None:
                    _logger.warning('Cannot specify a script if source code (content) is given.'
                                    'Ignore script option')
                makescreenshot(script_or_module=output_script,
                               filename=image_file_source_path)
            else:
                # script
                if script is None:
                    _logger.warning('no source code or script defined in the snapshot'
                                    'directive, fail to generate a screenshot')
                else:
                    script_path = os.path.join(home, script)
                    makescreenshot(script_or_module=script_path,
                                   filename=image_file_source_path)

            #
            # Use created image as in Figure
            return super(SnapshotQtDirective, self).run()

    def setup(app):
        app.add_config_value('snapshotqt_image_type', 'png', 'env')
        app.add_config_value('snapshotqt_script_dir', '..', 'env')
        app.add_directive('snapshotqt', SnapshotQtDirective)
        return {'version': '0.1'}

    # screensImageFileDialogH5.hot function ########################################################

    def makescreenshot(script_or_module, filename):
        _logger.info('generate screenshot for %s from %s, binding is %s'
                     '' % (filename, script_or_module, qt.BINDING))

        def grabWindow(winID):
            screen = qt.QApplication.primaryScreen()
            return screen.grabWindow(winID)

        global _count
        _count = 15
        global _TIMEOUT
        _TIMEOUT = 1000.  # in ms
        app = qt.QApplication.instance() or qt.QApplication([])
        _logger.debug('Using Qt bindings: %s', qt)

        def _grabActiveWindowAndClose():
            global _count
            activeWindow = qt.QApplication.activeWindow()
            if activeWindow is not None:
                if activeWindow.isVisible():
                    # hot fix since issue with pySide2 API
                    if qt.BINDING == 'PySide2':
                        pixmap = activeWindow.grab()
                    else:
                        pixmap = grabWindow(activeWindow.winId())
                    saveOK = pixmap.save(filename)
                    if not saveOK:
                        _logger.error(
                            'Cannot save snapshot to %s', filename)
                else:
                    _logger.error('activeWindow is not visible.')
                app.quit()
            else:
                _count -= 1
                if _count > 0:
                    # Only restart a timer if everything is OK
                    qt.QTimer.singleShot(_TIMEOUT,
                                         _grabActiveWindowAndClose)
                else:
                    app.quit()
                    raise TimeoutError(
                        'Aborted: It took too long to have an active window.')
        script_or_module = os.path.abspath(script_or_module)

        sys.argv = [script_or_module]
        sys.path.append(
            os.path.abspath(os.path.dirname(script_or_module)))
        qt.QTimer.singleShot(_TIMEOUT, _grabActiveWindowAndClose)
        with open(script_or_module) as f:
            code = compile(f.read(), script_or_module, 'exec')
            exec(code, globals(), locals())
