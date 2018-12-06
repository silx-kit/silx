"""RST directive to include snapshot of a Qt application in Sphinx doc.

Configuration variable in conf.py:

- snapshotqt_image_type: image file extension (default 'png').
- snapshotqt_script_dir: relative path of the root directory for scripts from
  the documentation source directory (i.e., the directory of conf.py)
  (default: '..').
"""
from __future__ import absolute_import
import os
import logging
import subprocess
import sys
import distutils
import shutil
from docutils.parsers.rst.directives.images import Image
from docutils.parsers.rst import Directive
from docutils.nodes import fully_normalize_name
from docutils import nodes

# from docutils.par
# note: conf.py is patching the PATH so this will be the 'current' qt version

def _distutils_dir_name(dname="lib"):
    """
    Returns the name of a distutils build directory
    """
    platform = distutils.util.get_platform()
    architecture = "%s.%s-%i.%i" % (dname, platform,
                                    sys.version_info[0], sys.version_info[1])
    return architecture

home = os.path.abspath(os.path.join(__file__, "..", "..", "..", '..'))
home = os.path.abspath(home)
LIBPATH = os.path.join(home, 'build', _distutils_dir_name('lib'))

if not os.path.exists(LIBPATH):
    raise RuntimeError("%s is not on the path. Fix your PYTHONPATH and restart sphinx." % project)

sys.path.append(LIBPATH)
env = os.environ.copy()
env.update(
    {"PYTHONPATH": LIBPATH + os.pathsep + os.environ.get("PYTHONPATH", ""),
     "PATH": os.environ.get("PATH", "")})
import silx


if not os.environ.get('SILX_GENERATE_SCREENSHOT') == 'True':
    """
    In case we don't wan't to regenerate screenshot, simply apply Figure
    directive
    """
    class SnapshotQtDirective(Image):
        option_spec = Image.option_spec.copy()
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
    
            .. snapshotqt: ../examples/demo.py
               :align: center
               :height: 5cm
    
               Caption of the image.
        """
        option_spec = Image.option_spec.copy()
        has_content = True

        # TODO this should be configured in conf.py
        SNAPSHOTS_QT = os.path.join('snapshotsqt_directive')
        """The path where to store images relative to doc directory."""

        def run(self):
            assert len(self.arguments) > 0
            # Run script stored in arguments and replace by snapshot filename
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
            assert os.path.exists(os.path.dirname(image_file_source_path))

            has_source_code = False
            with open(output_script, 'w') as _file:
                _file.write("from silx.gui import qt\n")
                # _file.write("app = qt.QApplication().instance\n")
                for _line in self.content:
                    _towrite = _line.lstrip(' ')
                    if not _towrite.startswith(':'):
                        _file.write(_towrite + '\n')
                        has_source_code = True
                _file.write("app.exec_()")
            self.content = []
            if not has_source_code:
                _logger.warning('no source code defined in the snapshot'
                                 'directive, fail to generate a screenshot')
            else:
                makescreenshot(script_or_module=output_script,
                               filename=image_file_source_path)
            self.options['figwidth'] = 'image'

            #
            # Use created image as in Figure
            return super(SnapshotQtDirective, self).run()


    def setup(app):
        app.add_config_value('snapshotqt_image_type', 'png', 'env')
        app.add_config_value('snapshotqt_script_dir', '..', 'env')
        app.add_directive('snapshotqt', SnapshotQtDirective)
        return {'version': '0.1'}

    # screenshot function ########################################################

    def makescreenshot(script_or_module, filename):
        _logger.info('generate screenshot for %s from %s, binding is %s'
                     '' % (filename, script_or_module, qt.BINDING))

        # Probe Qt binding
        if qt.BINDING == 'PyQt4':
            def grabWindow(winID):
                return qt.QPixmap.grabWindow(winID)
        elif qt.BINDING == 'PyQt5':
            def grabWindow(winID):
                screen = qt.QApplication.primaryScreen()
                return screen.grabWindow(winID)
        elif qt.BINDING == 'PySide2':
            def grabWindow(winID):
                screen = qt.QApplication.primaryScreen()
                import PySide2.QtGui
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
                    _logger.error(
                        'Aborted: It took too long to have an active window.')
        script_or_module = os.path.abspath(script_or_module)
        try:
            sys.argv = [script_or_module]
            sys.path.append(
                os.path.abspath(os.path.dirname(script_or_module)))
            qt.QTimer.singleShot(_TIMEOUT, _grabActiveWindowAndClose)
            if sys.version_info < (3, ):
                execfile(script_or_module)
            else:
                with open(script_or_module) as f:
                    code = compile(f.read(), "somefile.py", 'exec')
                    exec(code)
        except Exception as e:
            _logger.error(e)


# main ########################################################################
if __name__ == '__main__':
    import argparse

    # Parse argv
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="""Arguments provided after the script or module name are passed
        to the script or module.""")
    parser.add_argument(
        '-o', '--output', nargs=1, type=str,
        default='snapshot.png',
        help='Image filename of the snapshot (default: snapshot.png).')
    parser.add_argument(
        '--bindings', nargs='?',
        choices=('PySide2', 'PyQt4', 'PyQt5', 'auto'),
        default='auto',
        help="""Qt bindings used by the script/module.
        If 'auto' (the default), it is probed from available python modules.
        """)
    parser.add_argument(
        '-m', '--module', action='store_true',
        help='Run the corresponding module as a script.')
    parser.add_argument(
        'script_or_module', nargs=1, type=str,
        help='Python script to run for the snapshot.')
    args, unknown = parser.parse_known_args()
    script_or_module = args.script_or_module[0]
    # arguments provided after the script or module
    extra_args = sys.argv[sys.argv.index(script_or_module) + 1:]
    if unknown != extra_args:
        parser.print_usage()
        _logger.error(
            '%s: incorrect arguments', os.path.basename(sys.argv[0]))
        sys.exit(1)
    makescreenshot(script_or_module, args.output[0])
