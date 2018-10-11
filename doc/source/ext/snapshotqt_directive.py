"""RST directive to include snapshot of a Qt application in Sphinx doc.

Configuration variable in conf.py:

- snapshotqt_image_type: image file extension (default 'png').
- snapshotqt_script_dir: relative path of the root directory for scripts from
  the documentation source directory (i.e., the directory of conf.py)
  (default: '..').
"""

from __future__ import absolute_import

import logging
import os
import subprocess
import sys

from docutils.parsers.rst.directives.images import Figure

logging.basicConfig()
_logger = logging.getLogger(__name__)

# TODO:
# - Check if it is needed to patch block_text?

# RST directive ###############################################################

class SnapshotQtDirective(Figure):
    """Figure of a Qt application snapshot.

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

           Caption of the figure.
    """

    # TODO this should be configured in conf.py
    SNAPSHOTS_QT = os.path.join('snapshotsqt_directive')
    """The path where to store images relative to doc directory."""

    def run(self):
        def createNeededDirs(_dir):
            parentDir = os.path.dirname(_dir)
            if parentDir not in ('', os.sep):
                createNeededDirs(parentDir)
            if os.path.exists(_dir) is False:
                os.mkdir(_dir)

        # Run script stored in arguments and replace by snapshot filename

        env = self.state.document.settings.env

        # Create an image filename from arguments
        image_ext = env.config.snapshotqt_image_type.lower()
        image_name = '_'.join(self.arguments) + '.' + image_ext
        image_name = image_name.replace('./\\', '_')
        image_name = ''.join([c for c in image_name
                              if c.isalnum() or c in '_-.'])
        snapshot_dir = os.path.join(env.app.outdir, self.SNAPSHOTS_QT)
        image_name = os.path.join(snapshot_dir, image_name)
        createNeededDirs(os.path.dirname(image_name))
        assert os.path.isdir(snapshot_dir)

        # Change path to absolute path to run the script
        script_dir = os.path.join(env.srcdir, env.config.snapshotqt_script_dir)
        script_cmd = self.arguments[:]
        script_cmd[0] = os.path.join(script_dir, script_cmd[0])

        # Run snapshot
        snapshot_tool = os.path.abspath(__file__)
        _logger.info('Running script: %s', script_cmd)
        _logger.info('Saving snapshot to: %s', image_name)
        abs_image_name = os.path.join(env.srcdir, image_name)
        cmd = [sys.executable, snapshot_tool, '--output', abs_image_name]
        cmd += script_cmd
        subprocess.check_call(cmd)

        # Use created image as in Figure
        self.arguments = [os.sep + image_name]
        return super(SnapshotQtDirective, self).run()


def setup(app):
    app.add_config_value('snapshotqt_image_type', 'png', 'env')
    app.add_config_value('snapshotqt_script_dir', '../..', 'env')
    app.add_directive('snapshotqt', SnapshotQtDirective)
    return {'version': '0.1'}


# Qt monkey-patch #############################################################

def monkeyPatchQApplication(filename, qt=None):
    """Monkey-patch QApplication to take a snapshot and close the application.

    :param str filename: The image filename where to save the snapshot.
    :param str qt: The Qt binding to patch.
                   This MUST be the same as the one used by the script
                   for which to take a snapshot.
                   In: 'PyQt4', 'PyQt5', 'PySide2' or None (the default).
                   If None, it will try to use PyQt4, then PySide2 and
                   finally PyQt4.
    """

    # Probe Qt binding
    if qt is None:
        try:
            import PyQt4.QtCore  # noqa
            qt = 'PyQt4'
        except ImportError:
            try:
                import PySide2.QtCore  # noqa
                qt = 'PySide2'
            except ImportError:
                try:
                    import PyQt5.QtCore  # noqa
                    qt = 'PyQt5'
                except ImportError:
                    raise RuntimeError('Cannot find any supported Qt binding.')

    if qt == 'PyQt4':
        from PyQt4.QtGui import QApplication, QPixmap
        from PyQt4.QtCore import QTimer
        import PyQt4.QtGui as _QApplicationPackage

        def grabWindow(winID):
            return QPixmap.grabWindow(winID)

    elif qt == 'PyQt5':
        from PyQt5.QtGui import QPixmap
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QTimer
        import PyQt5.QtWidgets as _QApplicationPackage

        def grabWindow(winID):
            screen = QApplication.primaryScreen()
            return screen.grabWindow(winID)

    elif qt == 'PySide2':
        from PySide2.QtGui import QApplication, QPixmap
        from PySide2.QtCore import QTimer
        import PySide2.QtGui as _QApplicationPackage

        def grabWindow(winID):
            return QPixmap.grabWindow(winID)

    else:
        raise ValueError('Unsupported Qt binding: %s' % qt)

    _logger.info('Using Qt bindings: %s', qt)

    class _QApplication(QApplication):

        _TIMEOUT = 1000.
        _FILENAME = filename

        def _grabActiveWindowAndClose(self):
            activeWindow = QApplication.activeWindow()
            if activeWindow is not None:
                if activeWindow.isVisible():
                    pixmap = grabWindow(activeWindow.winId())
                    saveOK = pixmap.save(self._FILENAME)
                    if not saveOK:
                        _logger.error(
                            'Cannot save snapshot to %s', self._FILENAME)
                else:
                    _logger.error('activeWindow is not visible.')
                self.quit()
            else:
                self._count -= 1
                if self._count > 0:
                    # Only restart a timer if everything is OK
                    QTimer.singleShot(self._TIMEOUT,
                                      self._grabActiveWindowAndClose)
                else:
                    raise RuntimeError(
                        'Aborted: It took too long to have an active window.')

        def exec_(self):
            self._count = 10
            QTimer.singleShot(self._TIMEOUT, self._grabActiveWindowAndClose)

            return super(_QApplication, self).exec_()

    _QApplicationPackage.QApplication = _QApplication


# main ########################################################################

if __name__ == '__main__':
    import argparse
    import runpy

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

    # Monkey-patch Qt
    monkeyPatchQApplication(args.output[0],
                            args.bindings if args.bindings != 'auto' else None)

    # Update sys.argv and sys.path
    sys.argv = [script_or_module] + extra_args
    sys.path.insert(0, os.path.abspath(os.path.dirname(script_or_module)))

    if args.module:
        _logger.info('Running module: %s', ' '.join(sys.argv))
        runpy.run_module(script_or_module, run_name='__main__')

    else:
        with open(script_or_module) as f:
            code = f.read()

        _logger.info('Running script: %s', ' '.join(sys.argv))
        exec(code)
