Trouble shooting
================

Using OpenGL through ssh
------------------------

``silx view`` can use OpenGL>=2.1:

- for the "3D" visualization tab
- when the plot rendering backend was set to OpenGL in the Options menu.

If ``silx view`` was set to use its OpenGL plot rendering backend beforehand,
it is possible to disable it by running ``silx view --fresh``.

Some widgets in :mod:`silx.gui` are using OpenGL>=2.1:

- Widgets in :mod:`silx.gui.plot3d`, and
- The OpenGL backend of :class:`~silx.gui.plot.PlotWidget` and related widgets in :mod:`silx.gui.plot`.

OpenGL-based applications can run over SSH only under specific conditions:

- Use ``ssh -X`` to enable X11 forwarding.
- The operating system must enable OpenGL through X11 forwarding. This is not the case by default on Debian/Ubuntu. See `Enabling OpenGL forwarding`_.
- Depending on the GPU drivers installed on both the server and the client computers only OpenGL1.4 may be available and the display manager may be unstable.

To get the currently available version of OpenGL, run from the command line::

  glxinfo | grep "OpenGL version string"

On Debian/Ubuntu, ``glxinfo`` is available as part of the ``mesa-utils`` package.

If you encounter issues with OpenGL through ssh, try accessing the computer with a remote desktop solution instead.

Enabling OpenGL forwarding
..........................

"Indirect GLX" must be enabled on the local computer.
If it is disabled, setting it up requires root access.
The way to set it up depends on the configuration of the system (the operating system and the display manager).
